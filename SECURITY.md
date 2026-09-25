# Security Notes

SupportPilot AI is a portfolio project, not a production system. This document describes the guardrails that ARE implemented, and is explicit about what is NOT — both matter in an interview setting.

## Threat model

The primary threat modeled is **prompt injection**: a customer message, or content later added to the knowledge base, attempting to make the agent take an unauthorized action (create a ticket, claim a refund happened, reveal system instructions) or leak internal configuration.

## Guardrails implemented

1. **Untrusted data is fenced and re-fenced.** Customer messages, retrieved documents, and tool results are wrapped in `<customer_message>`, `<retrieved_documents>`, `<tool_results>` tags. Any of those tag strings appearing *inside* the untrusted text is stripped first (`guardrails.neutralize`), so injected text cannot forge a closing tag and "escape" into the instruction context. See `tests/test_guardrails.py::test_document_text_cannot_close_the_fence`.
2. **Two-layer injection detection.** An LLM classifier (`understand_query`) flags `injection_suspected`; a regex-based heuristic (`guardrails.heuristic_injection_signal`) backs it up and can only turn the flag ON, never off. When either fires, `decide_action` **skips the decision LLM entirely** and forces `action=answer` — no tool is even considered, regardless of what the model might have been tricked into deciding. See `test_a_fooled_model_still_cannot_create_a_ticket_without_approval`.
3. **Policy is enforced in code, after the LLM.** `_enforce_policy()` in `nodes.py` overrides the model's decision in two cases the business cannot allow to depend on a model's judgment: sensitive requests always route to human escalation, and an order ID never typed by the customer is never looked up (`_order_id_in_query`).
4. **Every write action requires human approval, enforced twice.** First, structurally: there is no graph edge from `decide_action` to `execute_tool` for a ticket creation — it must pass through `human_review`. Second, defensively: `execute_tool_node` independently checks `review_status == "approved"` before running any tool that `requires_approval()`, so even a graph-wiring mistake could not bypass approval. See `test_execute_tool_node_refuses_an_unapproved_ticket`.
5. **Tool inputs are validated by an allowlist, not trusted from the LLM.** `execute_tool()` is the only entry point to any tool; unknown tool names are rejected, and arguments are validated by a strict Pydantic model (`extra="forbid"`) before any tool code runs.
6. **Output is validated two ways before being shown to the customer.** An LLM validator checks relevance and groundedness; deterministic checks (`guardrails.deterministic_answer_issues`) independently catch invented ticket/order IDs (cross-referenced against actual tool results), leaked API-key patterns, and leaked system-prompt fragments. A failed validation triggers exactly one redraft, then escalation to a human — never an infinite loop, and never a silent pass.
7. **No secrets in logs.** A `RedactingFilter` masks API-key-shaped strings, `key=`/`token=`/`password=` assignments, and email addresses on every log record, application-wide.
8. **API-layer hardening.** Request bodies are capped (`413` over ~10KB), unknown JSON fields are rejected (`422`), input length and control characters are validated via Pydantic, and every unhandled exception returns a generic `500` with a request ID — never a stack trace or internal detail.
9. **Ticket creation is idempotent per conversation thread.** A double-submitted approval cannot create two tickets (`test_double_approval_cannot_create_two_tickets`).

## Explicitly NOT guaranteed

- **These defenses reduce risk; they do not guarantee complete protection against prompt injection.** A sufficiently novel adversarial prompt could still fool the classifier. Defense-in-depth (guardrail #4) is what limits the *impact* of a missed detection, not detection itself.
- **No authentication or authorization.** Anyone who can reach the API or dashboard can send messages and approve/reject actions. This project has no concept of a support-agent identity.
- **Approval state is in-memory (`InMemorySaver`), single-process.** A backend restart loses all pending reviews. `SqliteSaver` (or a Postgres checkpointer) would be the production swap for durability, and would also be required to run more than one uvicorn worker.
- **The request-body size limit reads `Content-Length`** and can be bypassed by a client using chunked transfer encoding. A production deployment would enforce this at a reverse proxy / gateway layer instead.
- **Groundedness evaluation is keyword-based, not semantic.** It cannot detect a subtly wrong paraphrase that happens to include the right keywords, nor penalize a correct answer phrased without them.
- **Rate limiting is per-call-retry only** (one retry after a transient failure), not a request-level rate limiter. This project relies on Gemini's own quota as the outer bound.
- **The knowledge base is static and synthetic.** Poisoning tests (`test_a_fooled_model_still_cannot_create_a_ticket_without_approval`, `test_retrieved_text_is_fenced_and_cannot_close_the_fence`) use hand-crafted adversarial content, not a red-team-sourced corpus.

## Reporting

This is a portfolio project with no production deployment; there is no disclosure program. Questions about the design are welcome via the GitHub repo.