# SupportPilot AI

An agentic customer support assistant for a fictional SaaS company ("TaskNest"), built to demonstrate an AI Forward Deployed Engineer / AI Engineer workflow: a genuine LangGraph agent, grounded RAG, guarded tool use, human-in-the-loop approval, and a deterministic evaluation harness — not just a prompt wrapped in a chat UI.

## What it does

A customer asks a support question. The agent:
1. Classifies intent and flags sensitive or injection-suspicious requests.
2. Retrieves relevant TaskNest policy documents from a ChromaDB knowledge base.
3. Decides whether to answer directly, call a tool, or escalate to a human — policy is enforced in code, not just prompted.
4. Runs a mock order-lookup tool automatically, but **pauses for human approval** before creating any support ticket.
5. Drafts a grounded reply and validates it (LLM checks + deterministic checks) before returning it, retrying once if validation fails.

## Architecture

![Architecture](docs/architecture.png)
*(Generated from the live graph — see "Regenerating the diagram" below.)*

- **LangGraph** — orchestrates the workflow above as an explicit state machine (not a free-form ReAct loop), with `interrupt()`/resume for human review.
- **FastAPI** — `/api/chat`, `/api/approve`, `/api/tickets`, `/health`. Thin routing layer only; all logic is in `backend/app/`.
- **Streamlit** — a support-agent dashboard: chat, review queue, ticket list.
- **Gemini** (`langchain-google-genai`) — structured-output LLM calls for classification, decisioning, drafting, and validation.
- **ChromaDB** — vector store over 5 synthetic TaskNest policy documents.

Full design rationale — why an explicit graph over ReAct, why policy sits in code rather than prompts, why the guardrails are layered — is in [SECURITY.md](SECURITY.md) and the phase-by-phase build log in this repo's commit history.

## Quickstart

### Option A: Docker (recommended)
```bash
cp .env.example .env        # add your GOOGLE_API_KEY
docker compose up --build
```
Open http://localhost:8501 (dashboard) and http://localhost:8000/docs (API).

### Option B: Local (Windows PowerShell)
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r backend\requirements-dev.txt -r frontend\requirements.txt
Copy-Item .env.example .env    # add your GOOGLE_API_KEY

# Terminal 1
cd backend; python -m uvicorn app.main:app --port 8000
# Terminal 2
cd frontend; streamlit run app.py
```

## Testing

```powershell
cd backend
python -m pytest -q      # 105 tests: RAG, tools, schemas, LLM retry logic,
                          # guardrails, graph routing/approval/injection cases, API
cd ..
python -m pytest evals -q   # 17 tests: evaluation metrics
```

## Evaluation

```powershell
python -m evals.run_evaluation            # offline: scripted LLM, tests graph wiring only
python -m evals.run_evaluation --live     # real Gemini calls, throttled for free-tier quota
```

**Offline** (architecture check, not a model-quality claim) — 19/19 dataset rows:

| Metric | Result |
|---|---|
| Intent accuracy | 16/16 (100%) |
| Retrieval hit rate | 11/11 (100%) |
| Tool selection accuracy | 19/19 (100%) |
| Escalation/approval accuracy | 19/19 (100%) |
| Groundedness | 10/10 (100%) |
| Structured output validity | 19/19 (100%) |

**Live** (real Gemini quality, `gemini-3.5-flash`): *not yet run to completion — free-tier quota was exhausted during development. Run `python -m evals.run_evaluation --live --delay 10` once quota resets and paste the results table here before treating this as a finished deliverable.* Offline mode proves the graph's routing and guardrails are correct; only a completed live run measures how well the model itself classifies, retrieves, and drafts.

Both modes write a full JSON report to `evals/reports/`, and each report is stamped `"mode": "offline"` or `"mode": "live"` so the two are never conflated.

## Known limitations

See [SECURITY.md](SECURITY.md) for the full list. The short version: approval state is in-memory (single backend process only), there's no authentication, and groundedness scoring is keyword-based rather than semantic.

## Regenerating the diagram
```powershell
cd backend
python -c "from app.graph import get_graph; open('../docs/architecture.mmd','w',encoding='utf-8').write(get_graph().get_graph().draw_mermaid())"
```
Paste the `.mmd` content into https://mermaid.live, export as PNG, save to `docs/architecture.png`.

## Project structure
See the folder tree in this repo, or the "Final folder structure" section of the original build plan (kept in `docs/` for reference, if you choose to save it).

## Tech stack
Python, LangGraph, FastAPI, Streamlit, Google Gemini, ChromaDB, LangChain, Pydantic, pytest, Docker.