import pytest

from evals.metrics import (
    escalation_accuracy,
    evaluate_row,
    groundedness,
    intent_accuracy,
    retrieval_hit_rate,
    structured_output_validity,
    summarize,
    tool_selection_accuracy,
)

ROW = {
    "id": "r1",
    "expected_intent": "billing",
    "expected_documents": ["billing_policy.md"],
    "expected_action": "lookup_order_status",
    "expected_review": "not_required",
    "expected_facts": ["30 minutes", "hold"],
}


# ---- intent_accuracy -----------------------------------------------------
def test_intent_accuracy_pass_and_fail():
    assert intent_accuracy(ROW, {"intent": "billing"}).passed is True
    assert intent_accuracy(ROW, {"intent": "product_info"}).passed is False


def test_intent_accuracy_skips_null_expectation():
    row = {**ROW, "expected_intent": None}
    result = intent_accuracy(row, {"intent": "billing"})
    assert result.passed is None


# ---- retrieval_hit_rate ---------------------------------------------------
def test_retrieval_hit_rate_pass_on_any_overlap():
    result = {"sources": [{"source": "refund_policy.md"}, {"source": "billing_policy.md"}]}
    assert retrieval_hit_rate(ROW, result).passed is True


def test_retrieval_hit_rate_fails_with_no_overlap():
    result = {"sources": [{"source": "product_faq.md"}]}
    assert retrieval_hit_rate(ROW, result).passed is False


def test_retrieval_hit_rate_skips_when_nothing_expected():
    row = {**ROW, "expected_documents": []}
    assert retrieval_hit_rate(row, {"sources": []}).passed is None


# ---- tool_selection_accuracy -----------------------------------------------
def test_tool_selection_expects_lookup_only():
    ok = {"tool_actions": [{"tool": "lookup_order_status"}], "pending_action": None}
    assert tool_selection_accuracy(ROW, ok).passed is True

    wrong_tool = {"tool_actions": [{"tool": "create_support_ticket"}], "pending_action": None}
    assert tool_selection_accuracy(ROW, wrong_tool).passed is False


def test_tool_selection_expects_no_tool_for_answer():
    row = {**ROW, "expected_action": "answer"}
    assert tool_selection_accuracy(row, {"tool_actions": [], "pending_action": None}).passed is True
    assert tool_selection_accuracy(
        row, {"tool_actions": [{"tool": "lookup_order_status"}], "pending_action": None}
    ).passed is False


def test_tool_selection_expects_escalation_proposed_or_executed():
    row = {**ROW, "expected_action": "escalate"}
    proposed = {"tool_actions": [], "pending_action": {"tool": "create_support_ticket"}}
    assert tool_selection_accuracy(row, proposed).passed is True

    executed = {"tool_actions": [{"tool": "create_support_ticket"}], "pending_action": None}
    assert tool_selection_accuracy(row, executed).passed is True

    neither = {"tool_actions": [], "pending_action": None}
    assert tool_selection_accuracy(row, neither).passed is False


# ---- escalation_accuracy ---------------------------------------------------
def test_escalation_accuracy():
    assert escalation_accuracy(ROW, {"review_status": "not_required"}).passed is True
    assert escalation_accuracy(ROW, {"review_status": "pending_approval"}).passed is False


# ---- groundedness -----------------------------------------------------------
def test_groundedness_requires_all_facts_case_insensitive():
    result = {"answer": "The hold lasts up to 30 Minutes before it clears."}
    assert groundedness(ROW, result).passed is True


def test_groundedness_fails_on_missing_fact():
    result = {"answer": "The charge is temporary."}
    outcome = groundedness(ROW, result)
    assert outcome.passed is False
    assert "30 minutes" in outcome.detail


def test_groundedness_skips_when_no_expected_facts():
    row = {**ROW, "expected_facts": []}
    assert groundedness(row, {"answer": "anything"}).passed is None


# ---- structured_output_validity ---------------------------------------------
def test_structured_output_validity_clean_run():
    assert structured_output_validity(ROW, {"errors": [], "run_error": None}).passed is True


@pytest.mark.parametrize(
    "result",
    [{"errors": ["Could not draft a reply."], "run_error": None}, {"errors": [], "run_error": "RuntimeError: boom"}],
)
def test_structured_output_validity_flags_errors(result):
    assert structured_output_validity(ROW, result).passed is False


# ---- evaluate_row / summarize -------------------------------------------------
def test_evaluate_row_runs_all_six_metrics():
    result = {
        "intent": "billing",
        "sources": [{"source": "billing_policy.md"}],
        "tool_actions": [{"tool": "lookup_order_status"}],
        "pending_action": None,
        "review_status": "not_required",
        "answer": "The hold lasts 30 minutes, it's a hold not a charge.",
        "errors": [],
        "run_error": None,
    }
    metrics = evaluate_row(ROW, result)
    assert len(metrics) == 6
    assert all(m.passed is True for m in metrics)


def test_summarize_computes_pass_rates_and_excludes_skips():
    from evals.metrics import MetricResult

    all_results = [
        [MetricResult("intent_accuracy", True), MetricResult("groundedness", None)],
        [MetricResult("intent_accuracy", False), MetricResult("groundedness", True)],
    ]
    summary = summarize(all_results)
    assert summary["intent_accuracy"] == {"passed": 1, "applicable": 2, "skipped": 0, "pass_rate": 0.5}
    assert summary["groundedness"] == {"passed": 1, "applicable": 1, "skipped": 1, "pass_rate": 1.0}