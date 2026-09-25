"""Deterministic evaluation metrics. Every function takes one dataset row (dict)
and one run result (dict, shape defined in run_evaluation.py) and returns a
MetricResult. No LLM calls happen here — this module is pure comparison logic.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class MetricResult:
    name: str
    passed: Optional[bool]  # None = not applicable to this row (excluded from the average)
    detail: str = ""


def intent_accuracy(row: dict[str, Any], result: dict[str, Any]) -> MetricResult:
    expected = row.get("expected_intent")
    if expected is None:
        return MetricResult("intent_accuracy", None, "not applicable (adversarial query)")
    actual = result.get("intent")
    return MetricResult(
        "intent_accuracy", actual == expected, f"expected={expected} actual={actual}"
    )


def retrieval_hit_rate(row: dict[str, Any], result: dict[str, Any]) -> MetricResult:
    expected = set(row.get("expected_documents") or [])
    if not expected:
        return MetricResult("retrieval_hit_rate", None, "no expected documents for this row")
    actual = {s["source"] for s in result.get("sources", [])}
    hit = bool(expected & actual)
    return MetricResult(
        "retrieval_hit_rate", hit, f"expected any of {sorted(expected)}; got {sorted(actual)}"
    )


def tool_selection_accuracy(row: dict[str, Any], result: dict[str, Any]) -> MetricResult:
    expected = row["expected_action"]
    tools_called = [t["tool"] for t in result.get("tool_actions", [])]
    proposed = (result.get("pending_action") or {}).get("tool")

    if expected == "answer":
        actual_ok = not tools_called and not proposed
        detail = f"expected no tool call; tools_called={tools_called} proposed={proposed}"
    elif expected == "lookup_order_status":
        actual_ok = tools_called == ["lookup_order_status"]
        detail = f"expected lookup_order_status only; got {tools_called}"
    elif expected == "escalate":
        actual_ok = proposed == "create_support_ticket" or "create_support_ticket" in tools_called
        detail = f"expected a ticket to be proposed or created; proposed={proposed} tools_called={tools_called}"
    else:
        actual_ok = False
        detail = f"unknown expected_action {expected!r} in dataset"
    return MetricResult("tool_selection_accuracy", actual_ok, detail)


def escalation_accuracy(row: dict[str, Any], result: dict[str, Any]) -> MetricResult:
    expected = row["expected_review"]
    actual = result.get("review_status")
    return MetricResult("escalation_accuracy", actual == expected, f"expected={expected} actual={actual}")


def groundedness(row: dict[str, Any], result: dict[str, Any]) -> MetricResult:
    expected_facts = row.get("expected_facts") or []
    if not expected_facts:
        return MetricResult("groundedness", None, "no expected facts for this row")
    answer = (result.get("answer") or "").lower()
    missing = [fact for fact in expected_facts if fact.lower() not in answer]
    return MetricResult(
        "groundedness",
        not missing,
        "all facts present" if not missing else f"missing: {missing}",
    )


def structured_output_validity(row: dict[str, Any], result: dict[str, Any]) -> MetricResult:
    """True unless the run itself hit an internal error (safe_error / a graph exception)."""
    errors = result.get("errors") or []
    run_error = result.get("run_error")
    ok = not errors and not run_error
    detail = f"errors={errors} run_error={run_error}" if not ok else "no errors"
    return MetricResult("structured_output_validity", ok, detail)


METRICS = [
    intent_accuracy,
    retrieval_hit_rate,
    tool_selection_accuracy,
    escalation_accuracy,
    groundedness,
    structured_output_validity,
]


def evaluate_row(row: dict[str, Any], result: dict[str, Any]) -> list[MetricResult]:
    return [metric(row, result) for metric in METRICS]


_WS = re.compile(r"\s+")


def summarize(all_results: list[list[MetricResult]]) -> dict[str, dict[str, Any]]:
    """Per-metric-name aggregate: pass rate over applicable rows."""
    by_name: dict[str, list[MetricResult]] = {}
    for row_results in all_results:
        for metric_result in row_results:
            by_name.setdefault(metric_result.name, []).append(metric_result)

    summary: dict[str, dict[str, Any]] = {}
    for name, results in by_name.items():
        applicable = [r for r in results if r.passed is not None]
        passed = sum(1 for r in applicable if r.passed)
        summary[name] = {
            "passed": passed,
            "applicable": len(applicable),
            "skipped": len(results) - len(applicable),
            "pass_rate": round(passed / len(applicable), 3) if applicable else None,
        }
    return summary