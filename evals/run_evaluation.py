"""Evaluation runner.

    python -m evals.run_evaluation                # offline (mocked LLM), default
    python -m evals.run_evaluation --live          # real Gemini calls, throttled
    python -m evals.run_evaluation --live --delay 5

Offline mode tests the GRAPH'S ARCHITECTURE (routing, guardrails, policy
enforcement) with a scripted fake LLM: deterministic, free, no network.
Live mode tests actual MODEL QUALITY through the real graph and Gemini:
useful but subject to your API quota, hence --delay and --live being opt-in.
Every report is stamped with which mode produced it. Never read an offline
score as a claim about live model quality, or vice versa.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Make `backend/app` importable when running from the repo root.
BACKEND_DIR = Path(__file__).resolve().parent.parent / "backend"
sys.path.insert(0, str(BACKEND_DIR))

DATASET_PATH = Path(__file__).resolve().parent / "dataset.json"
REPORT_DIR = Path(__file__).resolve().parent / "reports"


def load_dataset() -> list[dict[str, Any]]:
    return json.loads(DATASET_PATH.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------
# Offline mode: a scripted fake LLM keyed off dataset expectations.
# It answers plausibly for EACH dataset row so the graph takes the expected
# path; it is not a language model and proves wiring, not model quality.
# --------------------------------------------------------------------------
def _fake_llm_for_row(row: dict[str, Any]):
    from app.schemas import (
        ActionDecision,
        ActionType,
        DraftedAnswer,
        Intent,
        Priority,
        QueryAnalysis,
        ValidationResult,
    )

    injection = row["category"] == "prompt_injection"
    sensitive = row["expected_action"] == "escalate" and row["category"] != "unanswerable"
    intent = Intent(row["expected_intent"]) if row.get("expected_intent") else Intent.OTHER

    analysis = QueryAnalysis(
        intent=intent, summary=f"Synthetic summary for {row['id']}", sensitive=sensitive,
        injection_suspected=injection,
    )

    action_map = {
        "answer": ActionType.ANSWER,
        "lookup_order_status": ActionType.LOOKUP_ORDER,
        "escalate": ActionType.ESCALATE,
    }
    decision_kwargs: dict[str, Any] = {}
    if row["expected_action"] == "lookup_order_status":
        match = re.search(r"ORD-\d{4}", row["query"])
        decision_kwargs["order_id"] = match.group(0) if match else "ORD-0000"
    elif row["expected_action"] == "escalate":
        decision_kwargs["ticket_issue"] = f"Escalation for {row['id']}: {row['query'][:100]}"
        decision_kwargs["ticket_priority"] = Priority.MEDIUM
    decision = ActionDecision(
        action=action_map[row["expected_action"]], reasoning="synthetic", **decision_kwargs
    )

    fact_text = ". ".join(row.get("expected_facts") or []) or "Here is the information you asked for."
    draft = DraftedAnswer(answer=f"{fact_text}.", cited_sources=list(row.get("expected_documents") or []))

    def fake(schema, system, user):
        return {
            "QueryAnalysis": analysis,
            "ActionDecision": decision,
            "DraftedAnswer": draft,
            "ValidationResult": ValidationResult(is_relevant=True),
        }[schema.__name__].model_copy(deep=True)

    return fake


import re  # noqa: E402 (kept near use above for readability)


def _fake_retriever_for_row(row: dict[str, Any]):
    sources = row.get("expected_documents") or []

    def fake_retrieve(query: str, k: int = 4):
        return [
            {"source": s, "chunk_id": f"{s}::0", "content": f"Synthetic content for {s}", "score": 0.9}
            for s in sources
        ]

    return fake_retrieve


def run_offline(dataset: list[dict[str, Any]]) -> list[dict[str, Any]]:
    from app import graph as graph_module
    from app import llm, nodes
    from app.graph import build_graph, resume_chat, run_chat

    results = []
    for row in dataset:
        fresh_graph = build_graph()
        graph_module.get_graph = lambda g=fresh_graph: g  # type: ignore[assignment]
        llm.structured_call = _fake_llm_for_row(row)
        nodes.retrieve = _fake_retriever_for_row(row)

        try:
            resp = run_chat(row["query"])
            results.append(resp.model_dump(mode="json") | {"run_error": None})
        except Exception as exc:  # noqa: BLE001
            results.append({"run_error": f"{type(exc).__name__}: {exc}"})
    return results


# --------------------------------------------------------------------------
# Live mode: the real graph, real Gemini, throttled between requests.
# --------------------------------------------------------------------------
def run_live(dataset: list[dict[str, Any]], delay: float) -> list[dict[str, Any]]:
    from app.graph import resume_chat, run_chat
    from app.rag import ensure_ingested

    ensure_ingested()
    results = []
    for i, row in enumerate(dataset):
        if i > 0:
            time.sleep(delay)
        try:
            resp = run_chat(row["query"])
            results.append(resp.model_dump(mode="json") | {"run_error": None})
        except Exception as exc:  # noqa: BLE001
            print(f"  [{row['id']}] run failed: {type(exc).__name__}: {exc}")
            results.append({"run_error": f"{type(exc).__name__}: {exc}"})
    return results


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------
def build_report(mode: str, dataset: list[dict[str, Any]], results: list[dict[str, Any]]) -> dict[str, Any]:
    from evals.metrics import evaluate_row, summarize

    per_row_metrics = [evaluate_row(row, result) for row, result in zip(dataset, results)]
    rows_report = [
        {
            "id": row["id"],
            "category": row["category"],
            "query": row["query"],
            "review_status": result.get("review_status"),
            "answer": result.get("answer"),
            "run_error": result.get("run_error"),
            "metrics": [
                {"name": m.name, "passed": m.passed, "detail": m.detail} for m in metrics
            ],
        }
        for row, result, metrics in zip(dataset, results, per_row_metrics)
    ]
    return {
        "mode": mode,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset_size": len(dataset),
        "summary": summarize(per_row_metrics),
        "rows": rows_report,
    }


def print_summary(report: dict[str, Any]) -> None:
    print(f"\n{'=' * 60}")
    print(f"  EVALUATION MODE: {report['mode'].upper()}")
    if report["mode"] == "offline":
        print("  (tests graph wiring/routing with a scripted fake LLM)")
        print("  (NOT a measure of real Gemini model quality)")
    else:
        print("  (real Gemini calls through the real graph)")
    print(f"{'=' * 60}")
    print(f"Dataset: {report['dataset_size']} queries\n")
    for name, stats in report["summary"].items():
        rate = f"{stats['pass_rate'] * 100:.0f}%" if stats["pass_rate"] is not None else "n/a"
        print(f"  {name:<28} {stats['passed']:>2}/{stats['applicable']:<2} ({rate})  skipped={stats['skipped']}")
    print()
    failures = [
        (row["id"], m["name"], m["detail"])
        for row in report["rows"]
        for m in row["metrics"]
        if m["passed"] is False
    ]
    if failures:
        print(f"Failures ({len(failures)}):")
        for row_id, metric_name, detail in failures:
            print(f"  - {row_id} / {metric_name}: {detail}")
    else:
        print("No failures.")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="SupportPilot evaluation runner")
    parser.add_argument("--live", action="store_true", help="use real Gemini calls instead of the offline fake")
    parser.add_argument("--delay", type=float, default=3.0, help="seconds between live requests (default 3)")
    args = parser.parse_args()

    dataset = load_dataset()
    mode = "live" if args.live else "offline"
    print(f"Running {len(dataset)} queries in {mode.upper()} mode...")

    results = run_live(dataset, args.delay) if args.live else run_offline(dataset)
    report = build_report(mode, dataset, results)
    print_summary(report)

    REPORT_DIR.mkdir(exist_ok=True)
    out_path = REPORT_DIR / f"eval_{mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Full report saved to {out_path}")


if __name__ == "__main__":
    main()