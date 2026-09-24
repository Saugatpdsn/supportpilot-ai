"""Deterministic guardrails (no LLM involved).

Phase 3 covers: untrusted-data wrapping, a heuristic injection signal, and
checks on the drafted answer. Phase 4 adds the HTTP-layer pieces and SECURITY.md.

These defenses reduce risk; they cannot guarantee protection against prompt injection.
"""
from __future__ import annotations

import json
import re
from typing import Any

# Tags we use to fence untrusted data. Stripped from any untrusted text so a
# document or message cannot "close" the fence and smuggle in instructions.
_TAG_PATTERN = re.compile(
    r"</?\s*(?:customer_message|retrieved_documents|document|tool_results|draft_reply)\b[^>]*>",
    re.IGNORECASE,
)

# Heuristic BACKUP signal only. The primary detector is the LLM classifier in
# understand_query; regexes alone are trivially bypassed.
_INJECTION_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in (
        r"ignore\s+(?:all\s+|any\s+|the\s+)?(?:previous|prior|above|earlier)\b",
        r"disregard\s+(?:all\s+|any\s+|the\s+)?(?:previous|prior|above|earlier)\b",
        r"(?:reveal|show|print|repeat|leak)\s+(?:me\s+)?(?:your\s+|the\s+)?"
        r"(?:system\s+prompt|hidden\s+prompt|instructions)",
        r"you\s+are\s+now\s+(?:a|an|in)\b",
        r"\bdeveloper\s+mode\b",
        r"\bjailbreak\b",
        r"act\s+as\s+(?:the\s+)?(?:system|admin|administrator|developer)\b",
    )
]

_ID_PATTERN = re.compile(r"\b(?:TCK|ORD)-\d+\b", re.IGNORECASE)
_LEAK_PATTERNS = [
    re.compile(r"AIza[0-9A-Za-z_\-]{20,}"),                  # Google API key shape
    re.compile(r"Security rules \(highest priority"),        # our own system prompt
    re.compile(r"</?(?:customer_message|retrieved_documents)", re.IGNORECASE),
]
MAX_ANSWER_CHARS = 2000


def neutralize(text: str) -> str:
    """Remove our fence tags from untrusted text."""
    return _TAG_PATTERN.sub("", text)


def wrap_customer_message(text: str) -> str:
    return f"<customer_message>\n{neutralize(text)}\n</customer_message>"


def wrap_documents(chunks: list[dict[str, Any]]) -> str:
    if not chunks:
        return "<retrieved_documents>\n(no documents were retrieved)\n</retrieved_documents>"
    docs = [
        f'<document source="{c["source"]}">\n{neutralize(c["content"])}\n</document>'
        for c in chunks
    ]
    return "<retrieved_documents>\n" + "\n".join(docs) + "\n</retrieved_documents>"


def wrap_tool_results(results: list[dict[str, Any]]) -> str:
    if not results:
        return "<tool_results>\n(no tools were run)\n</tool_results>"
    return "<tool_results>\n" + neutralize(json.dumps(results, indent=2, default=str)) + "\n</tool_results>"


def heuristic_injection_signal(text: str) -> bool:
    return any(p.search(text) for p in _INJECTION_PATTERNS)


def deterministic_answer_issues(
    answer: str, *, query: str, tool_results: list[dict[str, Any]]
) -> list[str]:
    """Checks that do not depend on any LLM. Returns human-readable issues."""
    if not answer or not answer.strip():
        return ["the answer is empty"]

    issues: list[str] = []
    if len(answer) > MAX_ANSWER_CHARS:
        issues.append("the answer is too long")
    if any(p.search(answer) for p in _LEAK_PATTERNS):
        issues.append("the answer appears to leak internal instructions or a secret")

    ticket_ids = {
        str(r["data"]["ticket_id"]).upper()
        for r in tool_results
        if r.get("success") and isinstance(r.get("data"), dict) and "ticket_id" in r["data"]
    }
    tool_blob = json.dumps(tool_results, default=str).upper()

    for ident in sorted({m.upper() for m in _ID_PATTERN.findall(answer)}):
        if ident.startswith("TCK-"):
            if ident not in ticket_ids:
                issues.append(f"the answer mentions {ident}, which no tool created")
        elif ident not in tool_blob and ident.split("-")[1] not in query:
            issues.append(f"the answer mentions {ident}, which is not in the tool results or message")
    return issues