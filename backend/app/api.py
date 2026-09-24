"""HTTP routes. A thin layer: validation and error mapping only.
The agent logic lives in graph.py, tools.py and rag.py."""
from __future__ import annotations

import threading
from typing import Any

from fastapi import APIRouter, HTTPException

from app.config import get_settings
from app.graph import NoPendingReview, resume_chat, run_chat
from app.schemas import ApproveRequest, ChatRequest, ChatResponse, Ticket
from app.tools import list_tickets

router = APIRouter()
_resume_lock = threading.Lock()  # serializes approvals (simple, fine for one worker)


@router.get("/health", tags=["system"], summary="Liveness check")
def health() -> dict[str, Any]:
    settings = get_settings()
    return {
        "status": "ok",
        "llm_model": settings.llm_model,
        "api_key_configured": bool(settings.google_api_key),
    }


@router.post(
    "/api/chat",
    response_model=ChatResponse,
    tags=["agent"],
    summary="Send a customer message to the agent",
    description="Runs the LangGraph workflow. The response is either a final answer or "
    "`review_status=pending_approval` with a `pending_action` awaiting human review.",
)
def chat(payload: ChatRequest) -> ChatResponse:
    return run_chat(payload.message)


@router.post(
    "/api/approve",
    response_model=ChatResponse,
    tags=["agent"],
    summary="Approve or reject a pending action",
    responses={404: {"description": "No pending review for this thread (unknown or already handled)."}},
)
def approve(payload: ApproveRequest) -> ChatResponse:
    try:
        with _resume_lock:
            return resume_chat(payload.thread_id, payload.approve, payload.reviewer_note)
    except NoPendingReview:
        raise HTTPException(
            status_code=404,
            detail="No pending review for this thread (unknown or already handled).",
        ) from None


@router.get("/api/tickets", response_model=list[Ticket], tags=["tickets"], summary="List created tickets")
def tickets() -> list[Ticket]:
    return list_tickets()