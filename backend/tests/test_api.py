import pytest
from fastapi.testclient import TestClient

from app import api as api_module
from app import graph as graph_module
from app import llm, nodes
from app.config import ConfigError
from app.graph import NoPendingReview, build_graph
from app.main import app
from app.schemas import (
    ActionDecision,
    ActionType,
    ChatResponse,
    DraftedAnswer,
    Intent,
    Priority,
    QueryAnalysis,
    Ticket,
    ValidationResult,
)


@pytest.fixture
def client():
    # No `with`: this skips the startup hook, so tests never build the real knowledge base.
    return TestClient(app, raise_server_exceptions=False)


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert "api_key_configured" in body


def test_openapi_lists_all_routes(client):
    assert client.get("/docs").status_code == 200
    paths = client.get("/openapi.json").json()["paths"]
    assert {"/health", "/api/chat", "/api/approve", "/api/tickets"} <= set(paths)


def test_chat_returns_the_service_response(client, monkeypatch):
    monkeypatch.setattr(api_module, "run_chat", lambda message: ChatResponse(thread_id="t-1", answer=f"echo: {message}"))
    resp = client.post("/api/chat", json={"message": "hello"})
    assert resp.status_code == 200
    assert resp.json()["answer"] == "echo: hello"
    assert resp.json()["review_status"] == "not_required"


@pytest.mark.parametrize(
    "payload",
    [
        {"message": ""},
        {"message": "x" * 1001},
        {"message": "hi", "thread_id": "abc"},  # unknown field
        {},
        {"message": 123},
    ],
)
def test_chat_rejects_invalid_input(client, payload):
    assert client.post("/api/chat", json=payload).status_code == 422


def test_validation_errors_do_not_echo_the_input(client):
    resp = client.post("/api/chat", json={"message": "x" * 1001})
    assert resp.status_code == 422
    assert "x" * 50 not in resp.text
    assert set(resp.json()["detail"][0]) == {"loc", "msg"}


def test_oversized_body_is_rejected(client):
    resp = client.post("/api/chat", content="x" * 20_000, headers={"content-type": "application/json"})
    assert resp.status_code == 413


def test_unexpected_error_returns_a_generic_500(client, monkeypatch):
    def boom(message):
        raise RuntimeError("secret-internal-detail")

    monkeypatch.setattr(api_module, "run_chat", boom)
    resp = client.post("/api/chat", json={"message": "hello"})
    assert resp.status_code == 500
    assert "secret-internal-detail" not in resp.text
    assert "request_id" in resp.json()


def test_missing_api_key_returns_503(client, monkeypatch):
    def no_key(message):
        raise ConfigError("GOOGLE_API_KEY is not set")

    monkeypatch.setattr(api_module, "run_chat", no_key)
    resp = client.post("/api/chat", json={"message": "hello"})
    assert resp.status_code == 503
    assert "GOOGLE_API_KEY" not in resp.text


def test_approve_unknown_thread_is_404(client, monkeypatch):
    def none_pending(thread_id, approve, note):
        raise NoPendingReview(thread_id)

    monkeypatch.setattr(api_module, "resume_chat", none_pending)
    resp = client.post("/api/approve", json={"thread_id": "nope", "approve": True})
    assert resp.status_code == 404


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"thread_id": "t"},
        {"thread_id": "t", "approve": True, "reviewer_note": "x" * 501},
    ],
)
def test_approve_rejects_invalid_input(client, payload):
    assert client.post("/api/approve", json=payload).status_code == 422


def test_tickets_endpoint(client, monkeypatch):
    ticket = Ticket(ticket_id="TCK-1001", issue="Example issue text", priority=Priority.LOW)
    monkeypatch.setattr(api_module, "list_tickets", lambda: [ticket])
    resp = client.get("/api/tickets")
    assert resp.status_code == 200
    assert resp.json()[0]["ticket_id"] == "TCK-1001"


def test_end_to_end_approval_flow_over_http(client, monkeypatch, tools_env):
    """Real graph, real tools, real HTTP layer. Only the LLM and retriever are faked."""

    def fake_llm(schema, system, user):
        return {
            "QueryAnalysis": QueryAnalysis(
                intent=Intent.REFUND_REQUEST, summary="Customer requests a refund.", sensitive=True
            ),
            "ActionDecision": ActionDecision(
                action=ActionType.ESCALATE,
                reasoning="sensitive",
                ticket_issue="Customer requests a refund for order ORD-1003.",
                ticket_priority=Priority.MEDIUM,
            ),
            "DraftedAnswer": DraftedAnswer(
                answer="Your refund request was submitted as ticket TCK-1001.", cited_sources=["refund_policy.md"]
            ),
            "ValidationResult": ValidationResult(is_relevant=True),
        }[schema.__name__].model_copy(deep=True)

    fresh_graph = build_graph()
    monkeypatch.setattr(graph_module, "get_graph", lambda: fresh_graph)
    monkeypatch.setattr(llm, "structured_call", fake_llm)
    monkeypatch.setattr(
        nodes,
        "retrieve",
        lambda query, k=4: [{"source": "refund_policy.md", "chunk_id": "r::0", "content": "Refunds need approval.", "score": 0.9}],
    )

    chat = client.post("/api/chat", json={"message": "I want a refund for order ORD-1003"})
    assert chat.status_code == 200
    body = chat.json()
    assert body["review_status"] == "pending_approval"
    assert body["pending_action"]["tool"] == "create_support_ticket"
    assert client.get("/api/tickets").json() == []  # nothing happens before approval

    approved = client.post("/api/approve", json={"thread_id": body["thread_id"], "approve": True})
    assert approved.status_code == 200
    assert approved.json()["review_status"] == "approved"
    assert approved.json()["tool_actions"][0]["data"]["ticket_id"] == "TCK-1001"
    assert len(client.get("/api/tickets").json()) == 1

    again = client.post("/api/approve", json={"thread_id": body["thread_id"], "approve": True})
    assert again.status_code == 404  # cannot be approved twice