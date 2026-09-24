import pytest
from pydantic import ValidationError

from app.config import get_settings
from app.schemas import ActionDecision, ActionType, ChatRequest, Priority, ValidationResult


def test_chat_request_strips_whitespace():
    assert ChatRequest(message="  hello there  ").message == "hello there"


@pytest.mark.parametrize("bad", ["", "   ", "hi\x00there"])
def test_chat_request_rejects_empty_and_control_chars(bad):
    with pytest.raises(ValidationError):
        ChatRequest(message=bad)


def test_chat_request_enforces_length_limit():
    limit = get_settings().max_query_chars
    ChatRequest(message="x" * limit)
    with pytest.raises(ValidationError):
        ChatRequest(message="x" * (limit + 1))


def test_action_decision_requires_arguments_per_action():
    ActionDecision(action=ActionType.ANSWER, reasoning="FAQ")
    with pytest.raises(ValidationError):
        ActionDecision(action=ActionType.LOOKUP_ORDER, reasoning="no id")
    with pytest.raises(ValidationError):
        ActionDecision(action=ActionType.CREATE_TICKET, reasoning="no issue", ticket_issue="x")
    ok = ActionDecision(
        action=ActionType.CREATE_TICKET,
        reasoning="needs a human",
        ticket_issue="Duplicate charge on ORD-1002",
        ticket_priority=Priority.HIGH,
    )
    assert ok.ticket_priority == Priority.HIGH


def test_validation_result_passes_only_when_everything_is_clean():
    assert ValidationResult(is_relevant=True).passes
    assert not ValidationResult(is_relevant=False).passes
    assert not ValidationResult(is_relevant=True, unsupported_claims=["invented policy"]).passes
    assert not ValidationResult(is_relevant=True, tool_result_misrepresented=True).passes
    assert not ValidationResult(is_relevant=True, missing_info_acknowledged=False).passes