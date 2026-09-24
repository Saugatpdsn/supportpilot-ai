import pytest

from app import llm
from app.config import ConfigError


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(llm.time, "sleep", lambda seconds: None)


def test_success_is_not_retried():
    calls = []

    def fn():
        calls.append(1)
        return "ok"

    assert llm.with_one_retry(fn, "test") == "ok"
    assert len(calls) == 1


def test_retries_once_then_succeeds():
    attempts = {"n": 0}

    def fn():
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise RuntimeError("transient")
        return "ok"

    assert llm.with_one_retry(fn, "test") == "ok"
    assert attempts["n"] == 2


def test_gives_up_after_exactly_one_retry():
    attempts = {"n": 0}

    def fn():
        attempts["n"] += 1
        raise RuntimeError("still failing")

    with pytest.raises(llm.LLMError):
        llm.with_one_retry(fn, "test")
    assert attempts["n"] == 2


def test_config_error_is_not_retried():
    attempts = {"n": 0}

    def fn():
        attempts["n"] += 1
        raise ConfigError("no key")

    with pytest.raises(ConfigError):
        llm.with_one_retry(fn, "test")
    assert attempts["n"] == 1