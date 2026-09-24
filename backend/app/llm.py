"""Gemini access: the client, structured output, and the retry policy in one place.

Policy: a failed model or embedding call is retried ONCE. After that we raise
LLMError and the graph routes to safe_error. Only exception class names are
logged, never messages or prompts.
"""
from __future__ import annotations

import logging
import time
from functools import lru_cache
from typing import Callable, TypeVar

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from app.config import ConfigError, get_settings

logger = logging.getLogger(__name__)

T = TypeVar("T")
S = TypeVar("S", bound=BaseModel)
RETRY_DELAY_SECONDS = 2.0


class LLMError(RuntimeError):
    """A model/embedding call failed even after one retry."""


def with_one_retry(fn: Callable[[], T], what: str) -> T:
    for attempt in (1, 2):
        try:
            return fn()
        except ConfigError:
            raise  # misconfiguration is not transient: fail loudly
        except Exception as exc:  # noqa: BLE001
            logger.warning("%s failed (attempt %d): %s", what, attempt, type(exc).__name__)
            if attempt == 2:
                raise LLMError(f"{what} failed after one retry") from exc
            time.sleep(RETRY_DELAY_SECONDS)
    raise LLMError(what)  # unreachable, keeps type checkers happy


@lru_cache
def _client():
    from langchain_google_genai import ChatGoogleGenerativeAI

    settings = get_settings()
    return ChatGoogleGenerativeAI(
        model=settings.llm_model,
        temperature=settings.llm_temperature,
        google_api_key=settings.require_api_key(),
    )


def structured_call(schema: type[S], system: str, user: str) -> S:
    """Call Gemini and parse the reply into `schema` (validated by Pydantic)."""

    def _once() -> S:
        runnable = _client().with_structured_output(schema)
        result = runnable.invoke([SystemMessage(content=system), HumanMessage(content=user)])
        if result is None:
            raise ValueError("empty structured output")
        return result if isinstance(result, schema) else schema.model_validate(result)

    return with_one_retry(_once, what=schema.__name__)