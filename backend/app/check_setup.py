"""Dev helper. Run from backend/:

    python -m app.check_setup            # offline: config + schema checks
    python -m app.check_setup --live     # also calls Gemini (uses your API key)
"""
from __future__ import annotations

import sys

from pydantic import ValidationError

from app.config import get_settings
from app.schemas import ActionDecision, ActionType, ChatRequest, QueryAnalysis


def main() -> None:
    settings = get_settings()
    print(f"LLM model:          {settings.llm_model}")
    print(f"Embedding model:    {settings.embedding_model}")
    print(f"API key configured: {bool(settings.google_api_key)}")  # never print the key

    # Schema sanity checks (offline)
    ActionDecision(action=ActionType.ANSWER, reasoning="FAQ question")
    try:
        ActionDecision(action=ActionType.LOOKUP_ORDER, reasoning="missing order id")
    except ValidationError:
        print("OK: ActionDecision rejects lookup without order_id")
    try:
        ChatRequest(message="x" * (settings.max_query_chars + 1))
    except ValidationError:
        print("OK: ChatRequest rejects over-long messages")

    if "--live" in sys.argv:
        from langchain_google_genai import ChatGoogleGenerativeAI

        llm = ChatGoogleGenerativeAI(
            model=settings.llm_model,
            temperature=settings.llm_temperature,
            google_api_key=settings.require_api_key(),
        )
        print("Gemini plain reply:", str(llm.invoke("Reply with exactly: OK").content).strip())

        # The important one: does Gemini return our Pydantic schema?
        analysis = llm.with_structured_output(QueryAnalysis).invoke(
            "Classify this customer message: 'Why was I charged twice this month?'"
        )
        print("Gemini structured output:", analysis)


if __name__ == "__main__":
    main()