"""Central configuration. Everything comes from environment variables
(loaded from <repo>/.env during local development)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent  # .../backend
PROJECT_ROOT = BASE_DIR.parent                      # repo root

# Local dev: read <repo>/.env. In Docker, compose injects the variables and
# load_dotenv silently does nothing when the file is absent.
load_dotenv(PROJECT_ROOT / ".env")


class ConfigError(RuntimeError):
    """Raised when required configuration is missing."""


@dataclass(frozen=True)
class Settings:
    # repr=False so the key can never leak through logging or printing Settings
    google_api_key: str = field(repr=False)
    llm_model: str
    embedding_model: str
    llm_temperature: float
    max_query_chars: int
    knowledge_base_dir: Path
    mock_data_path: Path
    tickets_path: Path
    chroma_dir: Path
    log_level: str

    def require_api_key(self) -> str:
        if not self.google_api_key:
            raise ConfigError(
                "GOOGLE_API_KEY is not set. Copy .env.example to .env and add your key."
            )
        return self.google_api_key


@lru_cache
def get_settings() -> Settings:
    return Settings(
        google_api_key=os.getenv("GOOGLE_API_KEY", ""),
        llm_model=os.getenv("LLM_MODEL", "gemini-2.5-flash"),
        embedding_model=os.getenv("EMBEDDING_MODEL", "gemini-embedding-001"),
        llm_temperature=float(os.getenv("LLM_TEMPERATURE", "0")),
        max_query_chars=int(os.getenv("MAX_QUERY_CHARS", "1000")),
        knowledge_base_dir=Path(
            os.getenv("KNOWLEDGE_BASE_DIR", BASE_DIR / "data" / "knowledge_base")
        ),
        mock_data_path=Path(os.getenv("MOCK_DATA_PATH", BASE_DIR / "data" / "mock_data.json")),
        tickets_path=Path(os.getenv("TICKETS_PATH", BASE_DIR / "data" / "tickets.json")),
        chroma_dir=Path(os.getenv("CHROMA_DIR", BASE_DIR / "chroma_db")),
        log_level=os.getenv("LOG_LEVEL", "INFO"),
    )