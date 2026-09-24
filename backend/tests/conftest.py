"""Shared fixtures. Nothing here calls Gemini."""
from __future__ import annotations

import dataclasses
import re
import zlib
from math import sqrt

import pytest
from langchain_core.embeddings import Embeddings

from app.config import get_settings
from app.rag import build_vectorstore, ingest_knowledge_base

_STOPWORDS = {"the", "and", "for", "you", "your", "are", "with", "this", "that", "can",
              "was", "why", "how", "what", "from", "have", "not", "will", "our", "but"}


class KeywordEmbeddings(Embeddings):
    """Deterministic bag-of-words embeddings for OFFLINE tests.

    They test our pipeline mechanics (load, split, store, retrieve, rank), NOT the
    semantic quality of real embeddings. That is measured by the live evaluation.
    """

    DIM = 512

    def _embed(self, text: str) -> list[float]:
        vec = [0.0] * self.DIM
        for token in re.findall(r"[a-z0-9]+", text.lower()):
            if len(token) < 3 or token in _STOPWORDS:
                continue
            vec[zlib.crc32(token.encode()) % self.DIM] += 1.0
        norm = sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


@pytest.fixture
def kb_embeddings() -> KeywordEmbeddings:
    return KeywordEmbeddings()


@pytest.fixture(scope="session")
def kb_store(tmp_path_factory):
    """A real Chroma store built from the real knowledge base with fake embeddings."""
    persist_dir = tmp_path_factory.mktemp("chroma")
    embeddings = KeywordEmbeddings()
    ingest_knowledge_base(embeddings=embeddings, persist_dir=persist_dir)
    return build_vectorstore(embeddings, persist_dir)


@pytest.fixture
def tools_env(tmp_path, monkeypatch):
    """Point the ticket store at a temp file so tests never touch real data."""
    settings = dataclasses.replace(get_settings(), tickets_path=tmp_path / "tickets.json")
    monkeypatch.setattr("app.tools.get_settings", lambda: settings)
    return settings