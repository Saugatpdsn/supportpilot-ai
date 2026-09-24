"""Simple RAG pipeline: load -> split -> embed -> store (ChromaDB) -> retrieve.

Retrieved chunks are returned as plain dicts so they can live in AgentState.
Chunk text is UNTRUSTED data; wrapping it safely for prompts happens in guardrails.
"""
from __future__ import annotations

import argparse
import logging
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import get_settings
from app.schemas import SourceDoc

logger = logging.getLogger(__name__)

COLLECTION_NAME = "supportpilot_kb"
CHUNK_SIZE = 600
CHUNK_OVERLAP = 80
DEFAULT_TOP_K = 4


# --------------------------------------------------------------------------
# Embeddings + vector store
# --------------------------------------------------------------------------
def get_embeddings() -> Embeddings:
    """Gemini embeddings. Tests inject a fake instead, so they never call the API."""
    from langchain_google_genai import GoogleGenerativeAIEmbeddings

    settings = get_settings()
    model = settings.embedding_model
    if not model.startswith("models/"):
        model = f"models/{model}"
    return GoogleGenerativeAIEmbeddings(model=model, google_api_key=settings.require_api_key())


def build_vectorstore(
    embeddings: Optional[Embeddings] = None, persist_dir: Optional[Path] = None
) -> Chroma:
    settings = get_settings()
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings or get_embeddings(),
        persist_directory=str(persist_dir or settings.chroma_dir),
        # cosine distance, so score = 1 - distance is a similarity in [-1, 1]
        collection_metadata={"hnsw:space": "cosine"},
    )


@lru_cache
def get_vectorstore() -> Chroma:
    return build_vectorstore()


# --------------------------------------------------------------------------
# Ingestion
# --------------------------------------------------------------------------
def load_documents(kb_dir: Optional[Path] = None) -> list[Document]:
    kb_dir = Path(kb_dir or get_settings().knowledge_base_dir)
    docs: list[Document] = []
    for path in sorted(kb_dir.glob("*.md")):
        docs.append(
            Document(
                page_content=path.read_text(encoding="utf-8"),
                metadata={"source": path.name},
            )
        )
    return docs


def split_documents(docs: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        # prefer splitting at markdown "## " section boundaries
        separators=["\n## ", "\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    counters: dict[str, int] = {}
    for chunk in chunks:
        source = chunk.metadata["source"]
        index = counters.get(source, 0)
        counters[source] = index + 1
        chunk.metadata["chunk_index"] = index
        chunk.metadata["chunk_id"] = f"{source}::{index}"  # deterministic ID
    return chunks


def ingest_knowledge_base(
    embeddings: Optional[Embeddings] = None,
    kb_dir: Optional[Path] = None,
    persist_dir: Optional[Path] = None,
) -> int:
    """Rebuild the collection from scratch (idempotent). Returns the chunk count."""
    chunks = split_documents(load_documents(kb_dir))
    if not chunks:
        raise ValueError("No knowledge base documents found.")

    embeddings = embeddings or get_embeddings()
    build_vectorstore(embeddings, persist_dir).delete_collection()
    store = build_vectorstore(embeddings, persist_dir)
    store.add_documents(chunks, ids=[c.metadata["chunk_id"] for c in chunks])
    get_vectorstore.cache_clear()
    logger.info("Ingested %d chunks from %d documents", len(chunks), len({c.metadata['source'] for c in chunks}))
    return len(chunks)


def ensure_ingested() -> None:
    """Called at API startup: ingest only if the collection is empty."""
    if not get_vectorstore().get(limit=1)["ids"]:
        ingest_knowledge_base()


# --------------------------------------------------------------------------
# Retrieval
# --------------------------------------------------------------------------
def retrieve(
    query: str,
    k: int = DEFAULT_TOP_K,
    store: Optional[Chroma] = None,
    min_score: Optional[float] = None,
) -> list[dict[str, Any]]:
    """Top-k chunks, best first. `score` is cosine similarity (higher = closer)."""
    if not query.strip():
        return []
    if store is None:
        store = get_vectorstore()

    chunks: list[dict[str, Any]] = []
    for doc, distance in store.similarity_search_with_score(query, k=k):
        score = round(1.0 - float(distance), 4)
        if min_score is not None and score < min_score:
            continue
        chunks.append(
            {
                "source": doc.metadata.get("source", "unknown"),
                "chunk_id": doc.metadata.get("chunk_id", ""),
                "content": doc.page_content,
                "score": score,
            }
        )
    return chunks


def to_source_docs(chunks: list[dict[str, Any]], snippet_chars: int = 200) -> list[SourceDoc]:
    """One SourceDoc per document (best-scoring chunk), for the API and UI."""
    seen: set[str] = set()
    sources: list[SourceDoc] = []
    for chunk in sorted(chunks, key=lambda c: c["score"], reverse=True):
        if chunk["source"] in seen:
            continue
        seen.add(chunk["source"])
        sources.append(
            SourceDoc(
                source=chunk["source"],
                snippet=chunk["content"][:snippet_chars].strip(),
                score=chunk["score"],
            )
        )
    return sources


# --------------------------------------------------------------------------
# CLI:  python -m app.rag --ingest      python -m app.rag --query "..."
# --------------------------------------------------------------------------
def _main() -> None:
    logging.basicConfig(level=get_settings().log_level)
    parser = argparse.ArgumentParser(description="SupportPilot knowledge base tools")
    parser.add_argument("--ingest", action="store_true", help="(re)build the ChromaDB collection")
    parser.add_argument("--query", type=str, help="run a retrieval query")
    args = parser.parse_args()

    if args.ingest:
        print(f"Ingested {ingest_knowledge_base()} chunks.")
    if args.query:
        for i, chunk in enumerate(retrieve(args.query), start=1):
            print(f"\n#{i}  {chunk['source']}  (score={chunk['score']})")
            print(chunk["content"][:300].replace("\n", " "))


if __name__ == "__main__":
    _main()