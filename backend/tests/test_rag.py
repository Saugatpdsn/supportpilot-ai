import pytest

from app.rag import (
    build_vectorstore,
    ingest_knowledge_base,
    load_documents,
    retrieve,
    split_documents,
    to_source_docs,
)

EXPECTED_DOCS = {
    "billing_policy.md",
    "refund_policy.md",
    "account_access_guide.md",
    "troubleshooting_guide.md",
    "product_faq.md",
}


def test_documents_load_and_split_with_unique_ids():
    docs = load_documents()
    assert {d.metadata["source"] for d in docs} == EXPECTED_DOCS

    chunks = split_documents(docs)
    ids = [c.metadata["chunk_id"] for c in chunks]
    assert len(chunks) >= len(docs)
    assert len(ids) == len(set(ids))
    assert all(c.page_content.strip() for c in chunks)


@pytest.mark.parametrize(
    "query, expected_source",
    [
        ("Why was I charged twice for my subscription?", "billing_policy.md"),
        ("How do I get a refund on my annual plan?", "refund_policy.md"),
        ("I forgot my password and need to reset it", "account_access_guide.md"),
        ("The app is slow and pages will not load", "troubleshooting_guide.md"),
    ],
)
def test_retrieval_finds_expected_document(kb_store, query, expected_source):
    results = retrieve(query, k=4, store=kb_store)
    assert expected_source in [r["source"] for r in results]


def test_results_have_expected_shape_and_are_ranked(kb_store):
    results = retrieve("refund policy for annual plans", k=4, store=kb_store)
    assert 0 < len(results) <= 4
    assert {"source", "chunk_id", "content", "score"} <= set(results[0])
    scores = [r["score"] for r in results]
    assert scores == sorted(scores, reverse=True)


def test_empty_query_returns_nothing(kb_store):
    assert retrieve("   ", store=kb_store) == []


def test_min_score_filters_results(kb_store):
    assert retrieve("refund", k=4, store=kb_store, min_score=1.1) == []


def test_to_source_docs_deduplicates_by_document():
    chunks = [
        {"source": "a.md", "chunk_id": "a.md::0", "content": "x" * 300, "score": 0.5},
        {"source": "a.md", "chunk_id": "a.md::1", "content": "y" * 300, "score": 0.9},
        {"source": "b.md", "chunk_id": "b.md::0", "content": "z", "score": 0.7},
    ]
    sources = to_source_docs(chunks)
    assert [s.source for s in sources] == ["a.md", "b.md"]
    assert sources[0].score == 0.9
    assert len(sources[0].snippet) <= 200


def test_ingestion_is_idempotent(tmp_path, kb_embeddings):
    persist_dir = tmp_path / "chroma"
    first = ingest_knowledge_base(embeddings=kb_embeddings, persist_dir=persist_dir)
    second = ingest_knowledge_base(embeddings=kb_embeddings, persist_dir=persist_dir)
    store = build_vectorstore(kb_embeddings, persist_dir)
    assert first == second == len(store.get()["ids"])