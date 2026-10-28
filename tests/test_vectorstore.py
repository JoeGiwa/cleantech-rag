from __future__ import annotations

import logging

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding

from cleantech_rag.config import Settings
from cleantech_rag.vectorstore import build_index, estimate_cost, load_index


@pytest.fixture
def settings(tmp_path):
    return Settings(chroma_dir=tmp_path / "chroma", collection_name="test_coll")


@pytest.fixture
def emb():
    return DeterministicFakeEmbedding(size=8)


def _docs(n=5):
    return [
        Document(
            page_content=f"text {i}",
            metadata={"article_id": str(i), "chunk_id": f"{i}-0", "title": f"T{i}"},
        )
        for i in range(n)
    ]


def test_build_twice_no_duplicates(settings, emb):
    build_index(_docs(), settings, emb, batch_size=2)
    store = build_index(_docs(), settings, emb, batch_size=2)
    assert store._collection.count() == 5


def test_load_and_search_returns_metadata(settings, emb):
    build_index(_docs(), settings, emb)
    res = load_index(settings, emb).similarity_search("text 3", k=2)
    assert len(res) == 2
    assert all({"article_id", "chunk_id", "title"} <= set(r.metadata) for r in res)
    assert isinstance(res[0].metadata["article_id"], str)


def test_failing_batch_is_retried(settings, emb, monkeypatch):
    calls = {"n": 0}
    orig = emb.embed_documents

    def flaky(texts):
        calls["n"] += 1
        if calls["n"] <= 2:
            raise RuntimeError("boom")
        return orig(texts)

    monkeypatch.setattr(type(emb), "embed_documents", lambda self, t: flaky(t))
    sleeps: list[float] = []
    store = build_index(_docs(3), settings, emb, sleep=sleeps.append)
    assert calls["n"] == 3
    assert sleeps == [1.0, 2.0]
    assert store._collection.count() == 3


def test_persistent_failure_raises(settings, emb, monkeypatch):
    def bad(texts):
        raise RuntimeError("down")

    monkeypatch.setattr(type(emb), "embed_documents", lambda self, t: bad(t))
    with pytest.raises(RuntimeError):
        build_index(_docs(1), settings, emb, max_retries=2, sleep=lambda s: None)


def test_logs_estimate_before_build(settings, emb, caplog):
    with caplog.at_level(logging.INFO, logger="cleantech_rag.vectorstore"):
        build_index(_docs(2), settings, emb)
    assert "est. cost" in caplog.text


def test_estimate_cost():
    assert estimate_cost(1_000_000, "text-embedding-3-small") == pytest.approx(0.02)
    assert estimate_cost(10, "unknown") is None
