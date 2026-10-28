from __future__ import annotations

import logging

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding

from cleantech_rag.config import Settings
from cleantech_rag.vectorstore import build_index, estimate_cost, estimate_tokens, load_index


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


# ---- edge cases ----


def test_empty_docs_builds_empty_index(settings, emb):
    store = build_index([], settings, emb)
    assert store._collection.count() == 0


def test_missing_chunk_id_raises_keyerror(settings, emb):
    with pytest.raises(KeyError):
        build_index([Document(page_content="a", metadata={})], settings, emb)


def test_duplicate_chunk_ids_in_one_batch_raise(settings, emb):
    docs = [
        Document(page_content="a", metadata={"chunk_id": "d"}),
        Document(page_content="b", metadata={"chunk_id": "d"}),
    ]
    with pytest.raises(Exception, match="unique|duplicate"):
        build_index(docs, settings, emb, max_retries=0, sleep=lambda s: None)


def test_duplicate_chunk_ids_across_batches_last_wins(settings, emb):
    docs = [
        Document(page_content="old", metadata={"chunk_id": "d"}),
        Document(page_content="new", metadata={"chunk_id": "d"}),
    ]
    store = build_index(docs, settings, emb, batch_size=1)
    assert store._collection.count() == 1
    assert store._collection.get(ids=["d"])["documents"] == ["new"]


def test_rebuild_updates_changed_content(settings, emb):
    build_index(_docs(2), settings, emb)
    changed = _docs(2)
    changed[0].page_content = "changed"
    store = build_index(changed, settings, emb)
    assert store._collection.count() == 2
    assert store._collection.get(ids=["0-0"])["documents"] == ["changed"]


def test_non_str_chunk_id_is_stringified(settings, emb):
    store = build_index([Document(page_content="a", metadata={"chunk_id": 5})], settings, emb)
    assert store._collection.get(ids=["5"])["ids"] == ["5"]


@pytest.mark.parametrize(
    "n,bs,expected_calls", [(4, 2, 2), (5, 2, 3), (3, 3, 1), (3, 10, 1), (1, 1, 1)]
)
def test_batch_boundaries(settings, emb, monkeypatch, n, bs, expected_calls):
    sizes: list[int] = []
    orig = type(emb).embed_documents

    def spy(self, texts):
        sizes.append(len(texts))
        return orig(self, texts)

    monkeypatch.setattr(type(emb), "embed_documents", spy)
    store = build_index(_docs(n), settings, emb, batch_size=bs)
    assert len(sizes) == expected_calls
    assert max(sizes) <= bs
    assert sum(sizes) == n
    assert store._collection.count() == n


def test_batch_size_zero_raises(settings, emb):
    with pytest.raises(ValueError):
        build_index(_docs(1), settings, emb, batch_size=0)


def test_none_metadata_value_accepted(settings, emb):
    doc = Document(page_content="a", metadata={"chunk_id": "n", "author": None})
    store = build_index([doc], settings, emb)
    assert store._collection.count() == 1


def test_load_nonexistent_dir_gives_empty_store(settings, emb):
    assert not settings.chroma_dir.exists()
    assert load_index(settings, emb).similarity_search("x", k=2) == []


def test_retry_only_final_batch(settings, emb, monkeypatch):
    orig = type(emb).embed_documents
    state = {"calls": 0, "failed": False}

    def flaky(self, texts):
        state["calls"] += 1
        if "text 4" in texts and not state["failed"]:
            state["failed"] = True
            raise RuntimeError("boom")
        return orig(self, texts)

    monkeypatch.setattr(type(emb), "embed_documents", flaky)
    sleeps: list[float] = []
    store = build_index(_docs(5), settings, emb, batch_size=2, sleep=sleeps.append)
    assert state["calls"] == 4  # 3 batches + 1 retry of the last
    assert sleeps == [1.0]
    assert store._collection.count() == 5


def test_earlier_batches_persist_when_later_batch_exhausts_retries(settings, emb, monkeypatch):
    orig = type(emb).embed_documents

    def bad_last(self, texts):
        if "text 4" in texts:
            raise RuntimeError("down")
        return orig(self, texts)

    monkeypatch.setattr(type(emb), "embed_documents", bad_last)
    with pytest.raises(RuntimeError):
        build_index(_docs(5), settings, emb, batch_size=2, max_retries=1, sleep=lambda s: None)
    assert load_index(settings, emb)._collection.count() == 4


def test_upsert_failure_is_retried(settings, emb, monkeypatch):
    import chromadb.api.models.Collection as coll

    orig = coll.Collection.upsert
    n = {"c": 0}

    def flaky(self, *a, **k):
        n["c"] += 1
        if n["c"] == 1:
            raise RuntimeError("chroma hiccup")
        return orig(self, *a, **k)

    monkeypatch.setattr(coll.Collection, "upsert", flaky)
    store = build_index(_docs(2), settings, emb, sleep=lambda s: None)
    assert n["c"] == 2
    assert store._collection.count() == 2


def test_estimate_logged_before_any_embedding(settings, emb, caplog, monkeypatch):
    order: list[str] = []
    orig = type(emb).embed_documents

    def spy(self, texts):
        order.append("embed")
        return orig(self, texts)

    monkeypatch.setattr(type(emb), "embed_documents", spy)
    handler_records = caplog.records
    with caplog.at_level(logging.INFO, logger="cleantech_rag.vectorstore"):
        monkeypatch.setattr(
            logging.getLogger("cleantech_rag.vectorstore"),
            "info",
            lambda *a, **k: order.append("log"),
        )
        build_index(_docs(2), settings, emb)
    assert order[0] == "log" and "embed" in order
    del handler_records


def test_estimate_tokens_and_cost_values():
    docs = [Document(page_content="a" * 400)]
    assert estimate_tokens(docs) == 100
    assert estimate_tokens([]) == 0
    assert estimate_cost(0, "text-embedding-3-small") == 0.0


def test_log_unknown_model_does_not_crash(tmp_path, emb, caplog):
    s = Settings(chroma_dir=tmp_path / "c", collection_name="test_coll", embedding_model="weird")
    with caplog.at_level(logging.INFO, logger="cleantech_rag.vectorstore"):
        build_index(_docs(1), s, emb)
    assert "unknown" in caplog.text
