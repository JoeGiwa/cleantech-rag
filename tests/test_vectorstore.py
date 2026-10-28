from __future__ import annotations

import logging

import chromadb.api.models.Collection as coll
import httpx
import openai
import pytest
from chromadb.errors import DuplicateIDError
from langchain_core.documents import Document

from cleantech_rag import vectorstore
from cleantech_rag.config import Settings
from cleantech_rag.vectorstore import build_index, estimate_cost, estimate_tokens, load_index

_REQ = httpx.Request("POST", "https://api.openai.com/v1/embeddings")


class _FakeEncoding:
    """Whitespace tokenizer so tests never download tiktoken's encoding."""

    def encode(self, text):
        return text.split()


@pytest.fixture(autouse=True)
def _offline_tokenizer(monkeypatch):
    monkeypatch.setattr(vectorstore, "_get_encoding", lambda: _FakeEncoding())


@pytest.fixture
def settings(tmp_path):
    return Settings(chroma_dir=tmp_path / "chroma", collection_name="test_coll")


def _status_error(cls, status):
    return cls("err", response=httpx.Response(status, request=_REQ), body=None)


def _docs(n=5):
    return [
        Document(
            page_content=f"text {i}",
            metadata={"article_id": str(i), "chunk_id": f"{i}-0", "title": f"T{i}"},
        )
        for i in range(n)
    ]


def _count(store):
    return len(store.get()["ids"])


def _no_sleep(_):
    raise AssertionError("sleep must not be called")


def _fail_embed_times(monkeypatch, fake_embeddings, exc, times):
    orig = type(fake_embeddings).embed_documents
    state = {"calls": 0}

    def flaky(self, texts):
        state["calls"] += 1
        if state["calls"] <= times:
            raise exc
        return orig(self, texts)

    monkeypatch.setattr(type(fake_embeddings), "embed_documents", flaky)
    return state


def test_build_twice_no_duplicates(settings, fake_embeddings):
    build_index(_docs(), settings, fake_embeddings, batch_size=2)
    store = build_index(_docs(), settings, fake_embeddings, batch_size=2)
    assert _count(store) == 5


def test_load_and_search_returns_metadata(settings, fake_embeddings):
    build_index(_docs(), settings, fake_embeddings)
    res = load_index(settings, fake_embeddings).similarity_search("text 3", k=2)
    assert len(res) == 2
    assert all({"article_id", "chunk_id", "title"} <= set(r.metadata) for r in res)
    assert isinstance(res[0].metadata["article_id"], str)


# ---- retry policy ----


def test_rate_limit_is_retried(settings, fake_embeddings, monkeypatch):
    state = _fail_embed_times(
        monkeypatch, fake_embeddings, _status_error(openai.RateLimitError, 429), 2
    )
    sleeps: list[float] = []
    store = build_index(_docs(3), settings, fake_embeddings, sleep=sleeps.append)
    assert state["calls"] == 3
    assert sleeps == [1.0, 2.0]
    assert _count(store) == 3


def test_timeout_is_retried(settings, fake_embeddings, monkeypatch):
    state = _fail_embed_times(monkeypatch, fake_embeddings, openai.APITimeoutError(request=_REQ), 1)
    sleeps: list[float] = []
    store = build_index(_docs(2), settings, fake_embeddings, sleep=sleeps.append)
    assert state["calls"] == 2
    assert sleeps == [1.0]
    assert _count(store) == 2


def test_auth_error_reraised_without_sleep(settings, fake_embeddings, monkeypatch):
    state = _fail_embed_times(
        monkeypatch, fake_embeddings, _status_error(openai.AuthenticationError, 401), 99
    )
    with pytest.raises(openai.AuthenticationError):
        build_index(_docs(2), settings, fake_embeddings, sleep=_no_sleep)
    assert state["calls"] == 1


def test_other_4xx_not_retried(settings, fake_embeddings, monkeypatch):
    state = _fail_embed_times(
        monkeypatch, fake_embeddings, _status_error(openai.BadRequestError, 400), 99
    )
    with pytest.raises(openai.BadRequestError):
        build_index(_docs(2), settings, fake_embeddings, sleep=_no_sleep)
    assert state["calls"] == 1


def test_connection_error_not_retried(settings, fake_embeddings, monkeypatch):
    state = _fail_embed_times(
        monkeypatch, fake_embeddings, openai.APIConnectionError(request=_REQ), 99
    )
    with pytest.raises(openai.APIConnectionError):
        build_index(_docs(2), settings, fake_embeddings, sleep=_no_sleep)
    assert state["calls"] == 1


def test_non_transient_error_not_retried(settings, fake_embeddings, monkeypatch):
    state = _fail_embed_times(monkeypatch, fake_embeddings, ValueError("bad"), 99)
    with pytest.raises(ValueError):
        build_index(_docs(2), settings, fake_embeddings, sleep=_no_sleep)
    assert state["calls"] == 1


def test_retries_exhausted_raises(settings, fake_embeddings, monkeypatch):
    state = _fail_embed_times(monkeypatch, fake_embeddings, TimeoutError("slow"), 99)
    with pytest.raises(TimeoutError):
        build_index(_docs(1), settings, fake_embeddings, max_retries=2, sleep=lambda s: None)
    assert state["calls"] == 3


def test_upsert_retry_does_not_re_embed(settings, fake_embeddings, monkeypatch):
    embed_calls = {"n": 0}
    orig_embed = type(fake_embeddings).embed_documents

    def spy(self, texts):
        embed_calls["n"] += 1
        return orig_embed(self, texts)

    monkeypatch.setattr(type(fake_embeddings), "embed_documents", spy)
    orig_upsert = coll.Collection.upsert
    upserts: list[list] = []

    def flaky(self, *a, **k):
        upserts.append(k["embeddings"])
        if len(upserts) == 1:
            raise TimeoutError("chroma slow")
        return orig_upsert(self, *a, **k)

    monkeypatch.setattr(coll.Collection, "upsert", flaky)
    sleeps: list[float] = []
    store = build_index(_docs(2), settings, fake_embeddings, sleep=sleeps.append)
    assert len(upserts) == 2
    assert embed_calls["n"] == 1  # one batch, embedded exactly once
    assert upserts[0] is upserts[1]
    assert sleeps == [1.0]
    assert _count(store) == 2


def test_retry_only_final_batch(settings, fake_embeddings, monkeypatch):
    orig = type(fake_embeddings).embed_documents
    state = {"calls": 0, "failed": False}

    def flaky(self, texts):
        state["calls"] += 1
        if "text 4" in texts and not state["failed"]:
            state["failed"] = True
            raise TimeoutError("slow")
        return orig(self, texts)

    monkeypatch.setattr(type(fake_embeddings), "embed_documents", flaky)
    sleeps: list[float] = []
    store = build_index(_docs(5), settings, fake_embeddings, batch_size=2, sleep=sleeps.append)
    assert state["calls"] == 4  # 3 batches + 1 retry of the last
    assert sleeps == [1.0]
    assert _count(store) == 5


def test_earlier_batches_persist_when_later_batch_exhausts_retries(
    settings, fake_embeddings, monkeypatch
):
    orig = type(fake_embeddings).embed_documents

    def bad_last(self, texts):
        if "text 4" in texts:
            raise TimeoutError("down")
        return orig(self, texts)

    monkeypatch.setattr(type(fake_embeddings), "embed_documents", bad_last)
    with pytest.raises(TimeoutError):
        build_index(
            _docs(5), settings, fake_embeddings, batch_size=2, max_retries=1, sleep=lambda s: None
        )
    assert _count(load_index(settings, fake_embeddings)) == 4


# ---- cost estimate ----


def test_estimate_logged_before_any_embedding(settings, fake_embeddings, monkeypatch):
    order: list[str] = []
    orig = type(fake_embeddings).embed_documents

    def spy(self, texts):
        order.append("embed")
        return orig(self, texts)

    monkeypatch.setattr(type(fake_embeddings), "embed_documents", spy)
    monkeypatch.setattr(
        logging.getLogger("cleantech_rag.vectorstore"),
        "info",
        lambda *a, **k: order.append("log"),
    )
    build_index(_docs(2), settings, fake_embeddings)
    assert order[0] == "log" and "embed" in order


def test_logs_estimate(settings, fake_embeddings, caplog):
    with caplog.at_level(logging.INFO, logger="cleantech_rag.vectorstore"):
        build_index(_docs(2), settings, fake_embeddings)
    assert "est. cost" in caplog.text
    assert "4 tokens" in caplog.text  # fake tokenizer: 2 words x 2 docs


def test_log_unknown_model_does_not_crash(tmp_path, fake_embeddings, caplog):
    s = Settings(chroma_dir=tmp_path / "c", collection_name="test_coll", embedding_model="weird")
    with caplog.at_level(logging.INFO, logger="cleantech_rag.vectorstore"):
        build_index(_docs(1), s, fake_embeddings)
    assert "unknown" in caplog.text


def test_estimate_tokens_and_cost_values():
    assert estimate_tokens([Document(page_content="a b c d")]) == 4
    assert estimate_tokens([]) == 0
    assert estimate_tokens([Document(page_content="abcdef")], encode=lambda t: list(t)) == 6
    assert estimate_cost(0, "text-embedding-3-small") == 0.0
    assert estimate_cost(1_000_000, "text-embedding-3-small") == pytest.approx(0.02)
    assert estimate_cost(10, "unknown") is None


# ---- validation / edge cases ----


def test_empty_docs_builds_empty_index(settings, fake_embeddings):
    assert _count(build_index([], settings, fake_embeddings)) == 0


def test_missing_chunk_id_raises_before_any_embedding(settings, fake_embeddings, monkeypatch):
    def boom(self, texts):
        raise AssertionError("must not embed")

    monkeypatch.setattr(type(fake_embeddings), "embed_documents", boom)
    docs = _docs(2) + [Document(page_content="a", metadata={})]
    with pytest.raises(ValueError, match="chunk_id"):
        build_index(docs, settings, fake_embeddings)


def test_duplicate_chunk_ids_in_one_batch_raise(settings, fake_embeddings):
    docs = [
        Document(page_content="a", metadata={"chunk_id": "d"}),
        Document(page_content="b", metadata={"chunk_id": "d"}),
    ]
    with pytest.raises(DuplicateIDError):
        build_index(docs, settings, fake_embeddings, sleep=_no_sleep)


def test_duplicate_chunk_ids_across_batches_last_wins(settings, fake_embeddings):
    docs = [
        Document(page_content="old", metadata={"chunk_id": "d"}),
        Document(page_content="new", metadata={"chunk_id": "d"}),
    ]
    store = build_index(docs, settings, fake_embeddings, batch_size=1)
    assert _count(store) == 1
    assert store.get(ids=["d"])["documents"] == ["new"]


def test_rebuild_updates_changed_content(settings, fake_embeddings):
    build_index(_docs(2), settings, fake_embeddings)
    changed = _docs(2)
    changed[0].page_content = "changed"
    store = build_index(changed, settings, fake_embeddings)
    assert _count(store) == 2
    assert store.get(ids=["0-0"])["documents"] == ["changed"]


def test_non_str_chunk_id_is_stringified(settings, fake_embeddings):
    store = build_index(
        [Document(page_content="a", metadata={"chunk_id": 5})], settings, fake_embeddings
    )
    assert store.get(ids=["5"])["ids"] == ["5"]


@pytest.mark.parametrize(
    "n,bs,expected_calls", [(4, 2, 2), (5, 2, 3), (3, 3, 1), (3, 10, 1), (1, 1, 1)]
)
def test_batch_boundaries(settings, fake_embeddings, monkeypatch, n, bs, expected_calls):
    sizes: list[int] = []
    orig = type(fake_embeddings).embed_documents

    def spy(self, texts):
        sizes.append(len(texts))
        return orig(self, texts)

    monkeypatch.setattr(type(fake_embeddings), "embed_documents", spy)
    store = build_index(_docs(n), settings, fake_embeddings, batch_size=bs)
    assert len(sizes) == expected_calls
    assert max(sizes) <= bs
    assert sum(sizes) == n
    assert _count(store) == n


def test_invalid_batch_size_and_retries_raise_before_logging(settings, fake_embeddings, caplog):
    with caplog.at_level(logging.INFO, logger="cleantech_rag.vectorstore"):
        with pytest.raises(ValueError, match="batch_size"):
            build_index(_docs(1), settings, fake_embeddings, batch_size=0)
        with pytest.raises(ValueError, match="max_retries"):
            build_index(_docs(1), settings, fake_embeddings, max_retries=-1)
    assert "Building index" not in caplog.text


def test_none_metadata_value_accepted(settings, fake_embeddings):
    doc = Document(page_content="a", metadata={"chunk_id": "n", "author": None})
    assert _count(build_index([doc], settings, fake_embeddings)) == 1


def test_load_nonexistent_dir_gives_empty_store(settings, fake_embeddings):
    assert not settings.chroma_dir.exists()
    assert load_index(settings, fake_embeddings).similarity_search("x", k=2) == []
