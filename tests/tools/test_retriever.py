from pathlib import Path

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.messages import ToolMessage
from langchain_core.vectorstores import InMemoryVectorStore

import cleantech_rag
from cleantech_rag.tools.retriever import (
    format_context,
    make_retriever_tool,
    retrieve_articles,
)


def _store(docs):
    s = InMemoryVectorStore(DeterministicFakeEmbedding(size=32))
    s.add_documents(docs)
    return s


def _doc(text, aid, idx=0):
    return Document(
        page_content=text,
        metadata={
            "article_id": aid,
            "chunk_index": idx,
            "title": f"T{aid}",
            "date": "2024-01-01",
            "url": f"http://x/{aid}",
        },
    )


def test_chunks_collapse_to_best_rank():
    # Query equals text of A's chunk 1 exactly -> A is rank 1; B second; A's other chunk dropped.
    store = _store(
        [_doc("alpha", 1, 0), _doc("query text", 1, 1), _doc("beta", 2, 0), _doc("gamma", 3, 0)]
    )
    res = retrieve_articles(store, "query text", k=3)
    assert res.article_ids[0] == 1
    assert sorted(res.article_ids) == [1, 2, 3]
    assert len(res.documents) == 3
    assert res.documents[0].metadata["chunk_index"] == 1


def test_k_limits_articles():
    store = _store([_doc(f"t{i}", i) for i in range(10)])
    assert len(retrieve_articles(store, "t1", k=2).article_ids) == 2


def test_tool_content_and_artifact_per_call():
    store = _store([_doc("solar panels", 1), _doc("wind farms", 2)])
    tool = make_retriever_tool(store, k=1)

    def call(q):
        return tool.invoke(
            {"name": "retrieve", "args": {"query": q}, "id": "c1", "type": "tool_call"}
        )

    m1 = call("solar panels")
    m2 = call("wind farms")
    assert m1.artifact["article_ids"] == [1]
    assert m2.artifact["article_ids"] == [2]
    assert "Title: T1" in m1.content and "http://x/1" in m1.content and "solar panels" in m1.content
    assert not hasattr(tool, "last_retrieved_ids")


def test_empty_store():
    tool = make_retriever_tool(_store([]), k=3)
    msg = tool.invoke({"name": "retrieve", "args": {"query": "q"}, "id": "c", "type": "tool_call"})
    assert msg.artifact["article_ids"] == []
    assert "No relevant" in msg.content


def test_best_rank_when_worse_chunk_listed_first():
    # A's weak chunk is inserted first; the exact-match chunk must still win rank 1.
    store = _store([_doc("weak", 1, 0), _doc("beta", 2, 0), _doc("exact query", 1, 1)])
    res = retrieve_articles(store, "exact query", k=2)
    assert res.article_ids[0] == 1
    assert res.article_ids.count(1) == 1
    assert res.documents[0].page_content == "exact query"


def test_ids_unique_and_aligned_with_documents():
    store = _store([_doc(f"c{i}", i % 3, i) for i in range(9)])
    res = retrieve_articles(store, "c0", k=5)
    assert len(res.article_ids) == len(set(res.article_ids)) == 3
    assert [d.metadata["article_id"] for d in res.documents] == res.article_ids


def test_overfetch_fills_k_articles_from_many_chunks():
    # One article has many chunks that would crowd out others without over-fetching.
    docs = [_doc(f"query {i}", 1, i) for i in range(3)] + [_doc("other", 2), _doc("more", 3)]
    res = retrieve_articles(_store(docs), "query 0", k=3)
    assert sorted(res.article_ids) == [1, 2, 3]


@pytest.mark.parametrize("k", [0, -1])
def test_non_positive_k_returns_empty(k):
    res = retrieve_articles(_store([_doc("a", 1)]), "a", k=k)
    assert res.article_ids == [] and res.documents == []


def test_missing_article_id_skipped():
    noid = Document(page_content="orphan", metadata={"title": "x"})
    res = retrieve_articles(_store([noid, _doc("a", 1)]), "orphan", k=5)
    assert res.article_ids == [1]
    assert all(d.metadata.get("article_id") is not None for d in res.documents)


def test_article_id_zero_is_kept():
    res = retrieve_articles(_store([_doc("a", 0)]), "a", k=1)
    assert res.article_ids == [0]


def test_format_context_missing_metadata_fallbacks():
    out = format_context([Document(page_content="body", metadata={})])
    assert "Title: N/A" in out and "Date: N/A" in out and "URL: N/A" in out and "body" in out


def test_format_context_empty_and_numbering():
    assert format_context([]) == "No relevant articles found."
    out = format_context([_doc("one", 1), _doc("two", 2)])
    assert out.index("[1] Title: T1") < out.index("[2] Title: T2")
    assert "Date: 2024-01-01" in out


def test_tool_metadata_and_artifact_documents():
    tool = make_retriever_tool(_store([_doc("solar", 1)]), k=1)
    assert tool.name == "retrieve"
    assert tool.response_format == "content_and_artifact"
    msg = tool.invoke(
        {"name": "retrieve", "args": {"query": "solar"}, "id": "x", "type": "tool_call"}
    )
    assert isinstance(msg, ToolMessage)
    assert msg.tool_call_id == "x"
    assert isinstance(msg.artifact["documents"][0], Document)


def test_tool_direct_invoke_returns_content_only():
    tool = make_retriever_tool(_store([_doc("solar", 1)]), k=1)
    assert isinstance(tool.invoke({"query": "solar"}), str)


def test_artifact_not_leaked_across_calls_after_empty_result():
    store = _store([_doc("solar", 1)])
    tool = make_retriever_tool(store, k=1)
    call = lambda q: tool.invoke(  # noqa: E731
        {"name": "retrieve", "args": {"query": q}, "id": "c", "type": "tool_call"}
    )
    first = call("solar")
    first.artifact["article_ids"].append(999)  # mutate; must not affect later calls
    second = call("solar")
    assert second.artifact["article_ids"] == [1]


def test_no_instance_state_on_tool():
    tool = make_retriever_tool(_store([_doc("a", 1)]), k=1)
    assert not [a for a in vars(tool) if "retrieved" in a or "last" in a]


def test_no_summarizer_in_package():
    root = Path(cleantech_rag.__file__).parent
    assert not list(root.rglob("summarizer.py"))
    for py in root.rglob("*.py"):
        assert "summarize_and_answer" not in py.read_text()
