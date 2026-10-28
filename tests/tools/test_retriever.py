from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.vectorstores import InMemoryVectorStore

from cleantech_rag.tools.retriever import make_retriever_tool, retrieve_articles


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
