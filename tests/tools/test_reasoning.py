from langchain_core.documents import Document
from langchain_core.messages import AIMessage
from langchain_core.vectorstores import InMemoryVectorStore

from cleantech_rag.tools import reasoning
from cleantech_rag.tools.reasoning import make_reasoning_tool
from cleantech_rag.tools.retriever import RetrievalResult


def _call(tool, q="why?"):
    return tool.invoke(
        {"type": "tool_call", "name": "reasoning_chain", "args": {"question": q}, "id": "1"}
    )


def _store(fake_embeddings):
    s = InMemoryVectorStore(fake_embeddings)
    s.add_documents(
        [
            Document(page_content=f"text {i}", metadata={"article_id": str(i), "title": f"T{i}"})
            for i in range(1, 4)
        ]
    )
    return s


def test_returns_retrieved_ids_as_artifact(fake_embeddings, fake_chat_model_factory):
    llm = fake_chat_model_factory([AIMessage(content="step 1... conclusion")])
    msg = _call(make_reasoning_tool(_store(fake_embeddings), llm, k=2))
    assert msg.content == "step 1... conclusion"
    ids = msg.artifact["article_ids"]
    assert len(ids) == 2 and all(isinstance(i, str) for i in ids)
    assert "text" in llm.calls[0][0].content  # context reached the LLM


def test_uses_only_retriever_function(monkeypatch, fake_chat_model_factory):
    class Boom:
        def __getattr__(self, name):
            raise AssertionError(f"vectorstore touched directly: {name}")

    seen = {}

    def fake_retrieve(store, query, k, overfetch):
        seen["store"] = store
        return RetrievalResult(
            article_ids=["7"], documents=[Document(page_content="c", metadata={"article_id": "7"})]
        )

    monkeypatch.setattr(reasoning, "retrieve_articles", fake_retrieve)
    store = Boom()
    llm = fake_chat_model_factory([AIMessage(content="ok")])
    msg = _call(make_reasoning_tool(store, llm))
    assert seen["store"] is store
    assert msg.artifact["article_ids"] == ["7"]


def test_no_results_skips_llm(monkeypatch, fake_chat_model_factory):
    monkeypatch.setattr(reasoning, "retrieve_articles", lambda *a, **k: RetrievalResult())
    llm = fake_chat_model_factory([AIMessage(content="should not be used")])
    msg = _call(make_reasoning_tool(object(), llm))
    assert "No relevant articles" in msg.content
    assert msg.artifact["article_ids"] == []
    assert llm.calls == []
