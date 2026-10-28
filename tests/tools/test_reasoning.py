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


def test_artifact_has_documents_and_tool_call_id(fake_embeddings, fake_chat_model_factory):
    llm = fake_chat_model_factory([AIMessage(content="a")])
    msg = _call(make_reasoning_tool(_store(fake_embeddings), llm, k=2))
    assert msg.tool_call_id == "1"
    assert len(msg.artifact["documents"]) == len(msg.artifact["article_ids"]) == 2


def test_k_and_overfetch_forwarded(monkeypatch, fake_chat_model_factory):
    seen = {}

    def fake_retrieve(store, query, k, overfetch):
        seen.update(query=query, k=k, overfetch=overfetch)
        return RetrievalResult()

    monkeypatch.setattr(reasoning, "retrieve_articles", fake_retrieve)
    llm = fake_chat_model_factory([])
    _call(make_reasoning_tool(object(), llm, k=7, overfetch=4), q="Q?")
    assert seen == {"query": "Q?", "k": 7, "overfetch": 4}


def test_empty_store_skips_llm(fake_embeddings, fake_chat_model_factory):
    llm = fake_chat_model_factory([AIMessage(content="x")])
    msg = _call(make_reasoning_tool(InMemoryVectorStore(fake_embeddings), llm))
    assert "No relevant articles" in msg.content
    assert msg.artifact["article_ids"] == [] and llm.calls == []


def test_question_in_prompt_and_non_string_content(monkeypatch, fake_chat_model_factory):
    doc = Document(page_content="CTX", metadata={"article_id": "1"})
    monkeypatch.setattr(
        reasoning, "retrieve_articles", lambda *a, **k: RetrievalResult(["1"], [doc])
    )
    content = [
        {"type": "text", "text": "Step 1. "},
        {"type": "tool_use", "id": "t", "name": "x", "input": {}},
        "Step 2.",
    ]
    llm = fake_chat_model_factory([AIMessage(content=content)])
    msg = _call(make_reasoning_tool(object(), llm), q="MYQ")
    assert "MYQ" in llm.calls[0][0].content and "CTX" in llm.calls[0][0].content
    assert msg.content == "Step 1. Step 2."
