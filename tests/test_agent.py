from pathlib import Path

import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessage
from langchain_core.vectorstores import InMemoryVectorStore
from langgraph.errors import GraphRecursionError

from cleantech_rag.agent import AgentResult, create_agent
from cleantech_rag.config import Settings


def _store(emb, n=8):
    s = InMemoryVectorStore(emb)
    s.add_documents(
        [
            Document(page_content=f"solar text {i}", metadata={"article_id": str(i), "title": "T"})
            for i in range(n)
        ]
    )
    return s


def _call(name, q, cid):
    arg = "question" if name == "reasoning_chain" else "query"
    return AIMessage(content="", tool_calls=[{"name": name, "args": {arg: q}, "id": cid}])


def _agent(emb, factory, responses, **kw):
    llm = factory(responses)
    return create_agent(_store(emb), llm, Settings(top_k=3, _env_file=None), **kw)


def test_retrieve_then_answer(fake_embeddings, fake_chat_model_factory):
    agent = _agent(
        fake_embeddings,
        fake_chat_model_factory,
        [_call("retrieve", "solar", "1"), AIMessage(content="The answer.")],
    )
    res = agent.run("q")
    assert isinstance(res, AgentResult)
    assert res.answer == "The answer."
    assert len(res.retrieved_ids) == 3 and all(isinstance(i, str) for i in res.retrieved_ids)
    assert res.tools_used == ["retrieve"]


def test_web_search_run_does_not_leak_previous_ids(fake_embeddings, fake_chat_model_factory):
    llm = fake_chat_model_factory(
        [
            _call("retrieve", "solar", "1"),
            AIMessage(content="first"),
            _call("web_search", "news", "2"),
            AIMessage(content="second"),
        ]
    )
    agent = create_agent(_store(fake_embeddings), llm, Settings(top_k=3, _env_file=None))
    assert agent.run("a").retrieved_ids != []
    res = agent.run("b")
    assert res.retrieved_ids == []
    assert res.tools_used == ["web_search"]
    assert res.answer == "second"


def test_two_retrievals_dedup_union_cut_to_k(fake_embeddings, fake_chat_model_factory):
    store = _store(fake_embeddings)
    from cleantech_rag.tools.retriever import retrieve_articles

    first = retrieve_articles(store, "solar a", k=3).article_ids
    second = retrieve_articles(store, "solar b", k=3).article_ids
    expected = list(dict.fromkeys(first + second))[:3]
    agent = _agent(
        fake_embeddings,
        fake_chat_model_factory,
        [
            _call("retrieve", "solar a", "1"),
            _call("retrieve", "solar b", "2"),
            AIMessage(content="done"),
        ],
    )
    res = agent.run("q")
    assert res.retrieved_ids == expected
    assert len(set(res.retrieved_ids)) == len(res.retrieved_ids) <= 3


def test_reasoning_ids_toggle(fake_embeddings, fake_chat_model_factory):
    script = [_call("reasoning_chain", "why", "1"), AIMessage(content="r"), AIMessage(content="d")]
    on = _agent(fake_embeddings, fake_chat_model_factory, list(script)).run("q")
    off = _agent(
        fake_embeddings, fake_chat_model_factory, list(script), count_reasoning_ids=False
    ).run("q")
    assert on.retrieved_ids and off.retrieved_ids == []


def test_recursion_limit_enforced(fake_embeddings, fake_chat_model_factory):
    agent = _agent(
        fake_embeddings,
        fake_chat_model_factory,
        [_call("web_search", "loop", str(i)) for i in range(20)],  # never answers
        recursion_limit=5,
    )
    with pytest.raises(GraphRecursionError):
        agent.run("q")


def test_no_agent_executor_in_src():
    src = Path(__file__).resolve().parents[1] / "src"
    for f in src.rglob("*.py"):
        assert "AgentExecutor" not in f.read_text(), f
