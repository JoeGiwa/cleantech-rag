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


# ---- edge cases -------------------------------------------------------------------------------

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage  # noqa: E402
from langchain_core.tools import StructuredTool  # noqa: E402

from cleantech_rag.agent import Agent, build_graph, merge_ids  # noqa: E402


def _stub_tool(name, ids, artifact=True):
    """Tool returning fixed IDs (or a bare string with no artifact)."""
    if artifact:

        def fn(query: str):
            return "text", {"article_ids": list(ids)}

        return StructuredTool.from_function(
            fn, name=name, description="d", response_format="content_and_artifact"
        )

    def plain(query: str) -> str:
        return "text"

    return StructuredTool.from_function(plain, name=name, description="d")


def _stub_agent(factory, responses, tools, k=3, **kw):
    llm = factory(responses)
    return Agent(build_graph(llm, tools, k=k, **kw)), llm


def _multi(*calls):
    return AIMessage(
        content="",
        tool_calls=[
            {"name": n, "args": {"query": "q"}, "id": f"c{i}"} for i, n in enumerate(calls)
        ],
    )


def test_merge_ids_dedup_order_and_cut():
    assert merge_ids(["a", "b"], ["b", "c", "d"], 3) == ["a", "b", "c"]
    assert merge_ids([], [], 3) == []
    assert merge_ids(["a"], ["a"], 3) == ["a"]


def test_union_exceeding_k_is_cut_in_call_order(fake_chat_model_factory):
    t1, t2 = _stub_tool("retrieve", ["1", "2", "3"]), _stub_tool("reasoning_chain", ["3", "4", "5"])
    agent, _ = _stub_agent(
        fake_chat_model_factory,
        [_multi("retrieve", "reasoning_chain"), AIMessage(content="x")],
        [t1, t2],
        k=4,
    )
    assert agent.run("q").retrieved_ids == ["1", "2", "3", "4"]


def test_parallel_tool_calls_in_one_message(fake_chat_model_factory):
    tools = [
        _stub_tool("retrieve", ["1", "2"]),
        _stub_tool("web_search", [], artifact=False),
        _stub_tool("reasoning_chain", ["2", "9"]),
    ]
    agent, _ = _stub_agent(
        fake_chat_model_factory,
        [_multi("retrieve", "web_search", "reasoning_chain"), AIMessage(content="ok")],
        tools,
        k=5,
    )
    res = agent.run("q")
    assert res.retrieved_ids == ["1", "2", "9"]
    assert res.tools_used == ["retrieve", "web_search", "reasoning_chain"]


def test_tools_used_keeps_order_and_repeats(fake_chat_model_factory):
    tools = [_stub_tool("retrieve", ["1"]), _stub_tool("web_search", [], artifact=False)]
    agent, _ = _stub_agent(
        fake_chat_model_factory,
        [_multi("retrieve"), _multi("web_search"), _multi("retrieve"), AIMessage(content="ok")],
        tools,
    )
    assert agent.run("q").tools_used == ["retrieve", "web_search", "retrieve"]


def test_tool_without_artifact_does_not_break_or_add_ids(fake_chat_model_factory):
    agent, _ = _stub_agent(
        fake_chat_model_factory,
        [_multi("retrieve"), AIMessage(content="ok")],
        [_stub_tool("retrieve", [], artifact=False)],
    )
    res = agent.run("q")
    assert res.retrieved_ids == [] and res.answer == "ok"


def test_numeric_artifact_ids_become_str(fake_chat_model_factory):
    agent, _ = _stub_agent(
        fake_chat_model_factory,
        [_multi("retrieve"), AIMessage(content="ok")],
        [_stub_tool("retrieve", [7, 8])],
    )
    assert agent.run("q").retrieved_ids == ["7", "8"]


def test_raising_tool_is_reported_to_model_not_crash(fake_chat_model_factory):
    def boom(query: str) -> str:
        raise ValueError("kaboom")

    tool = StructuredTool.from_function(boom, name="retrieve", description="d")
    agent, llm = _stub_agent(
        fake_chat_model_factory, [_multi("retrieve"), AIMessage(content="recovered")], [tool]
    )
    res = agent.run("q")
    assert res.answer == "recovered" and res.retrieved_ids == []
    tool_msgs = [m for m in llm.calls[-1] if isinstance(m, ToolMessage)]
    assert tool_msgs and "kaboom" in tool_msgs[0].content


def test_unknown_tool_name_is_reported_not_crash(fake_chat_model_factory):
    agent, llm = _stub_agent(
        fake_chat_model_factory,
        [_multi("nonexistent"), AIMessage(content="ok")],
        [_stub_tool("retrieve", ["1"])],
    )
    res = agent.run("q")
    assert res.answer == "ok" and res.retrieved_ids == []


def test_direct_answer_without_tools(fake_chat_model_factory):
    agent, _ = _stub_agent(fake_chat_model_factory, [AIMessage(content="hi")], [])
    res = agent.run("q")
    assert (res.answer, res.retrieved_ids, res.tools_used) == ("hi", [], [])


def test_empty_final_answer(fake_chat_model_factory):
    agent, _ = _stub_agent(fake_chat_model_factory, [AIMessage(content="")], [])
    assert agent.run("q").answer == ""


def test_list_content_final_answer_is_joined(fake_chat_model_factory):
    msg = AIMessage(content=[{"type": "text", "text": "foo"}, {"type": "text", "text": "bar"}])
    agent, _ = _stub_agent(fake_chat_model_factory, [msg], [])
    answer = agent.run("q").answer
    assert "foo" in answer and "bar" in answer


def test_consecutive_runs_do_not_share_messages(fake_chat_model_factory):
    agent, llm = _stub_agent(
        fake_chat_model_factory,
        [_multi("retrieve"), AIMessage(content="one"), AIMessage(content="two")],
        [_stub_tool("retrieve", ["1"])],
    )
    agent.run("first question")
    assert agent.run("second question").answer == "two"
    second_call = llm.calls[-1]
    assert isinstance(second_call[0], SystemMessage)
    humans = [m.content for m in second_call if isinstance(m, HumanMessage)]
    assert humans == ["second question"]
    assert not any(isinstance(m, ToolMessage) for m in second_call)


def test_ids_reset_between_runs_with_stub_tools(fake_chat_model_factory):
    tools = [_stub_tool("retrieve", ["1", "2"]), _stub_tool("web_search", [], artifact=False)]
    agent, _ = _stub_agent(
        fake_chat_model_factory,
        [
            _multi("retrieve"),
            AIMessage(content="a"),
            _multi("web_search"),
            AIMessage(content="b"),
        ],
        tools,
    )
    assert agent.run("a").retrieved_ids == ["1", "2"]
    assert agent.run("b").retrieved_ids == []


def test_result_lists_are_independent_copies(fake_chat_model_factory):
    agent, _ = _stub_agent(
        fake_chat_model_factory,
        [_multi("retrieve"), AIMessage(content="a"), _multi("retrieve"), AIMessage(content="b")],
        [_stub_tool("retrieve", ["1"])],
    )
    r1 = agent.run("a")
    r1.retrieved_ids.append("junk")
    assert agent.run("b").retrieved_ids == ["1"]


def test_recursion_limit_default_and_custom(fake_embeddings, fake_chat_model_factory):
    agent = _agent(fake_embeddings, fake_chat_model_factory, [AIMessage(content="x")])
    assert agent.recursion_limit == 12
    agent = _agent(
        fake_embeddings, fake_chat_model_factory, [AIMessage(content="x")], recursion_limit=3
    )
    assert agent.recursion_limit == 3


def test_no_agent_executor_anywhere_in_repo_python():
    root = Path(__file__).resolve().parents[1]
    for base in ("src", "tests"):
        for f in (root / base).rglob("*.py"):
            if f == Path(__file__).resolve():
                continue
            text = f.read_text()
            assert "AgentExecutor" not in text and "create_tool_calling_agent" not in text, f
