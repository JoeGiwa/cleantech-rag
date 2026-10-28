from pathlib import Path

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import StructuredTool
from langchain_core.vectorstores import InMemoryVectorStore

from cleantech_rag.agent import (
    FIRST_TURN_NOTE,
    STEP_LIMIT_ANSWER,
    SYSTEM_PROMPT,
    Agent,
    AgentResult,
    build_graph,
    create_agent,
    merge_ids,
)
from cleantech_rag.config import Settings
from cleantech_rag.tools.retriever import retrieve_articles


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
    assert res.tools_used == [{"tool": "retrieve", "ok": True}]
    assert res.stopped_reason == "completed"


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
    # no BRAVE_API_KEY in tests: the disabled tool is a failure, not ok
    assert res.tools_used == [{"tool": "web_search", "ok": False}]
    assert res.answer == "second"


def test_two_retrievals_dedup_union_cut_to_k(fake_embeddings, fake_chat_model_factory):
    store = _store(fake_embeddings)

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


def test_retrieve_plus_reasoning_merged_deduped_cut_to_k(fake_embeddings, fake_chat_model_factory):
    store = _store(fake_embeddings)
    first = retrieve_articles(store, "solar a", k=3).article_ids
    second = retrieve_articles(store, "why", k=3).article_ids
    expected = list(dict.fromkeys(first + second))[:3]
    agent = _agent(
        fake_embeddings,
        fake_chat_model_factory,
        [
            _call("retrieve", "solar a", "1"),
            _call("reasoning_chain", "why", "2"),
            AIMessage(content="r"),  # reasoning tool's internal LLM call
            AIMessage(content="done"),
        ],
    )
    res = agent.run("q")
    assert res.retrieved_ids == expected and len(res.retrieved_ids) <= 3
    assert res.tools_used == [
        {"tool": "retrieve", "ok": True},
        {"tool": "reasoning_chain", "ok": True},
    ]


def test_step_limit_returns_fallback_with_partial_state(fake_chat_model_factory):
    counter = iter(range(100))

    def fn(query: str):
        n = next(counter)
        return "text", {"article_ids": [f"a{n}", f"b{n}"]}

    tool = StructuredTool.from_function(
        fn, name="retrieve", description="d", response_format="content_and_artifact"
    )
    agent, _ = _stub_agent(
        fake_chat_model_factory,
        [_multi("retrieve") for _ in range(20)],  # never answers
        [tool],
        k=10,
        recursion_limit=5,
    )
    res = agent.run("q")
    assert res.stopped_reason == "step_limit"
    assert res.answer == STEP_LIMIT_ANSWER
    assert res.retrieved_ids == ["a0", "b0", "a1", "b1"]
    assert res.tools_used == [{"tool": "retrieve", "ok": True}] * 2


def test_final_answer_on_last_allowed_step_is_completed(fake_embeddings, fake_chat_model_factory):
    agent = _agent(
        fake_embeddings,
        fake_chat_model_factory,
        [_call("retrieve", "solar", "1"), AIMessage(content="final")],
        recursion_limit=3,
    )
    res = agent.run("q")
    assert res.stopped_reason == "completed"
    assert res.answer == "final"
    assert len(res.retrieved_ids) == 3


# ---- edge cases ----------------------------------------------------------------------------


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


def _stub_agent(factory, responses, tools, k=3, recursion_limit=12, **kw):
    llm = factory(responses)
    return Agent(build_graph(llm, tools, k=k, **kw), recursion_limit), llm


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
    assert [t["tool"] for t in res.tools_used] == ["retrieve", "web_search", "reasoning_chain"]


def test_tools_used_keeps_order_and_repeats(fake_chat_model_factory):
    tools = [_stub_tool("retrieve", ["1"]), _stub_tool("web_search", [], artifact=False)]
    agent, _ = _stub_agent(
        fake_chat_model_factory,
        [_multi("retrieve"), _multi("web_search"), _multi("retrieve"), AIMessage(content="ok")],
        tools,
    )
    assert [t["tool"] for t in agent.run("q").tools_used] == ["retrieve", "web_search", "retrieve"]


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
    assert res.tools_used == [{"tool": "retrieve", "ok": False}]
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
    assert res.tools_used == [{"tool": "nonexistent", "ok": False}]


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


def test_recursion_limit_comes_from_settings(fake_embeddings, fake_chat_model_factory):
    llm = fake_chat_model_factory([AIMessage(content="x")])
    agent = create_agent(
        _store(fake_embeddings), llm, Settings(agent_recursion_limit=7, _env_file=None)
    )
    assert agent.recursion_limit == 7


# ---- issue 30 cleanups ---------------------------------------------------------------------


def test_parallel_tool_calls_at_step_limit_close_all_calls(fake_chat_model_factory):
    tools = [_stub_tool("retrieve", ["1"]), _stub_tool("web_search", [], artifact=False)]
    agent, llm = _stub_agent(
        fake_chat_model_factory,
        [_multi("retrieve", "web_search"), AIMessage(content="next turn ok")],
        tools,
        recursion_limit=1,  # the agent emits parallel calls, then the limit trips before tools
    )
    res = agent.run("q", thread_id="t")
    assert res.stopped_reason == "step_limit" and res.answer == STEP_LIMIT_ANSWER
    msgs = agent.history("t")
    ai = next(m for m in msgs if isinstance(m, AIMessage) and m.tool_calls)
    answered = {m.tool_call_id for m in msgs if isinstance(m, ToolMessage)}
    assert answered == {c["id"] for c in ai.tool_calls}
    assert isinstance(msgs[-1], AIMessage) and msgs[-1].content == STEP_LIMIT_ANSWER
    agent.recursion_limit = 5
    assert agent.run("again", thread_id="t").answer == "next turn ok"  # thread still valid


def test_web_search_failure_marked_not_ok(fake_chat_model_factory):
    def fn(query: str):
        return "Web search failed: HTTP 500.", {"article_ids": [], "ok": False}

    tool = StructuredTool.from_function(
        fn, name="web_search", description="d", response_format="content_and_artifact"
    )
    agent, _ = _stub_agent(
        fake_chat_model_factory, [_multi("web_search"), AIMessage(content="x")], [tool]
    )
    assert agent.run("q").tools_used == [{"tool": "web_search", "ok": False}]


def test_result_sources_have_title_and_url(fake_embeddings, fake_chat_model_factory):
    store = InMemoryVectorStore(fake_embeddings)
    store.add_documents(
        [
            Document(
                page_content="solar",
                metadata={"article_id": "5", "title": "Sun", "url": "http://s/5"},
            )
        ]
    )
    llm = fake_chat_model_factory([_call("retrieve", "solar", "1"), AIMessage(content="a")])
    agent = create_agent(store, llm, Settings(top_k=3, _env_file=None))
    res = agent.run("q")
    assert res.sources == [{"id": "5", "title": "Sun", "url": "http://s/5"}]


def _systems(llm):
    return [c[0].content for c in llm.calls]


def test_first_turn_note_only_on_chat_turn_one(fake_chat_model_factory):
    agent, llm = _stub_agent(
        fake_chat_model_factory, [AIMessage(content="a"), AIMessage(content="b")], []
    )
    agent.run("what did you say?", thread_id="t", chat=True)
    agent.run("and now?", thread_id="t", chat=True)
    first, second = _systems(llm)
    assert first == SYSTEM_PROMPT + FIRST_TURN_NOTE
    assert second == SYSTEM_PROMPT


def test_no_note_for_ask_style_runs(fake_chat_model_factory):
    agent, llm = _stub_agent(fake_chat_model_factory, [AIMessage(content="a")] * 3, [])
    agent.run("q")  # stateless (ask / evaluate)
    agent.run("q", thread_id="t")  # threaded but not chat
    agent.run("q", chat=True)  # chat flag is ignored without a thread
    assert _systems(llm) == [SYSTEM_PROMPT] * 3


def test_evaluate_runner_path_uses_base_prompt(fake_chat_model_factory):
    from cleantech_rag.evaluation.runner import EvalItem, evaluate_items

    agent, llm = _stub_agent(fake_chat_model_factory, [AIMessage(content="a")], [])
    evaluate_items(
        [EvalItem("q", "ref", [])], agent, None, encoder=lambda t: [[1.0, 0.0] for _ in t]
    )
    assert _systems(llm) == [SYSTEM_PROMPT]


def test_first_turn_note_not_repeated_with_zero_history(fake_chat_model_factory):
    agent, llm = _stub_agent(
        fake_chat_model_factory,
        [_multi("retrieve"), AIMessage(content="a"), _multi("retrieve"), AIMessage(content="b")],
        [_stub_tool("retrieve", ["1"])],
        max_history=0,
    )
    agent.run("one", thread_id="t", chat=True)
    agent.run("two", thread_id="t", chat=True)
    systems = _systems(llm)
    assert systems[:2] == [SYSTEM_PROMPT + FIRST_TURN_NOTE] * 2  # both steps of turn 1
    assert systems[2:] == [SYSTEM_PROMPT] * 2  # turn 2, even after trimming turn 1


def test_create_agent_wires_retrieval_overfetch(fake_chat_model_factory):
    seen = []

    class Store:
        def similarity_search(self, query, k=4, **kw):
            seen.append(k)
            return []

    llm = fake_chat_model_factory([_call("retrieve", "x", "1"), AIMessage(content="a")])
    agent = create_agent(Store(), llm, Settings(top_k=3, retrieval_overfetch=7, _env_file=None))
    agent.run("q")
    assert seen == [21]


def test_sources_ignore_error_tool_messages_and_untitled_docs(fake_chat_model_factory):
    def fn(query: str):
        docs = [
            Document(page_content="a", metadata={"article_id": "1"}),  # no title/url
            Document(page_content="b", metadata={"title": "no id"}),  # no article_id
        ]
        return "text", {"article_ids": ["1"], "documents": docs}

    tool = StructuredTool.from_function(
        fn, name="retrieve", description="d", response_format="content_and_artifact"
    )
    agent, _ = _stub_agent(
        fake_chat_model_factory, [_multi("retrieve"), AIMessage(content="x")], [tool]
    )
    assert agent.run("q").sources == [{"id": "1", "title": "", "url": ""}]


def test_sources_empty_when_no_retrieval(fake_chat_model_factory):
    agent, _ = _stub_agent(fake_chat_model_factory, [AIMessage(content="hi")], [])
    res = agent.run("q")
    assert res.sources == [] and res.retrieved_ids == []


def test_sources_not_leaked_from_previous_turn(fake_embeddings, fake_chat_model_factory):
    llm = fake_chat_model_factory(
        [_call("retrieve", "solar", "1"), AIMessage(content="a"), AIMessage(content="b")]
    )
    agent = create_agent(_store(fake_embeddings), llm, Settings(top_k=3, _env_file=None))
    assert agent.run("one", thread_id="t").sources
    second = agent.run("two", thread_id="t")
    assert second.sources == [{"id": i, "title": "", "url": ""} for i in second.retrieved_ids]
    assert len(second.sources) == len(second.retrieved_ids)


def test_non_dict_artifact_is_not_a_failure(fake_chat_model_factory):
    agent, _ = _stub_agent(
        fake_chat_model_factory,
        [_multi("web_search"), AIMessage(content="x")],
        [_stub_tool("web_search", [], artifact=False)],
    )
    assert agent.run("q").tools_used == [{"tool": "web_search", "ok": True}]


def test_mixed_ok_and_failed_tools_reported_per_call(fake_chat_model_factory):
    def bad(query: str):
        return "failed", {"article_ids": [], "ok": False}

    web = StructuredTool.from_function(
        bad, name="web_search", description="d", response_format="content_and_artifact"
    )
    agent, _ = _stub_agent(
        fake_chat_model_factory,
        [_multi("retrieve", "web_search"), AIMessage(content="x")],
        [_stub_tool("retrieve", ["1"]), web],
    )
    assert agent.run("q").tools_used == [
        {"tool": "retrieve", "ok": True},
        {"tool": "web_search", "ok": False},
    ]
