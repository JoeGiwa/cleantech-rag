from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import StructuredTool

from cleantech_rag.agent import Agent, build_graph


def _tool():
    def fn(query: str):
        return "text", {"article_ids": ["1"]}

    return StructuredTool.from_function(
        fn, name="retrieve", description="d", response_format="content_and_artifact"
    )


def _agent(factory, responses, **kw):
    llm = factory(responses)
    return Agent(build_graph(llm, [_tool()], k=3, **kw), 12, llm), llm


def _humans(msgs):
    return [m.content for m in msgs if isinstance(m, HumanMessage)]


def test_turn_two_sees_turn_one(fake_chat_model_factory):
    agent, llm = _agent(fake_chat_model_factory, [AIMessage(content="A1"), AIMessage(content="A2")])
    agent.run("Q1", thread_id="t")
    agent.run("Q2", thread_id="t")
    texts = [m.content for m in llm.calls[-1]]
    assert "Q1" in texts and "A1" in texts and "Q2" in texts


def test_other_thread_starts_empty(fake_chat_model_factory):
    agent, llm = _agent(fake_chat_model_factory, [AIMessage(content="A1"), AIMessage(content="A2")])
    agent.run("Q1", thread_id="t")
    agent.run("Q2", thread_id="other")
    assert _humans(llm.calls[-1]) == ["Q2"]
    assert agent.history("never-used") == []


def test_no_thread_id_is_stateless(fake_chat_model_factory):
    agent, llm = _agent(fake_chat_model_factory, [AIMessage(content="A")])
    agent.run("Q1")
    agent.run("Q2")
    assert _humans(llm.calls[-1]) == ["Q2"]


def test_trims_to_max_history_on_turn_boundaries(fake_chat_model_factory):
    call = AIMessage(
        content="", tool_calls=[{"name": "retrieve", "args": {"query": "x"}, "id": "c"}]
    )
    agent, llm = _agent(
        fake_chat_model_factory,
        [call, AIMessage(content="A1"), AIMessage(content="A2"), AIMessage(content="A3")],
        max_history=1,
    )
    for q in ("Q1", "Q2", "Q3"):
        agent.run(q, thread_id="t")
    assert _humans(llm.calls[-1]) == ["Q2", "Q3"]
    hist = agent.history("t")
    assert _humans(hist) == ["Q2", "Q3"]
    assert not any(isinstance(m, ToolMessage) for m in hist)  # Q1's tool traffic dropped whole


def test_retrieved_ids_reset_each_turn_in_thread(fake_chat_model_factory):
    call = AIMessage(
        content="", tool_calls=[{"name": "retrieve", "args": {"query": "x"}, "id": "c"}]
    )
    agent, _ = _agent(
        fake_chat_model_factory, [call, AIMessage(content="A1"), AIMessage(content="A2")]
    )
    assert agent.run("Q1", thread_id="t").retrieved_ids == ["1"]
    assert agent.run("Q2", thread_id="t").retrieved_ids == []


def test_clear(fake_chat_model_factory):
    agent, _ = _agent(fake_chat_model_factory, [AIMessage(content="A1")])
    agent.run("Q1", thread_id="t")
    assert agent.history("t") != []
    agent.clear("t")
    assert agent.history("t") == []


def test_no_partial_with_conversation_context():
    src = Path(__file__).parent.parent / "src" / "cleantech_rag"
    for f in src.rglob("*.py"):
        assert ".partial(" not in f.read_text(), f


# --- edge cases ---------------------------------------------------------------------------


def _call(cid="c"):
    return AIMessage(
        content="", tool_calls=[{"name": "retrieve", "args": {"query": "x"}, "id": cid}]
    )


def test_max_history_zero_keeps_only_current_turn(fake_chat_model_factory):
    agent, llm = _agent(
        fake_chat_model_factory,
        [_call("a"), AIMessage(content="A1"), _call("b"), AIMessage(content="A2")],
        max_history=0,
    )
    agent.run("Q1", thread_id="t")
    agent.run("Q2", thread_id="t")
    assert _humans(llm.calls[-1]) == ["Q2"]
    assert _humans(agent.history("t")) == ["Q2"]


def test_max_history_one_keeps_one_previous_turn_plus_current(fake_chat_model_factory):
    agent, llm = _agent(
        fake_chat_model_factory, [AIMessage(content=f"A{i}") for i in range(3)], max_history=1
    )
    for q in ("Q1", "Q2", "Q3"):
        agent.run(q, thread_id="t")
    assert _humans(llm.calls[-1]) == ["Q2", "Q3"]


def test_no_orphan_tool_messages_after_trimming(fake_chat_model_factory):
    responses = []
    for i in range(4):
        responses += [_call(f"c{i}"), AIMessage(content=f"A{i}")]
    agent, llm = _agent(fake_chat_model_factory, responses, max_history=1)
    for q in ("Q0", "Q1", "Q2", "Q3"):
        agent.run(q, thread_id="t")
    for msgs in [agent.history("t"), llm.calls[-1][1:]]:
        assert isinstance(msgs[0], HumanMessage)
        call_ids = {tc["id"] for m in msgs if isinstance(m, AIMessage) for tc in m.tool_calls}
        tool_ids = {m.tool_call_id for m in msgs if isinstance(m, ToolMessage)}
        assert tool_ids == call_ids
    assert _humans(agent.history("t")) == ["Q2", "Q3"]


def test_history_at_or_below_max_is_untouched(fake_chat_model_factory):
    agent, _ = _agent(
        fake_chat_model_factory,
        [AIMessage(content="A1"), AIMessage(content="A2")],
        max_history=1,
    )
    agent.run("Q1", thread_id="t")
    agent.run("Q2", thread_id="t")
    assert _humans(agent.history("t")) == ["Q1", "Q2"]


def test_clear_then_reuse_thread(fake_chat_model_factory):
    agent, llm = _agent(fake_chat_model_factory, [AIMessage(content="A1"), AIMessage(content="A2")])
    agent.run("Q1", thread_id="t")
    agent.clear("t")
    agent.run("Q2", thread_id="t")
    assert _humans(llm.calls[-1]) == ["Q2"]
    assert all("Q1" not in str(m.content) and "A1" not in str(m.content) for m in llm.calls[-1])


def test_clear_does_not_affect_other_threads_and_unknown_is_noop(fake_chat_model_factory):
    agent, _ = _agent(fake_chat_model_factory, [AIMessage(content="A")])
    agent.run("Q1", thread_id="keep")
    agent.run("Q2", thread_id="drop")
    agent.clear("drop")
    agent.clear("never-existed")
    assert _humans(agent.history("keep")) == ["Q1"]
    assert agent.history("drop") == []


def test_retrieved_ids_do_not_leak_when_turn_two_skips_retriever(fake_chat_model_factory):
    agent, _ = _agent(
        fake_chat_model_factory, [_call(), AIMessage(content="A1"), AIMessage(content="A2")]
    )
    r1 = agent.run("Q1", thread_id="t")
    r2 = agent.run("Q2", thread_id="t")
    assert r1.retrieved_ids == ["1"]
    assert r2.retrieved_ids == []


def test_tools_used_is_per_turn_not_cumulative(fake_chat_model_factory):
    agent, _ = _agent(
        fake_chat_model_factory, [_call(), AIMessage(content="A1"), AIMessage(content="A2")]
    )
    assert agent.run("Q1", thread_id="t").tools_used == [{"tool": "retrieve", "ok": True}]
    assert agent.run("Q2", thread_id="t").tools_used == []


def _assert_no_unanswered_tool_calls(msgs):
    call_ids = {tc["id"] for m in msgs if isinstance(m, AIMessage) for tc in m.tool_calls}
    tool_ids = {m.tool_call_id for m in msgs if isinstance(m, ToolMessage)}
    assert call_ids <= tool_ids


@pytest.mark.parametrize("limit", [3, 4, 5, 6])
def test_step_limit_turn_then_next_turn_still_works(fake_chat_model_factory, limit):
    llm = fake_chat_model_factory([_call(f"c{i}") for i in range(10)] + [AIMessage(content="A2")])
    agent = Agent(build_graph(llm, [_tool()], k=3), limit, llm)
    r1 = agent.run("Q1", thread_id="t")
    assert r1.stopped_reason == "step_limit"
    _assert_no_unanswered_tool_calls(agent.history("t"))  # saved state, before turn 2
    assert agent.history("t")[-1].content == r1.answer
    llm.index = 10
    res = agent.run("Q2", thread_id="t")
    assert res.answer == "A2" and res.stopped_reason == "completed"
    assert res.tools_used == []
    _assert_no_unanswered_tool_calls(llm.calls[-1])
    assert "Q1" in [m.content for m in llm.calls[-1]]


def test_clear_without_checkpointer_raises_and_run_still_works(fake_chat_model_factory):
    agent, _ = _agent(fake_chat_model_factory, [AIMessage(content="A")])
    agent._graph.checkpointer = None
    with pytest.raises(RuntimeError):
        agent.clear("t")
    assert agent.run("Q").answer == "A"  # ephemeral cleanup is skipped, not crashed


def test_system_prompt_is_not_persisted_in_history(fake_chat_model_factory):
    agent, _ = _agent(fake_chat_model_factory, [AIMessage(content="A1")])
    agent.run("Q1", thread_id="t")
    assert not any(m.type == "system" for m in agent.history("t"))


def test_messages_to_trim_pure():
    from cleantech_rag.memory import messages_to_trim

    msgs = [
        HumanMessage(content="a", id="1"),
        AIMessage(content="b", id="2"),
        HumanMessage(content="c", id="3"),
        AIMessage(content="d", id="4"),
    ]
    assert messages_to_trim(msgs, 2) == []
    assert messages_to_trim(msgs, 1) == []
    assert [r.id for r in messages_to_trim(msgs, 0)] == ["1", "2"]
    assert [r.id for r in messages_to_trim(msgs, -1)] == ["1", "2"]
    assert messages_to_trim([], 1) == []


def test_stateless_run_deletes_its_throwaway_thread(fake_chat_model_factory):
    agent, _ = _agent(fake_chat_model_factory, [AIMessage(content="A")])
    agent.run("Q1")
    agent.run("Q2", thread_id="keep")
    saver = agent._graph.checkpointer
    threads = {c.config["configurable"]["thread_id"] for c in saver.list(None)}
    assert threads == {"keep"}
