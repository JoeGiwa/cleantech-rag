from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import StructuredTool

from cleantech_rag.agent import Agent, build_graph
from cleantech_rag.memory import NO_CONVERSATION


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
        max_history=2,
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


def test_clear_and_summary(fake_chat_model_factory):
    agent, llm = _agent(
        fake_chat_model_factory,
        [AIMessage(content="A1"), AIMessage(content="the summary")],
    )
    assert agent.summary("t") == NO_CONVERSATION
    agent.run("Q1", thread_id="t")
    assert agent.summary("t") == "the summary"
    prompt = llm.calls[-1][0].content
    assert "User: Q1" in prompt and "Assistant: A1" in prompt
    agent.clear("t")
    assert agent.history("t") == []


def test_no_partial_with_conversation_context():
    src = Path(__file__).parent.parent / "src" / "cleantech_rag"
    for f in src.rglob("*.py"):
        assert ".partial(" not in f.read_text(), f
