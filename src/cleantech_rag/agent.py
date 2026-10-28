"""LangGraph agent: a hand-written agent <-> tools loop that tracks retrieved article IDs.

Graph: START -> agent -> (tool calls? -> tools -> agent | else END). State is per invocation:
``retrieved_ids`` starts empty on every ``run`` and is filled only from this run's tool
artifacts, so IDs cannot leak between queries. ``messages`` persist per ``thread_id`` through an
``InMemorySaver`` checkpointer (conversation memory); ``retrieved_ids`` is overwritten each run.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Annotated, Any, Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool
from langchain_core.vectorstores import VectorStore
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from typing_extensions import TypedDict

from cleantech_rag.config import Settings, get_settings
from cleantech_rag.memory import make_checkpointer, messages_to_trim, summarize
from cleantech_rag.tools.reasoning import make_reasoning_tool, message_text
from cleantech_rag.tools.retriever import make_retriever_tool
from cleantech_rag.tools.web_search import make_web_search_tool

SYSTEM_PROMPT = """You are an advanced AI assistant specializing in CleanTech and renewable energy.
You have access to multiple tools to provide comprehensive, accurate answers.

TOOLS:
1. retrieve: Search the CleanTech article database
2. web_search: Search the web for current information
3. reasoning_chain: Step-by-step reasoning over retrieved articles for complex questions

STRATEGY:
- For standard questions: use retrieve, then write the answer from the retrieved articles
- For recent events/statistics: add web_search for current data
- For complex analysis (comparison, multi-part, deduction): use reasoning_chain
- Answer directly, without tools, only if the question needs no outside information

Always prioritize accuracy and cite your sources when possible."""


@dataclass(frozen=True)
class AgentResult:
    answer: str
    retrieved_ids: list[str] = field(default_factory=list)
    # Executed tool calls in call order: [{"tool": name, "ok": bool}]. ok means the tool did not
    # raise/error (web_search soft-failure strings, e.g. "No web results found.", count as ok).
    tools_used: list[dict[str, Any]] = field(default_factory=list)
    stopped_reason: Literal["completed", "step_limit"] = "completed"


STEP_LIMIT_ANSWER = (
    "I could not finish answering within the allowed number of reasoning steps. "
    "Please try rephrasing or narrowing the question."
)


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    retrieved_ids: list[str]


def merge_ids(existing: list[str], new: list[str], k: int) -> list[str]:
    """Rank-ordered union in call order, de-duplicated, truncated to k."""
    merged = list(existing)
    for article_id in new:
        if article_id not in merged:
            merged.append(article_id)
    return merged[:k]


def build_graph(
    llm: BaseChatModel,
    tools: list[BaseTool],
    k: int = 5,
    max_history: int = 10,
    checkpointer: Any = None,
):
    """Compile the agent/tools loop. ``llm`` must support ``bind_tools``.

    Conversation history is kept per ``thread_id`` by ``checkpointer`` (default: in-memory) and
    trimmed to the last ``max_history`` turns.
    """
    llm_with_tools = llm.bind_tools(tools)
    tool_node = ToolNode(tools, handle_tool_errors=True)

    def agent_node(state: AgentState) -> dict[str, Any]:
        # Trim old turns here (not in a separate node) so it costs no extra graph step.
        drop = messages_to_trim(state["messages"], max_history)
        dropped = {r.id for r in drop}
        kept = [m for m in state["messages"] if m.id not in dropped]
        reply = llm_with_tools.invoke([SystemMessage(content=SYSTEM_PROMPT), *kept])
        return {"messages": [*drop, reply]}

    def tools_node(state: AgentState) -> dict[str, Any]:
        out = tool_node.invoke(state)
        ids = list(state["retrieved_ids"])
        for msg in out["messages"]:
            if not isinstance(msg, ToolMessage) or not isinstance(msg.artifact, dict):
                continue
            if msg.status == "error":
                continue
            ids = merge_ids(ids, [str(i) for i in msg.artifact.get("article_ids", [])], k)
        return {"messages": out["messages"], "retrieved_ids": ids}

    def route(state: AgentState) -> str:
        last = state["messages"][-1]
        return "tools" if isinstance(last, AIMessage) and last.tool_calls else END

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tools_node)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", route, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")
    return graph.compile(checkpointer=checkpointer or make_checkpointer())


class Agent:
    def __init__(self, graph: Any, recursion_limit: int = 12, llm: BaseChatModel | None = None):
        self._graph = graph
        self.recursion_limit = recursion_limit
        self._llm = llm

    def run(self, question: str, thread_id: str | None = None) -> AgentResult:
        """Answer one question; on hitting the step limit return a fallback with partial state.

        Turns sharing a ``thread_id`` see each other's history. Without one the run uses a fresh
        throwaway thread, so it is stateless (what evaluation needs).
        """
        thread_id = thread_id or f"ephemeral-{uuid.uuid4()}"
        state: dict[str, Any] = {"messages": [], "retrieved_ids": []}
        stopped: Literal["completed", "step_limit"] = "completed"
        try:
            for chunk in self._graph.stream(
                {"messages": [HumanMessage(content=question)], "retrieved_ids": []},
                config={
                    "recursion_limit": self.recursion_limit,
                    "configurable": {"thread_id": thread_id},
                },
                stream_mode="values",
            ):
                state = chunk  # keep the latest full state
        except GraphRecursionError:
            # The limit can trip right after the agent's final answer; that is still a completion.
            last = state["messages"][-1] if state["messages"] else None
            if not (isinstance(last, AIMessage) and not last.tool_calls):
                stopped = "step_limit"
        messages = state["messages"]
        tools_used = [
            {"tool": m.name, "ok": m.status != "error"}
            for m in messages
            if isinstance(m, ToolMessage)
        ]
        if stopped == "step_limit":
            answer = STEP_LIMIT_ANSWER
        else:
            answer = message_text(messages[-1].content) if messages else ""
        return AgentResult(answer, list(state["retrieved_ids"]), tools_used, stopped)

    def history(self, thread_id: str) -> list[BaseMessage]:
        """Live messages of a thread (empty for an unknown thread)."""
        config = {"configurable": {"thread_id": thread_id}}
        return list(self._graph.get_state(config).values.get("messages", []))

    def clear(self, thread_id: str) -> None:
        """Forget a thread's history."""
        self._graph.checkpointer.delete_thread(thread_id)

    def summary(self, thread_id: str) -> str:
        """LLM summary of the thread's current history."""
        if self._llm is None:
            raise RuntimeError("Agent was built without an llm; summary is unavailable.")
        return summarize(self._llm, self.history(thread_id))


def create_agent(
    store: VectorStore,
    llm: BaseChatModel,
    settings: Settings | None = None,
    recursion_limit: int | None = None,
) -> Agent:
    """Wire the three tools (chat model injected) into the graph."""
    settings = settings or get_settings()
    k, over = settings.top_k, settings.retrieval_overfetch
    tools = [
        make_retriever_tool(store, k=k, overfetch=over),
        make_web_search_tool(settings),
        make_reasoning_tool(store, llm, k=k, overfetch=over),
    ]
    graph = build_graph(llm, tools, k=k, max_history=settings.max_history)
    return Agent(
        graph,
        settings.agent_recursion_limit if recursion_limit is None else recursion_limit,
        llm,
    )
