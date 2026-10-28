"""LangGraph agent: a hand-written agent <-> tools loop that tracks retrieved article IDs.

Graph: START -> agent -> (tool calls? -> tools -> agent | else END). State is per invocation:
``retrieved_ids`` starts empty on every ``run`` and is filled only from this run's tool
artifacts, so IDs cannot leak between queries.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Annotated, Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool
from langchain_core.vectorstores import VectorStore
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from typing_extensions import TypedDict

from cleantech_rag.config import Settings, get_settings
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
    tools_used: list[str] = field(default_factory=list)


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
    count_reasoning_ids: bool = True,
):
    """Compile the agent/tools loop. ``llm`` must support ``bind_tools``."""
    llm_with_tools = llm.bind_tools(tools)
    tool_node = ToolNode(tools, handle_tool_errors=True)

    def agent_node(state: AgentState) -> dict[str, Any]:
        reply = llm_with_tools.invoke([SystemMessage(content=SYSTEM_PROMPT), *state["messages"]])
        return {"messages": [reply]}

    def tools_node(state: AgentState) -> dict[str, Any]:
        out = tool_node.invoke(state)
        ids = list(state["retrieved_ids"])
        for msg in out["messages"]:
            if not isinstance(msg, ToolMessage) or not isinstance(msg.artifact, dict):
                continue
            if msg.name == "reasoning_chain" and not count_reasoning_ids:
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
    return graph.compile()


class Agent:
    def __init__(self, graph: Any, recursion_limit: int = 12):
        self._graph = graph
        self.recursion_limit = recursion_limit

    def run(self, question: str) -> AgentResult:
        """Answer one question. Raises ``GraphRecursionError`` if the loop exceeds the limit."""
        final = self._graph.invoke(
            {"messages": [HumanMessage(content=question)], "retrieved_ids": []},
            config={"recursion_limit": self.recursion_limit},
        )
        messages = final["messages"]
        tools_used: list[str] = []
        for m in messages:
            if isinstance(m, AIMessage):
                tools_used.extend(c["name"] for c in m.tool_calls)
        answer = message_text(messages[-1].content) if messages else ""
        return AgentResult(answer, list(final["retrieved_ids"]), tools_used)


def create_agent(
    store: VectorStore,
    llm: BaseChatModel,
    settings: Settings | None = None,
    recursion_limit: int = 12,
    count_reasoning_ids: bool = True,
) -> Agent:
    """Wire the three tools (chat model injected) into the graph."""
    settings = settings or get_settings()
    k, over = settings.top_k, settings.retrieval_overfetch
    tools = [
        make_retriever_tool(store, k=k, overfetch=over),
        make_web_search_tool(settings),
        make_reasoning_tool(store, llm, k=k, overfetch=over),
    ]
    graph = build_graph(llm, tools, k=k, count_reasoning_ids=count_reasoning_ids)
    return Agent(graph, recursion_limit)
