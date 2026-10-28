"""Conversation memory: a LangGraph checkpointer keyed by ``thread_id`` plus history helpers.

Memory lives in graph state (the ``messages`` channel), so each turn sees the live history of
its thread. Nothing conversation-related is baked into the prompt at construction time.
"""

from __future__ import annotations

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AnyMessage, BaseMessage, HumanMessage, RemoveMessage
from langgraph.checkpoint.memory import InMemorySaver

from cleantech_rag.tools.reasoning import message_text

NO_CONVERSATION = "No conversation to summarize."

SUMMARY_PROMPT = """Summarize the following conversation concisely:

{conversation}

Provide a brief summary highlighting:
1. Main topics discussed
2. Key questions asked
3. Important conclusions or facts established"""


def make_checkpointer() -> InMemorySaver:
    return InMemorySaver()


def messages_to_trim(messages: list[AnyMessage], max_history: int) -> list[RemoveMessage]:
    """RemoveMessage ops dropping every turn but the last ``max_history`` (a turn starts at a
    human message and includes its tool calls and answer). Cuts only on turn boundaries."""
    starts = [i for i, m in enumerate(messages) if isinstance(m, HumanMessage)]
    if max_history < 1 or len(starts) <= max_history:
        return []
    return [RemoveMessage(id=m.id) for m in messages[: starts[-max_history]] if m.id]


def format_conversation(messages: list[BaseMessage]) -> str:
    """Render user questions and final assistant answers (tool traffic is skipped)."""
    lines = []
    for m in messages:
        text = message_text(m.content)
        if isinstance(m, HumanMessage):
            lines.append(f"User: {text}")
        elif m.type == "ai" and not getattr(m, "tool_calls", None) and text:
            lines.append(f"Assistant: {text}")
    return "\n".join(lines)


def summarize(llm: BaseChatModel, messages: list[BaseMessage]) -> str:
    conversation = format_conversation(messages)
    if not conversation:
        return NO_CONVERSATION
    reply = llm.invoke([HumanMessage(content=SUMMARY_PROMPT.format(conversation=conversation))])
    return message_text(reply.content)
