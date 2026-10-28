"""Conversation memory: a LangGraph checkpointer keyed by ``thread_id`` plus history trimming.

Memory lives in graph state (the ``messages`` channel), so each turn sees the live history of
its thread. Nothing conversation-related is baked into the prompt at construction time.
"""

from __future__ import annotations

from langchain_core.messages import AnyMessage, HumanMessage, RemoveMessage
from langgraph.checkpoint.memory import InMemorySaver


def make_checkpointer() -> InMemorySaver:
    return InMemorySaver()


def messages_to_trim(messages: list[AnyMessage], max_history: int) -> list[RemoveMessage]:
    """RemoveMessage ops keeping the current (last) turn plus the ``max_history`` turns before it.

    A turn starts at a human message and includes its tool calls and answer; cuts happen only on
    turn boundaries. ``max_history=0`` keeps just the current turn."""
    starts = [i for i, m in enumerate(messages) if isinstance(m, HumanMessage)]
    keep = max(max_history, 0) + 1
    if len(starts) <= keep:
        return []
    return [RemoveMessage(id=m.id) for m in messages[: starts[-keep]] if m.id]
