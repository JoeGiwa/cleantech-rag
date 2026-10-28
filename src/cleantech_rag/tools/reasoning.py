"""Reasoning-chain tool: retrieves context via the retriever function, then reasons step by step.

Retrieval goes only through ``retrieve_articles`` (never the vectorstore directly), and the
article IDs used are returned as the artifact so they are tracked like the retrieve tool's.
"""

from __future__ import annotations

from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool, StructuredTool
from langchain_core.vectorstores import VectorStore

from cleantech_rag.tools.retriever import OVERFETCH_FACTOR, format_context, retrieve_articles

REASONING_PROMPT = (
    "You are a careful analyst. Using ONLY the context below, answer the question by reasoning "
    "step by step: 1) identify the relevant facts, 2) connect them, 3) state a conclusion. "
    "If the context is insufficient, say so.\n\n"
    "Context:\n{context}\n\nQuestion: {question}"
)


def _message_text(content: str | list[Any]) -> str:
    """Join the text parts of a chat message's content (str or list of content blocks)."""
    if isinstance(content, str):
        return content
    parts = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict) and block.get("type") == "text":
            parts.append(str(block.get("text") or ""))
    return "".join(parts)


def make_reasoning_tool(
    store: VectorStore, llm: BaseChatModel, k: int = 5, overfetch: int = OVERFETCH_FACTOR
) -> BaseTool:
    """Build the `reasoning_chain` tool. Returns (content, {"article_ids", "documents"})."""

    def reasoning_chain(question: str) -> tuple[str, dict[str, Any]]:
        result = retrieve_articles(store, question, k=k, overfetch=overfetch)
        artifact = {"article_ids": result.article_ids, "documents": result.documents}
        if not result.documents:
            return "No relevant articles found to reason over.", artifact
        prompt = REASONING_PROMPT.format(
            context=format_context(result.documents), question=question
        )
        return _message_text(llm.invoke(prompt).content), artifact

    return StructuredTool.from_function(
        func=reasoning_chain,
        name="reasoning_chain",
        description=(
            "Answer a complex question with step-by-step reasoning over retrieved CleanTech "
            "articles. Use for multi-part or analytical questions."
        ),
        response_format="content_and_artifact",
    )
