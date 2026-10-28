"""Retriever tool: returns LLM-ready context plus this call's ranked article IDs as an artifact.

Chunk documents are expected to carry ``article_id`` and ``chunk_index`` metadata (see data.py).
No state is kept between calls; IDs travel only in the returned artifact.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from langchain_core.documents import Document
from langchain_core.tools import BaseTool, StructuredTool
from langchain_core.vectorstores import VectorStore

from cleantech_rag.config import DEFAULT_RETRIEVAL_OVERFETCH


@dataclass(frozen=True)
class RetrievalResult:
    """Ranked, article-level result of one retrieval call."""

    article_ids: list[str] = field(default_factory=list)
    documents: list[Document] = field(default_factory=list)  # best chunk per article, ranked


def _collapse(chunks: list[Document], k: int) -> RetrievalResult:
    """Collapse ranked chunks to at most k articles, each at its best rank. IDs are str."""
    ids: list[str] = []
    docs: list[Document] = []
    for doc in chunks:
        raw_id = doc.metadata.get("article_id")
        if raw_id is None:
            continue
        article_id = str(raw_id)
        if article_id in ids:
            continue
        ids.append(article_id)
        docs.append(doc)
        if len(ids) == k:
            break
    return RetrievalResult(article_ids=ids, documents=docs)


def retrieve_articles(
    store: VectorStore, query: str, k: int = 5, overfetch: int = DEFAULT_RETRIEVAL_OVERFETCH
) -> RetrievalResult:
    """Search chunks, collapse to articles at each article's best rank, return top-k articles.

    Fetches ``k * overfetch`` chunks; if that yields fewer than k unique articles and the
    store may hold more, retries once with twice as many chunks.
    """
    if k <= 0:
        return RetrievalResult()
    fetch_k = k * max(overfetch, 1)
    chunks = store.similarity_search(query, k=fetch_k)
    result = _collapse(chunks, k)
    if len(result.article_ids) < k and len(chunks) == fetch_k:
        result = _collapse(store.similarity_search(query, k=fetch_k * 2), k)
    return result


def format_context(docs: list[Document]) -> str:
    """Format documents (title, date, url, chunk text) for the LLM."""
    if not docs:
        return "No relevant articles found."
    blocks = []
    for i, doc in enumerate(docs, start=1):
        m = doc.metadata
        blocks.append(
            f"[{i}] Title: {m.get('title', 'N/A')}\n"
            f"Date: {m.get('date', 'N/A')}\n"
            f"URL: {m.get('url', 'N/A')}\n"
            f"{doc.page_content}"
        )
    return "\n\n".join(blocks)


def make_retriever_tool(
    store: VectorStore, k: int = 5, overfetch: int = DEFAULT_RETRIEVAL_OVERFETCH
) -> BaseTool:
    """Build the `retrieve` tool. Returns (content, {"article_ids", "documents"})."""

    def retrieve(query: str) -> tuple[str, dict[str, Any]]:
        result = retrieve_articles(store, query, k=k, overfetch=overfetch)
        artifact = {"article_ids": result.article_ids, "documents": result.documents}
        return format_context(result.documents), artifact

    return StructuredTool.from_function(
        func=retrieve,
        name="retrieve",
        description=(
            "Search the CleanTech news article database for passages relevant to the query. "
            "Returns article titles, dates, URLs and text excerpts."
        ),
        response_format="content_and_artifact",
    )
