"""Build and load the Chroma vector index."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from cleantech_rag.config import Settings

logger = logging.getLogger(__name__)

BATCH_SIZE = 100
MAX_RETRIES = 3
BACKOFF_BASE = 1.0  # seconds; doubles each retry
CHARS_PER_TOKEN = 4  # rough heuristic (avoids tiktoken's network download)

# USD per 1M tokens (embedding prices); unknown models log tokens only.
EMBEDDING_PRICE_PER_M: dict[str, float] = {
    "text-embedding-3-small": 0.02,
    "text-embedding-3-large": 0.13,
    "text-embedding-ada-002": 0.10,
}


def estimate_tokens(docs: list[Document]) -> int:
    return sum(len(d.page_content) for d in docs) // CHARS_PER_TOKEN


def estimate_cost(tokens: int, model: str) -> float | None:
    price = EMBEDDING_PRICE_PER_M.get(model)
    return None if price is None else tokens / 1_000_000 * price


def _default_embeddings(settings: Settings) -> Embeddings:
    from langchain_openai import OpenAIEmbeddings

    key = settings.openai_api_key
    return OpenAIEmbeddings(
        model=settings.embedding_model,
        api_key=key if key is not None else None,
    )


def _store(settings: Settings, embeddings: Embeddings | None) -> Chroma:
    return Chroma(
        collection_name=settings.collection_name,
        embedding_function=embeddings or _default_embeddings(settings),
        persist_directory=str(settings.chroma_dir),
    )


def build_index(
    docs: list[Document],
    settings: Settings,
    embeddings: Embeddings | None = None,
    *,
    batch_size: int = BATCH_SIZE,
    max_retries: int = MAX_RETRIES,
    sleep: Callable[[float], None] = time.sleep,
) -> Chroma:
    """Upsert chunks (ids = ``chunk_id``) into Chroma in batches; idempotent on rebuild.

    A failing batch is retried up to ``max_retries`` times with exponential backoff."""
    tokens = estimate_tokens(docs)
    cost = estimate_cost(tokens, settings.embedding_model)
    cost_s = f"~${cost:.4f}" if cost is not None else "unknown (no price for model)"
    logger.info(
        "Building index: %d chunks, ~%d tokens, est. cost %s (%s)",
        len(docs),
        tokens,
        cost_s,
        settings.embedding_model,
    )
    store = _store(settings, embeddings)
    for start in range(0, len(docs), batch_size):
        batch = docs[start : start + batch_size]
        ids = [str(d.metadata["chunk_id"]) for d in batch]
        texts = [d.page_content for d in batch]
        metas = [d.metadata for d in batch]
        for attempt in range(max_retries + 1):
            try:
                emb = store._embedding_function.embed_documents(texts)  # noqa: SLF001
                store._collection.upsert(  # noqa: SLF001
                    ids=ids, embeddings=emb, documents=texts, metadatas=metas
                )
                break
            except Exception as exc:
                if attempt >= max_retries:
                    raise
                delay = BACKOFF_BASE * 2**attempt
                logger.warning(
                    "Batch at %d failed (%s); retry %d/%d in %.1fs",
                    start,
                    exc,
                    attempt + 1,
                    max_retries,
                    delay,
                )
                sleep(delay)
    return store


def load_index(settings: Settings, embeddings: Embeddings | None = None) -> Chroma:
    return _store(settings, embeddings)
