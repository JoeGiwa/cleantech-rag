"""Build and load the Chroma vector index."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from functools import lru_cache
from typing import TypeVar

import chromadb
import openai
import tiktoken
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from cleantech_rag.config import Settings

logger = logging.getLogger(__name__)

BATCH_SIZE = 100
MAX_RETRIES = 3
BACKOFF_BASE = 1.0  # seconds; doubles each retry
ENCODING_NAME = "cl100k_base"  # tokenizer for text-embedding-3-*

# Only transient errors are retried; auth, other 4xx and everything else propagate immediately.
RETRYABLE_ERRORS: tuple[type[Exception], ...] = (
    openai.RateLimitError,
    openai.APITimeoutError,
    openai.APIConnectionError,
    TimeoutError,
    ConnectionError,
)

T = TypeVar("T")

# USD per 1M tokens (embedding prices); unknown models log tokens only.
EMBEDDING_PRICE_PER_M: dict[str, float] = {
    "text-embedding-3-small": 0.02,
    "text-embedding-3-large": 0.13,
    "text-embedding-ada-002": 0.10,
}


@lru_cache(maxsize=1)
def _get_encoding() -> tiktoken.Encoding:
    return tiktoken.get_encoding(ENCODING_NAME)


def estimate_tokens(docs: list[Document], encode: Callable[[str], list] | None = None) -> int:
    """Count tokens with cl100k_base (``encode`` is injectable so tests stay offline)."""
    enc = encode or _get_encoding().encode
    return sum(len(enc(d.page_content)) for d in docs)


def estimate_cost(tokens: int, model: str) -> float | None:
    price = EMBEDDING_PRICE_PER_M.get(model)
    return None if price is None else tokens / 1_000_000 * price


def _default_embeddings(settings: Settings) -> Embeddings:
    from langchain_openai import OpenAIEmbeddings

    return OpenAIEmbeddings(model=settings.embedding_model, api_key=settings.openai_api_key)


def _client(settings: Settings) -> chromadb.ClientAPI:
    return chromadb.PersistentClient(path=str(settings.chroma_dir))


def _with_retry(
    fn: Callable[[], T], max_retries: int, sleep: Callable[[float], None], what: str
) -> T:
    for attempt in range(max_retries + 1):
        try:
            return fn()
        except RETRYABLE_ERRORS as exc:
            if attempt >= max_retries:
                raise
            delay = BACKOFF_BASE * 2**attempt
            logger.warning(
                "%s failed (%s); retry %d/%d in %.1fs", what, exc, attempt + 1, max_retries, delay
            )
            sleep(delay)
    raise AssertionError("unreachable")  # pragma: no cover


def build_index(
    docs: list[Document],
    settings: Settings,
    embeddings: Embeddings | None = None,
    *,
    batch_size: int = BATCH_SIZE,
    max_retries: int = MAX_RETRIES,
    sleep: Callable[[float], None] = time.sleep,
    encode: Callable[[str], list] | None = None,
) -> Chroma:
    """Upsert chunks (ids = ``chunk_id``) into Chroma in batches; idempotent on rebuild.

    Each batch is embedded once, then upserted. Embedding and upserting are retried separately
    (rate limits / timeouts / connection errors only, exponential backoff); an upsert retry
    reuses the computed embeddings."""
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    if max_retries < 0:
        raise ValueError(f"max_retries must be >= 0, got {max_retries}")
    missing = [i for i, d in enumerate(docs) if "chunk_id" not in d.metadata]
    if missing:
        raise ValueError(
            f"{len(missing)} docs lack metadata['chunk_id'] (first index {missing[0]})"
        )

    tokens = estimate_tokens(docs, encode)
    cost = estimate_cost(tokens, settings.embedding_model)
    cost_s = f"~${cost:.4f}" if cost is not None else "unknown (no price for model)"
    logger.info(
        "Building index: %d chunks, %d tokens, est. cost %s (%s)",
        len(docs),
        tokens,
        cost_s,
        settings.embedding_model,
    )
    embedder = embeddings or _default_embeddings(settings)
    client = _client(settings)
    collection = client.get_or_create_collection(settings.collection_name)
    for start in range(0, len(docs), batch_size):
        batch = docs[start : start + batch_size]
        ids = [str(d.metadata["chunk_id"]) for d in batch]
        texts = [d.page_content for d in batch]
        metas = [d.metadata for d in batch]
        vectors = _with_retry(
            lambda texts=texts: embedder.embed_documents(texts),
            max_retries,
            sleep,
            f"Embedding batch at {start}",
        )
        _with_retry(
            lambda ids=ids, vectors=vectors, texts=texts, metas=metas: collection.upsert(
                ids=ids, embeddings=vectors, documents=texts, metadatas=metas
            ),
            max_retries,
            sleep,
            f"Upsert batch at {start}",
        )
    return _open(client, settings, embedder)


def _open(client: chromadb.ClientAPI, settings: Settings, embedder: Embeddings) -> Chroma:
    return Chroma(
        client=client,
        collection_name=settings.collection_name,
        embedding_function=embedder,
    )


def load_index(settings: Settings, embeddings: Embeddings | None = None) -> Chroma:
    return _open(_client(settings), settings, embeddings or _default_embeddings(settings))
