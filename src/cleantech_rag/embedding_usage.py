"""Count embedding tokens (e.g. query embeddings during eval), which usage_metadata doesn't cover.

The OpenAI embeddings wrapper reports no usage, so tokens are counted locally with tiktoken, and
only inside an ``embedding_usage_callback()`` block (outside one, nothing is tokenized).
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from functools import lru_cache

from langchain_core.embeddings import Embeddings

ENCODING_NAME = "cl100k_base"  # tokenizer for text-embedding-3-*

# Active collector (model name -> input tokens). The dict is shared, not copied, so calls made
# in worker threads that inherit the context (LangGraph tool execution) still add to it.
_active: ContextVar[dict[str, int] | None] = ContextVar("embedding_usage", default=None)


@lru_cache(maxsize=1)
def _encode() -> Callable[[str], list]:
    import tiktoken

    return tiktoken.get_encoding(ENCODING_NAME).encode


@contextmanager
def embedding_usage_callback() -> Iterator[dict[str, int]]:
    """Collect embedding input tokens per model for calls made inside the block."""
    usage: dict[str, int] = {}
    token = _active.set(usage)
    try:
        yield usage
    finally:
        _active.reset(token)


class CountingEmbeddings(Embeddings):
    """Delegates to ``inner``; adds the input tokens to the active collector, if any."""

    def __init__(
        self, inner: Embeddings, model: str, encode: Callable[[str], list] | None = None
    ) -> None:
        self.inner = inner
        self.model = model
        self._encode_fn = encode

    def _count(self, texts: list[str]) -> None:
        usage = _active.get()
        if usage is None:
            return
        enc = self._encode_fn or _encode()
        usage[self.model] = usage.get(self.model, 0) + sum(len(enc(t)) for t in texts)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self._count(texts)
        return self.inner.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        self._count([text])
        return self.inner.embed_query(text)
