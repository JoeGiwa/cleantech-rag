"""Load and clean the CleanTech CSV, convert to Documents, and chunk them."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import pandas as pd
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from cleantech_rag.config import get_settings

_TEXT_COLS = ["title", "date", "author", "content", "domain", "url"]


def load_articles(
    path: str | Path,
    limit: int | None = None,
    include_ids: Iterable[str | int] | None = None,
) -> pd.DataFrame:
    """Read the corpus CSV; rename ``Unnamed: 0`` to ``id`` and drop empty-content rows.

    ``limit`` keeps the first ``limit`` articles; ``include_ids`` articles are always
    kept in addition (original order preserved). ``include_ids`` without ``limit`` is a
    no-op because every article is already included. Duplicate ids keep the first row.
    Raises ``ValueError`` if the ``content`` or ``date`` column is missing.
    """
    df = pd.read_csv(path)
    if "Unnamed: 0" in df.columns:
        df = df.rename(columns={"Unnamed: 0": "id"})
    missing = [c for c in ("id", "content", "date") if c not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing required column(s): {', '.join(missing)}")
    for col in _TEXT_COLS:
        if col not in df.columns:
            df[col] = ""
    df["id"] = df["id"].astype(str)
    df = df.drop_duplicates(subset="id", keep="first")
    content = df["content"].fillna("").astype(str)
    df = df[content.str.strip() != ""]
    if limit is not None:
        mask = pd.Series(False, index=df.index)
        mask.iloc[: max(limit, 0)] = True
        if include_ids is not None:
            mask |= df["id"].isin({str(i) for i in include_ids})
        df = df[mask]
    return df.reset_index(drop=True)


def _s(value: object) -> str:
    return "" if pd.isna(value) else str(value)


def to_documents(df: pd.DataFrame) -> list[Document]:
    """One Document per article: ``"{title}\\n\\n{content}"`` plus metadata."""
    docs = []
    for row in df.itertuples(index=False):
        title, content = _s(row.title), _s(row.content)
        docs.append(
            Document(
                page_content=f"{title}\n\n{content}" if title else content,
                metadata={
                    "article_id": str(row.id),
                    "title": title,
                    "url": _s(row.url),
                    "date": _s(row.date),
                    "domain": _s(row.domain),
                    "author": _s(row.author),
                },
            )
        )
    return docs


def chunk_documents(
    docs: list[Document],
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[Document]:
    """Split article bodies; every chunk is prefixed with ``"{title}\\n\\n"`` and is <= chunk_size.

    Chunks get ``chunk_index`` and ``chunk_id`` = ``{article_id}-{index}``."""
    settings = get_settings()
    size = chunk_size if chunk_size is not None else settings.chunk_size
    overlap = chunk_overlap if chunk_overlap is not None else settings.chunk_overlap
    if overlap > size:
        raise ValueError(f"chunk_overlap ({overlap}) must not exceed chunk_size ({size})")
    chunks: list[Document] = []
    for doc in docs:
        title = doc.metadata.get("title", "")
        prefix = f"{title}\n\n" if title else ""
        body = doc.page_content
        if prefix and body.startswith(prefix):
            body = body[len(prefix) :]
        else:
            prefix = ""
        if len(prefix) > size // 2:  # keep room for body text
            prefix = title[: max(size // 2 - 2, 0)] + "\n\n" if size // 2 > 2 else ""
        eff = max(size - len(prefix), 1)
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=eff, chunk_overlap=max(min(overlap, eff - 1), 0)
        )
        for i, piece in enumerate(splitter.split_text(body)):
            meta = {**doc.metadata, "chunk_index": i}
            meta["chunk_id"] = f"{meta['article_id']}-{i}"
            chunks.append(Document(page_content=prefix + piece, metadata=meta))
    return chunks
