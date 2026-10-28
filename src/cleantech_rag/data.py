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
    kept in addition (original order preserved).
    """
    df = pd.read_csv(path)
    if "Unnamed: 0" in df.columns:
        df = df.rename(columns={"Unnamed: 0": "id"})
    for col in _TEXT_COLS:
        if col not in df.columns:
            df[col] = ""
    df["id"] = df["id"].astype(str)
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
    """Split documents; chunks get ``chunk_index`` and ``chunk_id`` = ``{article_id}-{index}``."""
    settings = get_settings()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size if chunk_size is not None else settings.chunk_size,
        chunk_overlap=chunk_overlap if chunk_overlap is not None else settings.chunk_overlap,
    )
    chunks: list[Document] = []
    for doc in docs:
        for i, piece in enumerate(splitter.split_text(doc.page_content)):
            meta = {**doc.metadata, "chunk_index": i}
            meta["chunk_id"] = f"{meta['article_id']}-{i}"
            chunks.append(Document(page_content=piece, metadata=meta))
    return chunks
