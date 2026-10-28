"""Parsers and validators for the evaluation datasets.

Article IDs are ``str`` everywhere.
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from collections.abc import Collection
from dataclasses import dataclass, field, replace
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

_HEADER_RE = re.compile(r"^\*\*(\d+)\.\s*Query:\*\*[ \t]*", re.MULTILINE)
_ANSWER_RE = re.compile(r"\*\*Desired Output:\*\*", re.IGNORECASE)
_REFS_RE = re.compile(
    r"\*\*Referenced Articles:\*\*([^\n]*(?:\n[ \t]*\d[\d,\t ]*(?=\n|$))*)", re.IGNORECASE
)


@dataclass(frozen=True)
class ComplexQA:
    query_num: int
    question: str
    answer: str
    article_ids: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class SimpleQA:
    example_id: str
    question_id: str
    question: str
    relevant_text: str
    answer: str
    article_url: str
    article_ids: list[str] = field(default_factory=list)


def parse_complex_qa(text: str, *, strict: bool = False) -> list[ComplexQA]:
    """Parse the 50-answers text. Each ``**N. Query:**`` block is parsed in isolation.

    Blocks without a ``Referenced Articles`` line are skipped with a warning
    (or raise ``ValueError`` when ``strict``); they never borrow the next block's IDs.
    """
    headers = list(_HEADER_RE.finditer(text))
    out: list[ComplexQA] = []
    for i, h in enumerate(headers):
        end = headers[i + 1].start() if i + 1 < len(headers) else len(text)
        block = text[h.end() : end]
        num = int(h.group(1))
        refs = _REFS_RE.search(block)
        ids = list(dict.fromkeys(re.findall(r"\d+", refs.group(1).split("(")[0]))) if refs else []
        if not ids:
            msg = f"query {num}: missing or empty 'Referenced Articles' line"
            if strict:
                raise ValueError(msg)
            logger.warning("%s; skipping", msg)
            continue
        ans = _ANSWER_RE.search(block)
        q_end = ans.start() if ans else refs.start()
        answer = block[ans.end() : refs.start()] if ans and ans.end() <= refs.start() else ""
        out.append(ComplexQA(num, block[:q_end].strip(), answer.strip(), ids))
    return out


def load_complex_qa(path: str | Path, *, strict: bool = False) -> list[ComplexQA]:
    return parse_complex_qa(Path(path).read_text(encoding="utf-8"), strict=strict)


def load_simple_qa(path: str | Path) -> list[SimpleQA]:
    """Load the ``;``-delimited simple QA CSV (all columns read as str)."""
    df = pd.read_csv(path, sep=";", dtype=str, keep_default_na=False)
    return [
        SimpleQA(
            example_id=r["example_id"],
            question_id=r["question_id"],
            question=r["question"],
            relevant_text=r["relevant_text"],
            answer=r["answer"],
            article_url=r["article_url"],
        )
        for r in df.to_dict("records")
    ]


def _norm_url(url: str) -> str:
    return url.strip().rstrip("/")


def load_url_index(corpus_path: str | Path) -> dict[str, list[str]]:
    """Map normalized corpus ``url`` -> article IDs (str, corpus order). Loads two columns only."""
    header = pd.read_csv(corpus_path, nrows=0).columns
    id_col = "Unnamed: 0" if "Unnamed: 0" in header else "id"  # same rule as data.load_articles
    df = pd.read_csv(corpus_path, usecols=["url", id_col], dtype=str, keep_default_na=False)
    index: dict[str, list[str]] = {}
    for url, aid in zip(df["url"], df[id_col], strict=True):
        key = _norm_url(url)
        if not key:
            continue
        ids = index.setdefault(key, [])
        if aid not in ids:
            ids.append(aid)
    return index


def map_article_ids(
    items: list[SimpleQA], url_index: dict[str, list[str]]
) -> tuple[list[SimpleQA], int]:
    """Fill ``article_ids`` from ``article_url``. Returns (items, unmapped_count).

    Unmapped rows are kept with empty ``article_ids`` (reversible: callers can filter).
    """
    out: list[SimpleQA] = []
    unmapped = 0
    for q in items:
        key = _norm_url(q.article_url)
        ids = url_index.get(key, []) if key else []
        unmapped += not ids
        out.append(replace(q, article_ids=list(ids)))
    if unmapped:
        logger.warning("%d of %d article URLs not found in corpus", unmapped, len(items))
    return out, unmapped


def validate(
    items: list[ComplexQA] | list[SimpleQA], corpus_ids: Collection[str] | None = None
) -> dict[str, int]:
    """Return counts only (no question/answer content)."""
    sets = Counter(frozenset(q.article_ids) for q in items if q.article_ids)
    shared = [k for k, c in sets.items() if c > 1]
    result = {
        "num_queries": len(items),
        "distinct_id_sets": len(sets),
        "queries_sharing_id_sets": sum(sets[k] for k in shared),
        "queries_without_ids": sum(not q.article_ids for q in items),
    }
    if corpus_ids is not None:
        corpus = {str(i) for i in corpus_ids}
        gold = {i for q in items for i in q.article_ids}
        result["gold_ids_missing_from_corpus"] = len(gold - corpus)
    return result
