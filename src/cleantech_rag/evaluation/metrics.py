"""Pure evaluation metrics: retrieval (P/R/MRR/NDCG), ROUGE, BLEU, semantic similarity.

``rouge-score``, ``nltk`` and ``sentence-transformers`` live in the optional ``[eval]`` extra and
are imported lazily, so importing this module never requires them.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Sequence
from typing import Any

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)

DEFAULT_ENCODER_MODEL = "all-MiniLM-L6-v2"

# An encoder maps a list of strings to a list of vectors (one per string).
Encoder = Callable[[list[str]], Sequence[Sequence[float]]]


class MissingEvalDependencyError(ImportError):
    """Raised when an optional evaluation dependency is not installed."""


def _require(module: str, package: str) -> Any:
    import importlib

    try:
        return importlib.import_module(module)
    except ImportError as exc:
        raise MissingEvalDependencyError(
            f"'{package}' is required for this metric. "
            "Install the evaluation extras: `uv sync --extra eval`."
        ) from exc


def tokenize(text: str) -> list[str]:
    """Lowercase regex word tokenizer (no nltk punkt download needed)."""
    return _TOKEN_RE.findall(text.lower())


# ---------------------------------------------------------------- retrieval metrics


def _dedupe(ids: Sequence[str]) -> list[str]:
    """Drop repeated IDs, keeping first occurrence order."""
    return list(dict.fromkeys(ids))


def _hits_at_k(retrieved: Sequence[str], relevant: Sequence[str], k: int) -> tuple[int, int]:
    top = _dedupe(retrieved)[:k]
    rel = set(relevant)
    return sum(1 for d in top if d in rel), len(top)


def precision_at_k(retrieved: Sequence[str], relevant: Sequence[str], k: int = 5) -> float:
    """Standard Precision@k: relevant hits in the top k, divided by ``k``.

    This is the primary metric and is comparable with the original notebook. If fewer than k
    documents were retrieved the missing slots count as misses. Duplicates count once.
    """
    if k <= 0:
        return 0.0
    hits, _ = _hits_at_k(retrieved, relevant, k)
    return hits / k


def precision_at_k_retrieved(
    retrieved: Sequence[str], relevant: Sequence[str], k: int = 5
) -> float:
    """Secondary Precision@k: hits divided by ``min(k, len(unique retrieved))``.

    Differs from :func:`precision_at_k` when fewer than k documents come back: 3 retrieved with
    1 hit gives 1/3 here but 1/5 for the standard metric. Returns 0.0 if nothing was retrieved.
    """
    if k <= 0:
        return 0.0
    hits, n = _hits_at_k(retrieved, relevant, k)
    return hits / n if n else 0.0


def recall_at_k(retrieved: Sequence[str], relevant: Sequence[str], k: int = 5) -> float:
    """Recall@k: unique relevant docs in the top k over the number of unique relevant docs."""
    rel = set(relevant)
    if not rel:
        return 0.0
    hits, _ = _hits_at_k(retrieved, relevant, k)
    return hits / len(rel)


def mrr(retrieved: Sequence[str], relevant: Sequence[str]) -> float:
    """Reciprocal rank of the first relevant document (0.0 if none)."""
    rel = set(relevant)
    for rank, doc_id in enumerate(_dedupe(retrieved), start=1):
        if doc_id in rel:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved: Sequence[str], relevant: Sequence[str], k: int = 5) -> float:
    """Binary-relevance NDCG@k with log2 discount."""
    rel = set(relevant)
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, doc_id in enumerate(_dedupe(retrieved)[:k], start=1)
        if doc_id in rel
    )
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, min(len(rel), k) + 1))
    return dcg / idcg if idcg else 0.0


# ---------------------------------------------------------------- response metrics


def rouge_scores(prediction: str, reference: str) -> dict[str, float]:
    """ROUGE-1/2/L F-measures (with stemming), keys ``rouge1``, ``rouge2``, ``rougeL``."""
    rouge_scorer = _require("rouge_score.rouge_scorer", "rouge-score")
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    scores = scorer.score(reference, prediction)
    return {name: scores[name].fmeasure for name in ("rouge1", "rouge2", "rougeL")}


def bleu_score(prediction: str, reference: str) -> float:
    """Sentence BLEU (smoothing method1) using a regex tokenizer.

    No exceptions are swallowed: the original notebook's bare ``except`` hid an nltk error and
    reported 0.0 for every query. Empty inputs return 0.0 explicitly.
    """
    bleu = _require("nltk.translate.bleu_score", "nltk")
    pred_tokens = tokenize(prediction)
    ref_tokens = tokenize(reference)
    if not pred_tokens or not ref_tokens:
        return 0.0
    return float(
        bleu.sentence_bleu(
            [ref_tokens], pred_tokens, smoothing_function=bleu.SmoothingFunction().method1
        )
    )


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def default_encoder(model_name: str = DEFAULT_ENCODER_MODEL) -> Encoder:
    """Build an encoder backed by sentence-transformers (lazy import; may download weights)."""
    st = _require("sentence_transformers", "sentence-transformers")
    model = st.SentenceTransformer(model_name)

    def encode(texts: list[str]) -> Sequence[Sequence[float]]:
        return [list(map(float, v)) for v in model.encode(texts)]

    return encode


def semantic_similarity(prediction: str, reference: str, encoder: Encoder | None = None) -> float:
    """Cosine similarity between embeddings of prediction and reference.

    ``encoder`` is injectable (callable: list[str] -> list of vectors). When omitted, the default
    ``all-MiniLM-L6-v2`` sentence-transformers model is loaded, which needs the ``[eval]`` extra.
    """
    enc = encoder if encoder is not None else default_encoder()
    pred_vec, ref_vec = enc([prediction, reference])
    return _cosine(pred_vec, ref_vec)
