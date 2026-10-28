"""Evaluation runner: runs the agent per query and scores retrieval, answer and judge quality.

Each query's retrieval metrics use only the ``retrieved_ids`` of that query's own
``agent.run()`` result, so IDs can never leak across queries (a query that did not retrieve
scores against an empty list).

Aggregation rules:

* Errored queries (``agent.run`` raised) are excluded from every average; ``n_errors`` is
  reported and the per-query ``error`` column is kept.
* Retrieval metrics are reported two ways: ``all_queries`` (a query that did not use retrieval
  scores 0) and ``retrieval_only`` (queries that did). Queries without gold article IDs are
  left out of both views and counted in ``n_no_gold``.
* Standard Precision@k is the primary precision; the retrieved-denominator variant is also given.
* Judge failures are excluded from judge averages (``n_failed`` reported). Fatal judge errors
  (auth, model not found) propagate and abort the run.

The agent, judge and sentence encoder are injected so tests can use fakes.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol

from cleantech_rag.evaluation import metrics
from cleantech_rag.evaluation.judge import JudgeResult, aggregate_judge_results

RETRIEVAL_METRICS = (
    "precision_at_k",
    "precision_at_k_retrieved",
    "recall_at_k",
    "mrr",
    "ndcg_at_k",
)
RESPONSE_METRICS = ("rouge1", "rouge2", "rougeL", "bleu", "semantic_similarity")
CSV_COLUMNS = (
    "index",
    "question",
    "retrieval_used",
    "n_retrieved",
    "retrieved_ids",
    "gold_ids",
    *RETRIEVAL_METRICS,
    *RESPONSE_METRICS,
    "judge_relevance",
    "judge_accuracy",
    "judge_completeness",
    "judge_clarity",
    "judge_overall",
    "judge_error",
    "error",
)


@dataclass(frozen=True)
class EvalItem:
    question: str
    reference: str
    gold_ids: list[str] = field(default_factory=list)


class AgentLike(Protocol):
    def run(self, question: str, thread_id: str | None = None) -> Any: ...


class JudgeLike(Protocol):
    def evaluate(self, query: str, prediction: str, reference: str) -> JudgeResult: ...


@dataclass
class QueryRecord:
    index: int
    question: str
    retrieved_ids: list[str] = field(default_factory=list)
    gold_ids: list[str] = field(default_factory=list)
    retrieval_used: bool = False
    answer: str = ""
    retrieval: dict[str, float] | None = None
    response: dict[str, float] = field(default_factory=dict)
    judge: JudgeResult | None = None
    error: str | None = None

    def csv_row(self) -> dict[str, Any]:
        row: dict[str, Any] = {
            "index": self.index,
            "question": self.question,
            "retrieval_used": self.retrieval_used,
            "n_retrieved": len(self.retrieved_ids),
            "retrieved_ids": ";".join(self.retrieved_ids),
            "gold_ids": ";".join(self.gold_ids),
            "error": self.error or "",
        }
        row.update(self.retrieval or {})
        row.update(self.response)
        if self.judge is not None:
            for name, value in self.judge.scores.items():
                row[f"judge_{name}"] = value
            row["judge_overall"] = self.judge.overall
            row["judge_error"] = self.judge.error or ""
        return row


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _retrieval_view(records: Sequence[QueryRecord]) -> dict[str, Any]:
    out: dict[str, Any] = {"n": len(records)}
    for name in RETRIEVAL_METRICS:
        out[name] = _mean([r.retrieval[name] for r in records if r.retrieval])
    return out


def _was_retrieval_used(result: Any, ids: list[str]) -> bool:
    tools = getattr(result, "tools_used", None) or []
    return bool(ids) or any(t.get("tool") == "retrieve" and t.get("ok") for t in tools)


def evaluate_items(
    items: Sequence[EvalItem],
    agent: AgentLike,
    judge: JudgeLike | None = None,
    encoder: metrics.Encoder | None = None,
    k: int = 5,
) -> list[QueryRecord]:
    """Run every item through the agent and score it. One record per item, in order."""
    records: list[QueryRecord] = []
    for i, item in enumerate(items):
        rec = QueryRecord(index=i, question=item.question, gold_ids=list(item.gold_ids))
        records.append(rec)
        try:
            result = agent.run(item.question)
            rec.answer = str(result.answer)
            rec.retrieved_ids = [str(x) for x in (getattr(result, "retrieved_ids", None) or [])]
            rec.retrieval_used = _was_retrieval_used(result, rec.retrieved_ids)
        except Exception as exc:  # noqa: BLE001 - recorded per query, excluded from averages
            rec.retrieved_ids = []
            rec.error = f"{type(exc).__name__}: {exc}"
            continue

        if rec.gold_ids:
            ids, gold = rec.retrieved_ids, rec.gold_ids
            rec.retrieval = {
                "precision_at_k": metrics.precision_at_k(ids, gold, k),
                "precision_at_k_retrieved": metrics.precision_at_k_retrieved(ids, gold, k),
                "recall_at_k": metrics.recall_at_k(ids, gold, k),
                "mrr": metrics.mrr(ids, gold),
                "ndcg_at_k": metrics.ndcg_at_k(ids, gold, k),
            }
        rouge = metrics.rouge_scores(rec.answer, item.reference)
        rec.response = {
            **rouge,
            "bleu": metrics.bleu_score(rec.answer, item.reference),
            "semantic_similarity": metrics.semantic_similarity(rec.answer, item.reference, encoder),
        }
        if judge is not None:
            rec.judge = judge.evaluate(item.question, rec.answer, item.reference)
    return records


def summarize(
    records: Sequence[QueryRecord], config: dict[str, Any], judge_enabled: bool
) -> dict[str, Any]:
    ok = [r for r in records if r.error is None]
    scored = [r for r in ok if r.retrieval is not None]
    summary: dict[str, Any] = {
        "config": config,
        "counts": {
            "n_queries": len(records),
            "n_errors": len(records) - len(ok),
            "n_evaluated": len(ok),
            "n_retrieval_used": sum(r.retrieval_used for r in ok),
            "n_no_gold": len(ok) - len(scored),
        },
        "retrieval": {
            "all_queries": _retrieval_view(scored),
            "retrieval_only": _retrieval_view([r for r in scored if r.retrieval_used]),
        },
        "response": {name: _mean([r.response[name] for r in ok]) for name in RESPONSE_METRICS},
        "judge": (
            aggregate_judge_results([r.judge for r in ok if r.judge is not None])
            if judge_enabled
            else None
        ),
    }
    return summary


def write_outputs(
    records: Sequence[QueryRecord], summary: dict[str, Any], out_dir: Path, timestamp: datetime
) -> tuple[Path, Path]:
    """Write ``summary_<ts>.json`` (committable) and ``per_query_<ts>.csv`` (gitignored)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = timestamp.strftime("%Y%m%dT%H%M%S")
    summary_path = out_dir / f"summary_{stamp}.json"
    csv_path = out_dir / f"per_query_{stamp}.csv"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS, restval="")
        writer.writeheader()
        writer.writerows(r.csv_row() for r in records)
    return summary_path, csv_path
