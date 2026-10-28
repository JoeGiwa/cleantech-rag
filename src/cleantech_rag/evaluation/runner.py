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

Incremental results: each finished query is appended to a JSONL run log (flushed per row), so a
crash, Ctrl-C or fatal judge error loses nothing already paid for. The first line is a header
with the run config; resuming a log whose header or question text does not match is refused.
Errored rows are not "completed": they are retried on resume, and the latest row per question
wins. Summary and CSV are computed from the log, so a resumed run covers all completed rows.

The agent, judge and sentence encoder are injected so tests can use fakes.
"""

from __future__ import annotations

import csv
import json
import logging
import os
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol

from cleantech_rag.evaluation import metrics
from cleantech_rag.evaluation.judge import JudgeResult, aggregate_judge_results

logger = logging.getLogger(__name__)

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

    def to_json(self) -> dict[str, Any]:
        return {"type": "row", **asdict(self)}

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> QueryRecord:
        data = {k: v for k, v in data.items() if k != "type"}
        judge = data.pop("judge", None)
        return cls(**data, judge=JudgeResult(**judge) if judge else None)

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


class ResumeMismatchError(ValueError):
    """The run log being resumed does not belong to this dataset/config."""


class CorruptLogError(ResumeMismatchError):
    """A run-log line (other than an interrupted final write) cannot be parsed."""


class RunLog:
    """Append-only JSONL log of per-query records with a config header on the first line."""

    def __init__(self, path: Path, header: dict[str, Any], resume: bool = False) -> None:
        self.path = path
        self.header = header
        self._latest: dict[int, QueryRecord] = {}
        if resume:
            self._load()
        elif path.exists():
            raise FileExistsError(f"{path} already exists; use --resume to continue it.")
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            self._write({"type": "header", "config": header})

    def _write(self, obj: dict[str, Any]) -> None:
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(obj) + "\n")
            fh.flush()
            os.fsync(fh.fileno())

    def _corrupt(self, lineno: int, why: object) -> CorruptLogError:
        return CorruptLogError(f"{self.path}: line {lineno} is not a valid run-log record ({why}).")

    def _load(self) -> None:
        if not self.path.exists():
            raise FileNotFoundError(f"resume file {self.path} not found.")
        raw = self.path.read_bytes()
        # (line number, byte offset where the line starts, text) for every non-blank line
        lines: list[tuple[int, int, str]] = []
        offset = 0
        for n, chunk in enumerate(raw.split(b"\n"), start=1):
            if chunk.strip():
                lines.append((n, offset, chunk.decode("utf-8", errors="replace")))
            offset += len(chunk) + 1
        if not lines:
            raise ResumeMismatchError(f"{self.path} is empty; not a run log.")
        try:
            first = json.loads(lines[0][2])
        except json.JSONDecodeError as exc:
            raise self._corrupt(lines[0][0], exc) from exc
        if not isinstance(first, dict) or first.get("type") != "header":
            raise ResumeMismatchError(f"{self.path} has no header record; not a run log.")
        if first.get("config") != self.header:
            raise ResumeMismatchError(
                f"{self.path} was written with a different config "
                f"({first.get('config')} != {self.header})."
            )
        for pos, (lineno, start, text) in enumerate(lines[1:], start=1):
            try:
                data = json.loads(text)
            except json.JSONDecodeError as exc:
                if pos == len(lines) - 1:  # interrupted final write: drop the partial line
                    logger.warning(
                        "%s: dropping truncated final line %d (interrupted write)",
                        self.path,
                        lineno,
                    )
                    with self.path.open("r+b") as fh:
                        fh.truncate(start)
                    break
                raise self._corrupt(lineno, exc) from exc
            try:
                rec = QueryRecord.from_json(data)
            except (TypeError, KeyError, ValueError, AttributeError) as exc:
                raise self._corrupt(lineno, f"{type(exc).__name__}: {exc}") from exc
            self._latest[rec.index] = rec  # last row per question wins
        # A complete last line may still lack its newline; never glue the next row onto it.
        with self.path.open("rb+") as fh:
            fh.seek(0, os.SEEK_END)
            if fh.tell() > 0:
                fh.seek(-1, os.SEEK_END)
                if fh.read(1) != b"\n":
                    fh.write(b"\n")

    def completed(self, index: int, question: str) -> bool:
        """True if this question finished without error. A different question text is refused."""
        rec = self._latest.get(index)
        if rec is None:
            return False
        if rec.question != question:
            raise ResumeMismatchError(f"{self.path}: question {index} differs from the dataset.")
        return rec.error is None

    def append(self, rec: QueryRecord) -> None:
        self._write(rec.to_json())
        self._latest[rec.index] = rec

    def records(self) -> list[QueryRecord]:
        return [self._latest[i] for i in sorted(self._latest)]


def evaluate_items(
    items: Sequence[EvalItem],
    agent: AgentLike,
    judge: JudgeLike | None = None,
    encoder: metrics.Encoder | None = None,
    k: int = 5,
    log: RunLog | None = None,
) -> list[QueryRecord]:
    """Run every item through the agent and score it. One record per item, in order.

    With a ``log``, questions already completed in it are skipped (the agent is not called),
    each new record is appended as soon as it is finished, and the returned records are the
    log's full contents.
    """
    records: list[QueryRecord] = []
    for i, item in enumerate(items):
        if log is not None and log.completed(i, item.question):
            continue
        rec = QueryRecord(index=i, question=item.question, gold_ids=list(item.gold_ids))
        records.append(rec)
        try:
            result = agent.run(item.question)
            rec.answer = result.answer or ""
            rec.retrieved_ids = [str(x) for x in (getattr(result, "retrieved_ids", None) or [])]
            rec.retrieval_used = _was_retrieval_used(result, rec.retrieved_ids)
        except Exception as exc:  # noqa: BLE001 - recorded per query, excluded from averages
            rec.retrieved_ids = []
            rec.error = f"{type(exc).__name__}: {exc}"
            if log is not None:
                log.append(rec)
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
        if log is not None:
            log.append(rec)
    return log.records() if log is not None else records


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
