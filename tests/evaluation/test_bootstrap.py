import json

import pytest
from typer.testing import CliRunner

from cleantech_rag import cli
from cleantech_rag.evaluation import report
from cleantech_rag.evaluation.bootstrap import bootstrap_ci
from cleantech_rag.evaluation.judge import CRITERIA, JudgeResult
from cleantech_rag.evaluation.runner import (
    QueryRecord,
    RunLog,
    confidence_intervals,
    load_records,
    summarize,
)

VALUES = [0.0, 0.2, 0.4, 0.4, 0.6, 1.0, 0.8, 0.1, 0.9, 0.5]


def test_deterministic_with_seed():
    assert bootstrap_ci(VALUES) == bootstrap_ci(VALUES)
    assert bootstrap_ci(VALUES, seed=1) != bootstrap_ci(VALUES, seed=2)


def test_ci_contains_mean():
    ci = bootstrap_ci(VALUES)
    assert ci["ci_low"] <= ci["mean"] <= ci["ci_high"]
    assert ci["mean"] == pytest.approx(sum(VALUES) / len(VALUES))
    assert ci["n"] == len(VALUES)


def test_single_value_collapses():
    assert bootstrap_ci([0.7]) == {"mean": 0.7, "ci_low": 0.7, "ci_high": 0.7, "n": 1}


def test_empty_is_none():
    assert bootstrap_ci([]) is None


def test_constant_values_zero_width():
    ci = bootstrap_ci([0.5] * 6)
    assert ci["ci_low"] == ci["ci_high"] == 0.5


def _rec(i, p, judge=True):
    retrieval = dict.fromkeys(
        ("precision_at_k", "precision_at_k_retrieved", "recall_at_k", "mrr", "ndcg_at_k"), p
    )
    response = dict.fromkeys(("rouge1", "rouge2", "rougeL", "bleu", "semantic_similarity"), p)
    j = JudgeResult(scores=dict.fromkeys(CRITERIA, 3 + i % 3), overall=3.0 + i % 3)
    return QueryRecord(
        index=i,
        question=f"q{i}",
        gold_ids=["a"],
        retrieval_used=True,
        answer="x",
        retrieval=retrieval,
        response=response,
        judge=j if judge else None,
    )


RECORDS = [_rec(i, v) for i, v in enumerate([0.1, 0.4, 0.2, 0.9, 0.5, 0.3])]


def test_confidence_intervals_match_summary_means():
    summary = summarize(RECORDS, {"k": 5}, judge_enabled=True, prices={})
    cis = summary["confidence_intervals"]
    assert cis["retrieval"]["all_queries"]["mrr"]["mean"] == pytest.approx(
        summary["retrieval"]["all_queries"]["mrr"]
    )
    assert cis["response"]["bleu"]["mean"] == pytest.approx(summary["response"]["bleu"])
    assert cis["judge"]["overall"]["mean"] == pytest.approx(summary["judge"]["overall"])
    assert summary["retrieval"]["all_queries"]["mrr"] == pytest.approx(0.4)  # scalars unchanged
    assert confidence_intervals(RECORDS, False)["judge"] is None


def test_summary_without_ci_still_reports(tmp_path):
    # backward compat: old summaries have no confidence_intervals key
    summary = summarize(RECORDS, {"k": 5}, judge_enabled=True, prices={})
    del summary["confidence_intervals"]
    assert "[" not in report.render_markdown(summary).split("## Retrieval")[1].split("## LLM")[0]


def _write_log(path, records):
    log = RunLog(path, {"dataset": "x"})
    for r in records:
        log.append(r)


def test_recompute_adds_cis_from_log(tmp_path):
    log_path = tmp_path / "run_x.jsonl"
    _write_log(log_path, RECORDS)
    config = {"k": 5, "judge_enabled": True, "run_log": "run_x.jsonl"}
    summary = summarize(RECORDS, config, judge_enabled=True, prices={})
    expected_cis = summary.pop("confidence_intervals")
    summary_path = tmp_path / "summary_x.json"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    before = log_path.read_bytes()

    result = CliRunner().invoke(cli.app, ["report", str(summary_path), "--recompute"])

    assert result.exit_code == 0, result.output
    data = json.loads(summary_path.read_text(encoding="utf-8"))
    assert data["confidence_intervals"] == json.loads(json.dumps(expected_cis))
    assert {k: v for k, v in data.items() if k != "confidence_intervals"} == summary
    assert log_path.read_bytes() == before  # run log untouched
    md = (tmp_path / "summary_x.md").read_text(encoding="utf-8")
    assert "[" in md and "Mean" in md
    assert (tmp_path / "summary_x_retrieval.png").exists()
    assert load_records(log_path)[1][0].question == "q0"


def test_recompute_missing_log(tmp_path):
    summary_path = tmp_path / "summary_x.json"
    summary_path.write_text(json.dumps({"config": {"run_log": "nope.jsonl"}}), encoding="utf-8")
    result = CliRunner().invoke(cli.app, ["report", str(summary_path), "--recompute"])
    assert result.exit_code != 0
    assert "not found" in result.output


def test_recompute_count_mismatch(tmp_path):
    log_path = tmp_path / "run_x.jsonl"
    _write_log(log_path, RECORDS[:2])
    summary_path = tmp_path / "summary_x.json"
    summary_path.write_text(
        json.dumps({"config": {"run_log": "run_x.jsonl"}, "counts": {"n_queries": 6}}),
        encoding="utf-8",
    )
    result = CliRunner().invoke(cli.app, ["report", str(summary_path), "--recompute"])
    assert result.exit_code != 0
