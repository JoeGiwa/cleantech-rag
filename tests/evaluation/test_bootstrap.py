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


# ---- edge cases added by the tester ----


def test_bootstrap_two_values_within_range():
    ci = bootstrap_ci([0.0, 1.0])
    assert 0.0 <= ci["ci_low"] <= ci["mean"] <= ci["ci_high"] <= 1.0
    assert ci["n"] == 2


def test_bootstrap_accepts_generators_and_numpy():
    assert bootstrap_ci(iter(VALUES)) == bootstrap_ci(VALUES)


def test_bootstrap_nan_propagates_not_crashes():
    ci = bootstrap_ci([0.1, float("nan"), 0.3])
    assert ci is not None and ci["n"] == 3


def test_bootstrap_matches_summary_for_other_seed_default():
    # defaults are the documented 1000 resamples / 95% / seed 0
    assert bootstrap_ci(VALUES) == bootstrap_ci(VALUES, n_resamples=1000, confidence=0.95, seed=0)


def test_cis_judge_disabled_is_none_and_summary_judge_none():
    summary = summarize(RECORDS, {"k": 5}, judge_enabled=False, prices={})
    assert summary["judge"] is None
    assert summary["confidence_intervals"]["judge"] is None


def test_cis_exclude_errors_and_failed_judge_and_no_gold():
    recs = [_rec(i, v) for i, v in enumerate([0.2, 0.4, 0.6, 0.8])]
    recs[0].error = "Boom: x"
    recs[0].retrieval, recs[0].response = None, {}
    recs[1].retrieval = None  # no gold ids
    recs[2].judge = JudgeResult(scores=dict.fromkeys(CRITERIA), overall=None, error="bad json")
    summary = summarize(recs, {"k": 5}, judge_enabled=True, prices={})
    cis = summary["confidence_intervals"]
    assert cis["retrieval"]["all_queries"]["mrr"]["n"] == 2
    assert cis["retrieval"]["all_queries"]["mrr"]["mean"] == pytest.approx(0.7)
    assert cis["response"]["bleu"]["n"] == 3
    assert cis["response"]["bleu"]["mean"] == pytest.approx(summary["response"]["bleu"])
    assert cis["judge"]["overall"]["n"] == summary["judge"]["n_scored"] == 2
    assert cis["judge"]["overall"]["mean"] == pytest.approx(summary["judge"]["overall"])


def test_cis_all_errors_gives_none_entries():
    recs = [_rec(0, 0.5)]
    recs[0].error = "Boom"
    cis = confidence_intervals(recs, True)
    assert cis["retrieval"]["all_queries"]["mrr"] is None
    assert cis["response"]["bleu"] is None
    assert cis["judge"]["overall"] is None


def test_cis_empty_records():
    cis = confidence_intervals([], False)
    assert cis["retrieval"]["retrieval_only"]["mrr"] is None
    assert cis["judge"] is None


def test_cis_single_question_collapses():
    cis = confidence_intervals(RECORDS[:1], True)
    m = cis["retrieval"]["all_queries"]["mrr"]
    assert m["ci_low"] == m["ci_high"] == m["mean"] and m["n"] == 1


def test_cis_retrieval_only_view_filters():
    recs = [_rec(i, v) for i, v in enumerate([0.2, 0.4, 0.6])]
    recs[0].retrieval_used = False
    cis = confidence_intervals(recs, False)
    assert cis["retrieval"]["retrieval_only"]["mrr"]["n"] == 2
    assert cis["retrieval"]["all_queries"]["mrr"]["n"] == 3


def test_cis_are_deterministic_across_calls():
    assert confidence_intervals(RECORDS, True) == confidence_intervals(RECORDS, True)


def _old_summary(tmp_path, records=RECORDS, judge=True, extra_config=None):
    log_path = tmp_path / "run_x.jsonl"
    _write_log(log_path, records)
    config = {"k": 5, "judge_enabled": judge, "run_log": "run_x.jsonl", **(extra_config or {})}
    summary = summarize(records, config, judge_enabled=judge, prices={})
    del summary["confidence_intervals"]
    path = tmp_path / "summary_x.json"
    path.write_text(json.dumps(summary), encoding="utf-8")
    return path, log_path, summary


def test_recompute_is_idempotent(tmp_path):
    path, _, _ = _old_summary(tmp_path)
    runner = CliRunner()
    assert runner.invoke(cli.app, ["report", str(path), "--recompute"]).exit_code == 0
    first = path.read_bytes()
    assert runner.invoke(cli.app, ["report", str(path), "--recompute"]).exit_code == 0
    assert path.read_bytes() == first


def test_recompute_judge_disabled_summary(tmp_path):
    path, _, _ = _old_summary(tmp_path, judge=False)
    result = CliRunner().invoke(cli.app, ["report", str(path), "--recompute"])
    assert result.exit_code == 0, result.output
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["confidence_intervals"]["judge"] is None
    assert data["judge"] is None
    assert not (tmp_path / "summary_x_judge.png").exists()


def test_recompute_run_log_override(tmp_path):
    path, log_path, _ = _old_summary(tmp_path)
    other = tmp_path / "elsewhere.jsonl"
    log_path.rename(other)
    result = CliRunner().invoke(
        cli.app, ["report", str(path), "--recompute", "--run-log", str(other)]
    )
    assert result.exit_code == 0, result.output
    assert "confidence_intervals" in json.loads(path.read_text(encoding="utf-8"))


def test_recompute_no_run_log_in_config(tmp_path):
    path = tmp_path / "summary_x.json"
    path.write_text(json.dumps({"config": {}}), encoding="utf-8")
    result = CliRunner().invoke(cli.app, ["report", str(path), "--recompute"])
    assert result.exit_code != 0
    assert "run_log" in result.output
    assert "confidence_intervals" not in json.loads(path.read_text(encoding="utf-8"))


def test_recompute_mismatch_leaves_summary_untouched(tmp_path):
    path, log_path, _ = _old_summary(tmp_path)
    before = path.read_bytes()
    _write_log(tmp_path / "short.jsonl", RECORDS[:2])
    result = CliRunner().invoke(
        cli.app, ["report", str(path), "--recompute", "--run-log", str(tmp_path / "short.jsonl")]
    )
    assert result.exit_code != 0
    assert path.read_bytes() == before


def test_recompute_does_not_call_network(tmp_path, monkeypatch):
    import socket

    def boom(*a, **k):
        raise AssertionError("network used")

    monkeypatch.setattr(socket.socket, "connect", boom)
    path, _, _ = _old_summary(tmp_path)
    assert CliRunner().invoke(cli.app, ["report", str(path), "--recompute"]).exit_code == 0


def test_report_without_recompute_does_not_modify_json(tmp_path):
    path, _, _ = _old_summary(tmp_path)
    before = path.read_bytes()
    result = CliRunner().invoke(cli.app, ["report", str(path)])
    assert result.exit_code == 0, result.output
    assert path.read_bytes() == before
    assert (
        "["
        not in (tmp_path / "summary_x.md")
        .read_text(encoding="utf-8")
        .split("## LLM")[0]
        .split("## Retrieval")[1]
    )


def test_markdown_shows_mean_low_high_format():
    summary = summarize(RECORDS, {"k": 5}, judge_enabled=True, prices={})
    md = report.render_markdown(summary)
    ci = summary["confidence_intervals"]["response"]["bleu"]
    assert f"{ci['mean']:.3f} [{ci['ci_low']:.3f}, {ci['ci_high']:.3f}]" in md
    j = summary["confidence_intervals"]["judge"]["overall"]
    assert f"{j['mean']:.2f} [{j['ci_low']:.2f}, {j['ci_high']:.2f}]" in md


def test_markdown_tolerates_malformed_or_null_cis():
    summary = summarize(RECORDS, {"k": 5}, judge_enabled=True, prices={})
    summary["confidence_intervals"]["response"]["bleu"] = None
    summary["confidence_intervals"]["retrieval"] = "junk"
    summary["confidence_intervals"]["judge"] = None
    summary["confidence_intervals"]["response"]["rouge1"] = {"ci_low": float("nan"), "ci_high": 1}
    assert "Response quality" in report.render_markdown(summary)


def test_charts_with_and_without_cis(tmp_path):
    summary = summarize(RECORDS, {"k": 5}, judge_enabled=True, prices={})
    with_ci = tmp_path / "a" / "summary_a.json"
    with_ci.parent.mkdir()
    with_ci.write_text(json.dumps(summary), encoding="utf-8")
    del summary["confidence_intervals"]
    without = tmp_path / "b" / "summary_b.json"
    without.parent.mkdir()
    without.write_text(json.dumps(summary), encoding="utf-8")
    pa = report.generate_report(with_ci, with_ci.parent)
    pb = report.generate_report(without, without.parent)
    assert len(pa) == len(pb) == 4 and all(p.exists() for p in pa + pb)


def test_bar_chart_draws_error_bars(tmp_path, monkeypatch):
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib.container import ErrorbarContainer

    captured = {}
    orig = report._pyplot

    def spy():
        plt = orig()
        real = plt.close

        def close(fig=None):
            if fig is not None and hasattr(fig, "axes"):
                captured["containers"] = list(fig.axes[0].containers)
            return real(fig) if fig is not None else real()

        monkeypatch.setattr(plt, "close", close)
        return plt

    monkeypatch.setattr(report, "_pyplot", spy)
    series = {"Mean": {"a": 0.5, "b": 0.7}}
    cis = {"Mean": {"a": {"ci_low": 0.4, "ci_high": 0.6}, "b": {"ci_low": 0.6, "ci_high": 0.8}}}
    report._bar_chart(series, {"a": "A", "b": "B"}, "t", 1.0, tmp_path / "c.png", cis)
    assert (tmp_path / "c.png").exists()
    if "containers" in captured:
        assert any(isinstance(c, ErrorbarContainer) for c in captured["containers"])
