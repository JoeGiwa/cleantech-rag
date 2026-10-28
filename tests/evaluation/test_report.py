import json
import re

import pytest
from typer.testing import CliRunner

from cleantech_rag import cli
from cleantech_rag.evaluation import report

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def _view(n, p):
    return {
        "n": n,
        "precision_at_k": p,
        "precision_at_k_retrieved": p * 1.5,
        "recall_at_k": 0.5,
        "mrr": 0.4,
        "ndcg_at_k": 0.3,
    }


def synthetic_summary(judge=True, none_value=False):
    return {
        "config": {"dataset": "complex", "k": 5, "agent_model": "m"},
        "counts": {"n_queries": 4, "n_errors": 1},
        "retrieval": {
            "all_queries": _view(3, 0.2),
            "retrieval_only": {**_view(2, 0.3), "mrr": None if none_value else 0.6},
        },
        "response": {
            "rouge1": 0.3,
            "rouge2": 0.1,
            "rougeL": 0.2,
            "bleu": 0.05,
            "semantic_similarity": 0.7,
        },
        "judge": (
            {
                "n_total": 3,
                "n_scored": 2,
                "n_failed": 1,
                "n_pending": 0,
                "relevance": 4.0,
                "accuracy": 3.5,
                "completeness": 3.0,
                "clarity": 4.5,
                "overall": 3.75,
            }
            if judge
            else None
        ),
    }


def _write(tmp_path, summary):
    path = tmp_path / "summary_x.json"
    path.write_text(json.dumps(summary), encoding="utf-8")
    return path


def test_generate_report_writes_pngs_and_markdown(tmp_path):
    out = tmp_path / "out"
    paths = report.generate_report(_write(tmp_path, synthetic_summary()), out)
    pngs = [p for p in paths if p.suffix == ".png"]
    assert {p.name for p in pngs} == {
        "summary_x_retrieval.png",
        "summary_x_response.png",
        "summary_x_judge.png",
    }
    for p in pngs:
        assert p.read_bytes().startswith(PNG_MAGIC)
    md = (out / "summary_x.md").read_text(encoding="utf-8")
    assert "| Recall@k | 0.500 | 0.500 |" in md
    assert "| Overall | 3.75 |" in md


def test_no_judge_and_none_values(tmp_path):
    out = tmp_path / "out"
    paths = report.generate_report(
        _write(tmp_path, synthetic_summary(judge=False, none_value=True)), out
    )
    assert not any(p.name.endswith("_judge.png") for p in paths)
    md = (out / "summary_x.md").read_text(encoding="utf-8")
    assert "Judge was not run." in md
    assert re.search(r"\| MRR \| 0\.400 \| n/a \|", md)


def test_report_cli(tmp_path):
    out = tmp_path / "charts"
    res = CliRunner().invoke(
        cli.app, ["report", str(_write(tmp_path, synthetic_summary())), "--output-dir", str(out)]
    )
    assert res.exit_code == 0, res.output
    assert (out / "summary_x_retrieval.png").exists()


def test_report_cli_missing_file(tmp_path):
    res = CliRunner().invoke(cli.app, ["report", str(tmp_path / "nope.json")])
    assert res.exit_code == 1


def test_empty_summary_does_not_crash(tmp_path):
    out = tmp_path / "out"
    paths = report.generate_report(_write(tmp_path, {}), out)
    assert all(p.exists() for p in paths)
    md = (out / "summary_x.md").read_text(encoding="utf-8")
    assert "Judge was not run." in md
    assert "n/a" in md


def test_partial_summary_missing_keys(tmp_path):
    summary = {"config": {"k": 5}, "retrieval": {"all_queries": {"n": 2, "mrr": 0.5}}}
    out = tmp_path / "out"
    paths = report.generate_report(_write(tmp_path, summary), out)
    assert any(p.name.endswith("_retrieval.png") for p in paths)
    md = (out / "summary_x.md").read_text(encoding="utf-8")
    assert "| MRR | 0.500 | n/a |" in md


def test_partial_judge_missing_criteria(tmp_path):
    out = tmp_path / "out"
    report.generate_report(_write(tmp_path, {"judge": {"overall": 4.0}}), out)
    md = (out / "summary_x.md").read_text(encoding="utf-8")
    assert "| Overall | 4.00 |" in md
    assert "| Relevance | n/a |" in md
    assert (out / "summary_x_judge.png").read_bytes().startswith(PNG_MAGIC)


def test_all_none_values(tmp_path):
    summary = synthetic_summary()
    for view in summary["retrieval"].values():
        for key in view:
            if key != "n":
                view[key] = None
    summary["response"] = dict.fromkeys(summary["response"])
    out = tmp_path / "out"
    paths = report.generate_report(_write(tmp_path, summary), out)
    assert all(p.exists() for p in paths)


def test_nan_values_do_not_crash(tmp_path):
    summary = synthetic_summary()
    summary["response"]["bleu"] = float("nan")
    out = tmp_path / "out"
    paths = report.generate_report(_write(tmp_path, summary), out)
    assert all(p.exists() for p in paths)


def test_output_dir_created_nested(tmp_path):
    out = tmp_path / "a" / "b" / "c"
    report.generate_report(_write(tmp_path, synthetic_summary()), out)
    assert (out / "summary_x.md").exists()


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        report.generate_report(tmp_path / "nope.json", tmp_path / "out")


def test_malformed_json_raises(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(json.JSONDecodeError):
        report.generate_report(path, tmp_path / "out")


def test_report_cli_malformed_json_clean_error(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text("{not json", encoding="utf-8")
    res = CliRunner().invoke(cli.app, ["report", str(path)])
    assert res.exit_code == 1
    assert isinstance(res.exception, SystemExit), repr(res.exception)


def test_report_cli_non_object_json(tmp_path):
    path = tmp_path / "list.json"
    path.write_text("[1, 2]", encoding="utf-8")
    res = CliRunner().invoke(cli.app, ["report", str(path)])
    assert res.exit_code == 1
    assert isinstance(res.exception, SystemExit), repr(res.exception)


def test_report_cli_defaults_next_to_json(tmp_path):
    path = _write(tmp_path, synthetic_summary())
    res = CliRunner().invoke(cli.app, ["report", str(path)])
    assert res.exit_code == 0, res.output
    assert (tmp_path / "summary_x_response.png").exists()


def test_load_summary_rejects_non_object(tmp_path):
    path = tmp_path / "list.json"
    path.write_text("[1]", encoding="utf-8")
    with pytest.raises(ValueError):
        report.load_summary(path)


@pytest.mark.parametrize("bad", [None, [1, 2], "text", 5])
@pytest.mark.parametrize("section", ["config", "counts", "retrieval", "response", "judge"])
def test_wrong_shaped_sections_do_not_crash(tmp_path, section, bad):
    summary = synthetic_summary()
    summary[section] = bad
    out = tmp_path / "out"
    paths = report.generate_report(_write(tmp_path, summary), out)
    assert (out / "summary_x.md").exists()
    assert all(p.exists() for p in paths)


def test_non_numeric_metric_values_do_not_crash(tmp_path):
    summary = synthetic_summary()
    summary["retrieval"]["all_queries"]["mrr"] = "high"
    summary["retrieval"]["all_queries"]["n"] = "three"
    summary["response"]["bleu"] = [0.1]
    summary["response"]["rouge1"] = True
    summary["judge"]["overall"] = {"x": 1}
    out = tmp_path / "out"
    report.generate_report(_write(tmp_path, summary), out)
    md = (out / "summary_x.md").read_text(encoding="utf-8")
    assert "| BLEU | n/a |" in md and "| MRR | n/a | 0.600 |" in md


def test_cli_directory_path_clean_error(tmp_path):
    res = CliRunner().invoke(cli.app, ["report", str(tmp_path)])
    assert res.exit_code == 1
    assert isinstance(res.exception, SystemExit), repr(res.exception)


def test_cli_unexpected_shape_errors_are_clean(tmp_path, monkeypatch):
    path = _write(tmp_path, synthetic_summary())
    for exc in (TypeError("t"), AttributeError("a"), PermissionError("p")):

        def boom(*a, _exc=exc, **k):
            raise _exc

        monkeypatch.setattr("cleantech_rag.evaluation.report.generate_report", boom)
        res = CliRunner().invoke(cli.app, ["report", str(path)])
        assert res.exit_code == 1 and isinstance(res.exception, SystemExit), repr(exc)


def test_missing_matplotlib_leaves_no_stray_markdown(tmp_path, monkeypatch):
    def no_mpl():
        raise ImportError("matplotlib is required")

    monkeypatch.setattr(report, "_pyplot", no_mpl)
    out = tmp_path / "out"
    with pytest.raises(ImportError):
        report.generate_report(_write(tmp_path, synthetic_summary()), out)
    assert not out.exists() or list(out.iterdir()) == []
