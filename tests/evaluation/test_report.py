import json
import re

import pytest

pytest.importorskip("matplotlib")

from typer.testing import CliRunner  # noqa: E402

from cleantech_rag import cli  # noqa: E402
from cleantech_rag.evaluation import report  # noqa: E402

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
