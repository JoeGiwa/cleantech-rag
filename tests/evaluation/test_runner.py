import csv
import json
from datetime import datetime
from types import SimpleNamespace

import pytest

from cleantech_rag.evaluation.judge import CRITERIA, JudgeResult
from cleantech_rag.evaluation.runner import (
    CSV_COLUMNS,
    EvalItem,
    evaluate_items,
    summarize,
    write_outputs,
)


def fake_encoder(texts):
    return [[1.0, 0.0] for _ in texts]


class FakeAgent:
    """Maps question -> (ids, tools) or an Exception; records call order."""

    def __init__(self, plan):
        self.plan = plan
        self.calls = []

    def run(self, question, thread_id=None):
        self.calls.append(question)
        out = self.plan[question]
        if isinstance(out, Exception):
            raise out
        ids, tools = out
        return SimpleNamespace(answer=f"answer {question}", retrieved_ids=ids, tools_used=tools)


class FakeJudge:
    def __init__(self, fail_on=()):
        self.fail_on = fail_on

    def evaluate(self, query, prediction, reference):
        if query in self.fail_on:
            return JudgeResult(error="ValidationError: bad")
        return JudgeResult(scores=dict.fromkeys(CRITERIA, 4), overall=4.0)


RET = [{"tool": "retrieve", "ok": True}]
ITEMS = [
    EvalItem("q0", "ref zero", ["a"]),
    EvalItem("q1", "ref one", ["b"]),  # no retrieval
    EvalItem("q2", "ref two", ["x"]),  # errors
    EvalItem("q3", "ref three", ["c", "d"]),
]
PLAN = {
    "q0": (["a", "z"], RET),  # P@5 .2, R 1, MRR 1
    "q1": ([], [{"tool": "web_search", "ok": True}]),
    "q2": RuntimeError("boom"),
    "q3": (["y", "c"], RET),  # P@5 .2, R .5, MRR .5
}


def run(judge=None):
    agent = FakeAgent(PLAN)
    records = evaluate_items(ITEMS, agent, judge, encoder=fake_encoder)
    return agent, records


def test_each_query_uses_its_own_ids():
    _, recs = run()
    assert recs[0].retrieved_ids == ["a", "z"]
    assert recs[1].retrieved_ids == []  # not the previous query's ids
    assert recs[1].retrieval["recall_at_k"] == 0.0
    assert recs[3].retrieved_ids == ["y", "c"]
    assert [r.retrieval_used for r in recs] == [True, False, False, True]


def test_both_aggregate_views():
    _, recs = run(FakeJudge())
    s = summarize(recs, {}, judge_enabled=True)
    allq = s["retrieval"]["all_queries"]
    only = s["retrieval"]["retrieval_only"]
    assert allq["n"] == 3 and only["n"] == 2
    assert allq["precision_at_k"] == pytest.approx((0.2 + 0 + 0.2) / 3)
    assert only["precision_at_k"] == pytest.approx(0.2)
    assert allq["recall_at_k"] == pytest.approx((1 + 0 + 0.5) / 3)
    assert only["recall_at_k"] == pytest.approx(0.75)
    assert only["mrr"] == pytest.approx(0.75)
    assert only["precision_at_k_retrieved"] == pytest.approx(0.5)


def test_errors_excluded_and_counted():
    _, recs = run()
    assert recs[2].error and "boom" in recs[2].error
    s = summarize(recs, {}, judge_enabled=False)
    assert s["counts"]["n_errors"] == 1
    assert s["counts"]["n_evaluated"] == 3
    assert s["response"]["semantic_similarity"] == pytest.approx(1.0)


def test_judge_failures_excluded_and_counted():
    _, recs = run(FakeJudge(fail_on={"q3"}))
    s = summarize(recs, {}, judge_enabled=True)
    assert s["judge"]["n_failed"] == 1
    assert s["judge"]["n_scored"] == 2
    assert s["judge"]["overall"] == 4.0


def test_no_judge_gives_null_judge():
    _, recs = run()
    assert summarize(recs, {}, judge_enabled=False)["judge"] is None


def test_summary_schema_and_files(tmp_path):
    _, recs = run(FakeJudge())
    s = summarize(recs, {"dataset": "complex"}, judge_enabled=True)
    jp, cp = write_outputs(recs, s, tmp_path / "res", datetime(2026, 1, 2, 3, 4, 5))
    assert jp.name == "summary_20260102T030405.json"
    data = json.loads(jp.read_text())
    assert set(data) == {"config", "counts", "retrieval", "response", "judge"}
    assert set(data["counts"]) == {
        "n_queries",
        "n_errors",
        "n_evaluated",
        "n_retrieval_used",
        "n_no_gold",
    }
    assert set(data["retrieval"]) == {"all_queries", "retrieval_only"}
    for view in data["retrieval"].values():
        assert set(view) == {
            "n",
            "precision_at_k",
            "precision_at_k_retrieved",
            "recall_at_k",
            "mrr",
            "ndcg_at_k",
        }
    assert set(data["response"]) == {"rouge1", "rouge2", "rougeL", "bleu", "semantic_similarity"}
    assert {"n_scored", "n_failed", "overall"} <= set(data["judge"])
    rows = list(csv.DictReader(cp.open()))
    assert len(rows) == 4 and tuple(rows[0]) == CSV_COLUMNS
    assert "boom" in rows[2]["error"]
    assert rows[1]["retrieval_used"] == "False"


def test_query_without_gold_ids_left_out_of_retrieval():
    items = [EvalItem("q0", "r", [])]
    recs = evaluate_items(items, FakeAgent({"q0": (["a"], RET)}), None, encoder=fake_encoder)
    s = summarize(recs, {}, judge_enabled=False)
    assert s["counts"]["n_no_gold"] == 1
    assert s["retrieval"]["all_queries"]["n"] == 0
    assert s["retrieval"]["all_queries"]["mrr"] is None
