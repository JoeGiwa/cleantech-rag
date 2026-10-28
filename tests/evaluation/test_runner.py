import csv
import json
from datetime import datetime
from types import SimpleNamespace

import pytest

from cleantech_rag.evaluation.judge import CRITERIA, JudgeResult
from cleantech_rag.evaluation.runner import (
    CSV_COLUMNS,
    CorruptLogError,
    EvalItem,
    ResumeMismatchError,
    RunLog,
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
        if isinstance(out, BaseException):
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


# ------------------------------------------------------------ extra edge cases


def _recs(items, plan, judge=None):
    return evaluate_items(items, FakeAgent(plan), judge, encoder=fake_encoder)


def test_all_queries_errored_gives_empty_averages():
    items = [EvalItem("q0", "r", ["a"]), EvalItem("q1", "r", ["b"])]
    plan = {"q0": ValueError("x"), "q1": RuntimeError("y")}
    s = summarize(_recs(items, plan, FakeJudge()), {}, judge_enabled=True)
    assert s["counts"]["n_errors"] == 2 and s["counts"]["n_evaluated"] == 0
    assert s["retrieval"]["all_queries"]["n"] == 0
    assert s["retrieval"]["all_queries"]["mrr"] is None
    assert all(v is None for v in s["response"].values())
    assert s["judge"]["n_scored"] == 0 and s["judge"]["overall"] is None
    json.dumps(s)  # serialisable


def test_zero_retrieval_only_queries():
    items = [EvalItem("q0", "r", ["a"]), EvalItem("q1", "r", ["b"])]
    plan = {"q0": ([], []), "q1": ([], [{"tool": "web_search", "ok": True}])}
    s = summarize(_recs(items, plan), {}, judge_enabled=False)
    assert s["retrieval"]["retrieval_only"]["n"] == 0
    assert s["retrieval"]["retrieval_only"]["precision_at_k"] is None
    assert s["retrieval"]["all_queries"]["n"] == 2
    assert s["retrieval"]["all_queries"]["recall_at_k"] == 0.0
    assert s["counts"]["n_retrieval_used"] == 0


def test_empty_items():
    recs = evaluate_items([], FakeAgent({}), None, encoder=fake_encoder)
    assert recs == []
    s = summarize(recs, {}, judge_enabled=False)
    assert s["counts"]["n_queries"] == 0
    assert s["retrieval"]["all_queries"]["n"] == 0


def test_retrieve_ok_with_no_hits_counts_as_retrieval_used():
    items = [EvalItem("q0", "r", ["a"])]
    recs = _recs(items, {"q0": ([], RET)})
    assert recs[0].retrieval_used is True
    s = summarize(recs, {}, judge_enabled=False)
    assert s["retrieval"]["retrieval_only"]["n"] == 1
    assert s["retrieval"]["retrieval_only"]["recall_at_k"] == 0.0


def test_failed_retrieve_tool_not_counted_as_used():
    recs = _recs([EvalItem("q0", "r", ["a"])], {"q0": ([], [{"tool": "retrieve", "ok": False}])})
    assert recs[0].retrieval_used is False


def test_ids_coerced_to_str_and_missing_attrs_tolerated():
    class Agent:
        def run(self, question, thread_id=None):
            return SimpleNamespace(answer="a", retrieved_ids=[1, 2])

    recs = evaluate_items([EvalItem("q", "r", ["1"])], Agent(), None, encoder=fake_encoder)
    assert recs[0].retrieved_ids == ["1", "2"]
    assert recs[0].retrieval["mrr"] == 1.0


def test_none_retrieved_ids_is_empty():
    class Agent:
        def run(self, question, thread_id=None):
            return SimpleNamespace(answer="a", retrieved_ids=None, tools_used=None)

    recs = evaluate_items([EvalItem("q", "r", ["1"])], Agent(), None, encoder=fake_encoder)
    assert recs[0].retrieved_ids == [] and recs[0].retrieval_used is False


def test_error_after_success_does_not_leak_ids():
    items = [EvalItem("q0", "r", ["a"]), EvalItem("q1", "r", ["a"])]
    recs = _recs(items, {"q0": (["a"], RET), "q1": RuntimeError("x")})
    assert recs[1].retrieved_ids == [] and recs[1].retrieval is None
    assert recs[1].error.startswith("RuntimeError")


def test_agent_called_once_per_query_in_order():
    agent, _ = run()
    assert agent.calls == ["q0", "q1", "q2", "q3"]


def test_judge_not_called_for_errored_query():
    seen = []

    class J(FakeJudge):
        def evaluate(self, query, prediction, reference):
            seen.append(query)
            return super().evaluate(query, prediction, reference)

    run(J())
    assert "q2" not in seen and len(seen) == 3


def test_fatal_judge_error_propagates():
    class J:
        def evaluate(self, *a):
            raise PermissionError("bad key")

    with pytest.raises(PermissionError):
        run(J())


def test_judge_failure_keeps_retrieval_metrics_in_aggregates():
    _, recs = run(FakeJudge(fail_on={"q0", "q1", "q3"}))
    s = summarize(recs, {}, judge_enabled=True)
    assert s["judge"]["n_scored"] == 0 and s["judge"]["overall"] is None
    assert s["retrieval"]["all_queries"]["n"] == 3
    assert s["counts"]["n_evaluated"] == 3


def test_csv_contents(tmp_path):
    _, recs = run(FakeJudge(fail_on={"q3"}))
    s = summarize(recs, {}, judge_enabled=True)
    _, cp = write_outputs(recs, s, tmp_path, datetime(2026, 1, 1))
    rows = list(csv.DictReader(cp.open()))
    assert rows[0]["retrieved_ids"] == "a;z" and rows[0]["gold_ids"] == "a"
    assert rows[0]["n_retrieved"] == "2" and rows[0]["judge_overall"] == "4.0"
    assert rows[1]["retrieved_ids"] == "" and rows[1]["retrieval_used"] == "False"
    assert rows[2]["error"].startswith("RuntimeError") and rows[2]["mrr"] == ""
    assert rows[3]["judge_error"].startswith("ValidationError")
    assert [r["index"] for r in rows] == ["0", "1", "2", "3"]


def test_write_outputs_no_judge_leaves_judge_columns_blank(tmp_path):
    _, recs = run()
    s = summarize(recs, {}, judge_enabled=False)
    jp, cp = write_outputs(recs, s, tmp_path, datetime(2026, 1, 1))
    assert json.loads(jp.read_text())["judge"] is None
    assert all(r["judge_overall"] == "" for r in csv.DictReader(cp.open()))


def test_question_with_delimiters_roundtrips_in_csv(tmp_path):
    items = [EvalItem('a, "b"\nc; d', "r", ["a"])]
    recs = _recs(items, {items[0].question: (["a"], RET)})
    _, cp = write_outputs(recs, summarize(recs, {}, False), tmp_path, datetime(2026, 1, 1))
    assert list(csv.DictReader(cp.open()))[0]["question"] == items[0].question


# ---------------------------------------------------------------- run log / resume

HEADER = {"dataset": "complex", "k": 5}
LOG_ITEMS = [EvalItem(f"q{i}", f"ref {i}", [str(i)]) for i in range(4)]
LOG_PLAN = {f"q{i}": ([str(i)], RET) for i in range(4)}


def _rows(path):
    return [json.loads(ln) for ln in path.read_text().splitlines()][1:]


def test_rows_appended_as_each_query_completes(tmp_path):
    plan = dict(LOG_PLAN)
    plan["q2"] = KeyboardInterrupt()  # not an Exception: aborts the run
    log = RunLog(tmp_path / "run.jsonl", HEADER)
    with pytest.raises(KeyboardInterrupt):
        evaluate_items(LOG_ITEMS, FakeAgent(plan), None, encoder=fake_encoder, log=log)
    assert [r["index"] for r in _rows(tmp_path / "run.jsonl")] == [0, 1]


def test_fatal_judge_error_keeps_earlier_rows(tmp_path):
    class Fatal:
        def evaluate(self, query, prediction, reference):
            if query == "q2":
                raise RuntimeError("auth")
            return JudgeResult(scores=dict.fromkeys(CRITERIA, 5), overall=5.0)

    log = RunLog(tmp_path / "run.jsonl", HEADER)
    with pytest.raises(RuntimeError):
        evaluate_items(LOG_ITEMS, FakeAgent(LOG_PLAN), Fatal(), encoder=fake_encoder, log=log)
    rows = _rows(tmp_path / "run.jsonl")
    assert [r["index"] for r in rows] == [0, 0, 1, 1, 2]  # agent row, judged row, per query
    assert rows[-1]["judge_pending"] is True and rows[-1]["judge"] is None


def test_resume_skips_completed_and_summary_covers_all(tmp_path):
    path = tmp_path / "run.jsonl"
    log = RunLog(path, HEADER)
    evaluate_items(LOG_ITEMS[:2], FakeAgent(LOG_PLAN), FakeJudge(), encoder=fake_encoder, log=log)
    agent = FakeAgent(LOG_PLAN)
    log2 = RunLog(path, HEADER, resume=True)
    recs = evaluate_items(LOG_ITEMS, agent, FakeJudge(), encoder=fake_encoder, log=log2)
    assert agent.calls == ["q2", "q3"]
    assert [r.index for r in recs] == [0, 1, 2, 3]
    s = summarize(recs, {}, judge_enabled=True)
    assert s["counts"]["n_evaluated"] == 4
    assert s["judge"]["n_scored"] == 4  # judge results survive the JSON round trip
    assert s["retrieval"]["all_queries"]["recall_at_k"] == 1.0


def test_resume_retries_errored_rows_latest_wins(tmp_path):
    path = tmp_path / "run.jsonl"
    plan = dict(LOG_PLAN)
    plan["q1"] = RuntimeError("flaky")
    evaluate_items(
        LOG_ITEMS[:2], FakeAgent(plan), None, encoder=fake_encoder, log=RunLog(path, HEADER)
    )
    agent = FakeAgent(LOG_PLAN)
    recs = evaluate_items(
        LOG_ITEMS[:2],
        agent,
        None,
        encoder=fake_encoder,
        log=RunLog(path, HEADER, resume=True),
    )
    assert agent.calls == ["q1"]
    assert all(r.error is None for r in recs) and len(recs) == 2


def test_resume_header_mismatch_refused(tmp_path):
    path = tmp_path / "run.jsonl"
    RunLog(path, HEADER)
    with pytest.raises(ResumeMismatchError):
        RunLog(path, {**HEADER, "dataset": "simple"}, resume=True)


def test_resume_different_question_text_refused(tmp_path):
    path = tmp_path / "run.jsonl"
    log = RunLog(path, HEADER)
    evaluate_items(LOG_ITEMS[:1], FakeAgent(LOG_PLAN), None, encoder=fake_encoder, log=log)
    other = [EvalItem("changed", "r", ["0"])]
    with pytest.raises(ResumeMismatchError):
        evaluate_items(
            other, FakeAgent({}), None, log=RunLog(path, HEADER, resume=True), encoder=fake_encoder
        )


def test_new_run_refuses_existing_file_and_resume_requires_header(tmp_path):
    path = tmp_path / "run.jsonl"
    RunLog(path, HEADER)
    with pytest.raises(FileExistsError):
        RunLog(path, HEADER)
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"type": "row"}\n')
    with pytest.raises(ResumeMismatchError):
        RunLog(bad, HEADER, resume=True)


def test_none_answer_becomes_empty_string():
    class A:
        def run(self, q, thread_id=None):
            return SimpleNamespace(answer=None, retrieved_ids=[], tools_used=[])

    recs = evaluate_items([EvalItem("q", "r", ["1"])], A(), None, encoder=fake_encoder)
    assert recs[0].answer == "" and recs[0].error is None


def _two_row_log(tmp_path):
    path = tmp_path / "run.jsonl"
    evaluate_items(
        LOG_ITEMS[:2],
        FakeAgent(LOG_PLAN),
        None,
        encoder=fake_encoder,
        log=RunLog(path, HEADER),
    )
    return path


def test_resume_truncated_final_line_is_dropped(tmp_path, caplog):
    path = _two_row_log(tmp_path)
    with path.open("a") as fh:
        fh.write('{"type": "row", "index": 2, "ques')  # crash mid-write, no newline
    agent = FakeAgent(LOG_PLAN)
    recs = evaluate_items(
        LOG_ITEMS,
        agent,
        None,
        encoder=fake_encoder,
        log=RunLog(path, HEADER, resume=True),
    )
    assert agent.calls == ["q2", "q3"]
    assert [r.index for r in recs] == [0, 1, 2, 3]
    assert "truncated" in caplog.text
    assert [r["index"] for r in _rows(path)] == [0, 1, 2, 3]  # every line parses


def test_resume_complete_last_line_without_newline_not_glued(tmp_path):
    path = _two_row_log(tmp_path)
    path.write_text(path.read_text().rstrip("\n"))
    rec = evaluate_items([LOG_ITEMS[2]], FakeAgent(LOG_PLAN), None, encoder=fake_encoder)[0]
    rec.index = 2
    RunLog(path, HEADER, resume=True).append(rec)
    assert [r["index"] for r in _rows(path)] == [0, 1, 2]


def test_resume_corrupt_middle_line_names_line_number(tmp_path):
    path = _two_row_log(tmp_path)
    lines = path.read_text().splitlines()
    lines[1] = "{not json"
    path.write_text("\n".join(lines) + "\n")
    with pytest.raises(CorruptLogError, match="line 2"):
        RunLog(path, HEADER, resume=True)


def test_resume_non_json_first_line(tmp_path):
    path = tmp_path / "run.jsonl"
    path.write_text("garbage\n")
    with pytest.raises(CorruptLogError, match="line 1"):
        RunLog(path, HEADER, resume=True)


@pytest.mark.parametrize("row", ['{"type": "row", "bogus": 1}', '{"type": "row"}', "[1, 2]"])
def test_resume_row_with_bad_keys_is_corrupt_error(tmp_path, row):
    path = _two_row_log(tmp_path)
    with path.open("a") as fh:
        fh.write(row + "\n")
        fh.write(json.dumps(_rows(path)[0]) + "\n")  # keep the bad row off the final line
    with pytest.raises(CorruptLogError, match="line 4"):
        RunLog(path, HEADER, resume=True)


def test_summary_identical_after_reload(tmp_path):
    path = tmp_path / "run.jsonl"
    mem = evaluate_items(
        ITEMS,
        FakeAgent(PLAN),
        FakeJudge(),
        encoder=fake_encoder,
        log=RunLog(path, HEADER),
    )
    reloaded = RunLog(path, HEADER, resume=True).records()
    assert summarize(mem, {}, True) == summarize(reloaded, {}, True)


class CountingJudge(FakeJudge):
    def __init__(self, fail_on=()):
        super().__init__(fail_on)
        self.seen = []

    def evaluate(self, query, prediction, reference):
        self.seen.append((query, prediction))
        return super().evaluate(query, prediction, reference)


def test_fatal_judge_error_saves_agent_row_and_resume_only_judges(tmp_path):
    path = tmp_path / "run.jsonl"

    class Fatal(CountingJudge):
        def evaluate(self, query, prediction, reference):
            if query == "q1":
                raise RuntimeError("auth")
            return super().evaluate(query, prediction, reference)

    with pytest.raises(RuntimeError):
        evaluate_items(
            LOG_ITEMS,
            FakeAgent(LOG_PLAN),
            Fatal(),
            encoder=fake_encoder,
            log=RunLog(path, HEADER),
        )
    pending = RunLog(path, HEADER, resume=True).lookup(1, "q1")
    assert pending.answer == "answer q1" and pending.judge is None and pending.judge_pending
    assert pending.retrieval is not None and pending.response

    s_mid = summarize(RunLog(path, HEADER, resume=True).records(), {}, True)
    assert s_mid["judge"]["n_pending"] == 1 and s_mid["judge"]["n_scored"] == 1

    agent, judge = FakeAgent(LOG_PLAN), CountingJudge()
    recs = evaluate_items(
        LOG_ITEMS,
        agent,
        judge,
        encoder=fake_encoder,
        log=RunLog(path, HEADER, resume=True),
    )
    assert agent.calls == ["q2", "q3"]  # q1 not re-run
    assert judge.seen[0] == ("q1", "answer q1")  # judged from the saved answer
    s = summarize(recs, {}, True)
    assert s["judge"]["n_scored"] == 4 and s["judge"]["n_pending"] == 0


def test_judge_failed_row_rejudged_without_agent_call(tmp_path):
    path = tmp_path / "run.jsonl"
    evaluate_items(
        LOG_ITEMS[:2],
        FakeAgent(LOG_PLAN),
        FakeJudge(fail_on={"q1"}),
        encoder=fake_encoder,
        log=RunLog(path, HEADER),
    )
    agent, judge = FakeAgent(LOG_PLAN), CountingJudge()
    recs = evaluate_items(
        LOG_ITEMS[:2], agent, judge, encoder=fake_encoder, log=RunLog(path, HEADER, resume=True)
    )
    assert agent.calls == [] and [q for q, _ in judge.seen] == ["q1"]
    assert all(r.judge.ok for r in recs)
    assert recs[1].retrieval is not None  # metrics came from the saved row


def test_reload_equals_in_memory_with_judge_rows(tmp_path):
    path = tmp_path / "run.jsonl"
    mem = evaluate_items(
        ITEMS,
        FakeAgent(PLAN),
        FakeJudge(fail_on={"q3"}),
        encoder=fake_encoder,
        log=RunLog(path, HEADER),
    )
    reloaded = RunLog(path, HEADER, resume=True).records()
    assert summarize(mem, {}, True) == summarize(reloaded, {}, True)


def test_newline_terminated_unparsable_final_line_is_corrupt(tmp_path):
    path = _two_row_log(tmp_path)
    with path.open("a") as fh:
        fh.write("{broken\n")
    size = path.stat().st_size
    with pytest.raises(CorruptLogError, match="line 4"):
        RunLog(path, HEADER, resume=True)
    assert path.stat().st_size == size  # nothing was cut


def test_judge_attempts_capped_at_two_across_resumes(tmp_path):
    path = tmp_path / "run.jsonl"
    items = LOG_ITEMS[:2]
    failing = CountingJudge(fail_on={"q1"})

    def go(judge, agent=None):
        agent = agent or FakeAgent(LOG_PLAN)
        recs = evaluate_items(
            items,
            agent,
            judge,
            encoder=fake_encoder,
            log=RunLog(path, HEADER, resume=path.exists()),
        )
        return agent, recs

    agent, recs = go(failing)  # 1st failure
    assert recs[1].judge_attempts == 1 and agent.calls == ["q0", "q1"]
    agent, recs = go(failing)  # re-judged on resume, fails again -> permanent
    assert agent.calls == [] and [q for q, _ in failing.seen] == ["q0", "q1", "q1"]
    assert recs[1].judge_attempts == 2
    s = summarize(recs, {}, True)
    assert s["judge"]["n_failed"] == 1 and s["judge"]["n_judge_failed_permanent"] == 1
    assert s["judge"]["n_scored"] == 1 and s["judge"]["n_pending"] == 0
    healthy = CountingJudge()
    agent, recs = go(healthy)  # third resume: not judged, agent not called
    assert agent.calls == [] and healthy.seen == []
    assert RunLog(path, HEADER, resume=True).lookup(1, "q1").judge_attempts == 2
    assert summarize(recs, {}, True) == s


def test_first_failure_counts_one_and_is_not_permanent(tmp_path):
    path = tmp_path / "run.jsonl"
    recs = evaluate_items(
        LOG_ITEMS[:2],
        FakeAgent(LOG_PLAN),
        FakeJudge(fail_on={"q0"}),
        encoder=fake_encoder,
        log=RunLog(path, HEADER),
    )
    s = summarize(recs, {}, True)
    assert s["judge"]["n_failed"] == 1 and s["judge"]["n_judge_failed_permanent"] == 0


def test_pending_interrupted_row_is_not_an_attempt(tmp_path):
    path = tmp_path / "run.jsonl"

    class Boom:
        def evaluate(self, *a):
            raise RuntimeError("auth")

    with pytest.raises(RuntimeError):
        evaluate_items(
            LOG_ITEMS[:1],
            FakeAgent(LOG_PLAN),
            Boom(),
            encoder=fake_encoder,
            log=RunLog(path, HEADER),
        )
    assert RunLog(path, HEADER, resume=True).lookup(0, "q0").judge_attempts == 0


def test_old_log_without_attempts_field_loads(tmp_path):
    path = _two_row_log(tmp_path)
    lines = path.read_text().splitlines()
    rows = [json.loads(ln) for ln in lines[1:]]
    for r in rows:
        r.pop("judge_attempts", None)
        r.pop("judge_pending", None)
    path.write_text(lines[0] + "\n" + "".join(json.dumps(r) + "\n" for r in rows))
    assert RunLog(path, HEADER, resume=True).lookup(0, "q0").judge_attempts == 0
