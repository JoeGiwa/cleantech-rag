"""Token/cost accounting in the eval runner, with fake chat models carrying usage_metadata."""

import json
from types import SimpleNamespace

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from cleantech_rag.evaluation import report
from cleantech_rag.evaluation.judge import CRITERIA, JudgeResult
from cleantech_rag.evaluation.runner import (
    RESPONSE_METRICS,
    EvalItem,
    QueryRecord,
    RunLog,
    evaluate_items,
    price_for,
    summarize,
    write_outputs,
)


class UsageModel(BaseChatModel):
    """Chat model reporting fixed token usage under a (dated) model name."""

    name: str
    tokens_in: int
    tokens_out: int

    @property
    def _llm_type(self) -> str:
        return "fake-usage"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        msg = AIMessage(
            content="ok",
            usage_metadata={
                "input_tokens": self.tokens_in,
                "output_tokens": self.tokens_out,
                "total_tokens": self.tokens_in + self.tokens_out,
            },
            response_metadata={"model_name": self.name},
        )
        return ChatResult(generations=[ChatGeneration(message=msg)])


AGENT_LLM = UsageModel(name="gpt-4o-mini-2024-07-18", tokens_in=1000, tokens_out=200)
JUDGE_LLM = UsageModel(name="gpt-4o-2024-08-06", tokens_in=500, tokens_out=50)


class LLMAgent:
    def __init__(self, fail=False):
        self.fail = fail

    def run(self, question, thread_id=None):
        AGENT_LLM.invoke(question)
        AGENT_LLM.invoke(question)  # two calls per question
        if self.fail:
            raise RuntimeError("boom")
        return SimpleNamespace(answer="a", retrieved_ids=[], tools_used=[])


class LLMJudgeFake:
    def evaluate(self, query, prediction, reference):
        JUDGE_LLM.invoke(query)
        return JudgeResult(scores=dict.fromkeys(CRITERIA, 4), overall=4.0)


def enc(texts):
    return [[1.0, 0.0] for _ in texts]


ITEMS = [EvalItem("q0", "r", ["a"]), EvalItem("q1", "r", ["b"])]


def test_usage_recorded_per_question_and_summed():
    recs = evaluate_items(ITEMS, LLMAgent(), LLMJudgeFake(), encoder=enc)
    assert recs[0].agent_usage == {
        "gpt-4o-mini-2024-07-18": {"input_tokens": 2000, "output_tokens": 400, "total_tokens": 2400}
    }
    assert recs[0].judge_usage["gpt-4o-2024-08-06"]["total_tokens"] == 550
    s = summarize(recs, {}, judge_enabled=True)
    u = s["usage"]
    assert u["tokens"]["gpt-4o-mini-2024-07-18"]["total_tokens"] == 4800
    assert u["tokens"]["gpt-4o-2024-08-06"]["input_tokens"] == 1000
    assert u["tokens_by_role"]["judge"]["gpt-4o-2024-08-06"]["output_tokens"] == 100
    mini = (4000 * 0.15 + 800 * 0.60) / 1e6
    full = (1000 * 2.50 + 100 * 10.00) / 1e6
    assert u["cost_usd"]["by_model"]["gpt-4o-mini-2024-07-18"] == pytest.approx(mini)
    assert u["cost_usd"]["total"] == pytest.approx(mini + full)


def test_errored_query_still_counts_tokens():
    recs = evaluate_items(ITEMS[:1], LLMAgent(fail=True), None, encoder=enc)
    assert recs[0].error
    assert recs[0].agent_usage["gpt-4o-mini-2024-07-18"]["total_tokens"] == 2400


def test_agents_without_llm_calls_have_empty_usage():
    class Plain:
        def run(self, q, thread_id=None):
            return SimpleNamespace(answer="a", retrieved_ids=[], tools_used=[])

    recs = evaluate_items(ITEMS, Plain(), None, encoder=enc)
    assert recs[0].agent_usage == {}
    assert summarize(recs, {}, False)["usage"]["cost_usd"]["total"] == 0


def test_custom_prices_and_unpriced_model():
    recs = evaluate_items(ITEMS[:1], LLMAgent(), LLMJudgeFake(), encoder=enc)
    prices = {"gpt-4o-mini": (1.0, 2.0)}  # gpt-4o judge has no price
    u = summarize(recs, {}, True, prices=prices)["usage"]
    assert u["cost_usd"]["by_model"]["gpt-4o-2024-08-06"] is None
    assert u["cost_usd"]["total"] == pytest.approx((2000 * 1.0 + 400 * 2.0) / 1e6)


def test_price_for_prefers_longest_prefix():
    prices = {"gpt-4o": (2.5, 10.0), "gpt-4o-mini": (0.15, 0.6)}
    assert price_for("gpt-4o-mini-2024-07-18", prices) == (0.15, 0.6)
    assert price_for("gpt-4o-2024-08-06", prices) == (2.5, 10.0)
    assert price_for("claude", prices) is None


def test_log_roundtrip_and_resume_keeps_usage(tmp_path):
    path = tmp_path / "run.jsonl"
    evaluate_items(ITEMS, LLMAgent(), None, encoder=enc, log=RunLog(path, {"x": 1}))
    log = RunLog(path, {"x": 1}, resume=True)
    assert log.records()[0].agent_usage["gpt-4o-mini-2024-07-18"]["total_tokens"] == 2400


def test_rejudge_adds_to_judge_usage(tmp_path):
    path = tmp_path / "run.jsonl"
    evaluate_items(ITEMS[:1], LLMAgent(), None, encoder=enc, log=RunLog(path, {"x": 1}))
    # resume the no-judge row with a judge: agent tokens kept, judge tokens added
    recs = evaluate_items(
        ITEMS[:1], LLMAgent(), LLMJudgeFake(), encoder=enc, log=RunLog(path, {"x": 1}, resume=True)
    )
    assert recs[0].agent_usage["gpt-4o-mini-2024-07-18"]["total_tokens"] == 2400
    assert recs[0].judge_usage["gpt-4o-2024-08-06"]["total_tokens"] == 550


def test_old_log_rows_without_usage_still_load(tmp_path):
    path = tmp_path / "old.jsonl"
    row = {
        "type": "row",
        "index": 0,
        "question": "q0",
        "answer": "a",
        "response": dict.fromkeys(RESPONSE_METRICS, 1.0),
    }
    path.write_text(
        json.dumps({"type": "header", "config": {"x": 1}}) + "\n" + json.dumps(row) + "\n"
    )
    rec = RunLog(path, {"x": 1}, resume=True).records()[0]
    assert rec.agent_usage == {} and rec.judge_usage == {}
    assert summarize([rec], {}, False)["usage"]["tokens"] == {}


def test_csv_has_token_columns(tmp_path):
    recs = evaluate_items(ITEMS[:1], LLMAgent(), LLMJudgeFake(), encoder=enc)
    s = summarize(recs, {}, True)
    _, csv_path = write_outputs(recs, s, tmp_path, __import__("datetime").datetime(2026, 1, 1))
    header, row = csv_path.read_text().splitlines()[:2]
    cols = header.split(",")
    assert row.split(",")[cols.index("agent_tokens")] == "2400"
    assert row.split(",")[cols.index("judge_tokens")] == "550"


def test_report_renders_cost_table_and_old_summary(tmp_path):
    recs = evaluate_items(ITEMS, LLMAgent(), LLMJudgeFake(), encoder=enc)
    md = report.render_markdown(summarize(recs, {}, True))
    assert "## Token usage and estimated cost" in md
    assert "gpt-4o-mini-2024-07-18 | 4000 | 800 | 4800 | $0.0011" in md
    assert "Estimated total: $" in md
    old = summarize(recs, {}, True)
    del old["usage"]
    assert "Token usage" not in report.render_markdown(old)


def test_queryrecord_default_usage_empty():
    assert QueryRecord(index=0, question="q").agent_usage == {}


def test_settings_model_prices_default_and_env(monkeypatch):
    from cleantech_rag.config import Settings

    assert Settings(_env_file=None).model_prices["gpt-4o"] == (2.5, 10.0)
    monkeypatch.setenv("MODEL_PRICES", '{"gpt-4o": [1, 2]}')
    assert Settings(_env_file=None).model_prices == {"gpt-4o": (1.0, 2.0)}


def test_report_no_total_line_when_nothing_priced():
    recs = evaluate_items(ITEMS[:1], LLMAgent(), None, encoder=enc)
    md = report.render_markdown(summarize(recs, {}, False, prices={}))
    assert "n/a (no price)" in md and "Estimated total" not in md
