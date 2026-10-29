"""Edge cases for token/cost accounting, MODEL_PRICES parsing and quiet HF logging."""

import json
import logging
import sys
import types

import pytest
from pydantic import ValidationError
from test_usage import JUDGE_LLM, LLMAgent, enc

from cleantech_rag import cli
from cleantech_rag.config import Settings
from cleantech_rag.evaluation import report
from cleantech_rag.evaluation.judge import JudgeResult
from cleantech_rag.evaluation.runner import (
    EvalItem,
    evaluate_items,
    merge_usage,
    price_for,
    summarize,
    summarize_usage,
)


def _rec(usage=None):
    """A scored record (summarize needs response metrics) with the given agent usage."""
    (rec,) = evaluate_items([EvalItem("q0", "r", ["a"])], _Plain(), None, encoder=enc)
    rec.agent_usage = usage or {}
    return rec


class _Plain:
    def run(self, q, thread_id=None):
        return types.SimpleNamespace(answer="a", retrieved_ids=[], tools_used=[])


def _u(i, o):
    return {"input_tokens": i, "output_tokens": o, "total_tokens": i + o}


def test_price_for_exact_match_and_order_independent():
    a = {"gpt-4o-mini": (1, 2), "gpt-4o": (3, 4)}
    b = dict(reversed(list(a.items())))
    for prices in (a, b):
        assert price_for("gpt-4o-mini", prices) == (1.0, 2.0)
        assert price_for("gpt-4o", prices) == (3.0, 4.0)
        assert price_for("gpt-4o-mini-2024", prices) == (1.0, 2.0)


def test_price_for_empty_table_and_non_prefix():
    assert price_for("gpt-4o", {}) is None
    assert price_for("my-gpt-4o", {"gpt-4o": (1, 2)}) is None  # prefix, not substring


def test_merge_usage_handles_empty_and_missing_keys():
    assert merge_usage() == {}
    assert merge_usage({}, {}) == {}
    merged = merge_usage({"m": {"input_tokens": 1}}, {"m": _u(2, 3)})
    assert merged == {"m": _u(2, 3) | {"input_tokens": 3, "total_tokens": 5}}


def test_summarize_usage_empty_records():
    out = summarize_usage([], {})
    assert out["tokens"] == {} and out["cost_usd"]["total"] == 0


def test_all_models_unpriced_total_zero_and_report_marks_na():
    rec = _rec({"weird-model": _u(10, 5)})
    s = summarize([rec], {}, False, prices={})
    assert s["usage"]["cost_usd"]["by_model"] == {"weird-model": None}
    assert s["usage"]["cost_usd"]["total"] == 0
    assert s["usage"]["prices_per_1m_tokens"] == {}
    md = report.render_markdown(s)
    assert "weird-model | 10 | 5 | 15 | n/a (no price)" in md


def test_summary_usage_is_json_serializable():
    rec = _rec({"gpt-4o": _u(10, 5), "x": _u(1, 1)})
    json.dumps(summarize([rec], {}, False)["usage"])


def test_prices_none_uses_defaults_but_empty_dict_means_no_prices():
    rec = _rec({"gpt-4o": _u(1_000_000, 0)})
    assert summarize([rec], {}, False)["usage"]["cost_usd"]["total"] == pytest.approx(2.5)
    assert summarize([rec], {}, False, prices={})["usage"]["cost_usd"]["total"] == 0


def test_usage_with_none_token_values_is_normalized():
    from cleantech_rag.evaluation.runner import _normalize_usage

    out = _normalize_usage({"m": {"input_tokens": None, "output_tokens": 3}})
    assert out == {"m": {"input_tokens": 0, "output_tokens": 3, "total_tokens": 0}}
    assert _normalize_usage(None) == {}


def test_usage_does_not_leak_between_questions():
    recs = evaluate_items(
        [EvalItem("q0", "r", ["a"]), EvalItem("q1", "r", ["b"])], LLMAgent(), None, encoder=enc
    )
    assert recs[0].agent_usage == recs[1].agent_usage  # not cumulative


def test_error_in_one_query_does_not_affect_next_usage():
    class Flaky(LLMAgent):
        def run(self, question, thread_id=None):
            self.fail = question == "q0"
            return super().run(question, thread_id)

    recs = evaluate_items(
        [EvalItem("q0", "r", ["a"]), EvalItem("q1", "r", ["b"])], Flaky(), None, encoder=enc
    )
    assert recs[0].error and not recs[1].error
    assert recs[1].agent_usage["gpt-4o-mini-2024-07-18"]["total_tokens"] == 2400


def test_failed_nonfatal_judge_tokens_still_recorded():
    class BadJudge:
        def evaluate(self, q, p, r):
            JUDGE_LLM.invoke(q)
            return JudgeResult(error="parse failure")

    recs = evaluate_items([EvalItem("q0", "r", ["a"])], LLMAgent(), BadJudge(), encoder=enc)
    assert recs[0].judge.ok is False
    assert recs[0].judge_usage["gpt-4o-2024-08-06"]["total_tokens"] == 550


def test_fatal_judge_error_persists_agent_and_judge_tokens(tmp_path):
    """A judge dying fatally still leaves the agent and judge tokens in the log row."""
    from cleantech_rag.evaluation.runner import RunLog

    class Fatal:
        def evaluate(self, q, p, r):
            JUDGE_LLM.invoke(q)
            raise KeyboardInterrupt

    path = tmp_path / "r.jsonl"
    with pytest.raises(KeyboardInterrupt):
        evaluate_items(
            [EvalItem("q0", "r", ["a"])],
            LLMAgent(),
            Fatal(),
            encoder=enc,
            log=RunLog(path, {"x": 1}),
        )
    rec = RunLog(path, {"x": 1}, resume=True).records()[0]
    assert rec.agent_usage["gpt-4o-mini-2024-07-18"]["total_tokens"] == 2400
    assert rec.judge_usage["gpt-4o-2024-08-06"]["total_tokens"] == 550


def test_report_ignores_empty_usage_tokens():
    s = summarize([_rec()], {}, False)
    assert "Token usage" not in report.render_markdown(s)


def test_judge_called_only_for_ok_rows_no_judge_usage_without_judge():
    recs = evaluate_items([EvalItem("q0", "r", ["a"])], LLMAgent(), None, encoder=enc)
    assert recs[0].judge_usage == {}


# ---- config ----


def test_model_prices_invalid_json_rejected(monkeypatch):
    monkeypatch.setenv("MODEL_PRICES", "not json")
    with pytest.raises(Exception):  # noqa: B017 - pydantic-settings SettingsError
        Settings(_env_file=None)


def test_model_prices_wrong_shape_rejected(monkeypatch):
    monkeypatch.setenv("MODEL_PRICES", '{"gpt-4o": [1]}')
    with pytest.raises(ValidationError):
        Settings(_env_file=None)
    monkeypatch.setenv("MODEL_PRICES", '{"gpt-4o": "cheap"}')
    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_model_prices_env_replaces_defaults_and_adds_models(monkeypatch):
    monkeypatch.setenv("MODEL_PRICES", '{"other": [0.1, 0.2]}')
    prices = Settings(_env_file=None).model_prices
    assert prices == {"other": (0.1, 0.2)}  # whole table replaced, defaults not merged


def test_model_prices_defaults_not_shared_between_instances():
    a = Settings(_env_file=None)
    a.model_prices["zzz"] = (1.0, 1.0)
    assert "zzz" not in Settings(_env_file=None).model_prices


# ---- quiet logging ----


@pytest.fixture
def restore_env_and_logs(monkeypatch):
    names = [*cli.NOISY_LOGGERS]
    saved = {n: logging.getLogger(n).level for n in names}
    monkeypatch.delenv("HF_HUB_DISABLE_PROGRESS_BARS", raising=False)
    monkeypatch.delenv("TRANSFORMERS_VERBOSITY", raising=False)
    yield
    for n, lvl in saved.items():
        logging.getLogger(n).setLevel(lvl)


def test_quiet_loaded_hf_noop_when_not_imported(monkeypatch):
    monkeypatch.delitem(sys.modules, "huggingface_hub.utils", raising=False)
    monkeypatch.delitem(sys.modules, "transformers.utils.logging", raising=False)
    cli.quiet_loaded_hf()  # must not import them or raise
    assert "transformers.utils.logging" not in sys.modules


def test_quiet_loaded_hf_calls_loaded_modules(monkeypatch):
    calls = []
    hub = types.SimpleNamespace(disable_progress_bars=lambda: calls.append("hub"))
    tf = types.SimpleNamespace(
        set_verbosity_error=lambda: calls.append("tf_err"),
        disable_progress_bar=lambda: calls.append("tf_bar"),
    )
    monkeypatch.setitem(sys.modules, "huggingface_hub.utils", hub)
    monkeypatch.setitem(sys.modules, "transformers.utils.logging", tf)
    cli.quiet_loaded_hf()
    assert calls == ["hub", "tf_err", "tf_bar"]


def test_quiet_loaded_hf_tolerates_modules_missing_functions(monkeypatch):
    monkeypatch.setitem(sys.modules, "huggingface_hub.utils", types.SimpleNamespace())
    monkeypatch.setitem(sys.modules, "transformers.utils.logging", types.SimpleNamespace())
    cli.quiet_loaded_hf()


def test_configure_logging_quiet_sets_levels_env_and_calls_quiet(monkeypatch, restore_env_and_logs):
    calls = []
    monkeypatch.setattr(cli, "quiet_loaded_hf", lambda: calls.append(1))
    cli.configure_logging(verbose=False)
    assert logging.getLogger("httpx").level == logging.WARNING
    assert logging.getLogger("huggingface_hub").level == logging.WARNING
    import os

    assert os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] == "1"
    assert os.environ["TRANSFORMERS_VERBOSITY"] == "error"
    assert calls == [1]
    monkeypatch.delenv("HF_HUB_DISABLE_PROGRESS_BARS")
    monkeypatch.delenv("TRANSFORMERS_VERBOSITY")


def test_configure_logging_verbose_leaves_hf_alone(monkeypatch, restore_env_and_logs):
    calls = []
    monkeypatch.setattr(cli, "quiet_loaded_hf", lambda: calls.append(1))
    cli.configure_logging(verbose=True)
    assert logging.getLogger("httpx").level == logging.INFO
    import os

    assert "HF_HUB_DISABLE_PROGRESS_BARS" not in os.environ
    assert calls == []


def test_configure_logging_respects_user_preset_env(monkeypatch, restore_env_and_logs):
    monkeypatch.setenv("HF_HUB_DISABLE_PROGRESS_BARS", "0")
    monkeypatch.setattr(cli, "quiet_loaded_hf", lambda: None)
    cli.configure_logging(verbose=False)
    import os

    assert os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] == "0"
    monkeypatch.delenv("TRANSFORMERS_VERBOSITY", raising=False)


def test_encoder_disables_progress_bar(monkeypatch):
    from cleantech_rag.evaluation import metrics

    seen = {}

    class FakeST:
        def __init__(self, name):
            pass

        def encode(self, texts, **kw):
            seen.update(kw)
            return [[1, 2] for _ in texts]

    monkeypatch.setitem(
        sys.modules, "sentence_transformers", types.SimpleNamespace(SentenceTransformer=FakeST)
    )
    assert metrics.default_encoder()(["a"]) == [[1.0, 2.0]]
    assert seen == {"show_progress_bar": False}
