import json
import socket

import pytest

from cleantech_rag.evaluation import judge as j


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def blocked(*a, **k):
        raise AssertionError("network access in test")

    monkeypatch.setattr(socket.socket, "connect", blocked)


class _Msg:
    def __init__(self, content):
        self.content = content


class FakeModel:
    def __init__(self, reply=None, exc=None):
        self.reply, self.exc, self.prompts = reply, exc, []

    def invoke(self, prompt):
        self.prompts.append(prompt)
        if self.exc:
            raise self.exc
        return _Msg(self.reply)


def _reply(**over):
    d = {"relevance": 5, "accuracy": 4, "completeness": 3, "clarity": 4}
    d.update(over)
    return json.dumps(d)


def test_valid_response():
    model = FakeModel(_reply())
    r = j.LLMJudge(model).evaluate("q?", "pred", "ref")
    assert r.ok and r.error is None
    assert r.scores == {"relevance": 5, "accuracy": 4, "completeness": 3, "clarity": 4}
    assert r.overall == pytest.approx(4.0)
    assert "q?" in model.prompts[0] and "pred" in model.prompts[0] and "ref" in model.prompts[0]


def test_code_fenced_json_accepted():
    r = j.LLMJudge(FakeModel("```json\n" + _reply() + "\n```")).evaluate("q", "p", "r")
    assert r.ok


def test_malformed_json_is_none_not_zero():
    r = j.LLMJudge(FakeModel("not json at all")).evaluate("q", "p", "r")
    assert not r.ok
    assert "malformed JSON" in r.error
    assert all(v is None for v in r.scores.values())
    assert r.overall is None


@pytest.mark.parametrize("bad", [0, 6, -1, 3.5, "5", True, None])
def test_out_of_range_or_invalid_score(bad):
    r = j.LLMJudge(FakeModel(_reply(accuracy=bad))).evaluate("q", "p", "r")
    assert not r.ok and "accuracy" in r.error
    assert all(v is None for v in r.scores.values())


def test_missing_criterion():
    data = json.loads(_reply())
    del data["clarity"]
    r = j.LLMJudge(FakeModel(json.dumps(data))).evaluate("q", "p", "r")
    assert not r.ok and "clarity" in r.error


def test_model_exception_recorded():
    r = j.LLMJudge(FakeModel(exc=RuntimeError("boom"))).evaluate("q", "p", "r")
    assert not r.ok and "boom" in r.error and r.overall is None


def test_aggregate_excludes_failures_and_reports_n_failed():
    good1 = j.LLMJudge(FakeModel(_reply())).evaluate("q", "p", "r")
    good2 = j.LLMJudge(FakeModel(_reply(relevance=1, accuracy=2))).evaluate("q", "p", "r")
    bad = j.LLMJudge(FakeModel("nope")).evaluate("q", "p", "r")
    agg = j.aggregate_judge_results([good1, bad, good2])
    assert agg["n_failed"] == 1 and agg["n_scored"] == 2 and agg["n_total"] == 3
    assert agg["relevance"] == pytest.approx(3.0)  # (5 + 1) / 2, failure not counted as 0
    assert agg["overall"] == pytest.approx((4.0 + 2.5) / 2)


def test_aggregate_all_failed():
    bad = j.JudgeResult(error="x")
    agg = j.aggregate_judge_results([bad, bad])
    assert agg["n_failed"] == 2 and agg["overall"] is None and agg["relevance"] is None


def test_default_judge_model_from_settings():
    from cleantech_rag.config import Settings

    assert Settings(_env_file=None).judge_model


@pytest.mark.parametrize("raw", ["NaN", "Infinity", "-Infinity"])
def test_nonfinite_scores_are_failures(raw):
    text = f'{{"relevance": {raw}, "accuracy": 4, "completeness": 3, "clarity": 4}}'
    r = j.LLMJudge(FakeModel(text)).evaluate("q", "p", "r")
    assert not r.ok and r.overall is None
    assert all(v is None for v in r.scores.values())


@pytest.mark.parametrize("val", [1, 5, 1.0, 5.0])
def test_boundary_scores_accepted(val):
    r = j.LLMJudge(FakeModel(_reply(relevance=val))).evaluate("q", "p", "r")
    assert r.ok and r.scores["relevance"] == int(val)
    assert isinstance(r.scores["relevance"], int)


@pytest.mark.parametrize("text", ["[1, 2, 3, 4]", '"hi"', "42", "null", "", "   "])
def test_non_object_or_empty_json_fails(text):
    r = j.LLMJudge(FakeModel(text)).evaluate("q", "p", "r")
    assert not r.ok and r.overall is None


def test_plain_string_model_response_supported():
    class StrModel:
        def invoke(self, prompt):
            return _reply()

    assert j.LLMJudge(StrModel()).evaluate("q", "p", "r").ok


def test_non_text_content_fails():
    r = j.LLMJudge(FakeModel([{"type": "text", "text": "x"}])).evaluate("q", "p", "r")
    assert not r.ok and "not text" in r.error


def test_none_content_fails():
    r = j.LLMJudge(FakeModel(None)).evaluate("q", "p", "r")
    assert not r.ok


def test_braces_in_inputs_do_not_break_prompt():
    model = FakeModel(_reply())
    r = j.LLMJudge(model).evaluate("what is {x}?", "pred {} }{", "ref {y}")
    assert r.ok and "{x}" in model.prompts[0]


def test_extra_keys_ignored():
    r = j.LLMJudge(FakeModel(_reply(bonus=9))).evaluate("q", "p", "r")
    assert r.ok and set(r.scores) == set(j.CRITERIA)


def test_failed_result_never_contains_zero():
    r = j.LLMJudge(FakeModel(_reply(clarity=0))).evaluate("q", "p", "r")
    assert 0 not in r.scores.values()
    assert all(v is None for v in r.scores.values())


def test_aggregate_empty():
    agg = j.aggregate_judge_results([])
    assert agg["n_total"] == 0 and agg["n_failed"] == 0 and agg["overall"] is None


def test_aggregate_accepts_generator_and_all_ok():
    good = j.LLMJudge(FakeModel(_reply())).evaluate("q", "p", "r")
    agg = j.aggregate_judge_results(x for x in [good, good])
    assert agg["n_failed"] == 0 and agg["clarity"] == pytest.approx(4.0)


def test_build_judge_model_uses_settings_model_name(monkeypatch):
    import sys
    import types

    captured = {}

    class FakeChat:
        def __init__(self, **kw):
            captured.update(kw)

    monkeypatch.setitem(sys.modules, "langchain_openai", types.SimpleNamespace(ChatOpenAI=FakeChat))
    settings = types.SimpleNamespace(judge_model="my-judge", openai_api_key="k")
    model = j.LLMJudge(settings=settings).model
    assert isinstance(model, FakeChat)
    assert captured["model"] == "my-judge"
