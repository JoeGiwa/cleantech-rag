import socket

import httpx
import openai
import pytest

from cleantech_rag.evaluation import judge as j


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def blocked(*a, **k):
        raise AssertionError("network access in test")

    monkeypatch.setattr(socket.socket, "connect", blocked)


class _FakeStructured:
    """Runnable-like returned by ``with_structured_output``."""

    def __init__(self, owner, schema):
        self.owner, self.schema = owner, schema

    def invoke(self, prompt):
        self.owner.prompts.append(prompt)
        self.owner.calls += 1
        if self.owner.errors:
            raise self.owner.errors.pop(0)
        if self.owner.exc:
            raise self.owner.exc
        if self.owner.payload is not None:
            # like real LangChain: raises pydantic.ValidationError on bad/missing/out-of-range
            return self.schema.model_validate(self.owner.payload)
        return self.owner.obj


class FakeModel:
    """Fake chat model: validate ``payload`` via the schema, return ``obj``, or raise ``exc``."""

    def __init__(self, payload=None, obj=None, exc=None, errors=()):
        self.payload, self.obj, self.exc = payload, obj, exc
        self.errors = list(errors)  # raised one per call, then normal behaviour
        self.calls = 0
        self.prompts, self.schemas = [], []

    def with_structured_output(self, schema):
        self.schemas.append(schema)
        return _FakeStructured(self, schema)


def _reply(**over):
    d = {"relevance": 5, "accuracy": 4, "completeness": 3, "clarity": 4}
    d.update(over)
    return d


def _judge(payload=None, **kw):
    return j.LLMJudge(FakeModel(payload, **kw)).evaluate("q", "p", "r")


def test_valid_response():
    model = FakeModel(_reply())
    r = j.LLMJudge(model).evaluate("q?", "pred", "ref")
    assert r.ok and r.error is None
    assert r.scores == {"relevance": 5, "accuracy": 4, "completeness": 3, "clarity": 4}
    assert r.overall == pytest.approx(4.0)
    assert "q?" in model.prompts[0] and "pred" in model.prompts[0] and "ref" in model.prompts[0]
    assert model.schemas == [j.JudgeScores]


def test_prompt_has_no_raw_json_instructions():
    assert "JSON" not in j.JUDGE_PROMPT


def test_returned_schema_object_accepted():
    obj = j.JudgeScores(relevance=1, accuracy=2, completeness=3, clarity=4)
    r = _judge(obj=obj)
    assert r.ok and r.overall == pytest.approx(2.5)


@pytest.mark.parametrize("bad", [0, 6, -1, 3.5, "5", True, None])
def test_out_of_range_or_invalid_score(bad):
    r = _judge(_reply(accuracy=bad))
    assert not r.ok and "accuracy" in r.error
    assert all(v is None for v in r.scores.values())
    assert r.overall is None


def test_missing_criterion():
    data = _reply()
    del data["clarity"]
    r = _judge(data)
    assert not r.ok and "clarity" in r.error


def test_malformed_output_is_none_not_zero():
    r = _judge("not a dict")
    assert not r.ok and r.overall is None
    assert all(v is None for v in r.scores.values())


@pytest.mark.parametrize("obj", [None, "text", {"relevance": 5}, 42])
def test_unexpected_return_fails(obj):
    r = _judge(obj=obj)
    assert not r.ok and r.overall is None and "unexpected" in r.error
    assert all(v is None for v in r.scores.values())


def test_model_exception_recorded():
    r = _judge(exc=RuntimeError("boom"))
    assert not r.ok and "boom" in r.error and r.overall is None


def test_with_structured_output_exception_recorded():
    class Broken:
        def with_structured_output(self, schema):
            raise NotImplementedError("no structured output")

    r = j.LLMJudge(Broken()).evaluate("q", "p", "r")
    assert not r.ok and "no structured output" in r.error


def test_aggregate_excludes_failures_and_reports_n_failed():
    good1 = _judge(_reply())
    good2 = _judge(_reply(relevance=1, accuracy=2))
    bad = _judge(_reply(clarity=9))
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

    assert Settings(_env_file=None).judge_model == "gpt-4o"


@pytest.mark.parametrize("val", [1, 5])
def test_boundary_scores_accepted(val):
    r = _judge(_reply(relevance=val))
    assert r.ok and r.scores["relevance"] == int(val)
    assert isinstance(r.scores["relevance"], int)


def test_braces_in_inputs_do_not_break_prompt():
    model = FakeModel(_reply())
    r = j.LLMJudge(model).evaluate("what is {x}?", "pred {} }{", "ref {y}")
    assert r.ok and "{x}" in model.prompts[0]


def test_extra_keys_ignored():
    r = _judge(_reply(bonus=9))
    assert r.ok and set(r.scores) == set(j.CRITERIA)


def test_failed_result_never_contains_zero():
    r = _judge(_reply(clarity=0))
    assert 0 not in r.scores.values()
    assert all(v is None for v in r.scores.values())


def test_aggregate_empty():
    agg = j.aggregate_judge_results([])
    assert agg["n_total"] == 0 and agg["n_failed"] == 0 and agg["overall"] is None


def test_aggregate_accepts_generator_and_all_ok():
    good = _judge(_reply())
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


_REQ = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")


def _status_error(cls, status):
    return cls("err", response=httpx.Response(status, request=_REQ), body=None)


def _rate_limit():
    return _status_error(openai.RateLimitError, 429)


def _timeout():
    return openai.APITimeoutError(request=_REQ)


def _retry_judge(model, **kw):
    delays = []
    judge = j.LLMJudge(model, base_delay=kw.pop("base_delay", 1.0), sleep=delays.append, **kw)
    return judge, delays


@pytest.mark.parametrize(
    "exc",
    [
        _status_error(openai.AuthenticationError, 401),
        _status_error(openai.PermissionDeniedError, 403),
        _status_error(openai.NotFoundError, 404),
    ],
)
def test_fatal_errors_reraise_without_retry(exc):
    model = FakeModel(_reply(), errors=[exc])
    judge, delays = _retry_judge(model)
    with pytest.raises(type(exc)):
        judge.evaluate("q", "p", "r")
    assert model.calls == 1 and delays == []


@pytest.mark.parametrize("make", [_rate_limit, _timeout, lambda: TimeoutError("slow")])
def test_transient_error_then_success(make):
    model = FakeModel(_reply(), errors=[make()])
    judge, delays = _retry_judge(model)
    r = judge.evaluate("q", "p", "r")
    assert r.ok and r.overall == pytest.approx(4.0)
    assert model.calls == 2 and delays == [1.0]


def test_two_transient_errors_then_success_backoff():
    model = FakeModel(_reply(), errors=[_rate_limit(), _timeout()])
    judge, delays = _retry_judge(model, base_delay=0.5)
    assert judge.evaluate("q", "p", "r").ok
    assert model.calls == 3 and delays == [0.5, 1.0]


@pytest.mark.parametrize("make", [_rate_limit, _timeout])
def test_transient_error_on_all_attempts_gives_none(make):
    model = FakeModel(_reply(), errors=[make(), make(), make()])
    judge, delays = _retry_judge(model, base_delay=2.0)
    r = judge.evaluate("q", "p", "r")
    assert not r.ok and r.overall is None and r.error
    assert all(v is None for v in r.scores.values())
    assert model.calls == 3 and delays == [2.0, 4.0]


def test_generic_error_not_retried():
    model = FakeModel(_reply(), errors=[RuntimeError("boom")])
    judge, delays = _retry_judge(model)
    r = judge.evaluate("q", "p", "r")
    assert not r.ok and "boom" in r.error
    assert model.calls == 1 and delays == []


def test_validation_error_not_retried():
    model = FakeModel(_reply(accuracy=9))
    judge, delays = _retry_judge(model)
    r = judge.evaluate("q", "p", "r")
    assert not r.ok and "accuracy" in r.error
    assert model.calls == 1 and delays == []
