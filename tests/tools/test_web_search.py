import pytest
import requests

from cleantech_rag.config import Settings
from cleantech_rag.tools import web_search
from cleantech_rag.tools.web_search import DISABLED_MESSAGE, brave_search, make_web_search_tool


class FakeResp:
    def __init__(self, status=200, payload=None, bad_json=False):
        self.status_code = status
        self._payload = payload
        self._bad = bad_json

    def json(self):
        if self._bad:
            raise ValueError("bad")
        return self._payload


def _settings(key="k"):
    return Settings(brave_api_key=key, _env_file=None)


def _patch_get(monkeypatch, fn):
    monkeypatch.setattr(web_search.requests, "get", fn)


def test_success(monkeypatch):
    seen = {}

    def get(url, headers, params, timeout):
        seen.update(headers=headers, params=params, timeout=timeout)
        return FakeResp(
            payload={"web": {"results": [{"title": "T", "url": "http://u", "description": "D"}]}}
        )

    _patch_get(monkeypatch, get)
    out = brave_search("solar", _settings())
    assert "Title: T" in out and "http://u" in out and "D" in out
    assert seen["headers"]["X-Subscription-Token"] == "k"
    assert seen["params"]["q"] == "solar"
    assert seen["timeout"] == 10.0


def test_non_200(monkeypatch):
    _patch_get(monkeypatch, lambda *a, **k: FakeResp(status=429))
    assert "HTTP 429" in brave_search("q", _settings())


def test_timeout(monkeypatch):
    def get(*a, **k):
        raise requests.Timeout()

    _patch_get(monkeypatch, get)
    assert "timed out" in brave_search("q", _settings())


def test_connection_error(monkeypatch):
    def get(*a, **k):
        raise requests.ConnectionError("secret-host")

    _patch_get(monkeypatch, get)
    out = brave_search("q", _settings())
    assert "failed" in out and "secret-host" not in out


def test_empty_results(monkeypatch):
    _patch_get(monkeypatch, lambda *a, **k: FakeResp(payload={"web": {"results": []}}))
    assert brave_search("q", _settings()) == "No web results found."
    _patch_get(monkeypatch, lambda *a, **k: FakeResp(payload={}))
    assert brave_search("q", _settings()) == "No web results found."


def test_invalid_json(monkeypatch):
    _patch_get(monkeypatch, lambda *a, **k: FakeResp(bad_json=True))
    assert "invalid response" in brave_search("q", _settings())


def test_missing_key_makes_no_request(monkeypatch):
    def get(*a, **k):
        raise AssertionError("must not call network")

    _patch_get(monkeypatch, get)
    assert brave_search("q", _settings(key=None)) == DISABLED_MESSAGE


def test_tool_returns_no_article_ids():
    tool = make_web_search_tool(_settings(key=None))
    msg = tool.invoke(
        {"type": "tool_call", "name": "web_search", "args": {"query": "x"}, "id": "1"}
    )
    assert msg.content == DISABLED_MESSAGE
    assert msg.artifact["article_ids"] == []


@pytest.mark.parametrize("key", [""])
def test_empty_key_disabled(key):
    assert brave_search("q", _settings(key=key)) == DISABLED_MESSAGE


def test_missing_fields_use_defaults(monkeypatch):
    _patch_get(monkeypatch, lambda *a, **k: FakeResp(payload={"web": {"results": [{}]}}))
    out = brave_search("q", _settings())
    assert "Title: N/A" in out and "URL: N/A" in out


def test_results_truncated_to_count(monkeypatch):
    items = [{"title": f"T{i}", "url": "u", "description": "d"} for i in range(30)]
    _patch_get(monkeypatch, lambda *a, **k: FakeResp(payload={"web": {"results": items}}))
    s = Settings(brave_api_key="k", brave_result_count=3, _env_file=None)
    out = brave_search("q", s)
    assert "[3]" in out and "[4]" not in out


@pytest.mark.xfail(strict=True, reason="BUG: non-dict JSON / null web / null results raise")
@pytest.mark.parametrize(
    "payload", [[], "str", None, {"web": None}, {"web": []}, {"web": {"results": None}}]
)
def test_malformed_json_shapes_do_not_raise(monkeypatch, payload):
    _patch_get(monkeypatch, lambda *a, **k: FakeResp(payload=payload))
    out = brave_search("q", _settings())
    assert isinstance(out, str)


@pytest.mark.xfail(strict=True, reason="BUG: non-dict result items raise")
def test_non_dict_result_items_do_not_raise(monkeypatch):
    _patch_get(monkeypatch, lambda *a, **k: FakeResp(payload={"web": {"results": ["x", None]}}))
    assert isinstance(brave_search("q", _settings()), str)


@pytest.mark.xfail(strict=True, reason="BUG: null description rendered as 'None'")
def test_null_description_not_rendered_as_none(monkeypatch):
    payload = {"web": {"results": [{"title": "T", "url": "u", "description": None}]}}
    _patch_get(monkeypatch, lambda *a, **k: FakeResp(payload=payload))
    assert "None" not in brave_search("q", _settings())


@pytest.mark.parametrize("status", [401, 500, 503])
def test_other_statuses(monkeypatch, status):
    _patch_get(monkeypatch, lambda *a, **k: FakeResp(status=status))
    assert f"HTTP {status}" in brave_search("q", _settings())


def test_api_key_never_leaked_in_errors(monkeypatch):
    def get(*a, **k):
        raise requests.RequestException("token=SECRETKEY")

    _patch_get(monkeypatch, get)
    assert "SECRETKEY" not in brave_search("q", _settings(key="SECRETKEY"))


def test_tool_invoke_success_artifact(monkeypatch):
    payload = {"web": {"results": [{"title": "T", "url": "u", "description": "d"}]}}
    _patch_get(monkeypatch, lambda *a, **k: FakeResp(payload=payload))
    msg = make_web_search_tool(_settings()).invoke(
        {"type": "tool_call", "name": "web_search", "args": {"query": "x"}, "id": "9"}
    )
    assert "Title: T" in msg.content
    assert msg.artifact == {"article_ids": [], "documents": []}
    assert msg.tool_call_id == "9"


def test_tool_timeout_does_not_raise(monkeypatch):
    def get(*a, **k):
        raise requests.Timeout()

    _patch_get(monkeypatch, get)
    msg = make_web_search_tool(_settings()).invoke(
        {"type": "tool_call", "name": "web_search", "args": {"query": "x"}, "id": "1"}
    )
    assert "timed out" in msg.content
