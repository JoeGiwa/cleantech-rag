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
