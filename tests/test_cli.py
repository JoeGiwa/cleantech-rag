from types import SimpleNamespace

import pandas as pd
from pydantic import SecretStr
from typer.testing import CliRunner

from cleantech_rag import cli
from cleantech_rag.config import Settings

runner = CliRunner()


def _settings(tmp_path, **kw):
    return Settings(
        _env_file=None,
        raw_data_dir=tmp_path,
        chroma_dir=tmp_path / "chroma",
        openai_api_key=SecretStr("sk-test"),
        **kw,
    )


def _corpus(tmp_path, n=10):
    pd.DataFrame(
        {
            "Unnamed: 0": list(range(n)),
            "title": [f"t{i}" for i in range(n)],
            "date": ["2024-01-01"] * n,
            "author": ["a"] * n,
            "content": [f"body {i}" for i in range(n)],
            "domain": ["d"] * n,
            "url": [f"http://x/{i}" for i in range(n)],
        }
    ).to_csv(tmp_path / cli.CORPUS_FILE, index=False)


def test_ask_prints_answer(tmp_path, monkeypatch):
    s = _settings(tmp_path)
    calls = {}

    class FakeAgent:
        def run(self, q, thread_id=None):
            calls["args"] = (q, thread_id)
            return SimpleNamespace(answer="the answer")

    monkeypatch.setattr(cli, "get_settings", lambda: s)
    monkeypatch.setattr(cli, "load_index", lambda settings: object())
    monkeypatch.setattr(cli, "_make_llm", lambda settings: object())
    monkeypatch.setattr(cli, "create_agent", lambda store, llm, settings: FakeAgent())
    res = runner.invoke(cli.app, ["ask", "What?", "--thread-id", "t1"])
    assert res.exit_code == 0
    assert "the answer" in res.output
    assert calls["args"] == ("What?", "t1")


def test_ask_missing_openai_key(monkeypatch):
    s = Settings(_env_file=None, openai_api_key=None)
    monkeypatch.setattr(cli, "get_settings", lambda: s)
    res = runner.invoke(cli.app, ["ask", "What?"])
    assert res.exit_code != 0
    assert "OPENAI_API_KEY" in res.output


def test_build_index_limit_includes_gold(tmp_path, monkeypatch):
    _corpus(tmp_path)
    s = _settings(tmp_path)
    built = {}
    monkeypatch.setattr(cli, "get_settings", lambda: s)
    monkeypatch.setattr(cli, "gold_article_ids", lambda raw: ["8", "9"])
    monkeypatch.setattr(cli, "build_index", lambda chunks, settings: built.update(chunks=chunks))
    res = runner.invoke(cli.app, ["build-index", "--limit", "5"])
    assert res.exit_code == 0, res.output
    ids = {c.metadata["article_id"] for c in built["chunks"]}
    assert ids == {"0", "1", "2", "3", "4", "8", "9"}


def test_full_build_requires_flag(tmp_path, monkeypatch):
    _corpus(tmp_path)
    s = _settings(tmp_path)
    called = []
    monkeypatch.setattr(cli, "get_settings", lambda: s)
    monkeypatch.setattr(cli, "build_index", lambda *a, **k: called.append(1))
    res = runner.invoke(cli.app, ["build-index"])
    assert res.exit_code != 0
    assert "estimated embedding cost" in res.output
    assert not called


def test_full_build_with_flag(tmp_path, monkeypatch):
    _corpus(tmp_path)
    s = _settings(tmp_path)
    called = []
    monkeypatch.setattr(cli, "get_settings", lambda: s)
    monkeypatch.setattr(cli, "build_index", lambda chunks, settings: called.append(len(chunks)))
    res = runner.invoke(cli.app, ["build-index", "--yes-full-corpus"])
    assert res.exit_code == 0
    assert called == [10]


def test_gold_article_ids_from_complex_file(tmp_path):
    (tmp_path / cli.COMPLEX_QA_FILE).write_text(
        "**Query 1:** q?\n**Desired Output:** a\n**Referenced Articles:** 5, 3\n", encoding="utf-8"
    )
    ids = cli.gold_article_ids(tmp_path)
    assert isinstance(ids, list) and all(isinstance(i, str) for i in ids)
