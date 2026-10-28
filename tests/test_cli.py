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
    monkeypatch.setattr(cli, "index_exists", lambda settings: True)
    monkeypatch.setattr(cli, "load_index", lambda settings: object())
    monkeypatch.setattr(cli, "_make_llm", lambda settings: object())
    monkeypatch.setattr(cli, "create_agent", lambda store, llm, settings: FakeAgent())
    res = runner.invoke(cli.app, ["ask", "What?"])
    assert res.exit_code == 0
    assert "the answer" in res.output
    assert calls["args"] == ("What?", None)


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
        "**1. Query:** q?\n**Desired Output:** a\n**Referenced Articles:** 5, 3\n", encoding="utf-8"
    )
    ids = cli.gold_article_ids(tmp_path)
    assert ids == ["3", "5"]
    assert isinstance(ids, list) and all(isinstance(i, str) for i in ids)


def _fake_agent(monkeypatch, s, calls):
    class FakeAgent:
        def run(self, q, thread_id=None):
            calls.append((q, thread_id))
            return SimpleNamespace(answer="ans")

    monkeypatch.setattr(cli, "get_settings", lambda: s)
    monkeypatch.setattr(cli, "index_exists", lambda settings: True)
    monkeypatch.setattr(cli, "load_index", lambda settings: object())
    monkeypatch.setattr(cli, "_make_llm", lambda settings: object())
    monkeypatch.setattr(cli, "create_agent", lambda store, llm, settings: FakeAgent())


def test_help_lists_commands():
    res = runner.invoke(cli.app, ["--help"])
    assert res.exit_code == 0
    assert "ask" in res.output and "chat" in res.output and "build-index" in res.output


def test_build_index_help_mentions_flags():
    res = runner.invoke(cli.app, ["build-index", "--help"])
    assert res.exit_code == 0
    assert "--limit" in res.output and "--yes-full-corpus" in res.output


def test_main_is_callable():
    assert callable(cli.main)


def test_ask_thread_id_defaults_to_none(tmp_path, monkeypatch):
    calls = []
    _fake_agent(monkeypatch, _settings(tmp_path), calls)
    res = runner.invoke(cli.app, ["ask", "Hi?"])
    assert res.exit_code == 0
    assert calls == [("Hi?", None)]


def test_ask_missing_key_does_not_load_index(monkeypatch):
    s = Settings(_env_file=None, openai_api_key=None)
    monkeypatch.setattr(cli, "get_settings", lambda: s)

    def boom(*a, **k):
        raise AssertionError("must not load index")

    monkeypatch.setattr(cli, "load_index", boom)
    res = runner.invoke(cli.app, ["ask", "What?"])
    assert res.exit_code != 0
    assert "OPENAI_API_KEY" in res.output


def test_ask_blank_key_rejected(monkeypatch):
    s = Settings(_env_file=None, openai_api_key=SecretStr("   "))
    monkeypatch.setattr(cli, "get_settings", lambda: s)
    res = runner.invoke(cli.app, ["ask", "What?"])
    assert res.exit_code != 0
    assert "OPENAI_API_KEY" in res.output


def test_ask_warns_without_brave_key(tmp_path, monkeypatch):
    calls = []
    s = _settings(tmp_path, brave_api_key=None)
    _fake_agent(monkeypatch, s, calls)
    res = runner.invoke(cli.app, ["ask", "Q"])
    assert res.exit_code == 0
    assert "BRAVE_API_KEY" in res.output
    assert "ans" in res.output


def test_ask_requires_question():
    res = runner.invoke(cli.app, ["ask"])
    assert res.exit_code != 0


def test_no_args_shows_help():
    res = runner.invoke(cli.app, [])
    assert "build-index" in res.output


def test_build_index_limit_calls_builder_with_5_plus_gold(tmp_path, monkeypatch):
    _corpus(tmp_path, n=20)
    s = _settings(tmp_path)
    built = {}
    monkeypatch.setattr(cli, "get_settings", lambda: s)
    monkeypatch.setattr(cli, "gold_article_ids", lambda raw: ["15", "17"])
    monkeypatch.setattr(
        cli, "build_index", lambda chunks, settings: built.update(chunks=chunks, s=settings)
    )
    res = runner.invoke(cli.app, ["build-index", "--limit", "5"])
    assert res.exit_code == 0, res.output
    ids = {c.metadata["article_id"] for c in built["chunks"]}
    assert len(ids) == 7 and {"15", "17"} <= ids
    assert built["s"] is s


def test_build_index_limit_requires_key(tmp_path, monkeypatch):
    _corpus(tmp_path)
    s = Settings(
        _env_file=None, raw_data_dir=tmp_path, chroma_dir=tmp_path / "c", openai_api_key=None
    )
    called = []
    monkeypatch.setattr(cli, "get_settings", lambda: s)
    monkeypatch.setattr(cli, "gold_article_ids", lambda raw: [])
    monkeypatch.setattr(cli, "build_index", lambda *a, **k: called.append(1))
    res = runner.invoke(cli.app, ["build-index", "--limit", "5"])
    assert res.exit_code != 0
    assert "OPENAI_API_KEY" in res.output
    assert not called


def test_full_build_with_flag_requires_key(tmp_path, monkeypatch):
    _corpus(tmp_path)
    s = Settings(
        _env_file=None, raw_data_dir=tmp_path, chroma_dir=tmp_path / "c", openai_api_key=None
    )
    called = []
    monkeypatch.setattr(cli, "get_settings", lambda: s)
    monkeypatch.setattr(cli, "build_index", lambda *a, **k: called.append(1))
    res = runner.invoke(cli.app, ["build-index", "--yes-full-corpus"])
    assert res.exit_code != 0
    assert "OPENAI_API_KEY" in res.output
    assert not called


def test_full_build_without_flag_exit_code_and_cost(tmp_path, monkeypatch):
    _corpus(tmp_path)
    monkeypatch.setattr(cli, "get_settings", lambda: _settings(tmp_path))
    monkeypatch.setattr(cli, "build_index", lambda *a, **k: None)
    res = runner.invoke(cli.app, ["build-index"])
    assert res.exit_code == 2
    assert "10 articles" in res.output
    assert "--yes-full-corpus" in res.output


def test_yes_full_corpus_with_limit_uses_limit(tmp_path, monkeypatch):
    _corpus(tmp_path, n=10)
    built = {}
    monkeypatch.setattr(cli, "get_settings", lambda: _settings(tmp_path))
    monkeypatch.setattr(cli, "gold_article_ids", lambda raw: [])
    monkeypatch.setattr(cli, "build_index", lambda chunks, settings: built.update(n=len(chunks)))
    res = runner.invoke(cli.app, ["build-index", "--limit", "3", "--yes-full-corpus"])
    assert res.exit_code == 0
    assert built["n"] == 3


def test_build_index_invalid_limit(tmp_path, monkeypatch):
    _corpus(tmp_path)
    called = []
    monkeypatch.setattr(cli, "get_settings", lambda: _settings(tmp_path))
    monkeypatch.setattr(cli, "build_index", lambda *a, **k: called.append(1))
    for bad in ("0", "-3", "abc"):
        res = runner.invoke(cli.app, ["build-index", "--limit", bad])
        assert res.exit_code != 0, bad
    assert not called


def test_build_index_missing_corpus(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "get_settings", lambda: _settings(tmp_path))
    called = []
    monkeypatch.setattr(cli, "build_index", lambda *a, **k: called.append(1))
    res = runner.invoke(cli.app, ["build-index", "--limit", "5"])
    assert res.exit_code != 0
    assert "Corpus CSV not found" in res.output
    assert not called


def test_gold_article_ids_missing_files(tmp_path):
    assert cli.gold_article_ids(tmp_path) == []


def test_gold_article_ids_dedup_sorted(tmp_path):
    (tmp_path / cli.COMPLEX_QA_FILE).write_text(
        "**1. Query:** q?\n**Desired Output:** a\n**Referenced Articles:** 5, 3\n"
        "**2. Query:** q2?\n**Desired Output:** b\n**Referenced Articles:** 3, 10\n",
        encoding="utf-8",
    )
    assert cli.gold_article_ids(tmp_path) == sorted({"5", "3", "10"})


def test_estimate_full_corpus_scales_with_length():
    s = Settings(_env_file=None)
    small = pd.DataFrame({"title": ["a"], "content": ["x" * 400]})
    big = pd.DataFrame({"title": ["a"], "content": ["x" * 4000]})
    t_small, _ = cli.estimate_full_corpus(small, s)
    t_big, _ = cli.estimate_full_corpus(big, s)
    assert 0 < t_small < t_big


def test_estimate_full_corpus_unknown_model_cost_none():
    s = Settings(_env_file=None, embedding_model="made-up-model")
    df = pd.DataFrame({"title": ["a"], "content": ["b"]})
    _, cost = cli.estimate_full_corpus(df, s)
    assert cost is None


def test_ask_has_no_thread_id_option():
    res = runner.invoke(cli.app, ["ask", "--help"])
    assert res.exit_code == 0
    assert "--thread-id" not in res.output
    res = runner.invoke(cli.app, ["ask", "Q", "--thread-id", "x"])
    assert res.exit_code != 0


def test_chat_help():
    res = runner.invoke(cli.app, ["chat", "--help"])
    assert res.exit_code == 0
    assert "exit" in res.output


def _chat_agent(monkeypatch, tmp_path, calls, ids=("7", "9")):
    class FakeAgent:
        def run(self, q, thread_id=None):
            calls.append((q, thread_id))
            return SimpleNamespace(answer=f"ans:{q}", retrieved_ids=list(ids) if q != "web" else [])

    monkeypatch.setattr(cli, "get_settings", lambda: _settings(tmp_path))
    monkeypatch.setattr(cli, "index_exists", lambda settings: True)
    monkeypatch.setattr(cli, "load_index", lambda settings: object())
    monkeypatch.setattr(cli, "_make_llm", lambda settings: object())
    monkeypatch.setattr(cli, "create_agent", lambda store, llm, settings: FakeAgent())


def test_chat_multi_turn_same_thread_and_sources(tmp_path, monkeypatch):
    calls = []
    _chat_agent(monkeypatch, tmp_path, calls)
    res = runner.invoke(cli.app, ["chat"], input="first\n\nweb\n  EXIT  \nnever\n")
    assert res.exit_code == 0, res.output
    assert [q for q, _ in calls] == ["first", "web"]
    assert calls[0][1] and calls[0][1] == calls[1][1]
    assert "ans:first" in res.output and "ans:web" in res.output
    assert "7, 9" in res.output
    assert "no articles retrieved" in res.output


def test_chat_exits_on_eof(tmp_path, monkeypatch):
    calls = []
    _chat_agent(monkeypatch, tmp_path, calls)
    res = runner.invoke(cli.app, ["chat"], input="hello\n")
    assert res.exit_code == 0, res.output
    assert len(calls) == 1


def test_chat_exits_on_ctrl_c(tmp_path, monkeypatch):
    calls = []
    _chat_agent(monkeypatch, tmp_path, calls)

    def interrupt(prompt=""):
        raise KeyboardInterrupt

    monkeypatch.setattr("builtins.input", interrupt)
    res = runner.invoke(cli.app, ["chat"])
    assert res.exit_code == 0
    assert calls == []


def test_chat_new_session_new_thread(tmp_path, monkeypatch):
    calls = []
    _chat_agent(monkeypatch, tmp_path, calls)
    runner.invoke(cli.app, ["chat"], input="a\nexit\n")
    runner.invoke(cli.app, ["chat"], input="b\nexit\n")
    assert calls[0][1] != calls[1][1]


def _no_index(monkeypatch, tmp_path):
    def boom(*a, **k):
        raise AssertionError("must not be called")

    monkeypatch.setattr(cli, "get_settings", lambda: _settings(tmp_path))
    monkeypatch.setattr(cli, "create_agent", boom)
    monkeypatch.setattr(cli, "_make_llm", boom)
    monkeypatch.setattr(cli, "load_index", boom)
    import langchain_openai

    monkeypatch.setattr(langchain_openai, "ChatOpenAI", boom)
    monkeypatch.setattr(langchain_openai, "OpenAIEmbeddings", boom)


def test_no_index_fails_fast_for_ask_and_chat(tmp_path, monkeypatch):
    _no_index(monkeypatch, tmp_path)
    for args in (["ask", "Q"], ["chat"]):
        res = runner.invoke(cli.app, args, input="hi\n")
        assert res.exit_code != 0, args
        assert "No index found. Run build-index --limit 200 first." in res.output
    assert not (tmp_path / "chroma").exists()
