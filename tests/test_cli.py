import importlib.util
import json
import logging
import re
from types import SimpleNamespace

import pandas as pd
import pytest
from pydantic import SecretStr
from typer.testing import CliRunner

from cleantech_rag import cli
from cleantech_rag.config import Settings

runner = CliRunner()

# Rich colours help output on CI (e.g. GitHub Actions), splitting "--limit" with
# ANSI codes; force plain, wide output and strip anything left over.
HELP_ENV = {"NO_COLOR": "1", "TERM": "dumb", "COLUMNS": "200"}
ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")


def _help(args):
    """Invoke the CLI for help text; return (result, ANSI-free output)."""
    res = runner.invoke(cli.app, args, env=HELP_ENV)
    return res, ANSI_RE.sub("", res.output)


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
        def run(self, q, thread_id=None, **kw):
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
        def run(self, q, thread_id=None, **kw):
            calls.append((q, thread_id))
            return SimpleNamespace(answer="ans")

    monkeypatch.setattr(cli, "get_settings", lambda: s)
    monkeypatch.setattr(cli, "index_exists", lambda settings: True)
    monkeypatch.setattr(cli, "load_index", lambda settings: object())
    monkeypatch.setattr(cli, "_make_llm", lambda settings: object())
    monkeypatch.setattr(cli, "create_agent", lambda store, llm, settings: FakeAgent())


def test_help_lists_commands():
    res, out = _help(["--help"])
    assert res.exit_code == 0
    assert "ask" in out and "chat" in out and "build-index" in out


def test_build_index_help_mentions_flags():
    res, out = _help(["build-index", "--help"])
    assert res.exit_code == 0
    assert "--limit" in out and "--yes-full-corpus" in out


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
    _, out = _help([])
    assert "build-index" in out


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
    res, out = _help(["ask", "--help"])
    assert res.exit_code == 0
    assert "Answer one question" in out  # guard: assert below must see real text
    assert "--thread-id" not in out
    res = runner.invoke(cli.app, ["ask", "Q", "--thread-id", "x"])
    assert res.exit_code != 0


def test_chat_help():
    res, out = _help(["chat", "--help"])
    assert res.exit_code == 0
    assert "exit" in out


def _chat_agent(monkeypatch, tmp_path, calls, ids=("7", "9")):
    class FakeAgent:
        def run(self, q, thread_id=None, **kw):
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
    assert "(ID 7)" in res.output and "(ID 9)" in res.output
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


# ---------------------------------------------------------------- evaluate


def _eval_setup(tmp_path, monkeypatch, n=5):
    s = _settings(tmp_path)
    lines = []
    for i in range(1, n + 1):
        lines.append(
            f"**{i}. Query:** Question {i}?\n**Desired Output:** answer {i}\n"
            f"**Referenced Articles:** {i}\n"
        )
    (tmp_path / cli.COMPLEX_QA_FILE).write_text("\n".join(lines), encoding="utf-8")
    asked = []

    class FakeAgent:
        def run(self, q, thread_id=None, **kw):
            asked.append(q)
            return SimpleNamespace(answer="an answer", retrieved_ids=["1"], tools_used=[])

    monkeypatch.setattr(cli, "get_settings", lambda: s)
    monkeypatch.setattr(cli, "_setup_agent", lambda settings: FakeAgent())
    monkeypatch.setattr(cli, "missing_eval_extras", lambda: [])
    monkeypatch.setattr(
        "cleantech_rag.evaluation.metrics.default_encoder",
        lambda *a, **k: lambda texts: [[1.0, 0.0] for _ in texts],
    )
    return asked


def test_evaluate_default_runs_three_queries(tmp_path, monkeypatch):
    asked = _eval_setup(tmp_path, monkeypatch)
    out = tmp_path / "out"
    res = runner.invoke(cli.app, ["evaluate", "--no-judge", "--output-dir", str(out)])
    assert res.exit_code == 0, res.output
    assert len(asked) == 3
    assert len(list(out.glob("summary_*.json"))) == 1
    assert len(list(out.glob("per_query_*.csv"))) == 1


def test_evaluate_full_run_requires_confirmation(tmp_path, monkeypatch):
    asked = _eval_setup(tmp_path, monkeypatch)
    res = runner.invoke(cli.app, ["evaluate", "--no-judge", "--max-queries", "0"])
    assert res.exit_code != 0
    assert asked == []
    res = runner.invoke(
        cli.app,
        ["evaluate", "--no-judge", "--yes-full-eval", "--output-dir", str(tmp_path / "o")],
    )
    assert res.exit_code == 0, res.output
    assert len(asked) == 5


def _eval_invoke(tmp_path, *args):
    return runner.invoke(cli.app, ["evaluate", "--output-dir", str(tmp_path / "out"), *args])


def test_evaluate_zero_max_queries_rejected(tmp_path, monkeypatch):
    asked = _eval_setup(tmp_path, monkeypatch)
    res = _eval_invoke(tmp_path, "--max-queries", "0")
    assert res.exit_code != 0
    assert asked == [] and not (tmp_path / "out").exists()


def test_evaluate_yes_full_eval_alone_runs_everything(tmp_path, monkeypatch):
    asked = _eval_setup(tmp_path, monkeypatch)
    res = _eval_invoke(tmp_path, "--no-judge", "--yes-full-eval")
    assert res.exit_code == 0, res.output
    assert len(asked) == 5


def test_evaluate_max_queries_with_yes_full_eval_still_limits(tmp_path, monkeypatch):
    asked = _eval_setup(tmp_path, monkeypatch)
    res = _eval_invoke(tmp_path, "--no-judge", "--max-queries", "2", "--yes-full-eval")
    assert res.exit_code == 0, res.output
    assert len(asked) == 2


def test_evaluate_max_queries_above_dataset_size(tmp_path, monkeypatch):
    asked = _eval_setup(tmp_path, monkeypatch, n=2)
    res = _eval_invoke(tmp_path, "--no-judge", "--max-queries", "10")
    assert res.exit_code == 0, res.output
    assert len(asked) == 2


def test_evaluate_negative_max_queries_rejected(tmp_path, monkeypatch):
    asked = _eval_setup(tmp_path, monkeypatch)
    res = _eval_invoke(tmp_path, "--max-queries", "-1")
    assert res.exit_code != 0 and asked == []


def test_evaluate_invalid_dataset_rejected(tmp_path, monkeypatch):
    asked = _eval_setup(tmp_path, monkeypatch)
    res = _eval_invoke(tmp_path, "--dataset", "bogus", "--no-judge")
    assert res.exit_code != 0 and asked == []


def test_evaluate_missing_eval_file_fails(tmp_path, monkeypatch):
    _eval_setup(tmp_path, monkeypatch)
    (tmp_path / cli.COMPLEX_QA_FILE).unlink()
    res = _eval_invoke(tmp_path, "--no-judge")
    assert res.exit_code == 1
    assert "Eval file not found" in res.output


def test_evaluate_summary_json_content_no_judge(tmp_path, monkeypatch):
    import json

    _eval_setup(tmp_path, monkeypatch)
    res = _eval_invoke(tmp_path, "--no-judge")
    assert res.exit_code == 0, res.output
    data = json.loads(next((tmp_path / "out").glob("summary_*.json")).read_text())
    assert data["judge"] is None
    cfg = data["config"]
    assert cfg["dataset"] == "complex" and cfg["judge_enabled"] is False
    assert cfg["judge_model"] is None and cfg["max_queries"] == 3
    assert cfg["n_queries_selected"] == 3
    assert data["counts"]["n_queries"] == 3 and data["counts"]["n_errors"] == 0
    # gold id for Question i is "i"; fake agent always returns ["1"], so only Q1 hits
    assert data["retrieval"]["all_queries"]["n"] == 3
    assert data["retrieval"]["all_queries"]["mrr"] == pytest.approx(1 / 3)


def test_evaluate_full_run_config_max_queries_null(tmp_path, monkeypatch):
    import json

    _eval_setup(tmp_path, monkeypatch)
    res = _eval_invoke(tmp_path, "--no-judge", "--yes-full-eval")
    data = json.loads(next((tmp_path / "out").glob("summary_*.json")).read_text())
    assert res.exit_code == 0 and data["config"]["max_queries"] is None


def test_evaluate_with_judge_and_judge_model_override(tmp_path, monkeypatch):
    import json

    from cleantech_rag.evaluation.judge import CRITERIA, JudgeResult

    _eval_setup(tmp_path, monkeypatch)
    made = {}

    class FakeJudge:
        def __init__(self, settings=None, **kw):
            made["model"] = settings.judge_model

        def evaluate(self, q, p, r):
            return JudgeResult(scores=dict.fromkeys(CRITERIA, 5), overall=5.0)

    monkeypatch.setattr("cleantech_rag.evaluation.judge.LLMJudge", FakeJudge)
    res = _eval_invoke(tmp_path, "--judge-model", "my-judge")
    assert res.exit_code == 0, res.output
    assert made["model"] == "my-judge"
    data = json.loads(next((tmp_path / "out").glob("summary_*.json")).read_text())
    assert data["config"]["judge_model"] == "my-judge" and data["config"]["judge_enabled"]
    assert data["judge"]["n_scored"] == 3 and data["judge"]["overall"] == 5.0


def test_evaluate_no_judge_never_constructs_judge(tmp_path, monkeypatch):
    _eval_setup(tmp_path, monkeypatch)

    def boom(*a, **k):
        raise AssertionError("judge constructed")

    monkeypatch.setattr("cleantech_rag.evaluation.judge.LLMJudge", boom)
    assert _eval_invoke(tmp_path, "--no-judge").exit_code == 0


def test_evaluate_agent_errors_reported_not_fatal(tmp_path, monkeypatch):
    import json

    _eval_setup(tmp_path, monkeypatch)

    class Boom:
        def run(self, q, thread_id=None, **kw):
            raise RuntimeError("down")

    monkeypatch.setattr(cli, "_setup_agent", lambda settings: Boom())
    res = _eval_invoke(tmp_path, "--no-judge")
    assert res.exit_code == 0, res.output
    assert "0 evaluated, 3 errors" in res.output
    data = json.loads(next((tmp_path / "out").glob("summary_*.json")).read_text())
    assert data["counts"]["n_errors"] == 3 and data["retrieval"]["all_queries"]["mrr"] is None


def test_evaluate_simple_dataset(tmp_path, monkeypatch):
    import json

    asked = _eval_setup(tmp_path, monkeypatch)
    _corpus(tmp_path, n=3)
    pd.DataFrame(
        {
            "example_id": ["e1", "e2"],
            "question_id": ["1", "2"],
            "question": ["Simple one?", "Simple two?"],
            "relevant_text": ["t", "t"],
            "answer": ["a1", "a2"],
            "article_url": ["http://x/1", "http://nowhere/z"],
        }
    ).to_csv(tmp_path / cli.SIMPLE_QA_FILE, sep=";", index=False)
    res = _eval_invoke(tmp_path, "--dataset", "simple", "--no-judge")
    assert res.exit_code == 0, res.output
    assert asked == ["Simple one?", "Simple two?"]
    data = json.loads(next((tmp_path / "out").glob("summary_*.json")).read_text())
    assert data["config"]["dataset"] == "simple"
    assert data["counts"]["n_no_gold"] == 1  # unmapped URL
    assert data["retrieval"]["all_queries"]["n"] == 1
    assert data["retrieval"]["all_queries"]["mrr"] == 1.0


def test_evaluate_simple_missing_corpus_fails(tmp_path, monkeypatch):
    _eval_setup(tmp_path, monkeypatch)
    res = _eval_invoke(tmp_path, "--dataset", "simple", "--no-judge")
    assert res.exit_code == 1 and "Eval file not found" in res.output


def test_evaluate_help_lists_flags():
    res, out = _help(["evaluate", "--help"])
    assert res.exit_code == 0
    for flag in (
        "--dataset",
        "--max-queries",
        "--no-judge",
        "--judge-model",
        "--yes-full-eval",
        "--output-dir",
    ):
        assert flag in out
    assert "complex" in out and "simple" in out


def test_evaluate_writes_jsonl_and_resume_skips_completed(tmp_path, monkeypatch):
    asked = _eval_setup(tmp_path, monkeypatch)
    res = _eval_invoke(tmp_path, "--no-judge", "--max-queries", "2")
    assert res.exit_code == 0, res.output
    log = next((tmp_path / "out").glob("run_*.jsonl"))
    assert len(log.read_text().splitlines()) == 3  # header + 2 rows
    asked.clear()
    res = _eval_invoke(tmp_path, "--no-judge", "--max-queries", "4", "--resume", str(log))
    assert res.exit_code == 0, res.output
    assert asked == ["Question 3?", "Question 4?"]
    assert len(log.read_text().splitlines()) == 5
    summaries = sorted((tmp_path / "out").glob("summary_*.json"))
    data = json.loads(summaries[-1].read_text())
    assert data["counts"]["n_evaluated"] == 4


def _first_log(tmp_path):
    return next((tmp_path / "out").glob("run_*.jsonl"))


def test_evaluate_resume_inherits_judge_setting(tmp_path, monkeypatch):
    asked = _eval_setup(tmp_path, monkeypatch)
    assert _eval_invoke(tmp_path, "--no-judge", "--max-queries", "1").exit_code == 0
    asked.clear()
    # no --no-judge passed: the log's "no judge" is inherited, so no judge is built
    res = _eval_invoke(tmp_path, "--resume", str(_first_log(tmp_path)), "--max-queries", "2")
    assert res.exit_code == 0, res.output
    assert len(asked) == 1


def test_evaluate_resume_judge_model_on_no_judge_log_refused(tmp_path, monkeypatch):
    asked = _eval_setup(tmp_path, monkeypatch)
    assert _eval_invoke(tmp_path, "--no-judge", "--max-queries", "1").exit_code == 0
    asked.clear()
    log = _first_log(tmp_path)
    res = _eval_invoke(tmp_path, "--resume", str(log), "--max-queries", "2", "--judge-model", "x")
    assert res.exit_code == 1
    assert "had no judge" in res.output
    assert asked == []


def test_evaluate_resume_conflicting_dataset_refused(tmp_path, monkeypatch):
    asked = _eval_setup(tmp_path, monkeypatch)
    assert _eval_invoke(tmp_path, "--no-judge", "--max-queries", "1").exit_code == 0
    asked.clear()
    res = _eval_invoke(
        tmp_path, "--resume", str(_first_log(tmp_path)), "--dataset", "simple", "--max-queries", "2"
    )
    assert res.exit_code != 0 and asked == []


def test_evaluate_resume_inherits_max_queries(tmp_path, monkeypatch):
    asked = _eval_setup(tmp_path, monkeypatch)
    assert _eval_invoke(tmp_path, "--no-judge", "--max-queries", "2").exit_code == 0
    asked.clear()
    res = _eval_invoke(tmp_path, "--no-judge", "--resume", str(_first_log(tmp_path)))
    assert res.exit_code == 0, res.output
    assert "0 of 2 questions remaining" in res.output  # not the default 3
    assert asked == []


def test_evaluate_resume_inherits_full_run_without_confirmation_flag(tmp_path, monkeypatch):
    asked = _eval_setup(tmp_path, monkeypatch)  # 5 questions
    first = _eval_invoke(tmp_path, "--no-judge", "--yes-full-eval")
    assert first.exit_code == 0, first.output
    log = _first_log(tmp_path)
    lines = log.read_text().splitlines()
    log.write_text("\n".join(lines[:3]) + "\n")  # header + 2 rows
    asked.clear()
    res = _eval_invoke(tmp_path, "--no-judge", "--resume", str(log))
    assert res.exit_code == 0, res.output
    assert len(asked) == 3
    assert "Summary covers 5 questions" in res.output


def test_evaluate_resume_explicit_max_queries_overrides(tmp_path, monkeypatch):
    asked = _eval_setup(tmp_path, monkeypatch)
    assert _eval_invoke(tmp_path, "--no-judge", "--max-queries", "2").exit_code == 0
    asked.clear()
    res = _eval_invoke(
        tmp_path, "--no-judge", "--resume", str(_first_log(tmp_path)), "--max-queries", "4"
    )
    assert res.exit_code == 0, res.output
    assert len(asked) == 2


def test_evaluate_resume_old_log_without_limit_needs_explicit_flag(tmp_path, monkeypatch):
    asked = _eval_setup(tmp_path, monkeypatch)
    assert _eval_invoke(tmp_path, "--no-judge", "--max-queries", "2").exit_code == 0
    log = _first_log(tmp_path)
    lines = log.read_text().splitlines()
    header = json.loads(lines[0])
    del header["config"]["max_queries"]  # simulate a log from before the field existed
    log.write_text("\n".join([json.dumps(header), *lines[1:]]) + "\n")
    asked.clear()
    res = _eval_invoke(tmp_path, "--no-judge", "--resume", str(log))
    assert res.exit_code == 1 and asked == []
    assert "max-queries" in res.output
    res = _eval_invoke(tmp_path, "--no-judge", "--resume", str(log), "--max-queries", "3")
    assert res.exit_code == 0, res.output
    assert len(asked) == 1


def test_evaluate_resume_missing_file_fails(tmp_path, monkeypatch):
    asked = _eval_setup(tmp_path, monkeypatch)
    res = _eval_invoke(tmp_path, "--no-judge", "--resume", str(tmp_path / "nope.jsonl"))
    assert res.exit_code != 0 and asked == []


def test_evaluate_resume_corrupt_log_clean_error(tmp_path, monkeypatch):
    asked = _eval_setup(tmp_path, monkeypatch)
    assert _eval_invoke(tmp_path, "--no-judge", "--max-queries", "2").exit_code == 0
    log = next((tmp_path / "out").glob("run_*.jsonl"))
    log.write_text(log.read_text() + '{"type": "row", "bogus": 1}\n{"type": "row", "bogus": 1}\n')
    asked.clear()
    res = _eval_invoke(tmp_path, "--no-judge", "--resume", str(log))
    assert res.exit_code == 1 and asked == []
    assert "line 4" in res.output
    assert not isinstance(res.exception, (TypeError, KeyError))


def test_evaluate_resume_truncated_tail_and_remaining_count(tmp_path, monkeypatch):
    asked = _eval_setup(tmp_path, monkeypatch)
    assert _eval_invoke(tmp_path, "--no-judge", "--max-queries", "2").exit_code == 0
    log = next((tmp_path / "out").glob("run_*.jsonl"))
    with log.open("a") as fh:
        fh.write('{"type": "row", "ind')
    asked.clear()
    res = _eval_invoke(tmp_path, "--no-judge", "--max-queries", "4", "--resume", str(log))
    assert res.exit_code == 0, res.output
    assert "2 of 4 questions remaining" in res.output
    assert asked == ["Question 3?", "Question 4?"]


def test_evaluate_interrupt_prints_resume_hint(tmp_path, monkeypatch):
    _eval_setup(tmp_path, monkeypatch)

    class Boom:
        n = 0

        def run(self, q, thread_id=None, **kw):
            self.n += 1
            if self.n == 2:
                raise KeyboardInterrupt
            return SimpleNamespace(answer="a", retrieved_ids=["1"], tools_used=[])

    monkeypatch.setattr(cli, "_setup_agent", lambda settings: Boom())
    res = _eval_invoke(tmp_path, "--no-judge")
    assert res.exit_code != 0
    assert "--resume" in res.output and ".jsonl" in res.output
    log = next((tmp_path / "out").glob("run_*.jsonl"))
    assert len(log.read_text().splitlines()) == 2  # header + 1 row


def test_evaluate_max_queries_over_ten_needs_confirmation(tmp_path, monkeypatch):
    asked = _eval_setup(tmp_path, monkeypatch, n=12)
    res = _eval_invoke(tmp_path, "--no-judge", "--max-queries", "11")
    assert res.exit_code == 2 and asked == []
    assert "--yes-full-eval" in res.output
    assert not (tmp_path / "out").exists()
    res = _eval_invoke(tmp_path, "--no-judge", "--max-queries", "11", "--yes-full-eval")
    assert res.exit_code == 0, res.output
    assert len(asked) == 11


def test_evaluate_max_queries_ten_needs_no_flag(tmp_path, monkeypatch):
    asked = _eval_setup(tmp_path, monkeypatch, n=12)
    res = _eval_invoke(tmp_path, "--no-judge", "--max-queries", "10")
    assert res.exit_code == 0, res.output
    assert len(asked) == 10


def test_evaluate_prints_summary_coverage_including_on_resume(tmp_path, monkeypatch):
    _eval_setup(tmp_path, monkeypatch)
    res = _eval_invoke(tmp_path, "--no-judge", "--max-queries", "2")
    assert "Summary covers 2 questions (2 from this run, 0 from resumed log)." in res.output
    log = next((tmp_path / "out").glob("run_*.jsonl"))
    res = _eval_invoke(tmp_path, "--no-judge", "--max-queries", "3", "--resume", str(log))
    assert "Summary covers 3 questions (1 from this run, 2 from resumed log)." in res.output
    res = _eval_invoke(tmp_path, "--no-judge", "--max-queries", "1", "--resume", str(log))
    assert "Summary covers 3 questions (0 from this run, 3 from resumed log)." in res.output


# ---------------------------------------------------------------- issue 30 cleanups


@pytest.fixture(autouse=True)
def _isolate_logging_and_hf_env(monkeypatch):
    """Keep configure_logging's global side effects (env vars, handlers, levels) from leaking."""
    for var in ("HF_HUB_DISABLE_PROGRESS_BARS", "TRANSFORMERS_VERBOSITY"):
        monkeypatch.delenv(var, raising=False)
    root = logging.getLogger()
    handlers, level = list(root.handlers), root.level
    yield
    root.handlers[:] = handlers
    root.setLevel(level)


@pytest.fixture
def restore_logging():
    names = [*cli.NOISY_LOGGERS, *cli.TORCH_NOISY_LOGGERS]
    saved = {n: logging.getLogger(n).level for n in names}
    yield
    for n, lvl in saved.items():
        logging.getLogger(n).setLevel(lvl)


def test_noisy_loggers_hidden_by_default(tmp_path, monkeypatch, restore_logging):
    _fake_agent(monkeypatch, _settings(tmp_path), [])
    res = runner.invoke(cli.app, ["ask", "Q"])
    assert res.exit_code == 0
    for name in ("httpx", "huggingface_hub", "sentence_transformers"):
        assert logging.getLogger(name).getEffectiveLevel() >= logging.WARNING


def test_verbose_shows_noisy_loggers(tmp_path, monkeypatch, restore_logging):
    _fake_agent(monkeypatch, _settings(tmp_path), [])
    res = runner.invoke(cli.app, ["--verbose", "ask", "Q"])
    assert res.exit_code == 0
    for name in ("httpx", "huggingface_hub", "sentence_transformers"):
        assert logging.getLogger(name).getEffectiveLevel() <= logging.INFO


def test_verbose_in_root_help():
    _, out = _help(["--help"])
    assert "--verbose" in out


def test_ask_prints_sources_with_title_url_id(tmp_path, monkeypatch):
    class FakeAgent:
        def run(self, q, thread_id=None, **kw):
            return SimpleNamespace(
                answer="A",
                retrieved_ids=["7", "9"],
                sources=[
                    {"id": "7", "title": "Solar boom", "url": "http://x/7"},
                    {"id": "9", "title": "", "url": ""},
                ],
            )

    _fake_agent(monkeypatch, _settings(tmp_path), [])
    monkeypatch.setattr(cli, "create_agent", lambda store, llm, settings: FakeAgent())
    res = runner.invoke(cli.app, ["ask", "Q"])
    assert res.exit_code == 0, res.output
    assert "Solar boom" in res.output and "http://x/7" in res.output and "ID 7" in res.output
    assert "ID 9" in res.output and "(untitled)" in res.output


def test_ask_without_retrieval_says_so(tmp_path, monkeypatch):
    _fake_agent(monkeypatch, _settings(tmp_path), [])
    res = runner.invoke(cli.app, ["ask", "Q"])
    assert "no articles retrieved" in res.output


def test_chat_shows_titles(tmp_path, monkeypatch):
    class FakeAgent:
        def run(self, q, thread_id=None, **kw):
            return SimpleNamespace(
                answer="A",
                retrieved_ids=["7"],
                sources=[{"id": "7", "title": "Wind news", "url": "http://x/7"}],
            )

    _fake_agent(monkeypatch, _settings(tmp_path), [])
    monkeypatch.setattr(cli, "create_agent", lambda store, llm, settings: FakeAgent())
    res = runner.invoke(cli.app, ["chat"], input="hi\n")
    assert "Wind news" in res.output and "ID 7" in res.output


def test_evaluate_checks_extras_before_any_paid_call(tmp_path, monkeypatch):
    asked = _eval_setup(tmp_path, monkeypatch)
    monkeypatch.setattr(cli, "missing_eval_extras", lambda: ["sentence-transformers"])

    def boom(*a, **k):
        raise AssertionError("must not be reached")

    monkeypatch.setattr(cli, "_setup_agent", boom)
    res = _eval_invoke(tmp_path, "--no-judge")
    assert res.exit_code != 0
    assert "sentence-transformers" in res.output and "--extra eval" in res.output
    assert asked == []
    assert not (tmp_path / "out").exists()


def test_configure_logging_quiet_sets_hf_env(monkeypatch, restore_logging):
    monkeypatch.delenv("HF_HUB_DISABLE_PROGRESS_BARS", raising=False)
    monkeypatch.delenv("TRANSFORMERS_VERBOSITY", raising=False)
    cli.configure_logging(False)
    import os

    assert os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] == "1"
    assert os.environ["TRANSFORMERS_VERBOSITY"] == "error"
    monkeypatch.delenv("HF_HUB_DISABLE_PROGRESS_BARS")
    monkeypatch.delenv("TRANSFORMERS_VERBOSITY")


def test_configure_logging_verbose_leaves_hf_env_alone(monkeypatch, restore_logging):
    import os

    monkeypatch.delenv("HF_HUB_DISABLE_PROGRESS_BARS", raising=False)
    cli.configure_logging(True)
    assert "HF_HUB_DISABLE_PROGRESS_BARS" not in os.environ


def test_configure_logging_does_not_override_user_env(monkeypatch, restore_logging):
    import os

    monkeypatch.setenv("HF_HUB_DISABLE_PROGRESS_BARS", "0")
    cli.configure_logging(False)
    assert os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] == "0"


def test_sources_with_ids_but_no_sources_attr(capsys):
    cli._echo_sources(SimpleNamespace(retrieved_ids=["3"]))
    out = capsys.readouterr().out
    assert "(untitled)" in out and "ID 3" in out


def test_sources_url_omitted_when_empty(capsys):
    cli._echo_sources(
        SimpleNamespace(retrieved_ids=["3"], sources=[{"id": "3", "title": "T", "url": ""}])
    )
    out = capsys.readouterr().out
    assert "T (ID 3)" in out and "http" not in out


@pytest.mark.parametrize(
    "args",
    [["--verbose", "ask", "Q"], ["ask", "Q", "--verbose"], ["ask", "--verbose", "Q"]],
)
def test_verbose_works_before_or_after_ask(tmp_path, monkeypatch, restore_logging, args):
    _fake_agent(monkeypatch, _settings(tmp_path), [])
    res = runner.invoke(cli.app, args)
    assert res.exit_code == 0, res.output
    assert logging.getLogger("httpx").getEffectiveLevel() <= logging.INFO


def test_verbose_after_chat(tmp_path, monkeypatch, restore_logging):
    _fake_agent(monkeypatch, _settings(tmp_path), [])
    res = runner.invoke(cli.app, ["chat", "--verbose"], input="exit\n")
    assert res.exit_code == 0, res.output
    assert logging.getLogger("httpx").getEffectiveLevel() <= logging.INFO


def test_verbose_after_build_index(tmp_path, monkeypatch, restore_logging):
    _corpus(tmp_path)
    monkeypatch.setattr(cli, "get_settings", lambda: _settings(tmp_path))
    monkeypatch.setattr(cli, "gold_article_ids", lambda raw: [])
    monkeypatch.setattr(cli, "build_index", lambda chunks, settings: None)
    res = runner.invoke(cli.app, ["build-index", "--limit", "3", "--verbose"])
    assert res.exit_code == 0, res.output
    assert logging.getLogger("httpx").getEffectiveLevel() <= logging.INFO


def test_verbose_after_evaluate(tmp_path, monkeypatch, restore_logging):
    _eval_setup(tmp_path, monkeypatch)
    res = _eval_invoke(tmp_path, "--no-judge", "--max-queries", "1", "--verbose")
    assert res.exit_code == 0, res.output
    assert logging.getLogger("httpx").getEffectiveLevel() <= logging.INFO


def test_subcommand_without_verbose_keeps_root_verbose(tmp_path, monkeypatch, restore_logging):
    _fake_agent(monkeypatch, _settings(tmp_path), [])
    runner.invoke(cli.app, ["--verbose", "ask", "Q"])
    assert logging.getLogger("httpx").getEffectiveLevel() <= logging.INFO


def test_chat_passes_chat_flag_ask_does_not(tmp_path, monkeypatch):
    seen = []

    class FakeAgent:
        def run(self, q, thread_id=None, **kw):
            seen.append((thread_id is not None, kw))
            return SimpleNamespace(answer="a")

    _fake_agent(monkeypatch, _settings(tmp_path), [])
    monkeypatch.setattr(cli, "create_agent", lambda store, llm, settings: FakeAgent())
    runner.invoke(cli.app, ["ask", "Q"])
    runner.invoke(cli.app, ["chat"], input="hi\nexit\n")
    assert seen == [(False, {}), (True, {"chat": True})]


def test_evaluate_summary_has_usage(tmp_path, monkeypatch):
    asked = _eval_setup(tmp_path, monkeypatch, n=2)
    res = _eval_invoke(tmp_path, "--max-queries", "1", "--no-judge")
    assert res.exit_code == 0, res.output
    assert asked == ["Question 1?"]
    data = json.loads(next((tmp_path / "out").glob("summary_*.json")).read_text())
    assert data["usage"]["cost_usd"]["total"] == 0  # fake agent made no LLM calls


def test_evaluate_quiets_loaded_hf_unless_verbose(tmp_path, monkeypatch, restore_logging):
    _eval_setup(tmp_path, monkeypatch, n=2)
    calls = []
    monkeypatch.setattr(cli, "quiet_loaded_hf", lambda: calls.append(1))
    assert _eval_invoke(tmp_path, "--max-queries", "1", "--no-judge").exit_code == 0
    assert len(calls) == 2  # root callback, then again after the agent is set up
    calls.clear()
    res = runner.invoke(
        cli.app,
        ["evaluate", "--output-dir", str(tmp_path / "out2"), "--verbose", "--max-queries", "1"]
        + ["--no-judge"],
    )
    assert res.exit_code == 0, res.output
    assert len(calls) == 1  # only the root callback (before the per-command --verbose)


def test_root_verbose_before_evaluate_does_not_requiet(tmp_path, monkeypatch, restore_logging):
    _eval_setup(tmp_path, monkeypatch, n=2)
    calls = []
    monkeypatch.setattr(cli, "quiet_loaded_hf", lambda: calls.append(1))
    res = runner.invoke(
        cli.app,
        ["--verbose", "evaluate", "--output-dir", str(tmp_path / "out"), "--max-queries", "1"]
        + ["--no-judge"],
    )
    assert res.exit_code == 0, res.output
    assert not calls


def test_quiet_logging_survives_real_eval_libraries(restore_logging, capsys, caplog):
    """Fakes never import the real stack; importing it must not undo the quiet setup."""
    for mod in ("sentence_transformers", "rouge_score", "torch"):
        if importlib.util.find_spec(mod) is None:
            pytest.skip(f"{mod} (eval extra) not installed")
    torch_name = cli.TORCH_NOISY_LOGGERS[0]
    saved = logging.getLogger(torch_name).level
    try:
        cli.configure_logging(False)  # first, as in the CLI: libraries import afterwards
        capsys.readouterr()
        pytest.importorskip("sentence_transformers")
        rouge_scorer = pytest.importorskip("rouge_score.rouge_scorer")
        import torch.distributed.elastic.multiprocessing.redirects  # noqa: F401

        rouge_scorer.RougeScorer(["rouge1"]).score("a b", "a c")  # logs at INFO via absl
        import httpx2  # noqa: F401  (vendored by huggingface_hub >= 1.x)

        logging.getLogger("httpx2").info("HTTP Request: GET x")
        logging.getLogger(torch_name).warning("NOTE: Redirects are currently not supported")
        for name in ("httpx", "httpx2", "httpcore2", "absl", "huggingface_hub"):
            assert logging.getLogger(name).getEffectiveLevel() >= logging.WARNING, name
        assert logging.getLogger(torch_name).getEffectiveLevel() >= logging.ERROR
        text = capsys.readouterr().err + caplog.text
        for noise in ("HTTP Request", "Using default tokenizer", "Redirects are currently"):
            assert noise not in text
        cli.configure_logging(True)
        assert logging.getLogger("httpx2").getEffectiveLevel() <= logging.INFO
        assert logging.getLogger("absl").getEffectiveLevel() <= logging.INFO
    finally:
        logging.getLogger(torch_name).setLevel(saved)


def test_new_noisy_loggers_registered():
    for name in ("httpx2", "httpcore2", "absl"):
        assert name in cli.NOISY_LOGGERS
    assert cli.TORCH_NOISY_LOGGERS == ("torch.distributed.elastic.multiprocessing.redirects",)


@pytest.mark.parametrize("verbose, check", [(False, logging.ERROR), (True, logging.NOTSET)])
def test_torch_logger_level_follows_verbose(restore_logging, verbose, check):
    name = cli.TORCH_NOISY_LOGGERS[0]
    cli.configure_logging(verbose)
    assert logging.getLogger(name).level == check


def test_torch_logger_requiet_then_verbose_roundtrip(restore_logging):
    name = cli.TORCH_NOISY_LOGGERS[0]
    cli.configure_logging(False)
    cli.configure_logging(True)
    assert logging.getLogger(name).level == logging.NOTSET
    cli.configure_logging(False)
    assert logging.getLogger(name).level == logging.ERROR


def test_per_command_verbose_unquiets_torch_and_absl(tmp_path, monkeypatch, restore_logging):
    _fake_agent(monkeypatch, _settings(tmp_path), [])
    res = runner.invoke(cli.app, ["ask", "--verbose", "Q"])
    assert res.exit_code == 0, res.output
    assert logging.getLogger(cli.TORCH_NOISY_LOGGERS[0]).level == logging.NOTSET
    assert logging.getLogger("absl").getEffectiveLevel() <= logging.INFO


def test_quiet_torch_import_before_configure(restore_logging, capsys):
    """Configuring after torch is already imported must still silence its warning."""
    pytest.importorskip("torch")
    import torch.distributed.elastic.multiprocessing.redirects  # noqa: F401

    cli.configure_logging(False)
    capsys.readouterr()
    logging.getLogger(cli.TORCH_NOISY_LOGGERS[0]).warning("NOTE: Redirects are currently x")
    assert "Redirects are currently" not in capsys.readouterr().err
