import json
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
        def run(self, q, thread_id=None):
            asked.append(q)
            return SimpleNamespace(answer="an answer", retrieved_ids=["1"], tools_used=[])

    monkeypatch.setattr(cli, "get_settings", lambda: s)
    monkeypatch.setattr(cli, "_setup_agent", lambda settings: FakeAgent())
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
        def run(self, q, thread_id=None):
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


def test_evaluate_resume_mismatched_config_refused(tmp_path, monkeypatch):
    asked = _eval_setup(tmp_path, monkeypatch)
    assert _eval_invoke(tmp_path, "--no-judge", "--max-queries", "1").exit_code == 0
    log = next((tmp_path / "out").glob("run_*.jsonl"))
    asked.clear()
    # judge enabled now differs from the log header
    res = _eval_invoke(tmp_path, "--resume", str(log), "--max-queries", "2")
    assert res.exit_code != 0 and asked == []


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

        def run(self, q, thread_id=None):
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
