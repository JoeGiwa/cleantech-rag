from pathlib import Path

from cleantech_rag.config import Settings


def test_defaults():
    s = Settings(_env_file=None)
    assert s.agent_model == "gpt-4o-mini"
    assert s.judge_model == "gpt-4o"
    assert s.embedding_model == "text-embedding-3-small"
    assert (s.chunk_size, s.chunk_overlap, s.top_k) == (1000, 200, 5)
    assert s.retrieval_overfetch == 4
    assert s.chroma_dir == Path("data/chroma")
    assert s.openai_api_key is None and s.brave_api_key is None


def test_env_overrides(monkeypatch):
    monkeypatch.setenv("AGENT_MODEL", "x-model")
    monkeypatch.setenv("TOP_K", "9")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-secret")
    s = Settings(_env_file=None)
    assert s.agent_model == "x-model"
    assert s.top_k == 9
    assert s.openai_api_key.get_secret_value() == "sk-secret"
    assert "sk-secret" not in repr(s)


def test_missing_keys_ok():
    assert Settings(_env_file=None).openai_api_key is None


def test_overfetch_default_has_single_source():
    import inspect

    from cleantech_rag.config import DEFAULT_RETRIEVAL_OVERFETCH, Settings
    from cleantech_rag.tools.reasoning import make_reasoning_tool
    from cleantech_rag.tools.retriever import make_retriever_tool, retrieve_articles

    assert Settings(_env_file=None).retrieval_overfetch == DEFAULT_RETRIEVAL_OVERFETCH
    for fn in (retrieve_articles, make_retriever_tool, make_reasoning_tool):
        assert inspect.signature(fn).parameters["overfetch"].default == DEFAULT_RETRIEVAL_OVERFETCH
