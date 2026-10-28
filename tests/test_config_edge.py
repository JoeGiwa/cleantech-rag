import socket
import subprocess
import sys
from pathlib import Path

import pytest
import requests
from pydantic import SecretStr, ValidationError

from cleantech_rag.config import Settings, get_settings


@pytest.fixture(autouse=True)
def _isolated_cwd(tmp_path, monkeypatch):
    """Run in an empty dir so a real dotenv file in the repo root is never read."""
    monkeypatch.chdir(tmp_path)


def test_defaults_remaining_fields():
    s = Settings(_env_file=None)
    assert s.temperature == 0.0
    assert s.raw_data_dir == Path("data/raw")
    assert s.collection_name
    assert s.brave_endpoint.startswith("https://")
    assert s.brave_timeout > 0
    assert s.brave_result_count == 5


def test_get_settings_returns_settings_without_keys():
    s = get_settings()
    assert isinstance(s, Settings)
    assert s.openai_api_key is None


def test_env_overrides_types_and_case(monkeypatch):
    monkeypatch.setenv("chunk_size", "500")
    monkeypatch.setenv("TEMPERATURE", "0.7")
    monkeypatch.setenv("CHROMA_DIR", "/tmp/xyz")
    monkeypatch.setenv("BRAVE_API_KEY", "brave-secret")
    s = Settings(_env_file=None)
    assert s.chunk_size == 500
    assert s.temperature == 0.7
    assert s.chroma_dir == Path("/tmp/xyz")
    assert isinstance(s.brave_api_key, SecretStr)
    assert "brave-secret" not in repr(s) and "brave-secret" not in str(s)


def test_invalid_env_type_raises(monkeypatch):
    monkeypatch.setenv("TOP_K", "not-an-int")
    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_dotenv_file_loaded_and_env_wins(tmp_path, monkeypatch):
    dotenv = tmp_path / "custom.dotenv"
    dotenv.write_text("AGENT_MODEL=from-file\nTOP_K=3\nUNKNOWN_VAR=1\n")
    s = Settings(_env_file=dotenv)
    assert (s.agent_model, s.top_k) == ("from-file", 3)
    monkeypatch.setenv("AGENT_MODEL", "from-env")
    assert Settings(_env_file=dotenv).agent_model == "from-env"


def test_default_construction_in_empty_cwd_has_no_keys():
    s = Settings()
    assert s.openai_api_key is None and s.brave_api_key is None


def test_import_without_key_in_clean_subprocess(tmp_path):
    env = {"PATH": "", "HOME": str(tmp_path)}
    code = (
        "import cleantech_rag, cleantech_rag.config as c;"
        "s=c.get_settings(); assert s.openai_api_key is None"
    )
    r = subprocess.run(
        [sys.executable, "-c", code], cwd=tmp_path, env=env, capture_output=True, text=True
    )
    assert r.returncode == 0, r.stderr


def test_raw_socket_blocked():
    with pytest.raises(RuntimeError, match="blocked"):
        socket.create_connection(("example.com", 80), timeout=1)
    with pytest.raises(RuntimeError, match="blocked"):
        socket.socket().connect(("127.0.0.1", 9))
    with pytest.raises(RuntimeError, match="blocked"):
        socket.getaddrinfo("example.com", 80)


def test_requests_post_and_session_blocked():
    with pytest.raises(RuntimeError, match="blocked"):
        requests.post("http://example.com", json={}, timeout=1)
    with pytest.raises(RuntimeError, match="blocked"):
        requests.Session().get("http://example.com")
