from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_chat_tape_is_consistent_with_cli():
    tape = (ROOT / "demo" / "chat.tape").read_text()
    assert "Output demo/chat.gif" in tape
    assert "cleantech-rag chat" in tape
    assert 'Type "exit"' in tape


def test_readme_has_no_cost_tbd_and_has_badge():
    readme = (ROOT / "README.md").read_text()
    assert "TBD" not in readme
    assert "actions/workflows/ci.yml/badge.svg" in readme
    assert "$0.76" in readme
