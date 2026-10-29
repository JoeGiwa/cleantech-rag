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


def _tape_lines():
    return [
        ln.strip()
        for ln in (ROOT / "demo" / "chat.tape").read_text().splitlines()
        if ln.strip() and not ln.strip().startswith("#")
    ]


def test_tape_has_one_question_one_followup_then_exit():
    typed = [ln for ln in _tape_lines() if ln.startswith("Type ")]
    assert len(typed) == 4  # launch, question, follow-up, exit
    assert typed[0] == 'Type "uv run cleantech-rag chat"'
    assert typed[-1] == 'Type "exit"'
    assert typed[1].endswith('?"') and typed[2].endswith('?"')


def test_tape_exit_matches_cli_exit_handling():
    import inspect

    from cleantech_rag import cli

    src = inspect.getsource(cli.chat)
    assert 'question.lower() == "exit"' in src
    assert "chat" in {c.name or c.callback.__name__ for c in cli.app.registered_commands}


def test_tape_is_not_executed_by_tests_or_ci():
    ci = (ROOT / ".github" / "workflows" / "ci.yml").read_text()
    assert "vhs" not in ci


def test_badge_url_matches_workflow_and_remote():
    import re

    readme = (ROOT / "README.md").read_text()
    m = re.search(r"\[!\[CI\]\((\S+)\)\]\((\S+)\)", readme)
    assert m, "CI badge missing"
    badge, link = m.groups()
    assert badge.startswith("https://github.com/JoeGiwa/cleantech-rag/actions/workflows/ci.yml/")
    assert link == "https://github.com/JoeGiwa/cleantech-rag/actions/workflows/ci.yml"
    assert (ROOT / ".github" / "workflows" / "ci.yml").is_file()
    assert readme.index("[![CI]") < readme.index("An agentic RAG system")


def test_readme_demo_placeholder_and_rerecord_note():
    readme = (ROOT / "README.md").read_text()
    assert "demo/chat.gif" in readme
    assert "vhs demo/chat.tape" in readme
    if not (ROOT / "demo" / "chat.gif").exists():
        # placeholder: embed must be commented out so no broken image renders
        before = readme.split("![Chat demo](demo/chat.gif)")[0]
        assert before.rfind("<!--") > before.rfind("-->")


def test_readme_cost_details():
    readme = (ROOT / "README.md").read_text()
    assert "142,673" in readme
    assert "usage dashboard" in readme
