---
description: Take GitHub issue #N from code to open PR (implement → test → review → PR)
argument-hint: <issue-number>
---
Ship GitHub issue #$ARGUMENTS end to end on the current branch. Do not switch branches.

1. Confirm you are NOT on `main` (`git branch --show-current`). If you are, stop.
2. Use the **implementer** subagent to implement issue #$ARGUMENTS with tests.
3. Use the **tester** subagent to check coverage of the acceptance criteria. If it reports bugs,
   send them back to the implementer. Max 2 rounds.
4. Use the **reviewer** subagent. If VERDICT is FAIL, have the implementer fix the BLOCKING items,
   then review again. Max 2 rounds; if still failing, continue and list the open items in the PR.
5. Run `uv run pytest -q` and `uv run ruff check .` one final time yourself.
6. `git push -u origin HEAD`, then open a PR with `gh pr create --draft` using
   .github/pull_request_template.md. Title = Conventional Commit style. Body must include
   `Closes #$ARGUMENTS`, the real test output summary, the reviewer's verdict, and any
   "Questions for Joe".
7. Reply with the PR URL and a 3-line summary. Then stop. Never merge.
