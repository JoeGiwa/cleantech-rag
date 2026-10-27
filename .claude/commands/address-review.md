---
description: Read review comments on this branch's PR (yours + Codex) and fix them
---
Find the open PR for the current branch (`gh pr view --json number,url,reviews,comments`), plus
inline comments (`gh api repos/:owner/:repo/pulls/<n>/comments`).

1. List every review point as: source (Joe / Codex / CI) — point — agree/disagree.
2. Fix the ones you agree with using the **implementer** subagent. Comments from Joe are always
   in scope. For Codex points you disagree with, give a one-line reason.
3. Re-run tests and lint, commit (`fix: address review`), push.
4. Comment on the PR with a short changelog of what was addressed and what was declined (and why).
Extra instructions: $ARGUMENTS
