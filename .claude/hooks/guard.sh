#!/usr/bin/env bash
# PreToolUse guard for Bash calls. Exit 2 = block the command and tell the agent why.
# Backstop for the permission rules: catches dangerous git/gh commands however they're phrased.
set -uo pipefail

input="$(cat)"
read -r -d '' PY <<'EOF'
import json, sys
d = json.load(sys.stdin)
print(d.get("cwd") or "")
print((d.get("tool_input") or {}).get("command") or "")
EOF
parsed="$(printf '%s' "$input" | python3 -c "$PY" 2>/dev/null)" || exit 0
cwd="$(printf '%s\n' "$parsed" | head -n1)"
cmd="$(printf '%s\n' "$parsed" | tail -n +2)"
[ -z "$cmd" ] && exit 0

block() { echo "BLOCKED by guard.sh: $1" >&2; exit 2; }

# Force pushes in any form
if printf '%s' "$cmd" | grep -Eq 'git[[:space:]]+push([^|;&]*)[[:space:]](-f|--force|--force-with-lease)([[:space:]=]|$)'; then
  block "force push is not allowed. Push normally or ask Joe."
fi
# Pushing to main/master explicitly (git push origin main, HEAD:main, etc.)
if printf '%s' "$cmd" | grep -Eq 'git[[:space:]]+push[^|;&]*([[:space:]:+]|^)(main|master)([[:space:]]|$)'; then
  block "pushing to main is not allowed. Push your feature branch and open a PR."
fi
# Merging PRs or deleting things on GitHub
if printf '%s' "$cmd" | grep -Eq 'gh[[:space:]]+(pr[[:space:]]+merge|repo[[:space:]]+delete|release[[:space:]]+delete)'; then
  block "merging/deleting on GitHub is Joe's job."
fi
# Destructive git operations
if printf '%s' "$cmd" | grep -Eq 'git[[:space:]]+(reset[[:space:]]+--hard|clean[[:space:]]+-[a-zA-Z]*f|branch[[:space:]]+-D|checkout[[:space:]]+(main|master)([[:space:]]|$)|switch[[:space:]]+(main|master)([[:space:]]|$))'; then
  block "destructive or branch-switching git command. Stay on your feature branch."
fi
# Committing while on main
if printf '%s' "$cmd" | grep -Eq 'git[[:space:]]+commit'; then
  branch="$(git -C "${cwd:-.}" branch --show-current 2>/dev/null || true)"
  if [ "$branch" = "main" ] || [ "$branch" = "master" ]; then
    block "you're on $branch. Work happens on feature branches (use bin/task)."
  fi
fi
# Secrets
if printf '%s' "$cmd" | grep -Eq '(^|[[:space:]/])\.env([[:space:]]|$|\.)' && ! printf '%s' "$cmd" | grep -Eq '\.env\.example'; then
  block "don't touch .env files."
fi
exit 0
