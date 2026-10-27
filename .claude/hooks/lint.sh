#!/usr/bin/env bash
# PostToolUse hook: auto-format + lint every Python file an agent edits.
# Exit 2 sends remaining lint errors back to the agent so it fixes them immediately.
set -uo pipefail

file="$(python3 -c 'import json,sys; d=json.load(sys.stdin); print((d.get("tool_input") or {}).get("file_path") or "")' 2>/dev/null)" || exit 0
case "$file" in *.py) ;; *) exit 0 ;; esac
[ -f "$file" ] || exit 0

if uv run --quiet ruff --version >/dev/null 2>&1; then RUFF=(uv run --quiet ruff); else RUFF=(uvx --quiet ruff); fi

"${RUFF[@]}" format --quiet "$file" >/dev/null 2>&1 || true
if ! out="$("${RUFF[@]}" check --fix --quiet "$file" 2>&1)"; then
  echo "ruff found issues in $file — fix them before continuing:" >&2
  echo "$out" >&2
  exit 2
fi
exit 0
