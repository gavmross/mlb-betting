"""
PostToolUse hook — fires after every file edit.

Behaviour:
- If a .py file was edited: runs Ruff format + check --fix automatically
- If mlb/features.py was edited: also runs the leakage test suite
- Async (does not block Claude's response)

Exit codes:
  PostToolUse exit codes are informational only — they do not block.
  Errors are printed to stderr for visibility.
"""

import subprocess
import os
import json
import sys

tool_name = os.environ.get("CLAUDE_TOOL_NAME", "")
tool_input_raw = os.environ.get("CLAUDE_TOOL_INPUT", "{}")

try:
    tool_input = json.loads(tool_input_raw)
except json.JSONDecodeError:
    sys.exit(0)

file_path = tool_input.get("path", "")

# Only act on Python file edits
if tool_name not in ("Edit", "Write", "Create"):
    sys.exit(0)

if not file_path.endswith(".py"):
    sys.exit(0)

# ── Auto-format with Ruff ─────────────────────────────────────────────────────

fmt = subprocess.run(
    ["ruff", "format", file_path],
    capture_output=True,
    text=True,
)
if fmt.returncode != 0:
    print(f"Ruff format warning on {file_path}:\n{fmt.stderr}", file=sys.stderr)

fix = subprocess.run(
    ["ruff", "check", "--fix", "--quiet", file_path],
    capture_output=True,
    text=True,
)
if fix.returncode != 0:
    # Non-fixable lint issues — surface them to Claude
    print(
        f"Ruff lint issues in {file_path} (could not auto-fix):\n{fix.stdout}",
        file=sys.stderr,
    )

# ── Auto-run leakage tests when features.py is touched ───────────────────────

if "features" in file_path.lower():
    result = subprocess.run(
        ["pytest", "tests/unit/test_features.py", "-v", "--tb=short", "-q"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(
            "⚠ LEAKAGE TESTS FAILED after editing features.py:\n"
            + result.stdout[-2000:]  # last 2000 chars to avoid flooding
            + result.stderr[-500:],
            file=sys.stderr,
        )
    else:
        passed = result.stdout.count(" passed")
        print(f"✓ Leakage tests passed after features.py edit ({passed})", file=sys.stderr)

sys.exit(0)
