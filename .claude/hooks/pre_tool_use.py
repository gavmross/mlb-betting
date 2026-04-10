"""
PreToolUse hook — fires before every Bash command and file edit.

Blocks:
- Destructive database operations in production paths
- Committing sensitive files (DB, raw data, credentials)

Warns:
- Rolling window operations without .shift(1)
- Direct INSERT statements (should use INSERT OR IGNORE/REPLACE)

Exit codes:
  0 = allow
  2 = block the action (Claude sees stderr message and stops)
"""

import sys
import json
import os

tool_name = os.environ.get("CLAUDE_TOOL_NAME", "")
tool_input_raw = os.environ.get("CLAUDE_TOOL_INPUT", "{}")

try:
    tool_input = json.loads(tool_input_raw)
except json.JSONDecodeError:
    sys.exit(0)

command = tool_input.get("command", "")
file_path = tool_input.get("path", "")
new_content = tool_input.get("new_str", "") + tool_input.get("file_text", "")


# ── Bash command checks ──────────────────────────────────────────────────────

if tool_name == "Bash":

    # Block destructive DB operations
    destructive = [
        "DROP TABLE",
        "DROP INDEX",
        "TRUNCATE",
        "DELETE FROM games",
        "DELETE FROM team_stats",
        "DELETE FROM pitchers",
        "DELETE FROM predictions",
        "DELETE FROM sportsbook_odds",
        "DELETE FROM kalshi_markets",
        "rm -rf data/mlb.db",
        "rm data/mlb.db",
    ]
    for pattern in destructive:
        if pattern.lower() in command.lower():
            print(
                f"BLOCKED: Dangerous operation detected: '{pattern}'\n"
                f"Use soft deletes or versioning instead. "
                f"If you genuinely need this, do it manually.",
                file=sys.stderr,
            )
            sys.exit(2)

    # Block committing sensitive files
    if "git add" in command or "git commit" in command:
        banned_in_commit = ["mlb.db", "data/raw/", "data/models/", ".env", "private_key"]
        for b in banned_in_commit:
            if b in command:
                print(
                    f"BLOCKED: Attempting to commit '{b}' — this file is gitignored "
                    f"and must never be committed.",
                    file=sys.stderr,
                )
                sys.exit(2)

    # Warn on rolling without shift (common leakage pattern)
    if ".rolling(" in command and ".shift(" not in command:
        print(
            "WARNING: .rolling() detected without preceding .shift(1). "
            "This may cause data leakage. Verify temporal integrity.",
            file=sys.stderr,
        )
        # Don't block — just warn

    # Warn on bare INSERT (should use INSERT OR IGNORE / INSERT OR REPLACE)
    if "INSERT INTO" in command.upper() and "INSERT OR" not in command.upper():
        print(
            "WARNING: Raw INSERT detected. Use INSERT OR IGNORE or "
            "INSERT OR REPLACE to keep writes idempotent.",
            file=sys.stderr,
        )


# ── File edit checks ──────────────────────────────────────────────────────────

if tool_name in ("Edit", "Write", "Create") and file_path.endswith(".py"):

    # Warn on squared_error loss in model files
    if "model" in file_path.lower() and "squared_error" in new_content:
        print(
            "WARNING: 'squared_error' loss detected in a model file. "
            "Run scoring requires Poisson loss. "
            "Use GradientBoostingRegressor(loss='poisson') or PoissonRegressor.",
            file=sys.stderr,
        )

    # Warn on Normal CDF for probability conversion
    if "norm.cdf" in new_content and "calibration" not in file_path.lower():
        print(
            "WARNING: norm.cdf detected outside calibration.py. "
            "P(over) must be computed via Poisson convolution, not Normal CDF.",
            file=sys.stderr,
        )

    # Warn on rolling without shift in feature files
    if "features" in file_path.lower():
        if ".rolling(" in new_content and ".shift(1)" not in new_content:
            print(
                "WARNING: .rolling() in features.py without .shift(1). "
                "This will cause data leakage. Add .shift(1) before .rolling().",
                file=sys.stderr,
            )

sys.exit(0)
