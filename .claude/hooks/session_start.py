"""
SessionStart hook — fires when a new Claude Code session begins.

Prints to stdout so Claude sees it in context:
- Current git branch and status
- DB table row counts (quick health check)
- Current phase from MEMORY.md (if it exists)
- Any open predictions from today (if any)

This orients Claude immediately without needing manual prompting.
"""

import subprocess
import sqlite3
import os
import sys
from datetime import date

DB_PATH = "data/mlb.db"
MEMORY_PATH = "MEMORY.md"

print("=" * 60)
print("MLB BETTING SYSTEM — SESSION START")
print("=" * 60)

# ── Git status ────────────────────────────────────────────────────────────────

try:
    branch = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True, text=True
    ).stdout.strip()

    status = subprocess.run(
        ["git", "status", "--short"],
        capture_output=True, text=True
    ).stdout.strip()

    print(f"\n📌 Git branch: {branch}")
    if status:
        print(f"   Modified files:\n   " + "\n   ".join(status.splitlines()))
    else:
        print("   Working tree clean")
except Exception as e:
    print(f"   Git status unavailable: {e}")

# ── DB health ─────────────────────────────────────────────────────────────────

print(f"\n🗄  Database: {DB_PATH}")

if not os.path.exists(DB_PATH):
    print("   ⚠ DB not found — run Step 2 (mlb/db.py) to initialise schema")
else:
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.execute("PRAGMA journal_mode=WAL")

        tables = [
            "games", "team_stats", "pitchers", "weather",
            "sportsbook_odds", "kalshi_markets", "predictions",
            "elo_ratings", "stadiums", "scrape_log"
        ]

        for table in tables:
            try:
                count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                print(f"   {table:<20} {count:>8,} rows")
            except sqlite3.OperationalError:
                print(f"   {table:<20}  (table not found)")

        # Latest data date
        try:
            latest = conn.execute(
                "SELECT MAX(date) FROM games WHERE status = 'final'"
            ).fetchone()[0]
            print(f"\n   Latest completed game: {latest or 'none'}")
        except Exception:
            pass

        # Today's predictions
        today = date.today().isoformat()
        try:
            bets = conn.execute(
                "SELECT COUNT(*) FROM predictions p "
                "JOIN games g ON p.game_id = g.game_id "
                "WHERE g.date = ? AND p.bet_side != 'PASS'",
                (today,)
            ).fetchone()[0]
            if bets > 0:
                print(f"\n   🎯 {bets} bet recommendation(s) for today ({today})")
        except Exception:
            pass

        conn.close()

    except Exception as e:
        print(f"   Could not read DB: {e}")

# ── MEMORY.md — current session state ────────────────────────────────────────

print(f"\n📋 Session memory:")
if os.path.exists(MEMORY_PATH):
    with open(MEMORY_PATH) as f:
        content = f.read().strip()
    # Print first 20 lines only to avoid flooding context
    lines = content.splitlines()[:20]
    for line in lines:
        print(f"   {line}")
    if len(content.splitlines()) > 20:
        print(f"   ... ({len(content.splitlines()) - 20} more lines in MEMORY.md)")
else:
    print(f"   MEMORY.md not found — create it to track session state")
    print(f"   Template:")
    print(f"   # Current State")
    print(f"   ## Phase")
    print(f"   Phase 0 — Foundation")
    print(f"   ## Last completed")
    print(f"   (nothing yet)")
    print(f"   ## In progress")
    print(f"   (starting fresh)")
    print(f"   ## Blockers")
    print(f"   (none)")

print("\n" + "=" * 60)
print("Ready. Reference @docs/ARCHITECTURE.md for full spec.")
print("=" * 60 + "\n")
