"""
SBR odds scraper.

Fetches opening and closing totals, moneylines, and over/under odds from
SportsBooksReview.com via sbrscrape for all available sportsbooks.

Writes to
---------
- ``sportsbook_odds`` — one row per (date, home_team, away_team, book)
- ``scrape_log``      — one row per run

Notes
-----
- Pinnacle is no longer tracked by SBR (US market exit). Available books
  as of 2022-present: bet365, betmgm, caesars, draftkings, fanduel.
- The consensus / average of these books is used as the market benchmark
  where a single sharp line is needed.
- Rate limit: 1.5 s between SBR requests.
- Cache: raw game dicts saved to ``data/raw/sbr/``.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from datetime import date, timedelta
from pathlib import Path

import sbrscrape

from mlb.db import get_conn

logger = logging.getLogger(__name__)

RATE_LIMIT_S: float = 1.5
RAW_DIR: Path = Path("data/raw/sbr")

# SBR abbreviation → our canonical (pybaseball / BRef) abbreviation
SBR_TO_CANONICAL: dict[str, str] = {
    "ATH": "OAK",   # Athletics rebranding (2024+)
    "AZ": "ARI",    # SBR abbrev change (2024+)
    "KC": "KCR",
    "SD": "SDP",
    "SF": "SFG",
    "TB": "TBR",
    "WAS": "WSN",
}

# Books to store (SBR key names as returned by sbrscrape)
TRACKED_BOOKS: list[str] = ["bet365", "betmgm", "caesars", "draftkings", "fanduel"]

# ── Cache ─────────────────────────────────────────────────────────────────────


def _cache_path(filename: str) -> Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    return RAW_DIR / filename


def _load_cache(filename: str) -> list | None:
    p = _cache_path(filename)
    if p.exists():
        try:
            return json.loads(p.read_text())
        except json.JSONDecodeError:
            logger.warning("Corrupt cache %s — ignoring", p)
    return None


def _save_cache(filename: str, data: list) -> None:
    _cache_path(filename).write_text(json.dumps(data))


# ── Normalisation helpers ─────────────────────────────────────────────────────


def _canonical(abbr: str) -> str:
    """Map SBR team abbreviation to our canonical abbreviation."""
    return SBR_TO_CANONICAL.get(abbr, abbr)


def _date_str(iso_date: str) -> str:
    """Extract YYYY-MM-DD from an ISO datetime string."""
    return iso_date[:10]


def _safe_int(val: int | float | None) -> int | None:
    """Return val as int or None — American odds are integers."""
    try:
        return int(val) if val is not None else None
    except (ValueError, TypeError):
        return None


def _safe_float(val: float | str | None) -> float | None:
    try:
        return float(val) if val is not None else None
    except (ValueError, TypeError):
        return None


# ── Fetch ─────────────────────────────────────────────────────────────────────


def fetch_odds_for_date(date_str: str) -> tuple[list[dict], list[dict]]:
    """
    Fetch opening and closing odds for all MLB games on a given date.

    Uses a two-pass approach: one call for closing lines and one for opening
    lines. Both are cached to ``data/raw/sbr/``.

    Parameters
    ----------
    date_str : str
        Date in ``YYYY-MM-DD`` format.

    Returns
    -------
    tuple[list[dict], list[dict]]
        ``(closing_games, opening_games)`` — raw game dicts from sbrscrape.
    """
    close_key = f"sbr_close_{date_str}.json"
    open_key = f"sbr_open_{date_str}.json"

    # Closing lines
    close_games = _load_cache(close_key)
    if close_games is None:
        time.sleep(RATE_LIMIT_S)
        sb = sbrscrape.Scoreboard(sport="MLB", date=date_str, current_line=True)
        close_games = sb.games or []
        _save_cache(close_key, close_games)
        logger.debug("Fetched SBR closing lines for %s — %d games", date_str, len(close_games))

    # Opening lines
    open_games = _load_cache(open_key)
    if open_games is None:
        time.sleep(RATE_LIMIT_S)
        sb = sbrscrape.Scoreboard(sport="MLB", date=date_str, current_line=False)
        open_games = sb.games or []
        _save_cache(open_key, open_games)
        logger.debug("Fetched SBR opening lines for %s — %d games", date_str, len(open_games))

    return close_games, open_games


# ── Parsing ───────────────────────────────────────────────────────────────────


def _build_rows(
    date_str: str,
    close_games: list[dict],
    open_games: list[dict],
) -> list[dict]:
    """
    Merge opening and closing games into DB-ready rows, one per (game, book).

    Parameters
    ----------
    date_str : str
        Calendar date for this batch.
    close_games : list[dict]
        Closing-line game dicts from sbrscrape.
    open_games : list[dict]
        Opening-line game dicts from sbrscrape.

    Returns
    -------
    list[dict]
        Rows ready for insertion into ``sportsbook_odds``.
    """
    # Index opening games by (home_abbr, away_abbr) for fast merge
    open_index: dict[tuple[str, str], dict] = {}
    for g in open_games:
        h = _canonical(g.get("home_team_abbr", ""))
        a = _canonical(g.get("away_team_abbr", ""))
        if h and a:
            open_index[(h, a)] = g

    rows: list[dict] = []
    for g in close_games:
        home = _canonical(g.get("home_team_abbr", ""))
        away = _canonical(g.get("away_team_abbr", ""))
        if not home or not away:
            continue

        open_g = open_index.get((home, away), {})

        for book in TRACKED_BOOKS:
            total_close = _safe_float(g.get("total", {}).get(book))
            total_open = _safe_float(open_g.get("total", {}).get(book))

            # Only insert if we have at least a closing total for this book
            if total_close is None and total_open is None:
                continue

            rows.append({
                "date": date_str,
                "home_team": home,
                "away_team": away,
                "book": book,
                "total_open": total_open,
                "total_close": total_close,
                "over_odds_open": _safe_int(open_g.get("over_odds", {}).get(book)),
                "under_odds_open": _safe_int(open_g.get("under_odds", {}).get(book)),
                "over_odds_close": _safe_int(g.get("over_odds", {}).get(book)),
                "under_odds_close": _safe_int(g.get("under_odds", {}).get(book)),
                "home_ml_open": _safe_int(open_g.get("home_ml", {}).get(book)),
                "away_ml_open": _safe_int(open_g.get("away_ml", {}).get(book)),
                "home_ml_close": _safe_int(g.get("home_ml", {}).get(book)),
                "away_ml_close": _safe_int(g.get("away_ml", {}).get(book)),
                "source": "sbr",
            })

    return rows


# ── DB insertion ──────────────────────────────────────────────────────────────


def _insert_odds(conn, rows: list[dict]) -> int:
    """
    Upsert odds rows and link game_id where a matching game exists.

    Parameters
    ----------
    conn : sqlite3.Connection
        Open DB connection.
    rows : list[dict]
        Rows from ``_build_rows``.

    Returns
    -------
    int
        Number of rows inserted.
    """
    inserted = 0
    for row in rows:
        # Try to resolve game_id from games table
        game = conn.execute(
            """SELECT game_id FROM games
               WHERE date = ? AND home_team = ? AND away_team = ?
               LIMIT 1""",
            (row["date"], row["home_team"], row["away_team"]),
        ).fetchone()
        game_id = game["game_id"] if game else None

        conn.execute(
            """INSERT OR REPLACE INTO sportsbook_odds
               (game_id, date, home_team, away_team, book,
                total_open, total_close,
                over_odds_open, under_odds_open,
                over_odds_close, under_odds_close,
                home_ml_open, away_ml_open,
                home_ml_close, away_ml_close,
                source)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                game_id,
                row["date"],
                row["home_team"],
                row["away_team"],
                row["book"],
                row["total_open"],
                row["total_close"],
                row["over_odds_open"],
                row["under_odds_open"],
                row["over_odds_close"],
                row["under_odds_close"],
                row["home_ml_open"],
                row["away_ml_open"],
                row["home_ml_close"],
                row["away_ml_close"],
                row["source"],
            ),
        )
        inserted += 1

    return inserted


# ── Orchestration ─────────────────────────────────────────────────────────────


def _date_range(start: date, end: date) -> list[str]:
    days = (end - start).days + 1
    return [(start + timedelta(days=i)).isoformat() for i in range(days)]


def run(
    start_date: str,
    end_date: str,
    incremental: bool = False,
    db_path: str = "data/mlb.db",
) -> None:
    """
    Scrape SBR odds for a date range and write to ``sportsbook_odds``.

    Parameters
    ----------
    start_date : str
        First date, ``YYYY-MM-DD``.
    end_date : str
        Last date, ``YYYY-MM-DD``.
    incremental : bool
        If True, skip dates already fully represented in ``sportsbook_odds``.
    db_path : str
        Path to the SQLite database.
    """
    start = date.fromisoformat(start_date)
    end = date.fromisoformat(end_date)
    all_dates = _date_range(start, end)

    if incremental:
        with get_conn(db_path) as conn:
            latest = conn.execute(
                "SELECT MAX(date) FROM sportsbook_odds"
            ).fetchone()[0]
        if latest:
            resume = (date.fromisoformat(latest) + timedelta(days=1)).isoformat()
            all_dates = [d for d in all_dates if d >= resume]
            logger.info("Incremental mode: resuming from %s", resume)

    total_inserted = 0
    errors: list[str] = []

    for date_str in all_dates:
        # Skip if already in DB (non-incremental re-run protection)
        with get_conn(db_path) as conn:
            already = conn.execute(
                "SELECT 1 FROM sportsbook_odds WHERE date = ? LIMIT 1",
                (date_str,),
            ).fetchone()
        if already:
            continue

        try:
            close_games, open_games = fetch_odds_for_date(date_str)
        except Exception as exc:
            logger.error("SBR fetch failed for %s: %s", date_str, exc)
            errors.append(f"{date_str}: {exc}")
            continue

        if not close_games:
            logger.debug("No SBR games for %s", date_str)
            continue

        rows = _build_rows(date_str, close_games, open_games)
        if not rows:
            continue

        with get_conn(db_path) as conn:
            n = _insert_odds(conn, rows)
        total_inserted += n
        logger.info("%s — inserted %d odds rows", date_str, n)

    # scrape_log
    with get_conn(db_path) as conn:
        conn.execute(
            """INSERT INTO scrape_log
               (source, date_range_start, date_range_end,
                rows_inserted, status, error_msg)
               VALUES (?,?,?,?,?,?)""",
            (
                "sbrscrape",
                start_date,
                end_date,
                total_inserted,
                "success" if not errors else "partial",
                "; ".join(errors[:10]) if errors else None,
            ),
        )

    logger.info(
        "Done — dates=%d inserted=%d errors=%d",
        len(all_dates),
        total_inserted,
        len(errors),
    )


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="SBR odds scraper")
    parser.add_argument("--start", default="2022-04-07", help="Start date YYYY-MM-DD")
    parser.add_argument(
        "--end",
        default=date.today().isoformat(),
        help="End date YYYY-MM-DD",
    )
    parser.add_argument("--incremental", action="store_true")
    parser.add_argument("--date", help="Scrape a single date (overrides --start/--end)")
    args = parser.parse_args()

    if args.date:
        run(start_date=args.date, end_date=args.date)
    else:
        run(start_date=args.start, end_date=args.end, incremental=args.incremental)
