"""
MLB game schedule and box score scraper.

Data sources
------------
- MLB Stats API (via mlbstatsapi) — schedule, results, box scores
  Rate limit: 1 s between requests

Coverage
--------
Run with ``--start``/``--end`` for a backfill, or ``--incremental`` to
fetch only games not yet present in the DB.

Writes to
---------
- ``games``      — one row per game
- ``team_stats`` — one row per team per game (batting stats)
- ``pitchers``   — one row per pitcher per game (game + season cumulative stats)
- ``scrape_log`` — one row per run
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import mlbstatsapi

from mlb.db import get_conn

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

RATE_LIMIT_S: float = 1.0          # seconds between Stats API calls
RAW_DIR: Path = Path("data/raw/statsapi")
FIP_CONSTANT: float = 3.1          # approximate league-average FIP constant

# Stats API team ID → pybaseball / Baseball Reference abbreviation
# (used as the primary key in the stadiums table)
TEAM_ID_TO_ABBREV: dict[int, str] = {
    108: "LAA",
    109: "ARI",
    110: "BAL",
    111: "BOS",
    112: "CHC",
    113: "CIN",
    114: "CLE",
    115: "COL",
    116: "DET",
    117: "HOU",
    118: "KCR",
    119: "LAD",
    120: "WSN",
    121: "NYM",
    133: "OAK",   # Athletics (Oakland through 2024)
    134: "PIT",
    135: "SDP",
    136: "SEA",
    137: "SFG",
    138: "STL",
    139: "TBR",
    140: "TEX",
    141: "TOR",
    142: "MIN",
    143: "PHI",
    144: "ATL",
    145: "CHW",
    146: "MIA",
    147: "NYY",
    158: "MIL",
}

# ── Cache helpers ─────────────────────────────────────────────────────────────


def _cache_path(filename: str) -> Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    return RAW_DIR / filename


def _load_cache(filename: str) -> dict | list | None:
    p = _cache_path(filename)
    if p.exists():
        try:
            return json.loads(p.read_text())
        except json.JSONDecodeError:
            logger.warning("Corrupt cache file %s — ignoring", p)
    return None


def _save_cache(filename: str, data: dict | list) -> None:
    _cache_path(filename).write_text(json.dumps(data))


# ── FIP helper ────────────────────────────────────────────────────────────────


def _calc_fip(
    hr: int | None,
    bb: int | None,
    hbp: int | None,
    k: int | None,
    ip: float | None,
    fip_const: float = FIP_CONSTANT,
) -> float | None:
    """
    Compute FIP from counting stats.

    Parameters
    ----------
    hr, bb, hbp, k : int or None
        Season-level home runs, walks, hit-by-pitch, strikeouts.
    ip : float or None
        Season innings pitched.
    fip_const : float
        League-average FIP constant (approx 3.1).

    Returns
    -------
    float or None
        FIP value, or None if any input is missing / IP is zero.
    """
    if any(v is None for v in (hr, bb, hbp, k, ip)):
        return None
    if ip == 0:
        return None
    return ((13 * hr + 3 * (bb + hbp) - 2 * k) / ip) + fip_const


def _parse_ip(ip_str: str | None) -> float | None:
    """Convert '6.2' innings-pitched string to float (6.2 → 6.667)."""
    if not ip_str:
        return None
    try:
        whole, frac = str(ip_str).split(".")
        return int(whole) + int(frac) / 3.0
    except (ValueError, AttributeError):
        try:
            return float(ip_str)
        except (ValueError, TypeError):
            return None


# ── Schedule scraping ─────────────────────────────────────────────────────────


def scrape_schedule_for_date(
    date_str: str,
    mlb: mlbstatsapi.Mlb,
) -> list[dict]:
    """
    Fetch and cache the MLB schedule for one calendar date.

    Parameters
    ----------
    date_str : str
        Date in ``YYYY-MM-DD`` format.
    mlb : mlbstatsapi.Mlb
        Initialised API client.

    Returns
    -------
    list[dict]
        Normalised game dicts ready for DB insertion.
    """
    cache_key = f"schedule_{date_str}.json"
    cached = _load_cache(cache_key)
    if cached is not None:
        return cached

    time.sleep(RATE_LIMIT_S)
    games_raw = mlb.get_scheduled_games_by_date(date_str)

    rows: list[dict] = []
    for g in games_raw:
        if g.game_type != "R":      # regular season only
            continue
        home_id = g.teams.home.team.id
        away_id = g.teams.away.team.id
        home_abbr = TEAM_ID_TO_ABBREV.get(home_id)
        away_abbr = TEAM_ID_TO_ABBREV.get(away_id)
        if not home_abbr or not away_abbr:
            logger.warning("Unknown team id(s): home=%s away=%s", home_id, away_id)
            continue

        status = g.status.abstract_game_state   # "Final", "Preview", "Live"
        home_score: int | None = None
        away_score: int | None = None
        total_runs: int | None = None

        if status == "Final":
            home_score = g.teams.home.score
            away_score = g.teams.away.score
            if home_score is not None and away_score is not None:
                total_runs = home_score + away_score

        # game_time_et: convert UTC ISO string to HH:MM ET (rough)
        game_time_et: str | None = None
        if g.game_date:
            try:
                dt_utc = datetime.fromisoformat(g.game_date.replace("Z", "+00:00"))
                # UTC-4 (ET summer); full tz awareness skipped for simplicity
                dt_et = dt_utc.replace(tzinfo=None) - timedelta(hours=4)
                game_time_et = dt_et.strftime("%H:%M")
            except (ValueError, AttributeError):
                pass

        season = int(g.season)
        rows.append({
            "game_id": str(g.game_pk),
            "date": date_str,
            "season": season,
            "home_team": home_abbr,
            "away_team": away_abbr,
            "home_score": home_score,
            "away_score": away_score,
            "total_runs": total_runs,
            "venue": g.venue.name if g.venue else None,
            "game_time_et": game_time_et,
            "status": status,
        })

    _save_cache(cache_key, rows)
    logger.debug("Fetched schedule %s — %d regular-season games", date_str, len(rows))
    return rows


# ── Box score scraping ────────────────────────────────────────────────────────


def scrape_box_score(game_pk: int, mlb: mlbstatsapi.Mlb) -> dict | None:
    """
    Fetch and cache the box score for a completed game.

    Parameters
    ----------
    game_pk : int
        MLB Stats API game primary key.
    mlb : mlbstatsapi.Mlb
        Initialised API client.

    Returns
    -------
    dict or None
        Raw box score data dict, or None on error.
    """
    cache_key = f"boxscore_{game_pk}.json"
    cached = _load_cache(cache_key)
    if cached is not None:
        return cached

    time.sleep(RATE_LIMIT_S)
    try:
        box = mlb.get_game_box_score(game_pk)
    except Exception as exc:
        logger.error("Failed to fetch box score for game %s: %s", game_pk, exc)
        return None

    data = box.model_dump()
    _save_cache(cache_key, data)
    return data


# ── DB insertion helpers ──────────────────────────────────────────────────────


def _insert_games(conn, rows: list[dict]) -> tuple[int, int]:
    inserted = updated = 0
    for row in rows:
        existing = conn.execute(
            "SELECT status, home_score FROM games WHERE game_id = ?",
            (row["game_id"],),
        ).fetchone()

        if existing is None:
            conn.execute(
                """INSERT OR IGNORE INTO games
                   (game_id, date, season, home_team, away_team, home_score,
                    away_score, total_runs, venue, game_time_et, status)
                   VALUES (:game_id,:date,:season,:home_team,:away_team,
                           :home_score,:away_score,:total_runs,:venue,
                           :game_time_et,:status)""",
                row,
            )
            inserted += 1
        elif row["status"] == "Final" and (
            existing["status"] != "Final" or existing["home_score"] is None
        ):
            # Update score once game is complete
            conn.execute(
                """UPDATE games SET home_score=:home_score, away_score=:away_score,
                   total_runs=:total_runs, status=:status,
                   updated_at=datetime('now')
                   WHERE game_id=:game_id""",
                row,
            )
            updated += 1

    return inserted, updated


def _insert_team_stats(conn, game_id: str, box) -> None:
    """Insert batting stats for both teams from a box score object."""
    sides = [("home", box.teams.home), ("away", box.teams.away)]
    for side_label, side in sides:
        is_home = 1 if side_label == "home" else 0
        team_id = side.team.id
        abbr = TEAM_ID_TO_ABBREV.get(team_id)
        if not abbr:
            logger.warning("Unknown team id %s in box score %s", team_id, game_id)
            continue

        batting = side.team_stats.get("batting", {})
        runs = batting.get("runs")
        hits = batting.get("hits")
        errors_stat = side.team_stats.get("fielding", {}).get("errors")
        at_bats = batting.get("atBats") or 0
        pa = batting.get("plateAppearances") or 0
        k_val = batting.get("strikeOuts")
        bb_val = batting.get("baseOnBalls")

        k_pct = (k_val / pa) if (k_val is not None and pa > 0) else None
        bb_pct = (bb_val / pa) if (bb_val is not None and pa > 0) else None

        def _safe_float(val: str | float | None) -> float | None:
            try:
                return float(val) if val is not None else None
            except (ValueError, TypeError):
                return None

        conn.execute(
            """INSERT OR IGNORE INTO team_stats
               (game_id, team, is_home, runs, hits, errors,
                obp, slg, ops, k_pct, bb_pct)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            (
                game_id,
                abbr,
                is_home,
                runs,
                hits,
                errors_stat,
                _safe_float(batting.get("obp")),
                _safe_float(batting.get("slg")),
                _safe_float(batting.get("ops")),
                k_pct,
                bb_pct,
            ),
        )


def _insert_pitchers(conn, game_id: str, box) -> None:
    """Insert pitcher game + season stats for all pitchers in a box score."""
    sides = [box.teams.home, box.teams.away]
    for side in sides:
        team_id = side.team.id
        abbr = TEAM_ID_TO_ABBREV.get(team_id)
        if not abbr:
            continue

        starters: set[int] = set()
        for p_id in side.pitchers:
            p = side.players.get(f"ID{p_id}")
            if p is None:
                continue
            p_stats = p.stats.get("pitching", {})
            if p_stats.get("gamesStarted", 0) >= 1:
                starters.add(p_id)

        for p_id in side.pitchers:
            p = side.players.get(f"ID{p_id}")
            if p is None:
                continue

            game_stats = p.stats.get("pitching", {})
            season_stats = p.season_stats.get("pitching", {})

            ip_game = _parse_ip(game_stats.get("inningsPitched"))
            er_game = game_stats.get("earnedRuns")
            is_starter = 1 if p_id in starters else 0

            # Season cumulative stats (YTD including this game)
            ip_season = _parse_ip(season_stats.get("inningsPitched"))
            era_season_raw = season_stats.get("era")
            try:
                # "-.--" means no innings pitched yet — genuinely missing
                era_season = float(era_season_raw) if era_season_raw not in (None, "-.--") else None
            except (ValueError, TypeError):
                era_season = None

            k9_raw = season_stats.get("strikeoutsPer9Inn")
            bb9_raw = season_stats.get("walksPer9Inn")
            hr9_raw = season_stats.get("homeRunsPer9")

            def _safe_float(v: str | float | None) -> float | None:
                try:
                    return float(v) if v not in (None, "-.--") else None
                except (ValueError, TypeError):
                    return None

            k9 = _safe_float(k9_raw)
            bb9 = _safe_float(bb9_raw)
            hr9 = _safe_float(hr9_raw)

            # FIP from season counting stats
            fip = _calc_fip(
                hr=season_stats.get("homeRuns"),
                bb=season_stats.get("baseOnBalls"),
                hbp=season_stats.get("hitByPitch"),
                k=season_stats.get("strikeOuts"),
                ip=ip_season,
            )

            player_info = p.person
            pitcher_name = (
                player_info.full_name
                if hasattr(player_info, "full_name") and player_info.full_name
                else str(p_id)
            )

            conn.execute(
                """INSERT OR IGNORE INTO pitchers
                   (game_id, pitcher_id, pitcher_name, team, is_starter,
                    ip, er, era_season, fip_season,
                    k9_season, bb9_season, hr9_season)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    game_id,
                    p_id,
                    pitcher_name,
                    abbr,
                    is_starter,
                    ip_game,
                    er_game,
                    era_season,
                    fip,
                    k9,
                    bb9,
                    hr9,
                ),
            )


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
    Scrape schedule and box scores for a date range.

    Parameters
    ----------
    start_date : str
        First date to scrape, ``YYYY-MM-DD``.
    end_date : str
        Last date to scrape, ``YYYY-MM-DD``.
    incremental : bool
        If True, skip dates already fully scraped in the DB.
    db_path : str
        Path to the SQLite database.
    """
    mlb_api = mlbstatsapi.Mlb()
    start = date.fromisoformat(start_date)
    end = date.fromisoformat(end_date)
    all_dates = _date_range(start, end)

    if incremental:
        with get_conn(db_path) as conn:
            latest = conn.execute(
                "SELECT MAX(date) FROM games WHERE status = 'Final'"
            ).fetchone()[0]
        if latest:
            # Fetch from the day after the last completed game
            resume = (date.fromisoformat(latest) + timedelta(days=1)).isoformat()
            all_dates = [d for d in all_dates if d >= resume]
            logger.info("Incremental mode: resuming from %s", resume)

    total_games = total_inserted = total_updated = total_box = 0
    errors: list[str] = []

    for date_str in all_dates:
        try:
            rows = scrape_schedule_for_date(date_str, mlb_api)
        except Exception as exc:
            logger.error("Schedule fetch failed for %s: %s", date_str, exc)
            errors.append(f"{date_str}: schedule error — {exc}")
            continue

        if not rows:
            continue

        with get_conn(db_path) as conn:
            ins, upd = _insert_games(conn, rows)
        total_games += len(rows)
        total_inserted += ins
        total_updated += upd

        # Scrape box scores for completed games
        final_game_ids = [r["game_id"] for r in rows if r["status"] == "Final"]

        for game_id in final_game_ids:
            # Skip games whose box score is already in the DB
            with get_conn(db_path) as conn:
                already = conn.execute(
                    "SELECT 1 FROM team_stats WHERE game_id = ? LIMIT 1",
                    (game_id,),
                ).fetchone()
            if already:
                continue

            # Single API call — rate-limited, cached to disk
            cache_key = f"boxscore_{game_id}.json"
            if not _cache_path(cache_key).exists():
                time.sleep(RATE_LIMIT_S)

            try:
                box = mlb_api.get_game_box_score(int(game_id))
                _save_cache(cache_key, box.model_dump())
            except Exception as exc:
                logger.error("Box score fetch failed for game %s: %s", game_id, exc)
                errors.append(f"{game_id}: fetch error — {exc}")
                continue

            try:
                with get_conn(db_path) as conn:
                    _insert_team_stats(conn, game_id, box)
                    _insert_pitchers(conn, game_id, box)
                total_box += 1
            except Exception as exc:
                logger.error("Box score parse failed for game %s: %s", game_id, exc)
                errors.append(f"{game_id}: parse error — {exc}")

    # ── scrape_log entry ──────────────────────────────────────────────────────
    with get_conn(db_path) as conn:
        conn.execute(
            """INSERT INTO scrape_log
               (source, date_range_start, date_range_end,
                rows_inserted, rows_updated, status, error_msg)
               VALUES (?,?,?,?,?,?,?)""",
            (
                "mlb_statsapi",
                start_date,
                end_date,
                total_inserted + total_box,
                total_updated,
                "success" if not errors else "partial",
                "; ".join(errors[:10]) if errors else None,
            ),
        )

    logger.info(
        "Done — dates=%d games=%d inserted=%d updated=%d box_scores=%d errors=%d",
        len(all_dates),
        total_games,
        total_inserted,
        total_updated,
        total_box,
        len(errors),
    )


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="MLB schedule + box score scraper")
    parser.add_argument("--start", default="2022-04-07", help="Start date YYYY-MM-DD")
    parser.add_argument(
        "--end",
        default=date.today().isoformat(),
        help="End date YYYY-MM-DD",
    )
    parser.add_argument(
        "--incremental",
        action="store_true",
        help="Only fetch dates not yet in the DB",
    )
    args = parser.parse_args()

    run(start_date=args.start, end_date=args.end, incremental=args.incremental)
