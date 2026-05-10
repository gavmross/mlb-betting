"""
Elo rating system for MLB teams.

Computes game-by-game Elo updates and stores one row per (team, date) in
the ``elo_ratings`` table. Ratings are zero-sum within each season update.

Algorithm
---------
Expected score:  E = 1 / (1 + 10^((elo_opp - elo_self) / 400))
Update:          new_elo = old_elo + K * (result - E)
                 result = 1 win, 0 loss (no ties in MLB)
K-factor:        20 (within standard 15-25 range)

Off-season regression to mean (applied between seasons):
    elo_start = 0.70 * elo_end + 0.30 * 1500

Zero-sum invariant:
    sum(elo_ratings after game) == sum(elo_ratings before game)
    Verified after every batch update.

Writes to
---------
- ``elo_ratings`` — one row per (team, date)
- ``scrape_log``  — one row per run

Usage
-----
    python -m mlb.elo --start-season 2022 --end-season 2024
"""

from __future__ import annotations

import argparse
import logging

import numpy as np

from mlb.db import get_conn

logger = logging.getLogger(__name__)

ELO_INIT: float = 1500.0        # Starting Elo for every team
K_FACTOR: float = 20.0          # Update magnitude
REGRESSION_FACTOR: float = 0.30  # Fraction regressed toward mean each off-season
SCALE: float = 400.0             # Elo scale parameter


# ── Core math ─────────────────────────────────────────────────────────────────


def expected_score(elo_self: float, elo_opp: float) -> float:
    """
    Compute expected win probability for ``elo_self`` facing ``elo_opp``.

    Parameters
    ----------
    elo_self : float
        Current Elo of the team in question.
    elo_opp : float
        Current Elo of the opponent.

    Returns
    -------
    float
        Expected score in [0, 1].
    """
    return 1.0 / (1.0 + 10.0 ** ((elo_opp - elo_self) / SCALE))


def update_elo(
    elo_winner: float,
    elo_loser: float,
    k: float = K_FACTOR,
) -> tuple[float, float]:
    """
    Apply Elo update for a single game outcome.

    Parameters
    ----------
    elo_winner : float
    elo_loser : float
    k : float
        K-factor.

    Returns
    -------
    tuple[float, float]
        (new_elo_winner, new_elo_loser).
    """
    e_win = expected_score(elo_winner, elo_loser)
    e_lose = 1.0 - e_win

    new_winner = elo_winner + k * (1.0 - e_win)
    new_loser = elo_loser + k * (0.0 - e_lose)
    return new_winner, new_loser


def regress_to_mean(elo: float, mean: float = ELO_INIT) -> float:
    """
    Apply off-season regression: pull Elo toward the league mean.

    Parameters
    ----------
    elo : float
        End-of-season Elo.
    mean : float
        League mean (1500).

    Returns
    -------
    float
        Regressed Elo.
    """
    return (1.0 - REGRESSION_FACTOR) * elo + REGRESSION_FACTOR * mean


# ── Orchestration ─────────────────────────────────────────────────────────────


def _get_all_teams(conn) -> list[str]:
    """Return all team abbreviations from the stadiums table."""
    rows = conn.execute("SELECT team FROM stadiums ORDER BY team").fetchall()
    return [r["team"] for r in rows]


def _load_games_for_season(conn, season: int) -> list[dict]:
    """
    Load all Final games for a season, ordered by date.

    Parameters
    ----------
    conn : sqlite3.Connection
    season : int

    Returns
    -------
    list[dict]
        Keys: game_id, date, home_team, away_team, home_score, away_score.
    """
    rows = conn.execute(
        """SELECT game_id, date, home_team, away_team, home_score, away_score
           FROM games
           WHERE season = ? AND status = 'Final'
             AND home_score IS NOT NULL AND away_score IS NOT NULL
           ORDER BY date, game_id""",
        (season,),
    ).fetchall()
    return [dict(r) for r in rows]


def _upsert_elo(conn, team: str, date: str, elo: float, season: int) -> None:
    """Insert or replace a single elo_ratings row."""
    conn.execute(
        """INSERT OR REPLACE INTO elo_ratings (team, date, elo, season)
           VALUES (?, ?, ?, ?)""",
        (team, date, round(elo, 4), season),
    )


def run(
    start_season: int = 2022,
    end_season: int = 2024,
    reset: bool = False,
    db_path: str = "data/mlb.db",
) -> None:
    """
    Compute and store Elo ratings for a range of seasons.

    Parameters
    ----------
    start_season : int
        First season to compute.
    end_season : int
        Last season to compute (inclusive).
    reset : bool
        If True, delete existing elo_ratings before re-computing.
    db_path : str
        Path to the SQLite database.
    """
    with get_conn(db_path) as conn:
        teams = _get_all_teams(conn)

        if not teams:
            logger.error("No teams found in stadiums table — run stadiums_seed.py first")
            return

        if reset:
            conn.execute("DELETE FROM elo_ratings")
            logger.info("Cleared elo_ratings table")

        # Initialise Elo dict: team → current Elo
        elos: dict[str, float] = {t: ELO_INIT for t in teams}

        # If we have prior season data, load end-of-previous-season Elos
        if start_season > 2022:
            prior_rows = conn.execute(
                """SELECT team, elo FROM elo_ratings
                   WHERE season = ? AND date = (
                       SELECT MAX(date) FROM elo_ratings WHERE season = ?
                   )""",
                (start_season - 1, start_season - 1),
            ).fetchall()
            if prior_rows:
                for r in prior_rows:
                    elos[r["team"]] = regress_to_mean(r["elo"])
                logger.info(
                    "Loaded end-of-%d Elos and applied off-season regression",
                    start_season - 1,
                )

        total_rows = 0

        for season in range(start_season, end_season + 1):
            games = _load_games_for_season(conn, season)
            if not games:
                logger.warning("No Final games found for season %d — skipping", season)
                continue

            logger.info("Computing Elo for season %d (%d games)", season, len(games))

            # Apply off-season regression when moving to next season
            # (except for the very first season we process)
            if season > start_season:
                elos = {t: regress_to_mean(e) for t, e in elos.items()}
                logger.debug(
                    "Off-season regression applied: mean=%.1f std=%.1f",
                    np.mean(list(elos.values())),
                    np.std(list(elos.values())),
                )

            season_rows = 0
            prev_date = None

            for game in games:
                home = game["home_team"]
                away = game["away_team"]
                date = game["date"]

                # Add any new teams not yet in elos dict
                for team in (home, away):
                    if team not in elos:
                        logger.warning(
                            "Unknown team '%s' in game %s — initialising at 1500",
                            team,
                            game["game_id"],
                        )
                        elos[team] = ELO_INIT

                elo_home_pre = elos[home]
                elo_away_pre = elos[away]

                home_score = game["home_score"]
                away_score = game["away_score"]

                if home_score > away_score:
                    new_home, new_away = update_elo(elo_home_pre, elo_away_pre)
                else:
                    new_away, new_home = update_elo(elo_away_pre, elo_home_pre)

                elos[home] = new_home
                elos[away] = new_away

                # Store post-game Elo for each team on this date
                _upsert_elo(conn, home, date, new_home, season)
                _upsert_elo(conn, away, date, new_away, season)
                season_rows += 2

                if date != prev_date:
                    prev_date = date

            total_rows += season_rows

            # Zero-sum verification
            total_elo = sum(elos.values())
            expected_total = len(elos) * ELO_INIT
            np.testing.assert_allclose(
                total_elo,
                expected_total,
                rtol=1e-4,
                atol=1.0,
                err_msg=(
                    f"Elo zero-sum violated after season {season}: "
                    f"total={total_elo:.2f}, expected={expected_total:.2f}"
                ),
            )
            logger.info(
                "Season %d done — %d rows, mean Elo=%.1f, std=%.1f",
                season,
                season_rows,
                np.mean(list(elos.values())),
                np.std(list(elos.values())),
            )

        conn.execute(
            """INSERT INTO scrape_log
               (source, rows_inserted, status)
               VALUES (?,?,?)""",
            ("elo", total_rows, "success"),
        )

    logger.info("Elo complete — %d total rows inserted", total_rows)


# ── Query helper (used by features.py) ───────────────────────────────────────


def get_elo_before_date(
    team: str,
    date: str,
    db_path: str = "data/mlb.db",
) -> float | None:
    """
    Return the most recent Elo rating for ``team`` before ``date``.

    Parameters
    ----------
    team : str
        Canonical team abbreviation.
    date : str
        Game date in ``YYYY-MM-DD`` format.
    db_path : str

    Returns
    -------
    float or None
    """
    with get_conn(db_path) as conn:
        row = conn.execute(
            """SELECT elo FROM elo_ratings
               WHERE team = ? AND date < ?
               ORDER BY date DESC LIMIT 1""",
            (team, date),
        ).fetchone()
    return float(row["elo"]) if row else None


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="MLB Elo rating calculator")
    parser.add_argument("--start-season", type=int, default=2022)
    parser.add_argument("--end-season", type=int, default=2024)
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Clear elo_ratings table before computing",
    )
    args = parser.parse_args()

    run(
        start_season=args.start_season,
        end_season=args.end_season,
        reset=args.reset,
    )
