"""
Statcast pitcher season-level enrichment.

Pulls xERA, est_wOBA, and luck delta (ERA - xERA) from Baseball Savant for
each season 2015-present.  Stores in ``pitcher_season_statcast`` table, keyed
by (pitcher_id, season).

Features added to the model (all prior-season, zero leakage risk):
    sp_xera_prev       : previous season xERA (contact-quality-based ERA estimator)
    sp_est_woba_prev   : previous season expected wOBA against
    sp_luck_prev       : previous season ERA - xERA  (positive = got lucky, mean-reverts)
    sp_xera_2prev      : two seasons ago xERA (stability signal)

Usage
-----
    python -m mlb.statcast_enricher --seasons 2015 2025
    python -m mlb.statcast_enricher --seasons 2024 2025   # update only latest
"""

from __future__ import annotations

import argparse
import logging
import time

import pandas as pd
import pybaseball as pb

from mlb.db import get_conn

logger = logging.getLogger(__name__)

RATE_LIMIT_S: float = 2.0   # seconds between Statcast API calls


def fetch_season(year: int) -> pd.DataFrame:
    """
    Fetch Statcast pitcher expected stats for one season.

    Parameters
    ----------
    year : int
        MLB season year.

    Returns
    -------
    pd.DataFrame
        Columns: pitcher_id, season, xera, est_woba, luck_factor, pa.
        Empty if the API call fails.
    """
    try:
        df = pb.statcast_pitcher_expected_stats(year)
        if df.empty:
            logger.warning("No Statcast data for season %d", year)
            return pd.DataFrame()
        df = df.rename(columns={
            "player_id": "pitcher_id",
            "year": "season",
            "est_woba": "est_woba",
            "era_minus_xera_diff": "luck_factor",
        })
        df["season"] = year
        # Positive luck_factor = actual ERA > xERA = pitcher was unlucky (mean-reverts up)
        # Negative luck_factor = actual ERA < xERA = pitcher was lucky (mean-reverts down)
        keep = ["pitcher_id", "season", "xera", "est_woba", "luck_factor", "pa"]
        available = [c for c in keep if c in df.columns]
        return df[available].copy()
    except Exception as exc:
        logger.error("Statcast fetch failed for %d: %s", year, exc)
        return pd.DataFrame()


def ensure_table(conn) -> None:
    """Create pitcher_season_statcast if it does not exist."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS pitcher_season_statcast (
            pitcher_id   INTEGER NOT NULL,
            season       INTEGER NOT NULL,
            xera         REAL,
            est_woba     REAL,
            luck_factor  REAL,
            pa           INTEGER,
            fetched_at   TEXT DEFAULT (datetime('now')),
            PRIMARY KEY (pitcher_id, season)
        )
    """)


def upsert_season(conn, df: pd.DataFrame) -> int:
    """
    Insert or replace rows for one season.

    Parameters
    ----------
    df : pd.DataFrame
        Output of fetch_season().

    Returns
    -------
    int
        Number of rows upserted.
    """
    if df.empty:
        return 0
    rows = 0
    for _, row in df.iterrows():
        conn.execute(
            """INSERT OR REPLACE INTO pitcher_season_statcast
               (pitcher_id, season, xera, est_woba, luck_factor, pa)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                int(row["pitcher_id"]),
                int(row["season"]),
                row.get("xera"),
                row.get("est_woba"),
                row.get("luck_factor"),
                int(row["pa"]) if pd.notna(row.get("pa")) else None,
            ),
        )
        rows += 1
    return rows


def run(seasons: list[int], db_path: str = "data/mlb.db") -> None:
    """
    Fetch and store Statcast expected stats for the given seasons.

    Parameters
    ----------
    seasons : list[int]
        Season years to fetch.
    db_path : str
    """
    with get_conn(db_path) as conn:
        ensure_table(conn)

    total = 0
    for year in sorted(seasons):
        logger.info("Fetching Statcast pitcher expected stats for %d ...", year)
        df = fetch_season(year)
        if df.empty:
            logger.warning("Season %d: no data returned", year)
            time.sleep(RATE_LIMIT_S)
            continue
        with get_conn(db_path) as conn:
            n = upsert_season(conn, df)
        logger.info("Season %d: upserted %d rows", year, n)
        total += n
        time.sleep(RATE_LIMIT_S)

    logger.info("Statcast enrichment complete: %d total rows", total)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    parser = argparse.ArgumentParser(description="Statcast pitcher enrichment")
    parser.add_argument(
        "--seasons",
        nargs=2,
        type=int,
        metavar=("START", "END"),
        default=[2015, 2025],
        help="Season range inclusive (default: 2015 2025)",
    )
    parser.add_argument("--db", default="data/mlb.db")
    args = parser.parse_args()
    run(list(range(args.seasons[0], args.seasons[1] + 1)), db_path=args.db)
