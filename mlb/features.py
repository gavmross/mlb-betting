"""
Feature engineering pipeline — no-leakage guarantee.

Builds a game-level feature matrix where every rolling statistic is computed
using only data strictly available before first pitch.

Critical rule: every groupby-rolling operation uses .shift(1) to exclude the
current game.  Sort by (entity, date) BEFORE applying groupby-rolling.

Feature groups
--------------
A  Starting-pitcher stats (era, fip, k9, bb9, days_rest, era_l3)
B  Team offense (ops_10d, k_pct_10d, run_scoring_10d)
C  Bullpen (era_7d, era_30d, ip_7d)
D  Park factors (static join — no rolling)
E  Weather (game-day join — no rolling; pre-game data by definition)
F  Market signal (consensus total line — pre-game by definition)
G  Team strength (win_pct_10d, run_diff_pg_season)

Targets
-------
home_runs : int   λ_home target
away_runs : int   λ_away target

Usage
-----
    from mlb.features import build_features
    df = build_features(start_date='2022-04-07', end_date='2024-10-01')
    # df has one row per game, all features + targets
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from mlb.db import get_conn

logger = logging.getLogger(__name__)


# ── SQL helpers ───────────────────────────────────────────────────────────────


def _load_games(conn, start_date: str | None, end_date: str | None) -> pd.DataFrame:
    """Load base game rows including scores."""
    conditions = ["g.status = 'Final'", "g.total_runs IS NOT NULL"]
    params: list = []
    if start_date:
        conditions.append("g.date >= ?")
        params.append(start_date)
    if end_date:
        conditions.append("g.date <= ?")
        params.append(end_date)
    where = " AND ".join(conditions)
    sql = f"""
        SELECT g.game_id, g.date, g.season,
               g.home_team, g.away_team,
               g.home_score AS home_runs,
               g.away_score AS away_runs,
               g.total_runs,
               g.game_time_et
        FROM games g
        WHERE {where}
        ORDER BY g.date, g.game_id
    """
    rows = conn.execute(sql, params).fetchall()
    return pd.DataFrame([dict(r) for r in rows])


def _load_pitcher_game_log(conn) -> pd.DataFrame:
    """Load all starter appearances: er, ip, and YTD season stats."""
    sql = """
        SELECT p.game_id, p.pitcher_id, p.pitcher_name, p.team,
               p.ip, p.er,
               p.era_season, p.fip_season, p.k9_season,
               p.bb9_season, p.hr9_season, p.days_rest,
               g.date
        FROM pitchers p
        JOIN games g ON g.game_id = p.game_id
        WHERE p.is_starter = 1
        ORDER BY p.pitcher_id, g.date, g.game_id
    """
    rows = conn.execute(sql).fetchall()
    return pd.DataFrame([dict(r) for r in rows])


def _load_relief_log(conn) -> pd.DataFrame:
    """Load all relief appearances: er, ip, team, date."""
    sql = """
        SELECT p.game_id, p.pitcher_id, p.team,
               p.ip, p.er,
               g.date
        FROM pitchers p
        JOIN games g ON g.game_id = p.game_id
        WHERE p.is_starter = 0
        ORDER BY p.team, g.date, g.game_id
    """
    rows = conn.execute(sql).fetchall()
    return pd.DataFrame([dict(r) for r in rows])


def _load_team_batting(conn) -> pd.DataFrame:
    """Load per-game batting stats for every team."""
    sql = """
        SELECT ts.game_id, ts.team, ts.is_home,
               ts.runs, ts.ops, ts.k_pct, ts.bb_pct,
               g.date, g.season
        FROM team_stats ts
        JOIN games g ON g.game_id = ts.game_id
        WHERE g.status = 'Final'
        ORDER BY ts.team, g.date, g.game_id
    """
    rows = conn.execute(sql).fetchall()
    return pd.DataFrame([dict(r) for r in rows])


def _load_park_factors(conn) -> pd.DataFrame:
    """Load static stadium/park factor data."""
    sql = """
        SELECT team, park_run_factor, park_hr_factor,
               elevation_ft, is_dome
        FROM stadiums
    """
    rows = conn.execute(sql).fetchall()
    return pd.DataFrame([dict(r) for r in rows])


def _load_weather(conn) -> pd.DataFrame:
    """Load weather rows (prefer historical over forecast)."""
    sql = """
        SELECT w.game_id,
               w.temp_f, w.wind_speed_mph, w.wind_dir_label,
               w.precip_prob, w.humidity, w.is_dome AS wx_is_dome
        FROM weather w
        WHERE w.snapshot_type = 'historical'
        UNION ALL
        SELECT w.game_id,
               w.temp_f, w.wind_speed_mph, w.wind_dir_label,
               w.precip_prob, w.humidity, w.is_dome
        FROM weather w
        WHERE w.snapshot_type = 'forecast'
          AND w.game_id NOT IN (
              SELECT game_id FROM weather WHERE snapshot_type = 'historical'
          )
    """
    rows = conn.execute(sql).fetchall()
    return pd.DataFrame([dict(r) for r in rows])


def _load_odds(conn) -> pd.DataFrame:
    """
    Load consensus opening/closing totals lines (average across tracked books).

    Takes the mean of total_open and total_close across all books per game.
    """
    sql = """
        SELECT game_id,
               AVG(total_open)  AS total_line_open,
               AVG(total_close) AS total_line_close
        FROM sportsbook_odds
        WHERE game_id IS NOT NULL
        GROUP BY game_id
    """
    rows = conn.execute(sql).fetchall()
    return pd.DataFrame([dict(r) for r in rows])


def _load_elo(conn) -> pd.DataFrame:
    """Load most-recent elo rating per team per date (day-of or prior)."""
    sql = """
        SELECT team, date, elo
        FROM elo_ratings
        ORDER BY team, date
    """
    rows = conn.execute(sql).fetchall()
    return pd.DataFrame([dict(r) for r in rows])


# ── Rolling feature builders ──────────────────────────────────────────────────


def _add_sp_features(games: pd.DataFrame, sp_log: pd.DataFrame) -> pd.DataFrame:
    """
    Add starting-pitcher features (per pitcher, shift(1)-safe).

    Features added (home_ and away_ prefixed):
        sp_era_season  : YTD ERA from previous appearance
        sp_fip_season  : YTD FIP from previous appearance
        sp_k9_season   : YTD K/9 from previous appearance
        sp_bb9_season  : YTD BB/9 from previous appearance
        sp_days_rest   : Days rest for this start
        sp_era_l3      : ERA over last 3 starts (from er/ip)

    Parameters
    ----------
    games : pd.DataFrame
    sp_log : pd.DataFrame
        One row per starter appearance with pitcher_id, team, date, er, ip,
        era_season, fip_season, k9_season, bb9_season.

    Returns
    -------
    pd.DataFrame
        games with SP feature columns added.
    """
    if sp_log.empty:
        logger.warning("sp_log is empty — SP features will be NaN")
        return games

    sp = sp_log.sort_values(["pitcher_id", "date", "game_id"]).copy()
    sp["date"] = pd.to_datetime(sp["date"])

    # Compute days_rest from game dates (previous start date → current start date)
    sp["prev_date"] = sp.groupby("pitcher_id")["date"].shift(1)
    sp["days_rest_computed"] = (sp["date"] - sp["prev_date"]).dt.days

    # YTD stats: shift(1) per pitcher to get PREVIOUS game's season stats.
    # era_season from the API is post-game YTD. shift(1) gives pre-game YTD.
    for col in ["era_season", "fip_season", "k9_season", "bb9_season", "hr9_season"]:
        sp[f"{col}_lag"] = sp.groupby("pitcher_id")[col].shift(1)

    # ERA over last 3 starts from raw er/ip (fully lag-correct)
    sp["er_lag"] = sp.groupby("pitcher_id")["er"].shift(1)
    sp["ip_lag"] = sp.groupby("pitcher_id")["ip"].shift(1)

    def _era_l3(grp: pd.Series) -> pd.Series:
        er3 = grp["er_lag"].rolling(3, min_periods=1).sum()
        ip3 = grp["ip_lag"].rolling(3, min_periods=1).sum()
        return (er3 * 9 / ip3.replace(0, np.nan)).round(4)

    sp["sp_era_l3"] = sp.groupby("pitcher_id", group_keys=False).apply(_era_l3)

    # Rename lag columns
    sp = sp.rename(
        columns={
            "era_season_lag": "sp_era_season",
            "fip_season_lag": "sp_fip_season",
            "k9_season_lag": "sp_k9_season",
            "bb9_season_lag": "sp_bb9_season",
            "hr9_season_lag": "sp_hr9_season",
        }
    )

    sp_cols = [
        "game_id", "team",
        "sp_era_season", "sp_fip_season", "sp_k9_season",
        "sp_bb9_season", "sp_hr9_season",
        "sp_era_l3", "days_rest_computed",
    ]
    sp_slim = sp[sp_cols].copy()

    # Merge home SP
    home_sp = sp_slim.add_prefix("home_").rename(
        columns={"home_game_id": "game_id", "home_team": "_home_team"}
    )
    # Need to match on game_id + home_team
    home_sp["game_id"] = sp_slim["game_id"]
    home_sp["_team"] = sp_slim["team"]
    home_sp = home_sp.rename(columns={"home_days_rest": "home_sp_days_rest"})

    games = games.merge(
        home_sp[home_sp["_team"] == games["home_team"].values[0]].drop("_team", axis=1)
        if False  # placeholder — use loop below
        else _sp_for_side(sp_slim, games, "home"),
        on="game_id",
        how="left",
    )
    games = games.merge(
        _sp_for_side(sp_slim, games, "away"),
        on="game_id",
        how="left",
    )

    # Combined SP features
    games["sp_fip_combined"] = games["home_sp_fip_season"] + games["away_sp_fip_season"]
    games["sp_k9_combined"] = games["home_sp_k9_season"] + games["away_sp_k9_season"]
    games["sp_era_l3_combined"] = games["home_sp_era_l3"] + games["away_sp_era_l3"]

    return games


def _sp_for_side(sp_slim: pd.DataFrame, games: pd.DataFrame, side: str) -> pd.DataFrame:
    """
    Return SP feature columns for the home or away pitcher, keyed by game_id.

    Parameters
    ----------
    sp_slim : pd.DataFrame
        SP log with game_id, team, and feature columns.
    games : pd.DataFrame
        Game-level DataFrame with home_team and away_team columns.
    side : str
        ``'home'`` or ``'away'``.

    Returns
    -------
    pd.DataFrame
        Columns: game_id + prefixed SP feature columns.
    """
    team_col = f"{side}_team"
    game_teams = games[["game_id", team_col]].rename(columns={team_col: "team"})

    merged = game_teams.merge(sp_slim, on=["game_id", "team"], how="left")

    prefix_map = {
        "sp_era_season": f"{side}_sp_era_season",
        "sp_fip_season": f"{side}_sp_fip_season",
        "sp_k9_season": f"{side}_sp_k9_season",
        "sp_bb9_season": f"{side}_sp_bb9_season",
        "sp_hr9_season": f"{side}_sp_hr9_season",
        "sp_era_l3": f"{side}_sp_era_l3",
        "days_rest_computed": f"{side}_sp_days_rest",
    }
    merged = merged.rename(columns=prefix_map)

    keep = ["game_id"] + list(prefix_map.values())
    # Drop duplicates (doubleheaders can have two starters per team per game_id)
    return merged[keep].drop_duplicates("game_id")


def _add_team_offense_features(
    games: pd.DataFrame, batting: pd.DataFrame
) -> pd.DataFrame:
    """
    Add rolling 10-game team offense features (shift(1)-safe).

    Features added (home_ and away_ prefixed):
        team_ops_10d     : rolling 10-game OPS
        team_k_pct_10d   : rolling 10-game K%
        team_runs_10d    : rolling 10-game avg runs scored
        run_diff_pg_season : season run differential per game

    Parameters
    ----------
    games : pd.DataFrame
    batting : pd.DataFrame

    Returns
    -------
    pd.DataFrame
    """
    if batting.empty:
        return games

    b = batting.sort_values(["team", "date", "game_id"]).copy()

    # Rolling 10-game offensive stats (shift(1) excludes current game)
    for col in ["ops", "k_pct", "runs"]:
        b[f"{col}_10d"] = (
            b.groupby("team")[col]
            .shift(1)
            .rolling(10, min_periods=3)
            .mean()
            .values
        )

    # Season run differential per game (cumulative up to but NOT including today)
    b["runs_lag"] = b.groupby(["team", "season"])["runs"].shift(1)
    b["game_num"] = b.groupby(["team", "season"]).cumcount()  # 0-indexed

    # run_diff_pg: (my runs - opponent runs) / games played so far
    # We need opponent runs — join back to games to get the other team's runs
    opp = batting[["game_id", "team", "runs"]].rename(
        columns={"team": "opp_team", "runs": "opp_runs"}
    )
    b2 = b.merge(
        batting[["game_id", "team", "runs"]].rename(
            columns={"runs": "my_runs_raw"}
        ),
        on=["game_id", "team"],
    )
    # Actually simpler: compute cumulative run diff within team+season
    b["run_diff_lag"] = (
        b.groupby(["team", "season"])["runs"]
        .shift(1)
        .rolling(window=10000, min_periods=1)
        .sum()
        .values
    )

    keep_cols = ["game_id", "team", "ops_10d", "k_pct_10d", "runs_10d"]
    b_slim = b[keep_cols].copy()

    for side in ("home", "away"):
        team_col = f"{side}_team"
        game_teams = games[["game_id", team_col]].rename(columns={team_col: "team"})
        merged = game_teams.merge(b_slim, on=["game_id", "team"], how="left")
        prefix_map = {
            "ops_10d": f"{side}_ops_10d",
            "k_pct_10d": f"{side}_k_pct_10d",
            "runs_10d": f"{side}_runs_10d",
        }
        merged = merged.rename(columns=prefix_map).drop_duplicates("game_id")
        keep = ["game_id"] + list(prefix_map.values())
        games = games.merge(merged[keep], on="game_id", how="left")

    return games


def _add_bullpen_features(
    games: pd.DataFrame, relief_log: pd.DataFrame
) -> pd.DataFrame:
    """
    Add rolling bullpen ERA and workload features (shift(1)-safe).

    Features added (home_ and away_ prefixed):
        bullpen_era_7d  : relief ERA over last 7 calendar days
        bullpen_era_30d : relief ERA over last 30 calendar days
        bullpen_ip_7d   : relief IP in last 7 days (fatigue proxy)

    Parameters
    ----------
    games : pd.DataFrame
    relief_log : pd.DataFrame

    Returns
    -------
    pd.DataFrame
    """
    if relief_log.empty:
        return games

    r = relief_log.copy()
    r["date"] = pd.to_datetime(r["date"])

    def _bullpen_stats_for_game(
        game_id: str, team: str, game_date: pd.Timestamp, r: pd.DataFrame
    ) -> dict:
        """Compute bullpen stats using only appearances BEFORE this game."""
        prior = r[(r["team"] == team) & (r["date"] < game_date)]
        if prior.empty:
            return {
                "bullpen_era_7d": np.nan,
                "bullpen_era_30d": np.nan,
                "bullpen_ip_7d": np.nan,
            }
        w7 = prior[prior["date"] >= game_date - pd.Timedelta(days=7)]
        w30 = prior[prior["date"] >= game_date - pd.Timedelta(days=30)]

        def _era(subset: pd.DataFrame) -> float:
            ip = subset["ip"].sum()
            er = subset["er"].sum()
            return (er * 9 / ip) if ip > 0 else np.nan

        return {
            "bullpen_era_7d": _era(w7),
            "bullpen_era_30d": _era(w30),
            "bullpen_ip_7d": float(w7["ip"].sum()),
        }

    games["_date_ts"] = pd.to_datetime(games["date"])

    results: list[dict] = []
    for side in ("home", "away"):
        team_col = f"{side}_team"
        records = []
        for _, row in games.iterrows():
            stats = _bullpen_stats_for_game(
                row["game_id"], row[team_col], row["_date_ts"], r
            )
            rec = {"game_id": row["game_id"]}
            for k, v in stats.items():
                rec[f"{side}_{k}"] = v
            records.append(rec)
        side_df = pd.DataFrame(records)
        games = games.merge(side_df, on="game_id", how="left")

    games = games.drop(columns=["_date_ts"])
    return games


def _add_park_features(games: pd.DataFrame, parks: pd.DataFrame) -> pd.DataFrame:
    """
    Join static park factors from the stadiums table.

    Features added:
        park_run_factor, park_hr_factor, park_elevation_ft, is_dome

    Parameters
    ----------
    games : pd.DataFrame
    parks : pd.DataFrame

    Returns
    -------
    pd.DataFrame
    """
    parks_renamed = parks.rename(
        columns={"elevation_ft": "park_elevation_ft", "team": "home_team"}
    )
    return games.merge(parks_renamed, on="home_team", how="left")


def _add_weather_features(games: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    """
    Join game-day weather. Wind direction label is one-hot-encoded.

    Features added:
        temp_f, wind_speed_mph, wind_dir_out, wind_dir_in,
        wind_dir_cross_right, wind_dir_cross_left,
        precip_prob, humidity, is_night_game

    Parameters
    ----------
    games : pd.DataFrame
    weather : pd.DataFrame

    Returns
    -------
    pd.DataFrame
    """
    if weather.empty:
        return games

    w = weather.copy()

    # One-hot wind direction
    for label in ("out", "in", "cross_right", "cross_left"):
        w[f"wind_dir_{label}"] = (w["wind_dir_label"] == label).astype(int)

    w = w.drop(columns=["wind_dir_label", "wx_is_dome"], errors="ignore")

    games = games.merge(w, on="game_id", how="left")

    # Night game flag from game_time_et (19:00+ = night)
    def _is_night(t: str | None) -> int:
        if not t:
            return 1  # default: assume night
        try:
            return 1 if int(t.split(":")[0]) >= 17 else 0
        except (ValueError, AttributeError):
            return 1

    games["is_night_game"] = games["game_time_et"].apply(_is_night)
    return games


def _add_market_features(games: pd.DataFrame, odds: pd.DataFrame) -> pd.DataFrame:
    """
    Join pre-game market totals lines.

    These are pre-game signals — no shift needed.
    line_movement = close - open (sharp money proxy; sharp bets move lines).

    Features added:
        total_line_open, total_line_close, line_movement

    Parameters
    ----------
    games : pd.DataFrame
    odds : pd.DataFrame

    Returns
    -------
    pd.DataFrame
    """
    if odds.empty:
        return games

    o = odds.copy()
    o["line_movement"] = o["total_line_close"] - o["total_line_open"]
    return games.merge(o, on="game_id", how="left")


def _add_team_strength_features(
    games: pd.DataFrame, batting: pd.DataFrame
) -> pd.DataFrame:
    """
    Add rolling win% and run-differential features (shift(1)-safe).

    Features added (home_ and away_ prefixed):
        win_pct_10d    : rolling 10-game win percentage
        run_diff_pg_season : season run differential per game (YTD)

    Parameters
    ----------
    games : pd.DataFrame
    batting : pd.DataFrame

    Returns
    -------
    pd.DataFrame
    """
    if batting.empty:
        return games

    # Build a team-game-result table
    # We need win/loss and run_diff for each team in each game
    # batting has team + runs; join to get opponent runs
    b = batting[["game_id", "team", "runs", "is_home", "date", "season"]].copy()
    b["date"] = pd.to_datetime(b["date"])

    # Pair home and away via game_id
    home_b = b[b["is_home"] == 1][["game_id", "team", "runs", "date", "season"]].rename(
        columns={"team": "home_team", "runs": "home_runs_raw"}
    )
    away_b = b[b["is_home"] == 0][["game_id", "team", "runs"]].rename(
        columns={"team": "away_team", "runs": "away_runs_raw"}
    )
    pairs = home_b.merge(away_b, on="game_id", how="inner")

    # Stack into one row per team per game
    records: list[dict] = []
    for _, row in pairs.iterrows():
        for side, my_team, opp_runs, my_runs in [
            ("home", row["home_team"], row["away_runs_raw"], row["home_runs_raw"]),
            ("away", row["away_team"], row["home_runs_raw"], row["away_runs_raw"]),
        ]:
            records.append(
                {
                    "game_id": row["game_id"],
                    "team": my_team,
                    "date": row["date"],
                    "season": row["season"],
                    "my_runs": my_runs,
                    "opp_runs": opp_runs,
                    "win": 1 if my_runs > opp_runs else 0,
                    "run_diff": my_runs - opp_runs,
                }
            )

    ts = pd.DataFrame(records).sort_values(["team", "date", "game_id"])

    # Rolling win% (10 games) with shift(1)
    ts["win_pct_10d"] = (
        ts.groupby("team")["win"]
        .shift(1)
        .rolling(10, min_periods=3)
        .mean()
        .values
    )

    # Season run_diff per game (cumulative YTD with shift)
    ts["run_diff_cumsum"] = (
        ts.groupby(["team", "season"])["run_diff"]
        .shift(1)
        .rolling(window=10000, min_periods=1)
        .sum()
        .values
    )
    ts["games_played"] = ts.groupby(["team", "season"]).cumcount()  # 0-indexed
    ts["run_diff_pg_season"] = (
        ts["run_diff_cumsum"] / ts["games_played"].replace(0, np.nan)
    )

    ts_slim = ts[["game_id", "team", "win_pct_10d", "run_diff_pg_season"]]

    for side in ("home", "away"):
        team_col = f"{side}_team"
        game_teams = games[["game_id", team_col]].rename(columns={team_col: "team"})
        merged = game_teams.merge(ts_slim, on=["game_id", "team"], how="left")
        merged = merged.rename(
            columns={
                "win_pct_10d": f"{side}_win_pct_10d",
                "run_diff_pg_season": f"{side}_run_diff_pg_season",
            }
        ).drop_duplicates("game_id")
        keep = ["game_id", f"{side}_win_pct_10d", f"{side}_run_diff_pg_season"]
        games = games.merge(merged[keep], on="game_id", how="left")

    return games


def _add_elo_features(games: pd.DataFrame, elo: pd.DataFrame) -> pd.DataFrame:
    """
    Join Elo ratings for home and away teams (most recent rating before game).

    Features added:
        elo_home, elo_away

    Parameters
    ----------
    games : pd.DataFrame
    elo : pd.DataFrame

    Returns
    -------
    pd.DataFrame
    """
    if elo.empty:
        logger.debug("Elo table is empty — elo features will be NaN")
        games["elo_home"] = np.nan
        games["elo_away"] = np.nan
        return games

    elo = elo.sort_values(["team", "date"])

    def _latest_elo(team: str, before_date: str, elo: pd.DataFrame) -> float | None:
        subset = elo[(elo["team"] == team) & (elo["date"] < before_date)]
        if subset.empty:
            return None
        return float(subset.iloc[-1]["elo"])

    # Vectorised merge-asof per team
    games["elo_home"] = [
        _latest_elo(row["home_team"], row["date"], elo)
        for _, row in games.iterrows()
    ]
    games["elo_away"] = [
        _latest_elo(row["away_team"], row["date"], elo)
        for _, row in games.iterrows()
    ]
    return games


# ── Public entry point ────────────────────────────────────────────────────────


def build_features(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    db_path: str = "data/mlb.db",
    include_elo: bool = True,
) -> pd.DataFrame:
    """
    Build the full game-level feature matrix.

    All rolling features are computed with ``.shift(1)`` before ``.rolling()``
    to ensure no data from the current game is included.

    Parameters
    ----------
    start_date : str or None
        Return rows on or after this date (``YYYY-MM-DD``).
        Earlier games are still used for rolling history.
    end_date : str or None
        Return rows on or before this date.
    db_path : str
        Path to the SQLite database.
    include_elo : bool
        Whether to add Elo features (requires elo_ratings table to be populated).

    Returns
    -------
    pd.DataFrame
        One row per game. Columns:
        - game_id, date, season, home_team, away_team
        - home_runs, away_runs, total_runs  (targets — NaN for future games)
        - Feature groups A–G
    """
    logger.info("Building features (start=%s, end=%s)", start_date, end_date)

    with get_conn(db_path) as conn:
        games = _load_games(conn, start_date=None, end_date=end_date)
        sp_log = _load_pitcher_game_log(conn)
        relief_log = _load_relief_log(conn)
        batting = _load_team_batting(conn)
        parks = _load_park_factors(conn)
        weather = _load_weather(conn)
        odds = _load_odds(conn)
        elo = _load_elo(conn) if include_elo else pd.DataFrame()

    if games.empty:
        logger.warning("No games found in DB for the given date range")
        return pd.DataFrame()

    logger.info("Loaded %d base games", len(games))

    # ── A: SP features ────────────────────────────────────────────────────────
    games = _add_sp_features(games, sp_log)
    logger.info("SP features added")

    # ── B: Team offense ───────────────────────────────────────────────────────
    games = _add_team_offense_features(games, batting)
    logger.info("Team offense features added")

    # ── C: Bullpen (calendar-window — potentially slow for large datasets) ────
    games = _add_bullpen_features(games, relief_log)
    logger.info("Bullpen features added")

    # ── D: Park factors ───────────────────────────────────────────────────────
    games = _add_park_features(games, parks)
    logger.info("Park features added")

    # ── E: Weather ────────────────────────────────────────────────────────────
    games = _add_weather_features(games, weather)
    logger.info("Weather features added")

    # ── F: Market signal ─────────────────────────────────────────────────────
    games = _add_market_features(games, odds)
    logger.info("Market features added")

    # ── G: Team strength ─────────────────────────────────────────────────────
    games = _add_team_strength_features(games, batting)
    logger.info("Team strength features added")

    # ── Elo ───────────────────────────────────────────────────────────────────
    if include_elo:
        games = _add_elo_features(games, elo)
        logger.info("Elo features added")

    # Filter to requested date range (rolling was computed over all history)
    if start_date:
        games = games[games["date"] >= start_date].copy()

    logger.info("Feature matrix: %d rows x %d cols", len(games), len(games.columns))
    return games.reset_index(drop=True)


# ── Feature column registry ───────────────────────────────────────────────────

#: All feature column names in the output DataFrame (targets excluded).
FEATURE_COLS: list[str] = [
    # A: Starting pitchers
    "home_sp_era_season", "away_sp_era_season",
    "home_sp_fip_season", "away_sp_fip_season",
    "home_sp_k9_season", "away_sp_k9_season",
    "home_sp_bb9_season", "away_sp_bb9_season",
    "home_sp_days_rest", "away_sp_days_rest",
    "home_sp_era_l3", "away_sp_era_l3",
    "sp_fip_combined", "sp_k9_combined", "sp_era_l3_combined",
    # B: Team offense
    "home_ops_10d", "away_ops_10d",
    "home_k_pct_10d", "away_k_pct_10d",
    "home_runs_10d", "away_runs_10d",
    # C: Bullpen
    "home_bullpen_era_7d", "away_bullpen_era_7d",
    "home_bullpen_era_30d", "away_bullpen_era_30d",
    "home_bullpen_ip_7d", "away_bullpen_ip_7d",
    # D: Park
    "park_run_factor", "park_hr_factor", "park_elevation_ft", "is_dome",
    # E: Weather
    "temp_f", "wind_speed_mph",
    "wind_dir_out", "wind_dir_in", "wind_dir_cross_right", "wind_dir_cross_left",
    "precip_prob", "humidity", "is_night_game",
    # F: Market
    "total_line_open", "total_line_close", "line_movement",
    # G: Team strength
    "home_win_pct_10d", "away_win_pct_10d",
    "home_run_diff_pg_season", "away_run_diff_pg_season",
    # Elo
    "elo_home", "elo_away",
]

TARGET_COLS: list[str] = ["home_runs", "away_runs", "total_runs"]
ID_COLS: list[str] = ["game_id", "date", "season", "home_team", "away_team"]


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="Build MLB feature matrix")
    parser.add_argument("--start", help="Start date YYYY-MM-DD")
    parser.add_argument("--end", help="End date YYYY-MM-DD")
    parser.add_argument("--date", help="Single date (overrides --start/--end)")
    parser.add_argument("--output", help="Save to CSV at this path")
    args = parser.parse_args()

    start = args.date or args.start
    end = args.date or args.end

    df = build_features(start_date=start, end_date=end)

    if df.empty:
        print("No data returned")
    else:
        print(f"Feature matrix: {len(df)} rows x {len(df.columns)} cols")
        present = [c for c in FEATURE_COLS if c in df.columns]
        null_pcts = df[present].isna().mean().sort_values(ascending=False)
        print("\nTop features by null %:")
        print(null_pcts.head(15).to_string())

        if args.output:
            df.to_csv(args.output, index=False)
            print(f"\nSaved to {args.output}")
