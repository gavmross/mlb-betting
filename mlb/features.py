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
               g.f5_home_score,
               g.f5_away_score,
               g.f5_total_runs,
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


def _load_kalshi_lines(conn) -> pd.DataFrame:
    """
    Load the most-recent Kalshi full-game and F5 mid-prices per game.

    Returns a DataFrame with one row per game_id containing:
        kalshi_fullgame_line  : integer line from the full-game market
        kalshi_f5_line        : integer line from the F5 market
        f5_ratio              : kalshi_f5_line / kalshi_fullgame_line
                                (1.0 if full-game only or no F5 market)

    This cross-signal captures mispricing between F5 and full-game markets:
    if f5_ratio >> 0.5, the market expects a strong back-half; < 0.5 implies
    a weak bullpen or late-game run environment.
    """
    sql = """
        WITH latest AS (
            SELECT game_id, market_type, line,
                   ROW_NUMBER() OVER (
                       PARTITION BY game_id, market_type
                       ORDER BY snapshot_ts DESC
                   ) AS rn
            FROM kalshi_markets
            WHERE game_id IS NOT NULL
              AND market_type IN ('total_over', 'f5_total_over')
        )
        SELECT game_id,
               MAX(CASE WHEN market_type='total_over'    THEN line END) AS kalshi_fullgame_line,
               MAX(CASE WHEN market_type='f5_total_over' THEN line END) AS kalshi_f5_line
        FROM latest
        WHERE rn = 1
        GROUP BY game_id
    """
    rows = conn.execute(sql).fetchall()
    df = pd.DataFrame([dict(r) for r in rows])
    if df.empty:
        return df
    df["f5_ratio"] = df["kalshi_f5_line"] / df["kalshi_fullgame_line"].replace(0, float("nan"))
    return df


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

    # Raw ER per start over last 5 starts — direct runs-allowed momentum.
    # Complements era_l3 (which normalises by IP): a 2-inning 5-ER blowout
    # has the same ERA as a 6-inning 15-ER game but very different total-run impact.
    sp["sp_er_pg_l5"] = sp.groupby("pitcher_id", group_keys=False).apply(
        lambda g: g["er_lag"].rolling(5, min_periods=2).mean()
    )

    # Clip extreme ERA/FIP values caused by early-season tiny-IP samples.
    # 13.5 ≈ 3× league-average ERA; preserves signal while suppressing noise.
    ERA_CLIP = 13.5
    FIP_CLIP_LO, FIP_CLIP_HI = -2.0, 13.5
    ER_PG_CLIP = 8.0  # ~3× league-average ER/start; rolling avg rarely exceeds this
    sp["era_season_lag"] = sp["era_season_lag"].clip(upper=ERA_CLIP)
    sp["fip_season_lag"] = sp["fip_season_lag"].clip(lower=FIP_CLIP_LO, upper=FIP_CLIP_HI)
    sp["sp_era_l3"] = sp["sp_era_l3"].clip(upper=ERA_CLIP)
    sp["sp_er_pg_l5"] = sp["sp_er_pg_l5"].clip(upper=ER_PG_CLIP)

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
        "sp_era_l3", "sp_er_pg_l5", "days_rest_computed",
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
    games["sp_er_pg_l5_combined"] = games["home_sp_er_pg_l5"] + games["away_sp_er_pg_l5"]

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
        "sp_er_pg_l5": f"{side}_sp_er_pg_l5",
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

    # run_diff_pg: cumulative season run differential / games played so far
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


def _add_market_features(
    games: pd.DataFrame,
    odds: pd.DataFrame,
    kalshi_lines: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Join pre-game market totals lines and Kalshi cross-market signals.

    These are pre-game signals — no shift needed.
    line_movement = close - open (sharp money proxy; sharp bets move lines).

    Features added:
        total_line_open, total_line_close, line_movement
        kalshi_fullgame_line : Kalshi integer full-game line
        kalshi_f5_line       : Kalshi integer F5 line
        f5_ratio             : F5 line / full-game line (mispricing signal)

    Parameters
    ----------
    games : pd.DataFrame
    odds : pd.DataFrame
    kalshi_lines : pd.DataFrame or None
        Output of ``_load_kalshi_lines()``.

    Returns
    -------
    pd.DataFrame
    """
    if not odds.empty:
        o = odds.copy()
        o["line_movement"] = o["total_line_close"] - o["total_line_open"]
        games = games.merge(o, on="game_id", how="left")

    if kalshi_lines is not None and not kalshi_lines.empty:
        games = games.merge(
            kalshi_lines[["game_id", "kalshi_fullgame_line", "kalshi_f5_line", "f5_ratio"]],
            on="game_id",
            how="left",
        )

    return games


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
        for _side, my_team, opp_runs, my_runs in [
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


# ── Statcast xERA (prior-season) ─────────────────────────────────────────────


# ── Public entry point ────────────────────────────────────────────────────────


def build_features(
    start_date: str | None = None,
    end_date: str | None = None,
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
        kalshi_lines = _load_kalshi_lines(conn)
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
    games = _add_market_features(games, odds, kalshi_lines)
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


# ── Predict-mode helpers ─────────────────────────────────────────────────────


def _load_upcoming_games(conn, date: str) -> pd.DataFrame:
    """Load scheduled/Preview games for a specific date (no scores yet)."""
    sql = """
        SELECT g.game_id, g.date, g.season,
               g.home_team, g.away_team,
               NULL as home_runs, NULL as away_runs, NULL as total_runs,
               NULL as f5_home_score, NULL as f5_away_score, NULL as f5_total_runs,
               g.game_time_et
        FROM games g
        WHERE g.date = ? AND g.status IN ('Preview', 'Scheduled', 'Pre-Game', 'Warmup', 'Live')
        ORDER BY g.game_time_et
    """
    rows = conn.execute(sql, (date,)).fetchall()
    return pd.DataFrame([dict(r) for r in rows])


def _fetch_probable_starters(date: str) -> pd.DataFrame:
    """
    Fetch probable starting pitcher IDs for all games on date via MLB Stats API.

    Parameters
    ----------
    date : str
        Game date YYYY-MM-DD.

    Returns
    -------
    pd.DataFrame
        Columns: game_id (str), side ('home'|'away'), pitcher_id (int).
        Empty if the API call fails or no probables are announced.
    """
    import requests

    url = "https://statsapi.mlb.com/api/v1/schedule"
    params = {"sportId": 1, "date": date, "hydrate": "probablePitcher"}
    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        logger.warning("Could not fetch probable pitchers for %s: %s", date, exc)
        return pd.DataFrame(columns=["game_id", "side", "pitcher_id"])

    rows = []
    for date_entry in data.get("dates", []):
        for game in date_entry.get("games", []):
            game_pk = str(game.get("gamePk", ""))
            for side in ("home", "away"):
                pp = game.get("teams", {}).get(side, {}).get("probablePitcher", {})
                if pp and pp.get("id"):
                    rows.append(
                        {"game_id": game_pk, "side": side, "pitcher_id": int(pp["id"])}
                    )
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=["game_id", "side", "pitcher_id"])


def _make_synthetic_sp_rows(
    upcoming: pd.DataFrame, probable: pd.DataFrame
) -> pd.DataFrame:
    """
    Create synthetic sp_log rows for upcoming games so shift(1) can pull
    the probable starter's most-recent season stats into the feature.

    All per-game stat columns (er, ip, era_season, …) are NaN — the existing
    _add_sp_features shift(1) logic will pick up the PREVIOUS row's values,
    which are the starter's actual last-start stats.

    Parameters
    ----------
    upcoming : pd.DataFrame
        Output of _load_upcoming_games().
    probable : pd.DataFrame
        Columns: game_id, side, pitcher_id.  Side resolved to team abbreviation
        by the caller.

    Returns
    -------
    pd.DataFrame
        Rows in the same schema as sp_log, to be appended before _add_sp_features.
    """
    if probable.empty or upcoming.empty:
        return pd.DataFrame()

    # Map side → team abbreviation using upcoming game data
    home_sides = (
        upcoming[["game_id", "home_team"]].rename(columns={"home_team": "team"}).assign(side="home")
    )
    away_sides = (
        upcoming[["game_id", "away_team"]].rename(columns={"away_team": "team"}).assign(side="away")
    )
    game_sides = pd.concat([home_sides, away_sides])
    probable = (
        probable.merge(game_sides, on=["game_id", "side"], how="left").dropna(subset=["team"])
    )
    if probable.empty:
        return pd.DataFrame()

    date_map = upcoming.set_index("game_id")["date"].to_dict()
    rows = []
    for _, prob in probable.iterrows():
        gid = prob["game_id"]
        if gid not in date_map:
            continue
        rows.append(
            {
                "game_id": gid,
                "pitcher_id": int(prob["pitcher_id"]),
                "pitcher_name": f"probable_{int(prob['pitcher_id'])}",
                "team": prob["team"],
                "ip": np.nan,
                "er": np.nan,
                "era_season": np.nan,
                "fip_season": np.nan,
                "k9_season": np.nan,
                "bb9_season": np.nan,
                "hr9_season": np.nan,
                "days_rest": np.nan,
                "date": date_map[gid],
            }
        )
    return pd.DataFrame(rows)


def build_predict_features(
    date: str,
    db_path: str = "data/mlb.db",
    include_elo: bool = True,
) -> pd.DataFrame:
    """
    Build features for upcoming (not yet played) games on date.

    Uses the same no-leakage rolling pipeline as build_features(), but adds
    synthetic sp_log rows for probable starters so the model gets real pitcher
    stats (from their last start) instead of NaN imputation.

    Parameters
    ----------
    date : str
        Target date YYYY-MM-DD (games must be in 'Preview'/'Scheduled' status).
    db_path : str
    include_elo : bool

    Returns
    -------
    pd.DataFrame
        One row per upcoming game on date.  Target columns (home_runs,
        away_runs, total_runs) will be NaN — they haven't been played yet.
        Empty DataFrame if no scheduled games found.
    """
    logger.info("Building predict features for %s", date)

    with get_conn(db_path) as conn:
        history = _load_games(conn, start_date=None, end_date=None)
        upcoming = _load_upcoming_games(conn, date)
        sp_log = _load_pitcher_game_log(conn)
        relief_log = _load_relief_log(conn)
        batting = _load_team_batting(conn)
        parks = _load_park_factors(conn)
        weather = _load_weather(conn)
        odds = _load_odds(conn)
        kalshi_lines = _load_kalshi_lines(conn)
        elo = _load_elo(conn) if include_elo else pd.DataFrame()

    if upcoming.empty:
        logger.warning("No scheduled games found for %s", date)
        return pd.DataFrame()

    logger.info("Found %d upcoming games for %s", len(upcoming), date)

    # Fetch probable starters and create synthetic sp_log rows
    probable = _fetch_probable_starters(date)
    synthetic = _make_synthetic_sp_rows(upcoming, probable)
    if not synthetic.empty:
        logger.info("Added %d probable-starter rows to sp_log", len(synthetic))
        sp_log = pd.concat([sp_log, synthetic], ignore_index=True)

    # Combine history + upcoming into a single games frame for rolling.
    # Keep date as string (YYYY-MM-DD sorts correctly lexicographically).
    # Upcoming games have NaN targets but are otherwise identical in schema.
    games = pd.concat([history, upcoming], ignore_index=True)
    games = games.sort_values(["date", "game_id"]).reset_index(drop=True)

    # Run the full feature pipeline (computes rolling over all data)
    games = _add_sp_features(games, sp_log)
    games = _add_team_offense_features(games, batting)
    games = _add_bullpen_features(games, relief_log)
    games = _add_park_features(games, parks)
    games = _add_weather_features(games, weather)
    games = _add_market_features(games, odds, kalshi_lines)
    games = _add_team_strength_features(games, batting)
    if include_elo:
        games = _add_elo_features(games, elo)

    # Return only the target date's upcoming games
    target_ids = set(upcoming["game_id"])
    result = games[games["game_id"].isin(target_ids)].copy()
    logger.info("Predict feature matrix: %d rows x %d cols", len(result), len(result.columns))
    return result.reset_index(drop=True)


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
    "home_sp_er_pg_l5", "away_sp_er_pg_l5",
    "sp_fip_combined", "sp_k9_combined", "sp_era_l3_combined", "sp_er_pg_l5_combined",
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
    # precip_prob excluded — always NULL from archive API
    "humidity", "is_night_game",
    # F: Market features excluded — closing line is the target benchmark, not a feature.
    # Using total_line_close as a predictor of (total_runs > total_line_close) is circular:
    # the model would learn market consensus, not independent signal. Columns are still
    # computed in build_features() and available for EV calculation at inference time.
    # F2: Kalshi cross-market features excluded from training — 100% NULL in 2021-2024
    # (Kalshi MLB data starts 2026-03-31). Columns still computed in build_features()
    # and will populate for 2026+ live predictions; keep out of model to avoid noise.
    # G: Team strength
    "home_win_pct_10d", "away_win_pct_10d",
    "home_run_diff_pg_season", "away_run_diff_pg_season",
    # Elo
    "elo_home", "elo_away",
]

TARGET_COLS: list[str] = [
    "home_runs", "away_runs", "total_runs",
    "f5_home_score", "f5_away_score", "f5_total_runs",
]
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
