"""
Unit tests for mlb/features.py.

Every new feature requires:
1. A leakage test — feature at row N must NOT use data from game N
2. A cold-start test — feature must be NaN for the first appearance

Run after every edit to features.py:
    pytest tests/unit/test_features.py -v
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlb.features import (
    _add_sp_features,
    _add_team_offense_features,
    _add_team_strength_features,
    _load_pitcher_game_log,
    _load_team_batting,
    build_features,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture(scope="session")
def sample_df():
    """
    Small feature DataFrame built from 2022 April data.
    Session-scoped since build_features is read-only and slow.
    """
    return build_features(start_date="2022-04-07", end_date="2022-05-15")


@pytest.fixture(scope="session")
def sp_log():
    """Raw pitcher game log from DB (session-scoped, read-only)."""
    from mlb.db import get_conn

    with get_conn() as conn:
        return _load_pitcher_game_log(conn)


@pytest.fixture(scope="session")
def batting_log():
    """Raw team batting log from DB (session-scoped, read-only)."""
    from mlb.db import get_conn

    with get_conn() as conn:
        return _load_team_batting(conn)


# ── Smoke tests ───────────────────────────────────────────────────────────────


def test_build_features_returns_dataframe(sample_df):
    """build_features should return a non-empty DataFrame."""
    assert isinstance(sample_df, pd.DataFrame)
    assert len(sample_df) > 0


def test_build_features_has_required_columns(sample_df):
    """Output must contain id, target, and a selection of feature columns."""
    required = [
        "game_id", "date", "season", "home_team", "away_team",
        "home_runs", "away_runs", "total_runs",
        "home_sp_era_season", "away_sp_era_season",
        "home_sp_fip_season", "away_sp_fip_season",
        "home_ops_10d", "away_ops_10d",
        "park_run_factor", "is_dome",
        "home_win_pct_10d", "away_win_pct_10d",
    ]
    missing = [c for c in required if c not in sample_df.columns]
    assert missing == [], f"Missing columns: {missing}"


def test_build_features_one_row_per_game(sample_df):
    """Each game_id must appear exactly once."""
    assert sample_df["game_id"].nunique() == len(sample_df)


def test_targets_non_negative(sample_df):
    """home_runs and away_runs must be >= 0 for completed games."""
    completed = sample_df.dropna(subset=["home_runs", "away_runs"])
    assert (completed["home_runs"] >= 0).all()
    assert (completed["away_runs"] >= 0).all()


def test_total_runs_equals_sum(sample_df):
    """total_runs must equal home_runs + away_runs for non-null rows."""
    mask = sample_df["total_runs"].notna()
    sub = sample_df[mask]
    computed = sub["home_runs"] + sub["away_runs"]
    np.testing.assert_array_equal(computed.values, sub["total_runs"].values)


# ── SP feature leakage tests ──────────────────────────────────────────────────


def test_sp_era_season_cold_start(sample_df):
    """
    home_sp_era_season must be NaN for a pitcher's first ever DB appearance.
    In April data, every pitcher's season debut should have NaN era_season.
    """
    # First game in dataset for each home team
    first_games = sample_df.sort_values("date").groupby("home_team").first()
    # Not all first games will have NaN (some pitchers may have prior history
    # from other seasons), but none should have season-start ERA without prior data
    # The key invariant: first game of a new pitcher has NaN
    from mlb.db import get_conn

    with get_conn() as conn:
        # Find pitchers whose FIRST EVER appearance in DB is in this date range
        debut_pitchers = conn.execute(
            """
            SELECT p.pitcher_id, MIN(g.date) as first_date, p.team
            FROM pitchers p JOIN games g ON g.game_id = p.game_id
            WHERE p.is_starter = 1
            GROUP BY p.pitcher_id
            HAVING MIN(g.date) >= '2022-04-07' AND MIN(g.date) <= '2022-04-15'
            LIMIT 5
            """
        ).fetchall()

    if not debut_pitchers:
        pytest.skip("No debut pitchers found in April 2022 data")

    from mlb.db import get_conn as _get_conn

    with _get_conn() as conn2:
        for dp in debut_pitchers:
            # Find the first game for this pitcher in our feature df
            first_game = conn2.execute(
                """SELECT g.game_id FROM pitchers p
                   JOIN games g ON g.game_id = p.game_id
                   WHERE p.pitcher_id = ? AND p.is_starter = 1
                   AND g.date = ?""",
                (dp["pitcher_id"], dp["first_date"]),
            ).fetchone()
            if first_game is None:
                continue
            game_row = sample_df[sample_df["game_id"] == first_game["game_id"]]
            if game_row.empty:
                continue
            row = game_row.iloc[0]
            # Check whichever side this pitcher is on
            team = dp["team"]
            if row["home_team"] == team:
                assert pd.isna(row["home_sp_era_season"]), (
                    f"Pitcher debut game should have NaN era_season but got "
                    f"{row['home_sp_era_season']} for {team} on {dp['first_date']}"
                )


def test_sp_era_season_uses_prior_game(sp_log, sample_df):
    """
    home_sp_era_season in features must equal era_season from the pitcher's
    PREVIOUS start, not the current one.
    """
    # Find a pitcher with at least 2 starts in our sample
    sp = sp_log.sort_values(["pitcher_id", "date"])
    sp_counts = sp.groupby("pitcher_id")["game_id"].count()
    multi_start = sp_counts[sp_counts >= 2].index[:5]

    if len(multi_start) == 0:
        pytest.skip("No pitcher with 2+ starts in sample")

    from mlb.db import get_conn

    with get_conn() as conn:
        for pid in multi_start:
            starts = conn.execute(
                """SELECT p.game_id, g.date, p.era_season, p.team
                   FROM pitchers p JOIN games g ON g.game_id = p.game_id
                   WHERE p.pitcher_id = ? AND p.is_starter = 1
                   ORDER BY g.date LIMIT 4""",
                (int(pid),),
            ).fetchall()

            if len(starts) < 2:
                continue

            # Game 2's feature era_season should equal Game 1's DB era_season
            game2 = starts[1]
            expected = starts[0]["era_season"]
            team = game2["team"]

            row = sample_df[sample_df["game_id"] == game2["game_id"]]
            if row.empty:
                continue

            row = row.iloc[0]
            if row["home_team"] == team:
                actual = row["home_sp_era_season"]
            elif row["away_team"] == team:
                actual = row["away_sp_era_season"]
            else:
                continue

            if expected is None:
                assert pd.isna(actual), (
                    f"Expected NaN era_season for game 2 of pitcher {pid} "
                    f"(game 1 era was None) but got {actual}"
                )
            else:
                np.testing.assert_allclose(
                    float(actual),
                    float(expected),
                    rtol=1e-4,
                    err_msg=(
                        f"Leakage: game 2 era_season should equal game 1's DB era "
                        f"({expected}), got {actual} for pitcher {pid}"
                    ),
                )
            break  # One passing pitcher is enough


def test_sp_days_rest_computed(sample_df):
    """days_rest must be positive (>0) when not null, typically 4-10 days."""
    home_rest = sample_df["home_sp_days_rest"].dropna()
    away_rest = sample_df["away_sp_days_rest"].dropna()

    if home_rest.empty or away_rest.empty:
        pytest.skip("No non-null days_rest values in sample")

    assert (home_rest > 0).all(), "days_rest must be positive"
    assert (away_rest > 0).all(), "days_rest must be positive"
    # Sanity: no one pitches with 0 or negative rest
    assert home_rest.min() >= 1


# ── Team offense leakage tests ────────────────────────────────────────────────


def test_team_ops_10d_cold_start(sample_df):
    """
    ops_10d must be NaN for games where a team has fewer than 3 prior games
    (min_periods=3 for rolling).
    """
    # Find teams in their very first 2 games of the dataset
    team_game_count = {}
    sorted_df = sample_df.sort_values("date")

    early_nans_home = 0
    for _, row in sorted_df.iterrows():
        team = row["home_team"]
        team_game_count[team] = team_game_count.get(team, 0) + 1
        if team_game_count[team] <= 2 and pd.isna(row["home_ops_10d"]):
            early_nans_home += 1

    # At least some early games should have NaN ops_10d (cold start)
    assert early_nans_home > 0, (
        "Expected NaN ops_10d for teams in their first 2 games of the season"
    )


def test_team_ops_10d_no_leakage(batting_log, sample_df):
    """
    home_ops_10d at game N must equal the rolling mean of ops for the 10 games
    BEFORE game N (not including game N itself).
    """
    b = batting_log.sort_values(["team", "date", "game_id"])

    # Pick a team with 12+ games in our range
    from mlb.db import get_conn

    with get_conn() as conn:
        teams_with_games = conn.execute(
            """SELECT home_team, COUNT(*) as n FROM games
               WHERE date >= '2022-04-07' AND date <= '2022-05-15'
               AND status = 'Final'
               GROUP BY home_team HAVING n >= 12
               LIMIT 1"""
        ).fetchone()

    if teams_with_games is None:
        pytest.skip("No team with 12+ home games in sample")

    team = teams_with_games["home_team"]
    team_games = sample_df[sample_df["home_team"] == team].sort_values("date")

    # For game at index 11 (12th home game), manually compute expected ops_10d
    if len(team_games) < 12:
        pytest.skip(f"Team {team} has fewer than 12 home games in sample")

    target_idx = 11
    target_row = team_games.iloc[target_idx]
    actual = target_row["home_ops_10d"]

    if pd.isna(actual):
        pytest.skip(f"ops_10d is NaN for {team} at game {target_idx}")

    # Prior ops: all ARI games BEFORE the target game (date-strict, excludes target)
    target_date = target_row["date"]
    team_batting = b[b["team"] == team].sort_values(["date", "game_id"])
    prior_ops = team_batting[team_batting["date"] < target_date]["ops"].dropna()

    if len(prior_ops) < 3:
        pytest.skip("Not enough prior batting data")

    expected = float(prior_ops.tail(10).mean())
    np.testing.assert_allclose(
        float(actual),
        expected,
        rtol=1e-4,
        err_msg=f"ops_10d leakage: {team} game {target_idx}",
    )


# ── Team strength leakage tests ───────────────────────────────────────────────


def test_win_pct_cold_start(sample_df):
    """win_pct_10d must be NaN for teams in their first 2 games (min_periods=3)."""
    sorted_df = sample_df.sort_values("date")
    home_counts: dict[str, int] = {}

    early_nans = 0
    for _, row in sorted_df.iterrows():
        team = row["home_team"]
        home_counts[team] = home_counts.get(team, 0) + 1
        if home_counts[team] <= 2 and pd.isna(row["home_win_pct_10d"]):
            early_nans += 1

    assert early_nans > 0, "Expected NaN win_pct_10d for early season games"


def test_win_pct_range(sample_df):
    """win_pct_10d must be in [0, 1]."""
    home_valid = sample_df["home_win_pct_10d"].dropna()
    away_valid = sample_df["away_win_pct_10d"].dropna()

    if home_valid.empty:
        pytest.skip("No non-null win_pct values")

    assert (home_valid >= 0).all() and (home_valid <= 1).all()
    assert (away_valid >= 0).all() and (away_valid <= 1).all()


# ── Park feature tests ────────────────────────────────────────────────────────


def test_park_run_factor_populated(sample_df):
    """park_run_factor must be non-null for all games (static join)."""
    null_count = sample_df["park_run_factor"].isna().sum()
    assert null_count == 0, f"{null_count} games missing park_run_factor"


def test_coors_field_has_highest_run_factor(sample_df):
    """Coors Field (COL) should have the highest park_run_factor."""
    col_games = sample_df[sample_df["home_team"] == "COL"]
    other_games = sample_df[sample_df["home_team"] != "COL"]

    if col_games.empty or other_games.empty:
        pytest.skip("No COL home games in sample")

    col_factor = col_games["park_run_factor"].iloc[0]
    max_other = other_games["park_run_factor"].max()
    assert col_factor > max_other, (
        f"Coors factor ({col_factor}) should exceed all others ({max_other})"
    )


# ── Weather feature tests ─────────────────────────────────────────────────────


def test_wind_dir_one_hot_valid(sample_df):
    """Wind direction one-hot columns must be 0 or 1, mutually exclusive."""
    wind_cols = ["wind_dir_out", "wind_dir_in", "wind_dir_cross_right", "wind_dir_cross_left"]
    present = [c for c in wind_cols if c in sample_df.columns]

    if not present:
        pytest.skip("No wind_dir columns in sample")

    non_null_mask = sample_df[present[0]].notna()
    if non_null_mask.sum() == 0:
        pytest.skip("No non-null wind direction rows")

    sub = sample_df.loc[non_null_mask, present]

    # Values must be 0 or 1
    assert sub.isin([0, 1]).all().all(), "Wind direction indicators must be 0 or 1"

    # Exactly one direction set per row (for non-dome outdoor games with wind data)
    row_sums = sub.sum(axis=1)
    # Allow 0 (dome / no wind data) or 1 (one direction active)
    assert (row_sums <= 1).all(), "At most one wind direction can be active per game"


# ── Market feature tests ──────────────────────────────────────────────────────


def test_line_movement_computed(sample_df):
    """line_movement = total_line_close - total_line_open for non-null rows."""
    mask = sample_df["total_line_open"].notna() & sample_df["total_line_close"].notna()
    sub = sample_df[mask]

    if sub.empty:
        pytest.skip("No games with both open and close lines")

    expected = sub["total_line_close"] - sub["total_line_open"]
    np.testing.assert_allclose(
        sub["line_movement"].values,
        expected.values,
        rtol=1e-6,
        err_msg="line_movement must equal total_line_close - total_line_open",
    )


def test_total_lines_reasonable(sample_df):
    """Opening total lines should be in realistic MLB range (4.0 to 20.0)."""
    lines = sample_df["total_line_open"].dropna()
    if lines.empty:
        pytest.skip("No total_line_open data")

    assert (lines >= 4.0).all(), f"Min line too low: {lines.min()}"
    assert (lines <= 20.0).all(), f"Max line too high: {lines.max()}"
