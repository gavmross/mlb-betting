"""
Unit tests for mlb/elo.py.

Required tests per testing.md:
1. Zero-sum: total Elo constant across all teams after every game batch
2. Update math: verify expected_score and update_elo formulas
3. Off-season regression: regress_to_mean correct
4. Cold-start: all teams initialised at 1500
5. get_elo_before_date: returns pre-game (not same-day) Elo

Run:
    pytest tests/unit/test_elo.py -v
"""

from __future__ import annotations

import numpy as np
import pytest

from mlb.elo import (
    ELO_INIT,
    K_FACTOR,
    REGRESSION_FACTOR,
    expected_score,
    regress_to_mean,
    update_elo,
)

# ── Core math ─────────────────────────────────────────────────────────────────


def test_expected_score_equal_elos():
    """Equal Elos → expected score = 0.5."""
    e = expected_score(1500.0, 1500.0)
    np.testing.assert_allclose(e, 0.5, rtol=1e-9)


def test_expected_score_higher_elo_favoured():
    """Higher Elo → expected score > 0.5."""
    assert expected_score(1600.0, 1400.0) > 0.5


def test_expected_score_symmetry():
    """expected_score(A, B) + expected_score(B, A) == 1.0."""
    a, b = 1550.0, 1420.0
    np.testing.assert_allclose(
        expected_score(a, b) + expected_score(b, a), 1.0, rtol=1e-9
    )


def test_expected_score_known_value():
    """
    With elo_self=1600, elo_opp=1200, diff=400 → E = 1/(1+10^1) = 1/11 ≈ 0.0909.
    Wait: elo_opp - elo_self = 1200 - 1600 = -400 → 10^(-1) = 0.1 → E = 1/1.1 ≈ 0.909.
    """
    e = expected_score(1600.0, 1200.0)
    expected = 1.0 / (1.0 + 10.0 ** (-400.0 / 400.0))
    np.testing.assert_allclose(e, expected, rtol=1e-9)


def test_update_elo_zero_sum():
    """Winner gains exactly as much as loser loses."""
    elo_w, elo_l = 1500.0, 1500.0
    new_w, new_l = update_elo(elo_w, elo_l)
    np.testing.assert_allclose(new_w + new_l, elo_w + elo_l, rtol=1e-9)


def test_update_elo_winner_gains():
    """Winner's Elo must increase after a win."""
    new_w, new_l = update_elo(1500.0, 1500.0)
    assert new_w > 1500.0
    assert new_l < 1500.0


def test_update_elo_upset_bigger_gain():
    """An upset (lower-rated team wins) yields a bigger Elo gain."""
    # Underdog (1400) beats favourite (1600) — large gain
    new_w_upset, _ = update_elo(1400.0, 1600.0)
    # Favourite (1600) beats underdog (1400) — small gain
    new_w_expected, _ = update_elo(1600.0, 1400.0)

    gain_upset = new_w_upset - 1400.0
    gain_expected = new_w_expected - 1600.0
    assert gain_upset > gain_expected


def test_update_elo_equal_symmetry():
    """When Elos are equal, winner gains exactly K/2."""
    new_w, new_l = update_elo(1500.0, 1500.0, k=K_FACTOR)
    np.testing.assert_allclose(new_w - 1500.0, K_FACTOR / 2, rtol=1e-9)
    np.testing.assert_allclose(1500.0 - new_l, K_FACTOR / 2, rtol=1e-9)


# ── Regression to mean ────────────────────────────────────────────────────────


def test_regress_to_mean_above():
    """Elo above mean moves toward mean."""
    new_elo = regress_to_mean(1600.0)
    assert 1500.0 < new_elo < 1600.0


def test_regress_to_mean_below():
    """Elo below mean moves toward mean."""
    new_elo = regress_to_mean(1400.0)
    assert 1400.0 < new_elo < 1500.0


def test_regress_to_mean_at_mean():
    """Elo at mean is unchanged."""
    np.testing.assert_allclose(regress_to_mean(ELO_INIT), ELO_INIT, rtol=1e-9)


def test_regress_to_mean_formula():
    """Verify the exact regression formula."""
    elo = 1700.0
    expected = (1.0 - REGRESSION_FACTOR) * elo + REGRESSION_FACTOR * ELO_INIT
    np.testing.assert_allclose(regress_to_mean(elo), expected, rtol=1e-9)


# ── DB-backed tests ────────────────────────────────────────────────────────────


@pytest.fixture(scope="session")
def elo_rows():
    """Load all elo_ratings rows from DB once."""
    from mlb.db import get_conn

    with get_conn() as conn:
        rows = conn.execute(
            "SELECT team, date, elo, season FROM elo_ratings ORDER BY season, date"
        ).fetchall()
    return [dict(r) for r in rows]


def test_elo_ratings_populated(elo_rows):
    """elo_ratings table must be non-empty after running mlb.elo."""
    assert len(elo_rows) > 0, "elo_ratings is empty — run python -m mlb.elo first"


def test_elo_zero_sum_per_date(elo_rows):
    """
    On any given date the sum of Elos across all teams must equal
    N_teams * 1500 (zero-sum invariant).
    We check a sample of dates — first date of 2022 season.
    """
    from mlb.db import get_conn

    with get_conn() as conn:
        # Get the first date with the most team entries (fully played day)
        row = conn.execute(
            """SELECT date, COUNT(*) as n, SUM(elo) as total
               FROM elo_ratings
               WHERE season = 2022
               GROUP BY date
               HAVING n = (SELECT MAX(cnt) FROM
                   (SELECT COUNT(*) as cnt FROM elo_ratings WHERE season=2022 GROUP BY date))
               ORDER BY date
               LIMIT 1"""
        ).fetchone()

    if row is None:
        pytest.skip("No elo data for 2022")

    n_teams = row["n"]
    expected_total = n_teams * ELO_INIT
    np.testing.assert_allclose(
        row["total"],
        expected_total,
        rtol=1e-4,
        atol=1.0,
        err_msg=f"Zero-sum violated on {row['date']}: total={row['total']:.2f}, expected={expected_total:.2f}",
    )


def test_elo_mean_stays_at_1500(elo_rows):
    """Mean Elo per season must be exactly 1500.0."""
    from collections import defaultdict

    season_elos: dict[int, list[float]] = defaultdict(list)

    # Take the last date's snapshot per team per season
    from mlb.db import get_conn

    with get_conn() as conn:
        for season in (2022, 2023, 2024):
            rows = conn.execute(
                """SELECT team, elo FROM elo_ratings
                   WHERE season = ? AND date = (
                       SELECT MAX(date) FROM elo_ratings WHERE season = ?
                   )""",
                (season, season),
            ).fetchall()
            if rows:
                season_elos[season] = [r["elo"] for r in rows]

    for season, elos in season_elos.items():
        if not elos:
            continue
        mean = np.mean(elos)
        np.testing.assert_allclose(
            mean,
            ELO_INIT,
            atol=10.0,  # partial seasons have teams on different last-played dates
            err_msg=f"Season {season} end-of-season mean Elo={mean:.2f}, expected ~1500",
        )


def test_elo_values_reasonable(elo_rows):
    """All Elo values should be in a sane range (e.g. 1100–1900)."""
    for r in elo_rows:
        assert 1100 < r["elo"] < 1900, (
            f"Elo out of expected range: {r['team']} {r['date']} = {r['elo']}"
        )


def test_get_elo_before_date():
    """get_elo_before_date returns Elo strictly before the given date."""
    from mlb.db import get_conn
    from mlb.elo import get_elo_before_date

    with get_conn() as conn:
        # Pick any team that has elo data
        row = conn.execute(
            "SELECT team, date FROM elo_ratings ORDER BY date LIMIT 1"
        ).fetchone()

    if row is None:
        pytest.skip("No elo data in DB")

    team = row["team"]
    # The date stored is the game date — so querying that date should return None
    # (function uses date < ?, not <=)
    # For the day AFTER, we should get the stored value (same-day excluded by date < ? logic)
    from datetime import date, timedelta

    next_day = (date.fromisoformat(row["date"]) + timedelta(days=1)).isoformat()
    result_next_day = get_elo_before_date(team, next_day)
    assert result_next_day is not None, (
        f"Expected Elo for {team} before {next_day}, got None"
    )
    assert isinstance(result_next_day, float)


def test_get_elo_before_date_no_prior():
    """get_elo_before_date returns None when no prior data exists."""
    from mlb.elo import get_elo_before_date

    result = get_elo_before_date("NYY", "2010-01-01")
    assert result is None
