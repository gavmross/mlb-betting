"""
MLB betting engine — EV, Kelly, CLV, daily pricer, and backtest simulator.

The betting engine sits downstream of the model (lambda_home, lambda_away) and
calibration (over_prob).  It prices each game against the Kalshi mid-price (live)
or SBR Pinnacle closing line (backtest), applies position filters, and writes
bet recommendations back to the predictions table.

Public API
----------
compute_ev(over_prob, kalshi_over_price)
    Expected value for over and under sides.

kelly_bet(win_prob, kalshi_price, kelly_mult, max_pct)
    Fractional Kelly bet size as a fraction of bankroll.

compute_clv(entry, closing, side)
    Closing line value — positive means we beat the close.

american_to_price(odds)
    Convert American moneyline odds to a decimal implied probability.

get_consensus(kalshi_mid, poly_mid)
    Cross-market pricing signal.

run_daily(date, db_path)
    Price today's games and write bet recommendations to predictions table.

simulate(start, end, ...)
    Walk-forward betting simulation over historical data.

Usage
-----
    python -m mlb.betting --date today
    python -m mlb.betting --simulate --start 2021-04-01 --end 2024-10-01
    python -m mlb.betting --simulate --min-edge 0.02 --output sensitivity_002.csv
"""

from __future__ import annotations

import argparse
import logging
from datetime import date as date_type
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from mlb.calibration import p_over_negbinom, p_over_poisson
from mlb.db import get_conn

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────

MIN_EDGE: float = 0.03  # minimum EV to place a bet (vs Kalshi mid)
KELLY_MULT: float = 0.25  # fractional Kelly multiplier — never full Kelly
MAX_BET_PCT: float = 0.05  # hard cap: 5 % of bankroll per game
MAX_POSITIONS: int = 3  # maximum simultaneous open positions
MIN_OPEN_INTEREST: float = 1000.0  # minimum Kalshi open interest ($)

# ── Under-bias filters ─────────────────────────────────────────────────────────
# Derived from EDA on 2021-2025 SBR data; both survive 2023-2025 OOS validation.
# day_k9_park: day games in pitcher-friendly parks with high-K starters AND good
#   recent form hit under 56.5% of the time (p<0.01); K9 >= 14 and era_l3 <= 4.0
#   both use pre-game shift(1) values. era_l3 guard fixes 2022 lockout degradation.
# high_line: when the posted total >= 11.0, the actual game goes under 57.1% of
#   the time (p=0.030, n=413); likely regression-to-mean in outlier high-run setups.
UNDER_PARKS: frozenset[str] = frozenset({"SFG", "CLE", "TEX", "CIN", "CHW", "SDP", "SEA", "DET"})
K9_COMBINED_MIN: float = 14.0  # combined K/9 threshold for day_k9_park filter
ERA_L3_MAX: float = 4.0  # max era_l3 per SP for day_k9_park (pre-game, both SPs)
HIGH_LINE_MIN: float = 11.0  # closing total threshold for high_line filter

# ── Over-bias filters ──────────────────────────────────────────────────────────
# hot_wind_out: outdoor games with temp >= 80°F and wind blowing out at 10-14 mph
#   hit over 55.9% of the time (p=0.002, n=288, 2021-2025); OOS 2023-25 = 60.0%.
#   Lines are set days in advance and underweight same-day wind/heat conditions.
#   Wind speed cap at 15 mph — above that the effect reverses (disrupts pitchers).
HOT_TEMP_MIN: float = 80.0  # minimum temperature (°F) for hot_wind_out filter
WIND_SPEED_MIN: float = 10.0  # minimum wind speed (mph) for hot_wind_out filter
WIND_SPEED_MAX: float = 15.0  # wind speed cap — above disrupts pitching command
SUMMER_MONTHS: frozenset[int] = frozenset({7, 8, 9})  # July-Sep restriction for summer_hot_wind_out

# Per-filter EDA win rates used as Kelly input when sizing='quarter_kelly'.
# Derived from 2021-2025 SBR data; treat as the "prior" you'd bring to each bet.
_FILTER_WIN_PROBS: dict[str, float] = {
    "day_k9_park": 0.564,
    "high_line": 0.575,
    "hot_wind_out": 0.560,
    "summer_hot_wind_out": 0.631,
}


# ── Core math ─────────────────────────────────────────────────────────────────


def compute_ev(
    over_prob: float,
    kalshi_over_price: float,
    min_edge: float = MIN_EDGE,
) -> dict[str, Any]:
    """
    Compute expected value for over and under sides of a totals bet.

    The Kalshi price is treated as the cost to buy a $1 YES contract.
    Payout on a WIN = $1.  Net profit = $1 - price.  Net loss = -price.

    Parameters
    ----------
    over_prob : float
        Model's P(total > line), from Poisson or NegBinom convolution.
    kalshi_over_price : float
        Kalshi YES mid-price for the over contract, in dollars (0–1).
    min_edge : float
        Minimum EV threshold to recommend a bet.

    Returns
    -------
    dict
        Keys: ev_over, ev_under, edge, bet_side.
    """
    ev_over = over_prob * (1.0 - kalshi_over_price) - (1.0 - over_prob) * kalshi_over_price
    ev_under = (1.0 - over_prob) * kalshi_over_price - over_prob * (1.0 - kalshi_over_price)
    edge = over_prob - kalshi_over_price

    if ev_over > min_edge:
        bet_side = "OVER"
    elif ev_under > min_edge:
        bet_side = "UNDER"
    else:
        bet_side = "PASS"

    return {
        "ev_over": round(ev_over, 6),
        "ev_under": round(ev_under, 6),
        "edge": round(edge, 6),
        "bet_side": bet_side,
    }


def kelly_bet(
    win_prob: float,
    kalshi_price: float,
    kelly_mult: float = KELLY_MULT,
    max_pct: float = MAX_BET_PCT,
) -> float:
    """
    Fractional Kelly bet size as a fraction of bankroll.

    Formula: f* = (p·b - q) / b  where b = (1/price) - 1, q = 1 - p.
    Applied at kelly_mult fraction.  Capped at max_pct regardless.

    Parameters
    ----------
    win_prob : float
        P(bet wins) — over_prob if OVER, (1 - over_prob) if UNDER.
    kalshi_price : float
        Kalshi mid-price for the winning side (0–1).
    kelly_mult : float
        Fractional Kelly multiplier. Default 0.25x.
    max_pct : float
        Hard cap as fraction of bankroll. Default 0.05 (5 %).

    Returns
    -------
    float
        Fraction of bankroll to bet (0 – max_pct).
    """
    if kalshi_price <= 0.0 or kalshi_price >= 1.0:
        return 0.0
    b = (1.0 / kalshi_price) - 1.0
    q = 1.0 - win_prob
    full_kelly = max(0.0, (win_prob * b - q) / b)
    return min(full_kelly * kelly_mult, max_pct)


def compute_clv(entry: float, closing: float, side: str) -> float:
    """
    Closing line value — positive means we got a better price than the close.

    For OVER:  CLV = closing_price - entry_price
               (higher closing price = market moved against us = we had edge)
    For UNDER: CLV = entry_price - (1 - closing_price)
               (lower closing price for over = under got more expensive = we had edge)

    Parameters
    ----------
    entry : float
        Price at which we placed the bet (0–1).
    closing : float
        Closing Kalshi price for the OVER side (0–1).
    side : str
        'OVER' or 'UNDER'.

    Returns
    -------
    float
        CLV in price units. Positive = beat the close.
    """
    if side == "OVER":
        return round(closing - entry, 6)
    elif side == "UNDER":
        return round(entry - (1.0 - closing), 6)
    return 0.0


def american_to_price(odds: int) -> float:
    """
    Convert American moneyline odds to an implied probability (Kalshi-equivalent).

    Parameters
    ----------
    odds : int
        American odds, e.g. -110, +120, -115.

    Returns
    -------
    float
        Implied probability in (0, 1).
    """
    if odds < 0:
        return abs(odds) / (abs(odds) + 100.0)
    else:
        return 100.0 / (odds + 100.0)


def devig_prices(over_price_raw: float, under_price_raw: float) -> tuple[float, float]:
    """
    Remove the bookmaker vig to recover fair implied probabilities.

    Uses the additive method (proportional devig): divide each implied
    probability by the total overround so they sum to 1.

    Parameters
    ----------
    over_price_raw : float
        Raw implied probability for the over (includes vig).
    under_price_raw : float
        Raw implied probability for the under (includes vig).

    Returns
    -------
    tuple[float, float]
        (fair_over, fair_under) summing to 1.0.
    """
    total = over_price_raw + under_price_raw
    if total <= 0:
        return 0.5, 0.5
    return over_price_raw / total, under_price_raw / total


def get_consensus(kalshi_mid: float, poly_mid: float) -> dict[str, Any]:
    """
    Cross-market pricing signal between Kalshi and Polymarket.

    A spread > 0.04 suggests one platform is lagging behind the other.
    When Kalshi is cheaper for the over (kalshi_mid < poly_mid), it is a
    more favourable entry for an OVER bet.

    Parameters
    ----------
    kalshi_mid : float
        Kalshi OVER mid-price (0–1).
    poly_mid : float
        Polymarket OVER mid-price (0–1).

    Returns
    -------
    dict
        Keys: spread, kalshi_is_cheap_for_over, market_avg.
    """
    spread = abs(kalshi_mid - poly_mid)
    return {
        "spread": round(spread, 4),
        "kalshi_is_cheap_for_over": kalshi_mid < poly_mid,
        "market_avg": round((kalshi_mid + poly_mid) / 2.0, 4),
    }


# ── Position filter ────────────────────────────────────────────────────────────


def passes_filters(
    bet_side: str,
    open_interest: float | None,
    current_positions: int,
    min_open_interest: float = MIN_OPEN_INTEREST,
    max_positions: int = MAX_POSITIONS,
) -> tuple[bool, str]:
    """
    Apply pre-trade position filters.

    Parameters
    ----------
    bet_side : str
        'OVER', 'UNDER', or 'PASS'.
    open_interest : float or None
        Kalshi open interest in dollars.
    current_positions : int
        Number of already-open bets today.
    min_open_interest : float
    max_positions : int

    Returns
    -------
    tuple[bool, str]
        (passes, reason_if_rejected)
    """
    if bet_side == "PASS":
        return False, "no edge"
    if current_positions >= max_positions:
        return False, f"max {max_positions} positions reached"
    if open_interest is not None and open_interest < min_open_interest:
        return False, f"open_interest {open_interest:.0f} < {min_open_interest:.0f}"
    return True, ""


# ── Daily pricer ───────────────────────────────────────────────────────────────


def run_daily(
    date: str | None = None,
    db_path: str = "data/mlb.db",
    model_name: str = "gbr",
    use_negbinom: bool = True,
    target: str = "fullgame",
) -> pd.DataFrame:
    """
    Price today's games and write bet recommendations to the predictions table.

    Reads lambda_home, lambda_away from the predictions table (written by
    mlb/model.py --predict), fetches live Kalshi mid-prices (joined by game_id),
    recomputes over_prob with the current line, and writes back:
    edge, ev, kelly_fraction, recommended_bet, bet_side.

    Parameters
    ----------
    date : str or None
        ISO date (YYYY-MM-DD). Defaults to today.
    db_path : str
    model_name : str
        Model variant to price (matches model_name in predictions table).
    use_negbinom : bool
        If True, use NegBinom convolution when dispersion_alpha is available.
    target : str
        ``'fullgame'`` reads ``total_over`` Kalshi markets (KXMLBTOTAL).
        ``'f5'`` reads ``f5_total_over`` Kalshi markets (KXMLBF5TOTAL).

    Returns
    -------
    pd.DataFrame
        Priced predictions for the date, including PASS rows.
    """
    if date is None:
        date = date_type.today().isoformat()

    kalshi_market_type = "f5_total_over" if target == "f5" else "total_over"

    with get_conn(db_path) as conn:
        rows = conn.execute(
            """
            SELECT p.id, p.game_id, p.model_name, p.model_version,
                   p.lambda_home, p.lambda_away, p.dispersion_alpha,
                   p.polymarket_mid_price,
                   k.ticker AS kalshi_ticker, k.mid_price AS kalshi_mid_price,
                   k.line, k.open_interest,
                   g.home_team, g.away_team, g.game_time_et
            FROM predictions p
            JOIN games g ON p.game_id = g.game_id
            LEFT JOIN kalshi_markets k
                ON k.game_id = p.game_id
                AND k.market_type = ?
                AND k.snapshot_ts = (
                    SELECT MAX(snapshot_ts)
                    FROM kalshi_markets
                    WHERE game_id = p.game_id
                      AND market_type = ?
                )
            WHERE g.date = ?
              AND p.model_name LIKE ?
            ORDER BY g.game_time_et
            """,
            (kalshi_market_type, kalshi_market_type, date, f"{model_name}%"),
        ).fetchall()

    if not rows:
        logger.warning("No predictions found for date=%s model=%s", date, model_name)
        return pd.DataFrame()

    results = []
    positions_today = 0

    for row in rows:
        lam_h = row["lambda_home"]
        lam_a = row["lambda_away"]
        alpha = row["dispersion_alpha"]
        line = row["line"]
        kalshi_price = row["kalshi_mid_price"]
        kalshi_ticker = row["kalshi_ticker"]
        poly_price = row["polymarket_mid_price"]
        oi = row["open_interest"]

        if lam_h is None or lam_a is None or line is None or kalshi_price is None:
            logger.debug("Skipping %s — missing predictions or Kalshi price", row["game_id"])
            continue

        # Compute over_prob with current line
        if use_negbinom and alpha is not None and alpha > 0:
            over_prob = p_over_negbinom(float(lam_h), float(lam_a), float(alpha), float(line))
        else:
            over_prob = p_over_poisson(float(lam_h), float(lam_a), float(line))

        ev_result = compute_ev(over_prob, float(kalshi_price))
        bet_side = ev_result["bet_side"]

        win_prob = over_prob if bet_side == "OVER" else (1.0 - over_prob)
        bet_price = float(kalshi_price) if bet_side == "OVER" else (1.0 - float(kalshi_price))

        kelly = kelly_bet(win_prob, bet_price) if bet_side != "PASS" else 0.0

        ok, reason = passes_filters(bet_side, oi, positions_today)
        if not ok and bet_side != "PASS":
            logger.info(
                "%s %s vs %s — bet filtered: %s",
                date,
                row["home_team"],
                row["away_team"],
                reason,
            )
            bet_side = "PASS"
            kelly = 0.0

        if bet_side != "PASS":
            positions_today += 1

        consensus = (
            get_consensus(float(kalshi_price), float(poly_price))
            if poly_price is not None
            else None
        )

        rec = {
            "pred_id": row["id"],
            "game_id": row["game_id"],
            "home_team": row["home_team"],
            "away_team": row["away_team"],
            "game_time_et": row["game_time_et"],
            "line": line,
            "lambda_home": lam_h,
            "lambda_away": lam_a,
            "over_prob": round(over_prob, 4),
            "kalshi_ticker": kalshi_ticker,
            "kalshi_mid_price": kalshi_price,
            "polymarket_mid_price": poly_price,
            "edge": ev_result["edge"],
            "ev": ev_result["ev_over"] if bet_side in ("OVER", "PASS") else ev_result["ev_under"],
            "kelly_fraction": round(kelly, 6),
            "bet_side": bet_side,
            "consensus_spread": consensus["spread"] if consensus else None,
            "kalshi_is_cheap": consensus["kalshi_is_cheap_for_over"] if consensus else None,
        }
        results.append(rec)

    df = pd.DataFrame(results)

    # Write bet recommendations back to predictions table
    if not df.empty:
        with get_conn(db_path) as conn:
            for _, r in df.iterrows():
                conn.execute(
                    """
                    UPDATE predictions
                    SET line             = ?,
                        over_prob        = ?,
                        kalshi_ticker    = ?,
                        kalshi_mid_price = ?,
                        edge             = ?,
                        ev               = ?,
                        kelly_fraction   = ?,
                        recommended_bet  = ?,
                        bet_side         = ?
                    WHERE id = ?
                    """,
                    (
                        r["line"],
                        r["over_prob"],
                        r.get("kalshi_ticker"),
                        r.get("kalshi_mid_price"),
                        r["edge"],
                        r["ev"],
                        r["kelly_fraction"],
                        r["kelly_fraction"],  # recommended_bet = kelly_fraction
                        r["bet_side"],
                        int(r["pred_id"]),
                    ),
                )
        logger.info(
            "Priced %d games for %s — %d bets recommended",
            len(df),
            date,
            int((df["bet_side"] != "PASS").sum()),
        )

    return df


# ── Backtest simulation ────────────────────────────────────────────────────────


def simulate(
    start: str = "2021-04-01",
    end: str = "2024-10-01",
    min_edge: float = MIN_EDGE,
    kelly_mult: float = KELLY_MULT,
    initial_bankroll: float = 1000.0,
    model_name: str = "glm_poisson",
    book: str = "draftkings",
    use_negbinom: bool = True,
    under_filters: list[str] | None = None,
    output_path: str | None = None,
    db_path: str = "data/mlb.db",
) -> dict[str, Any]:
    """
    Walk-forward betting simulation over historical data.

    Uses SBR closing lines as the market benchmark since Kalshi data starts
    April 2025.  Converts American odds to fair devigged probabilities, applies
    the same EV/Kelly/filter logic as run_daily().

    Parameters
    ----------
    start : str
        Simulation start date (inclusive).
    end : str
        Simulation end date (inclusive).
    min_edge : float
        Minimum EV threshold.
    kelly_mult : float
        Fractional Kelly multiplier.
    initial_bankroll : float
        Starting bankroll in dollars.
    model_name : str
        Filter on predictions.model_name.
    book : str
        Sportsbook to use for odds (default 'draftkings').
    use_negbinom : bool
        Use NegBinom convolution when dispersion_alpha available.
    under_filters : list[str] or None
        Zero or more pre-game under-bias filters to apply.  Only games that
        match at least one active filter will be bet.  Pass None or [] to
        disable all filters (bet every game with edge > min_edge).

        Available filters (each is non-leaky — uses only pre-game data):

        ``'day_k9_park'``
            Day games (first pitch before 18:00 ET) in pitcher-friendly parks
            (``UNDER_PARKS``) where combined starter K/9 >= ``K9_COMBINED_MIN``.
            Under hit rate: 56.2% (p=0.012, n=889 over 2021-2025).

        ``'high_line'``
            Games where the closing total >= ``HIGH_LINE_MIN`` (11.0).
            Under hit rate: 57.1% (p=0.030, n=413 over 2021-2025).

    output_path : str or None
        If set, write per-bet detail to this CSV path.
    db_path : str

    Returns
    -------
    dict
        Summary statistics: roi, win_rate, bets_placed, sharpe,
        max_drawdown, avg_clv, bankroll_final.
    """
    active_filters: list[str] = list(under_filters) if under_filters else []
    use_day_k9_park = "day_k9_park" in active_filters
    use_high_line = "high_line" in active_filters

    with get_conn(db_path) as conn:
        # Precompute pre-game K/9 and era_l3 per pitcher via shift(1).
        # k9_season / era_l3 are season-to-date values after each outing;
        # shift(1) gives the values entering this game (only prior starts counted).
        if use_day_k9_park:
            pitcher_df = pd.read_sql_query(
                """
                SELECT ph.game_id, ph.team, ph.pitcher_id, ph.k9_season, ph.era_l3
                FROM pitchers ph
                JOIN games g ON g.game_id = ph.game_id
                WHERE ph.is_starter = 1 AND ph.k9_season IS NOT NULL
                ORDER BY ph.pitcher_id, g.date
                """,
                conn,
            )
            pitcher_df["k9_pregame"] = pitcher_df.groupby("pitcher_id")["k9_season"].shift(1)
            pitcher_df["era_l3_pregame"] = pitcher_df.groupby("pitcher_id")["era_l3"].shift(1)
            _k9_idx = pitcher_df.set_index(["game_id", "team"])["k9_pregame"]
            _era_l3_idx = pitcher_df.set_index(["game_id", "team"])["era_l3_pregame"]
        else:
            _k9_idx = None
            _era_l3_idx = None

        # Join predictions → games → sportsbook_odds for the simulation window.
        # game_time_et included for day/night detection in day_k9_park filter.
        rows = conn.execute(
            """
            SELECT p.game_id,
                   p.model_name, p.model_version,
                   p.lambda_home, p.lambda_away,
                   p.dispersion_alpha,
                   p.over_prob AS stored_over_prob,
                   g.date, g.home_team, g.away_team,
                   g.home_score, g.away_score,
                   g.game_time_et,
                   o.total_close,
                   o.over_odds_close, o.under_odds_close
            FROM predictions p
            JOIN games g ON p.game_id = g.game_id
            JOIN sportsbook_odds o
                ON o.date        = g.date
               AND o.home_team   = g.home_team
               AND o.book        = ?
            WHERE g.date BETWEEN ? AND ?
              AND p.model_name   LIKE ?
              AND (
                    (p.lambda_home IS NOT NULL AND p.lambda_away IS NOT NULL)
                    OR p.over_prob IS NOT NULL
              )
              AND o.total_close  IS NOT NULL
              AND o.over_odds_close  IS NOT NULL
              AND o.under_odds_close IS NOT NULL
              AND g.home_score   IS NOT NULL
              AND g.away_score   IS NOT NULL
            ORDER BY g.date, g.game_id
            """,
            (book, start, end, f"{model_name}%"),
        ).fetchall()

    if not rows:
        logger.warning(
            "No simulation data found for %s–%s model=%s book=%s",
            start,
            end,
            model_name,
            book,
        )
        return {}

    logger.info("Simulation: %d candidate games (%s to %s)", len(rows), start, end)

    bankroll = initial_bankroll
    bet_log: list[dict[str, Any]] = []
    positions_by_date: dict[str, int] = {}

    for row in rows:
        game_date = row["date"]
        lam_h_raw = row["lambda_home"]
        lam_a_raw = row["lambda_away"]
        lam_h = float(lam_h_raw) if lam_h_raw is not None else 0.0
        lam_a = float(lam_a_raw) if lam_a_raw is not None else 0.0
        alpha = row["dispersion_alpha"]
        stored_over_prob = row["stored_over_prob"]
        line = float(row["total_close"])
        over_odds = int(row["over_odds_close"])
        under_odds = int(row["under_odds_close"])
        actual_total = int(row["home_score"]) + int(row["away_score"])

        # Under-bias filters — applied before EV calculation so we only size bets
        # on games where the structural under edge is expected to apply.
        game_id = row["game_id"]
        home_team = row["home_team"]
        away_team = row["away_team"]

        if active_filters:
            passes_any = False

            if use_day_k9_park:
                game_time = row["game_time_et"] or ""
                try:
                    hour = int(game_time.split(":")[0])
                except (ValueError, IndexError):
                    hour = 20
                is_day = hour < 18
                in_under_park = home_team in UNDER_PARKS
                if is_day and in_under_park and _k9_idx is not None:
                    home_k9 = _k9_idx.get((game_id, home_team))
                    away_k9 = _k9_idx.get((game_id, away_team))
                    home_era = (
                        _era_l3_idx.get((game_id, home_team)) if _era_l3_idx is not None else None
                    )  # noqa: E501
                    away_era = (
                        _era_l3_idx.get((game_id, away_team)) if _era_l3_idx is not None else None
                    )  # noqa: E501

                    def _valid(v: object) -> bool:
                        return v is not None and not (isinstance(v, float) and np.isnan(v))

                    k9_sum = float(home_k9 or 0) + float(away_k9 or 0)
                    # era_l3 guard: only enforced when data is available for both SPs.
                    # If era_l3 is NULL (current DB state), bypass and trust K9 alone.
                    both_era_available = _valid(home_era) and _valid(away_era)
                    era_ok = not both_era_available or (
                        float(home_era) <= ERA_L3_MAX and float(away_era) <= ERA_L3_MAX
                    )
                    if _valid(home_k9) and _valid(away_k9) and k9_sum >= K9_COMBINED_MIN and era_ok:
                        passes_any = True

            if use_high_line and float(row["total_close"]) >= HIGH_LINE_MIN:
                passes_any = True

            if not passes_any:
                continue

        # Raw implied prices from closing odds (include the vig)
        raw_over = american_to_price(over_odds)
        raw_under = american_to_price(under_odds)
        # Devigged fair price is kept only as a reference / logging column
        fair_over, _ = devig_prices(raw_over, raw_under)
        vig_pct = round((raw_over + raw_under - 1.0) * 100.0, 2)

        # Model P(over) against closing line:
        # binary model (lam_h == 0.0 sentinel) → use stored calibrated probability
        # Poisson/NegBinom model → compute via convolution
        if lam_h > 0.01:
            if use_negbinom and alpha is not None and alpha > 0:
                over_prob = p_over_negbinom(lam_h, lam_a, float(alpha), line)
            else:
                over_prob = p_over_poisson(lam_h, lam_a, line)
        else:
            if stored_over_prob is None:
                continue
            over_prob = float(stored_over_prob)

        # EV and Kelly use raw (vig-inclusive) prices — realistic sportsbook fill cost
        ev_result = compute_ev(over_prob, raw_over, min_edge=min_edge)
        bet_side = ev_result["bet_side"]

        # Position limit per date
        n_open = positions_by_date.get(game_date, 0)
        ok, reason = passes_filters(bet_side, None, n_open)
        if not ok:
            continue

        win_prob = over_prob if bet_side == "OVER" else (1.0 - over_prob)
        # Actual fill price includes vig — different for each side when line is off-centre
        bet_price = raw_over if bet_side == "OVER" else raw_under
        kelly = kelly_bet(win_prob, bet_price, kelly_mult=kelly_mult)

        if kelly <= 0:
            continue

        stake = bankroll * kelly
        won = (bet_side == "OVER" and actual_total > line) or (
            bet_side == "UNDER" and actual_total < line
        )
        push = actual_total == line  # whole-number line; half-line lines never push

        if push:
            pnl = 0.0
            outcome = "PUSH"
        elif won:
            b = (1.0 / bet_price) - 1.0  # net odds at actual fill price (with vig)
            pnl = stake * b
            outcome = "WIN"
        else:
            pnl = -stake
            outcome = "LOSS"

        bankroll += pnl
        positions_by_date[game_date] = n_open + 1

        # CLV: use closing line as both entry and close (same market, retrospective)
        # True CLV requires recording entry price before close — use 0 as placeholder
        clv = 0.0  # populated when closing Kalshi data available

        bet_log.append(
            {
                "date": game_date,
                "game_id": row["game_id"],
                "home_team": row["home_team"],
                "away_team": row["away_team"],
                "line": line,
                "over_prob": round(over_prob, 4),
                "raw_over": round(raw_over, 4),
                "fair_over": round(fair_over, 4),
                "vig_pct": vig_pct,
                "edge": round(ev_result["edge"], 4),
                "bet_side": bet_side,
                "stake": round(stake, 2),
                "kelly_fraction": round(kelly, 4),
                "outcome": outcome,
                "pnl": round(pnl, 2),
                "bankroll_after": round(bankroll, 2),
                "actual_total": actual_total,
                "clv": clv,
            }
        )

    if not bet_log:
        logger.warning("No bets placed in simulation window")
        return {
            "games_evaluated": len(rows),
            "bets_placed": 0,
            "roi": 0.0,
            "win_rate": 0.0,
            "max_drawdown": 0.0,
            "sharpe": 0.0,
            "avg_clv": 0.0,
            "bankroll_final": initial_bankroll,
        }

    df_bets = pd.DataFrame(bet_log)

    # ── Summary statistics ────────────────────────────────────────────────────

    total_staked = df_bets["stake"].sum()
    total_pnl = df_bets["pnl"].sum()
    roi = (total_pnl / total_staked) * 100.0 if total_staked > 0 else 0.0

    decided = df_bets[df_bets["outcome"] != "PUSH"]
    win_rate = (decided["outcome"] == "WIN").mean() * 100.0 if len(decided) > 0 else 0.0

    # Max drawdown on bankroll curve
    bankroll_curve = np.array([initial_bankroll] + df_bets["bankroll_after"].tolist())
    peak = np.maximum.accumulate(bankroll_curve)
    drawdowns = (peak - bankroll_curve) / peak
    max_drawdown = float(drawdowns.max()) * 100.0

    # Per-bet ROI stats — binary bets always have high per-bet std (~0.9),
    # so raw Sharpe is suppressed. Annualise by bet frequency for a comparable figure.
    bet_rois = df_bets["pnl"].values / df_bets["stake"].values
    per_bet_std = float(np.std(bet_rois)) if len(bet_rois) > 1 else 0.0
    sharpe_raw = float(np.mean(bet_rois) / per_bet_std) if per_bet_std > 0 else 0.0

    from datetime import date as _date

    _start_d = _date.fromisoformat(start)
    _end_d = _date.fromisoformat(end)
    years_elapsed = max((_end_d - _start_d).days / 365.25, 1 / 365.25)
    bets_per_year = len(df_bets) / years_elapsed
    sharpe_annualised = sharpe_raw * (bets_per_year**0.5)

    avg_clv = float(df_bets["clv"].mean())

    summary: dict[str, Any] = {
        "games_evaluated": len(rows),
        "bets_placed": len(df_bets),
        "bet_pct": round(len(df_bets) / len(rows) * 100.0, 1),
        "wins": int((df_bets["outcome"] == "WIN").sum()),
        "losses": int((df_bets["outcome"] == "LOSS").sum()),
        "pushes": int((df_bets["outcome"] == "PUSH").sum()),
        "win_rate": round(win_rate, 1),
        "total_staked": round(total_staked, 2),
        "total_pnl": round(total_pnl, 2),
        "roi": round(roi, 2),
        "max_drawdown": round(max_drawdown, 2),
        "per_bet_volatility": round(per_bet_std * 100.0, 1),
        "sharpe": round(sharpe_raw, 3),
        "sharpe_annualised": round(sharpe_annualised, 2),
        "bets_per_year": round(bets_per_year, 0),
        "avg_clv": round(avg_clv, 4),
        "bankroll_final": round(float(bankroll), 2),
        "bankroll_peak": round(float(bankroll_curve.max()), 2),
    }

    if output_path:
        _path = Path(output_path)
        _path.parent.mkdir(parents=True, exist_ok=True)
        df_bets.to_csv(_path, index=False)
        logger.info("Per-bet detail saved to %s", _path)

    return summary


# ── Structural-only simulation (no model needed) ───────────────────────────────


def simulate_structural(
    start: str = "2021-04-01",
    end: str = "2025-10-01",
    filters: list[str] | None = None,
    sizing: str = "flat",
    flat_bet_pct: float = 0.02,
    kelly_mult: float = 0.25,
    kelly_cap: float = 0.05,
    initial_bankroll: float = 1000.0,
    book: str = "draftkings",
    output_path: str | None = None,
    db_path: str = "data/mlb.db",
) -> dict[str, Any]:
    """
    Structural betting simulation — no model required.

    Bets UNDER or OVER on every game that passes the requested filters.
    Stake size is either a flat fraction of bankroll or quarter-Kelly sized
    using per-filter EDA win rates as the Kelly input probability.

    Under filters: ``'day_k9_park'``, ``'high_line'``
    Over filters:  ``'hot_wind_out'``, ``'summer_hot_wind_out'``

    If a game triggers both an under and an over filter simultaneously, it is
    skipped (contradictory signal).

    EDA results (2021-2025 SBR data, OOS 2023-2025):
    - day_k9_park + era_l3 guard → 56.5% under rate (n=418, OOS 55.4%)
    - high_line                  → 57.1% under rate (n=413, OOS 56.1%)
    - hot_wind_out               → 55.9% over rate  (n=288, OOS 60.0%)
    - summer_hot_wind_out        → 61.3% over rate  (n=150, Jul-Sep only)

    Parameters
    ----------
    start : str
    end : str
    filters : list[str] or None
        Filter names to activate.  At least one must be specified.
        Under filters: ``'day_k9_park'``, ``'high_line'``.
        Over filters:  ``'hot_wind_out'``, ``'summer_hot_wind_out'``.
    sizing : str
        ``'flat'`` — stake is ``flat_bet_pct`` of current bankroll each bet.
        ``'quarter_kelly'`` — stake is 0.25× Kelly fraction, capped at
        ``kelly_cap`` of bankroll.  Uses ``_FILTER_WIN_PROBS`` as the win
        probability input to Kelly.
    flat_bet_pct : float
        Fraction of bankroll per bet when ``sizing='flat'``.  Default 0.02.
    kelly_mult : float
        Kelly fraction multiplier when ``sizing='quarter_kelly'``.  Default 0.25.
    kelly_cap : float
        Hard cap on Kelly stake as fraction of bankroll.  Default 0.05 (5 %).
    initial_bankroll : float
    book : str
    output_path : str or None
    db_path : str

    Returns
    -------
    dict
        Summary statistics including per-filter breakdowns.
    """
    active_filters: list[str] = list(filters) if filters else []
    if not active_filters:
        raise ValueError("simulate_structural requires at least one filter")

    use_day_k9_park = "day_k9_park" in active_filters
    use_high_line = "high_line" in active_filters
    use_hot_wind_out = "hot_wind_out" in active_filters
    use_summer_hot_wind_out = "summer_hot_wind_out" in active_filters
    need_pitcher_data = use_day_k9_park
    need_weather_data = use_hot_wind_out or use_summer_hot_wind_out

    with get_conn(db_path) as conn:
        if need_pitcher_data:
            pitcher_df = pd.read_sql_query(
                """
                SELECT ph.game_id, ph.team, ph.pitcher_id, ph.k9_season, ph.era_l3
                FROM pitchers ph
                JOIN games g ON g.game_id = ph.game_id
                WHERE ph.is_starter = 1 AND ph.k9_season IS NOT NULL
                ORDER BY ph.pitcher_id, g.date
                """,
                conn,
            )
            pitcher_df["k9_pregame"] = pitcher_df.groupby("pitcher_id")["k9_season"].shift(1)
            pitcher_df["era_l3_pregame"] = pitcher_df.groupby("pitcher_id")["era_l3"].shift(1)
            _k9_idx = pitcher_df.set_index(["game_id", "team"])["k9_pregame"]
            _era_l3_idx = pitcher_df.set_index(["game_id", "team"])["era_l3_pregame"]
        else:
            _k9_idx = None
            _era_l3_idx = None

        _base_sql = """
            SELECT g.game_id, g.date, g.home_team, g.away_team,
                   g.home_score, g.away_score, g.game_time_et,
                   o.total_close, o.over_odds_close, o.under_odds_close
            FROM games g
            JOIN sportsbook_odds o
                ON o.date = g.date AND o.home_team = g.home_team AND o.book = ?
            WHERE g.date BETWEEN ? AND ?
              AND o.total_close IS NOT NULL
              AND o.over_odds_close IS NOT NULL
              AND o.under_odds_close IS NOT NULL
              AND g.home_score IS NOT NULL
              AND g.away_score IS NOT NULL
            ORDER BY g.date, g.game_id
        """
        _weather_sql = """
            SELECT g.game_id, g.date, g.home_team, g.away_team,
                   g.home_score, g.away_score, g.game_time_et,
                   o.total_close, o.over_odds_close, o.under_odds_close,
                   w.temp_f, w.wind_speed_mph, w.wind_dir_label, w.is_dome
            FROM games g
            JOIN sportsbook_odds o
                ON o.date = g.date AND o.home_team = g.home_team AND o.book = ?
            LEFT JOIN weather w
                ON w.game_id = g.game_id AND w.snapshot_type = 'historical'
            WHERE g.date BETWEEN ? AND ?
              AND o.total_close IS NOT NULL
              AND o.over_odds_close IS NOT NULL
              AND o.under_odds_close IS NOT NULL
              AND g.home_score IS NOT NULL
              AND g.away_score IS NOT NULL
            ORDER BY g.date, g.game_id
        """
        _query = _weather_sql if need_weather_data else _base_sql
        rows = conn.execute(_query, (book, start, end)).fetchall()

    logger.info("Structural sim: %d candidate games (%s to %s)", len(rows), start, end)

    bankroll = initial_bankroll
    bet_log: list[dict[str, Any]] = []

    def _valid(v: object) -> bool:
        return v is not None and not (isinstance(v, float) and np.isnan(v))

    for row in rows:
        game_id = row["game_id"]
        home_team = row["home_team"]
        away_team = row["away_team"]
        line = float(row["total_close"])
        actual_total = int(row["home_score"]) + int(row["away_score"])

        under_signal = False
        over_signal = False
        triggered_filter = ""

        # ── Under filters ──────────────────────────────────────────────────────
        if use_day_k9_park:
            game_time = row["game_time_et"] or ""
            try:
                hour = int(game_time.split(":")[0])
            except (ValueError, IndexError):
                hour = 20
            is_day = hour < 18
            if is_day and home_team in UNDER_PARKS and _k9_idx is not None:
                home_k9 = _k9_idx.get((game_id, home_team))
                away_k9 = _k9_idx.get((game_id, away_team))
                home_era = (
                    _era_l3_idx.get((game_id, home_team)) if _era_l3_idx is not None else None
                )  # noqa: E501
                away_era = (
                    _era_l3_idx.get((game_id, away_team)) if _era_l3_idx is not None else None
                )  # noqa: E501
                k9_sum = float(home_k9 or 0) + float(away_k9 or 0)
                both_era_available = _valid(home_era) and _valid(away_era)
                era_ok = not both_era_available or (
                    float(home_era) <= ERA_L3_MAX and float(away_era) <= ERA_L3_MAX
                )
                if _valid(home_k9) and _valid(away_k9) and k9_sum >= K9_COMBINED_MIN and era_ok:
                    under_signal = True
                    triggered_filter = "day_k9_park"

        if use_high_line and not under_signal and line >= HIGH_LINE_MIN:
            under_signal = True
            triggered_filter = "high_line"

        # ── Over filters ───────────────────────────────────────────────────────
        if use_hot_wind_out or use_summer_hot_wind_out:
            temp_f = row["temp_f"] if need_weather_data else None
            wind_mph = row["wind_speed_mph"] if need_weather_data else None
            wind_dir = row["wind_dir_label"] if need_weather_data else None
            is_dome = row["is_dome"] if need_weather_data else 1
            base_conditions = (
                _valid(temp_f)
                and _valid(wind_mph)
                and float(temp_f) >= HOT_TEMP_MIN
                and int(is_dome or 0) == 0
                and wind_dir == "out"
                and WIND_SPEED_MIN <= float(wind_mph) < WIND_SPEED_MAX
            )
            if use_hot_wind_out and base_conditions:
                over_signal = True
                triggered_filter = "hot_wind_out"
            elif use_summer_hot_wind_out and base_conditions:
                game_month = pd.to_datetime(row["date"]).month
                if game_month in SUMMER_MONTHS:
                    over_signal = True
                    triggered_filter = "summer_hot_wind_out"

        # Contradictory signals → skip
        if under_signal and over_signal:
            continue
        if not under_signal and not over_signal:
            continue

        bet_side = "UNDER" if under_signal else "OVER"
        if bet_side == "UNDER":
            raw_price = american_to_price(int(row["under_odds_close"]))
            won = actual_total < line
        else:
            raw_price = american_to_price(int(row["over_odds_close"]))
            won = actual_total > line

        if sizing == "quarter_kelly":
            win_prob = _FILTER_WIN_PROBS.get(triggered_filter, 0.55)
            b = (1.0 / raw_price) - 1.0
            full_kelly = max(0.0, (win_prob * b - (1.0 - win_prob)) / b)
            stake_pct = min(full_kelly * kelly_mult, kelly_cap)
            if stake_pct <= 0.0:
                continue  # no positive Kelly edge at these odds; skip
            stake = bankroll * stake_pct
        else:
            stake = bankroll * flat_bet_pct
        push = actual_total == line

        if push:
            pnl = 0.0
            outcome = "PUSH"
        elif won:
            b = (1.0 / raw_price) - 1.0
            pnl = stake * b
            outcome = "WIN"
        else:
            pnl = -stake
            outcome = "LOSS"

        bankroll += pnl
        bet_log.append(
            {
                "date": row["date"],
                "game_id": game_id,
                "home_team": home_team,
                "away_team": away_team,
                "line": line,
                "bet_side": bet_side,
                "filter": triggered_filter,
                "stake": round(stake, 2),
                "stake_pct": round(stake / bankroll * 100, 3) if bankroll > 0 else 0,
                "outcome": outcome,
                "pnl": round(pnl, 2),
                "bankroll_after": round(bankroll, 2),
                "actual_total": actual_total,
            }
        )

    if not bet_log:
        logger.warning("No bets placed in structural simulation")
        return {"games_evaluated": len(rows), "bets_placed": 0, "roi": 0.0}

    df_bets = pd.DataFrame(bet_log)
    total_staked = df_bets["stake"].sum()
    total_pnl = df_bets["pnl"].sum()
    roi = (total_pnl / total_staked) * 100.0 if total_staked > 0 else 0.0

    decided = df_bets[df_bets["outcome"] != "PUSH"]
    win_rate = (decided["outcome"] == "WIN").mean() * 100.0 if len(decided) > 0 else 0.0

    bankroll_curve = np.array([initial_bankroll] + df_bets["bankroll_after"].tolist())
    peak = np.maximum.accumulate(bankroll_curve)
    drawdowns = (peak - bankroll_curve) / peak
    max_drawdown = float(drawdowns.max()) * 100.0

    bet_rois = df_bets["pnl"].values / df_bets["stake"].values
    per_bet_std = float(np.std(bet_rois)) if len(bet_rois) > 1 else 0.0
    sharpe_raw = float(np.mean(bet_rois) / per_bet_std) if per_bet_std > 0 else 0.0

    from datetime import date as _date

    _days = (_date.fromisoformat(end) - _date.fromisoformat(start)).days
    years_elapsed = max(_days / 365.25, 1 / 365.25)
    bets_per_year = len(df_bets) / years_elapsed
    sharpe_annualised = sharpe_raw * (bets_per_year**0.5)

    if output_path:
        _path = Path(output_path)
        _path.parent.mkdir(parents=True, exist_ok=True)
        df_bets.to_csv(_path, index=False)
        logger.info("Per-bet detail saved to %s", _path)

    return {
        "games_evaluated": len(rows),
        "bets_placed": len(df_bets),
        "bet_pct": round(len(df_bets) / len(rows) * 100.0, 1),
        "wins": int((df_bets["outcome"] == "WIN").sum()),
        "losses": int((df_bets["outcome"] == "LOSS").sum()),
        "pushes": int((df_bets["outcome"] == "PUSH").sum()),
        "win_rate": round(win_rate, 1),
        "total_staked": round(total_staked, 2),
        "total_pnl": round(total_pnl, 2),
        "roi": round(roi, 2),
        "max_drawdown": round(max_drawdown, 2),
        "per_bet_volatility": round(per_bet_std * 100.0, 1),
        "sharpe": round(sharpe_raw, 3),
        "sharpe_annualised": round(sharpe_annualised, 2),
        "bets_per_year": round(bets_per_year, 0),
        "bankroll_final": round(float(bankroll), 2),
        "bankroll_peak": round(float(bankroll_curve.max()), 2),
        "filters": active_filters,
        "flat_bet_pct": flat_bet_pct,
    }


# ── Kalshi-based simulation (full-game and F5) ─────────────────────────────────


def simulate_kalshi(
    start: str = "2025-04-01",
    end: str = "2025-10-01",
    target: str = "fullgame",
    min_edge: float = MIN_EDGE,
    kelly_mult: float = KELLY_MULT,
    initial_bankroll: float = 1000.0,
    model_name: str = "hgbr_poisson",
    output_path: str | None = None,
    db_path: str = "data/mlb.db",
) -> dict[str, Any]:
    """
    Betting simulation using Kalshi market prices as the benchmark.

    For each game in the window, iterates over every available Kalshi line and
    picks the one with the highest |EV|.  Outcome is determined from the actual
    game score stored in the games table.

    Supports both full-game (``target='fullgame'``) and F5 (``target='f5'``)
    modes.  For F5 the simulation joins Kalshi ``f5_total_over`` markets and
    uses ``f5_total_runs`` as the actual outcome.

    Parameters
    ----------
    start : str
        Simulation start date (inclusive).
    end : str
        Simulation end date (inclusive).
    target : str
        ``'fullgame'`` or ``'f5'``.
    min_edge : float
        Minimum EV threshold to place a bet.
    kelly_mult : float
        Fractional Kelly multiplier.
    initial_bankroll : float
        Starting bankroll in dollars.
    model_name : str
        Filter on predictions.model_name.
    output_path : str or None
        If set, write per-bet detail to this CSV path.
    db_path : str

    Returns
    -------
    dict
        Summary statistics.
    """
    market_type = "f5_total_over" if target == "f5" else "total_over"

    with get_conn(db_path) as conn:
        # Load model predictions joined to Kalshi markets for the window.
        # Each row is one (prediction, Kalshi-line) pair — one game can have
        # multiple lines; we pick the best EV in Python.
        _select_f5 = (
            "SELECT p.game_id, p.model_name, p.model_version,"
            " p.lambda_home, p.lambda_away, p.dispersion_alpha,"
            " g.date, g.home_team, g.away_team,"
            " g.f5_total_runs AS actual_result,"
            " km.line AS kalshi_line, km.mid_price AS kalshi_price,"
            " km.result AS kalshi_result"
            " FROM predictions p JOIN games g ON p.game_id = g.game_id"
            " JOIN kalshi_markets km ON km.game_id = p.game_id"
            "   AND km.market_type = ?"
            " WHERE g.date BETWEEN ? AND ? AND p.model_name LIKE ?"
            "   AND p.lambda_home IS NOT NULL AND p.lambda_away IS NOT NULL"
            "   AND km.mid_price BETWEEN 0.03 AND 0.97"
            "   AND km.yes_bid > 0.01 AND km.yes_ask < 0.99"
            "   AND km.line IS NOT NULL"
            "   AND g.f5_total_runs IS NOT NULL"
            " ORDER BY g.date, p.game_id, km.line"
        )
        _select_fg = (
            "SELECT p.game_id, p.model_name, p.model_version,"
            " p.lambda_home, p.lambda_away, p.dispersion_alpha,"
            " g.date, g.home_team, g.away_team,"
            " (g.home_score + g.away_score) AS actual_result,"
            " km.line AS kalshi_line, km.mid_price AS kalshi_price,"
            " km.result AS kalshi_result"
            " FROM predictions p JOIN games g ON p.game_id = g.game_id"
            " JOIN kalshi_markets km ON km.game_id = p.game_id"
            "   AND km.market_type = ?"
            " WHERE g.date BETWEEN ? AND ? AND p.model_name LIKE ?"
            "   AND p.lambda_home IS NOT NULL AND p.lambda_away IS NOT NULL"
            "   AND km.mid_price BETWEEN 0.03 AND 0.97"
            "   AND km.yes_bid > 0.01 AND km.yes_ask < 0.99"
            "   AND km.line IS NOT NULL"
            "   AND g.home_score IS NOT NULL AND g.away_score IS NOT NULL"
            " ORDER BY g.date, p.game_id, km.line"
        )
        sql = _select_f5 if target == "f5" else _select_fg
        rows = conn.execute(sql, (market_type, start, end, f"{model_name}%")).fetchall()

    if not rows:
        logger.warning(
            "No Kalshi simulation data found for %s–%s target=%s model=%s",
            start,
            end,
            target,
            model_name,
        )
        return {}

    # Group rows by (date, game_id) — pick the best-EV line per game
    from collections import defaultdict

    game_lines: dict[tuple[str, str], list] = defaultdict(list)
    for row in rows:
        key = (row["date"], row["game_id"])
        game_lines[key].append(dict(row))

    logger.info(
        "Kalshi simulation: %d games across %d (game, line) pairs (%s to %s)",
        len(game_lines),
        len(rows),
        start,
        end,
    )

    bankroll = initial_bankroll
    bet_log: list[dict[str, Any]] = []
    positions_by_date: dict[str, int] = {}

    for (game_date, game_id), lines in sorted(game_lines.items()):
        n_open = positions_by_date.get(game_date, 0)
        if n_open >= MAX_POSITIONS:
            continue

        best_ev = 0.0
        best_bet: dict[str, Any] | None = None

        for row in lines:
            lam_h = float(row["lambda_home"])
            lam_a = float(row["lambda_away"])
            alpha = row["dispersion_alpha"]
            line = float(row["kalshi_line"])
            price = float(row["kalshi_price"])

            if alpha is not None and alpha > 0:
                over_prob = p_over_negbinom(lam_h, lam_a, float(alpha), line)
            else:
                over_prob = p_over_poisson(lam_h, lam_a, line)

            ev_result = compute_ev(over_prob, price, min_edge=min_edge)
            candidate_ev = max(abs(ev_result["ev_over"]), abs(ev_result["ev_under"]))

            if ev_result["bet_side"] != "PASS" and candidate_ev > best_ev:
                best_ev = candidate_ev
                best_bet = {
                    "line": line,
                    "price": price,
                    "over_prob": over_prob,
                    "bet_side": ev_result["bet_side"],
                    "edge": ev_result["edge"],
                    "row": row,
                }

        if best_bet is None:
            continue

        row = best_bet["row"]
        bet_side = best_bet["bet_side"]
        over_prob = best_bet["over_prob"]
        price = best_bet["price"]
        line = best_bet["line"]
        actual = row["actual_result"]

        win_prob = over_prob if bet_side == "OVER" else (1.0 - over_prob)
        bet_price = price if bet_side == "OVER" else (1.0 - price)
        kelly = kelly_bet(win_prob, bet_price, kelly_mult=kelly_mult)

        if kelly <= 0:
            continue

        stake = bankroll * kelly

        # Kalshi stores integer N meaning "Over N+0.5" in UI.
        # YES (OVER) wins if actual > N (i.e. actual >= N+1).
        # NO  (UNDER) wins if actual <= N (i.e. actual < N+1).
        # No pushes possible with half-line markets.
        if bet_side == "OVER":
            won = actual > line
            push = False
        else:
            won = actual <= line
            push = False

        if push:
            pnl = 0.0
            outcome = "PUSH"
        elif won:
            b = (1.0 / bet_price) - 1.0
            pnl = stake * b
            outcome = "WIN"
        else:
            pnl = -stake
            outcome = "LOSS"

        bankroll += pnl
        positions_by_date[game_date] = n_open + 1

        bet_log.append(
            {
                "date": game_date,
                "game_id": game_id,
                "home_team": row["home_team"],
                "away_team": row["away_team"],
                "target": target,
                "line": line,
                "over_prob": round(over_prob, 4),
                "kalshi_price": round(price, 4),
                "edge": round(best_bet["edge"], 4),
                "bet_side": bet_side,
                "stake": round(stake, 2),
                "kelly_fraction": round(kelly, 4),
                "outcome": outcome,
                "pnl": round(pnl, 2),
                "bankroll_after": round(bankroll, 2),
                "actual_result": actual,
            }
        )

    if not bet_log:
        logger.warning("No bets placed in Kalshi simulation")
        return {
            "games_evaluated": len(game_lines),
            "bets_placed": 0,
            "roi": 0.0,
            "win_rate": 0.0,
            "max_drawdown": 0.0,
            "sharpe": 0.0,
            "bankroll_final": initial_bankroll,
        }

    df_bets = pd.DataFrame(bet_log)

    total_staked = df_bets["stake"].sum()
    total_pnl = df_bets["pnl"].sum()
    roi = (total_pnl / total_staked) * 100.0 if total_staked > 0 else 0.0

    decided = df_bets[df_bets["outcome"] != "PUSH"]
    win_rate = (decided["outcome"] == "WIN").mean() * 100.0 if len(decided) > 0 else 0.0

    bankroll_curve = np.array([initial_bankroll] + df_bets["bankroll_after"].tolist())
    peak = np.maximum.accumulate(bankroll_curve)
    drawdowns = (peak - bankroll_curve) / peak
    max_drawdown = float(drawdowns.max()) * 100.0

    bet_rois = df_bets["pnl"].values / df_bets["stake"].values
    sharpe = float(np.mean(bet_rois) / np.std(bet_rois)) if np.std(bet_rois) > 0 else 0.0

    summary: dict[str, Any] = {
        "target": target,
        "market": market_type,
        "games_evaluated": len(game_lines),
        "bets_placed": len(df_bets),
        "bet_pct": round(len(df_bets) / len(game_lines) * 100.0, 1),
        "wins": int((df_bets["outcome"] == "WIN").sum()),
        "losses": int((df_bets["outcome"] == "LOSS").sum()),
        "pushes": int((df_bets["outcome"] == "PUSH").sum()),
        "win_rate": round(win_rate, 1),
        "total_staked": round(total_staked, 2),
        "total_pnl": round(total_pnl, 2),
        "roi": round(roi, 2),
        "max_drawdown": round(max_drawdown, 2),
        "sharpe": round(sharpe, 3),
        "bankroll_final": round(float(bankroll), 2),
        "bankroll_peak": round(float(bankroll_curve.max()), 2),
    }

    if output_path:
        _path = Path(output_path)
        _path.parent.mkdir(parents=True, exist_ok=True)
        df_bets.to_csv(_path, index=False)
        logger.info("Per-bet detail saved to %s", _path)

    return summary


# ── CLV updater ────────────────────────────────────────────────────────────────


def update_clv(
    date: str | None = None,
    db_path: str = "data/mlb.db",
) -> int:
    """
    Update CLV for bets placed on a given date using the final Kalshi price.

    Reads closing_kalshi_price from predictions (populated by the Kalshi
    scraper after market settlement) and writes back the clv column.

    Parameters
    ----------
    date : str or None
        ISO date. Defaults to yesterday (CLV only meaningful after close).
    db_path : str

    Returns
    -------
    int
        Number of rows updated.
    """
    if date is None:
        from datetime import timedelta

        date = (date_type.today() - timedelta(days=1)).isoformat()

    updated = 0
    with get_conn(db_path) as conn:
        rows = conn.execute(
            """
            SELECT id, bet_side, kalshi_mid_price, closing_kalshi_price
            FROM predictions
            WHERE game_id IN (SELECT game_id FROM games WHERE date = ?)
              AND bet_side IN ('OVER', 'UNDER')
              AND kalshi_mid_price IS NOT NULL
              AND closing_kalshi_price IS NOT NULL
              AND clv IS NULL
            """,
            (date,),
        ).fetchall()

        for row in rows:
            clv = compute_clv(
                float(row["kalshi_mid_price"]),
                float(row["closing_kalshi_price"]),
                row["bet_side"],
            )
            conn.execute(
                "UPDATE predictions SET clv = ? WHERE id = ?",
                (clv, row["id"]),
            )
            updated += 1

    logger.info("Updated CLV for %d predictions on %s", updated, date)
    return updated


# ── CLI ────────────────────────────────────────────────────────────────────────


def _print_simulation_report(summary: dict[str, Any], params: dict[str, Any]) -> None:
    """Print formatted simulation report to stdout."""
    print()
    print("=" * 60)
    print(f"BETTING SIMULATION: {params['start']} to {params['end']}")
    book = params["book"].capitalize()
    print(f"  Market benchmark:  SBR {book} closing line (vig-inclusive fill)")
    print(f"  Min edge:          ${params['min_edge']:.2f}")
    print(f"  Kelly multiplier:  {params['kelly_mult']:.2f}x")
    print(f"  Initial bankroll:  ${params['initial_bankroll']:,.0f}")
    if params.get("under_filters"):
        print(f"  Under filters:     {', '.join(params['under_filters'])}")
    print("=" * 60)
    print()
    print(f"  Games evaluated:   {summary['games_evaluated']:,}")
    print(f"  Bets placed:       {summary['bets_placed']:,}  ({summary['bet_pct']:.1f}% of games)")
    print(f"  Record:            {summary['wins']}W – {summary['losses']}L – {summary['pushes']}P")
    print(f"  Win rate:          {summary['win_rate']:.1f}%")
    print()
    print(f"  Total staked:      ${summary['total_staked']:,.2f}")
    print(f"  Total P&L:         ${summary['total_pnl']:+,.2f}")
    print(f"  ROI:               {summary['roi']:+.2f}%")
    print()
    print(f"  Max drawdown:      -{summary['max_drawdown']:.1f}%")
    print(f"  Per-bet volatility:{summary['per_bet_volatility']:+.1f}% std")
    print(f"  Sharpe (per-bet):  {summary['sharpe']:.3f}")
    print(
        f"  Sharpe (annual):   {summary['sharpe_annualised']:.2f}"
        f"  [{summary['bets_per_year']:.0f} bets/yr x sqrt]"
    )
    print(f"  Average CLV:       {summary['avg_clv']:+.4f}")
    print()
    print(
        f"  Final bankroll:    ${summary['bankroll_final']:,.2f}"
        f"  (peak: ${summary['bankroll_peak']:,.2f})"
    )
    print("=" * 60)

    # Go/no-go — use annualised Sharpe (>= 1.0) since per-bet Sharpe is suppressed
    # by binary payoff variance and is not a fair comparison to financial asset Sharpe.
    print()
    go = summary["roi"] > 0 and summary["sharpe_annualised"] >= 1.0
    if go:
        print("  GO: ROI positive, annualised Sharpe >= 1.0")
        print("    Model is ready for paper trading.")
    else:
        reasons = []
        if summary["roi"] <= 0:
            reasons.append(f"ROI {summary['roi']:+.2f}% <= 0")
        if summary["sharpe_annualised"] < 1.0:
            reasons.append(f"annualised Sharpe {summary['sharpe_annualised']:.2f} < 1.0")
        print(f"  NO-GO: {'; '.join(reasons)}")
        print("    Do not proceed to live trading.")
    print()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="MLB betting engine")
    subparsers = parser.add_subparsers(dest="command")

    # -- daily pricing
    daily_p = subparsers.add_parser("daily", help="Price today's games")
    daily_p.add_argument("--date", default=None, help="ISO date (default: today)")
    daily_p.add_argument("--model", default="gbr")
    daily_p.add_argument("--db", default="data/mlb.db")

    # -- backtest simulation
    sim_p = subparsers.add_parser("simulate", help="Walk-forward betting simulation")
    sim_p.add_argument("--start", default="2021-04-01")
    sim_p.add_argument("--end", default="2024-10-01")
    sim_p.add_argument("--min-edge", type=float, default=MIN_EDGE)
    sim_p.add_argument("--kelly-mult", type=float, default=KELLY_MULT)
    sim_p.add_argument("--initial-bankroll", type=float, default=1000.0)
    sim_p.add_argument("--model", default="gbr")
    sim_p.add_argument("--book", default="pinnacle")
    sim_p.add_argument(
        "--under-filter",
        action="append",
        dest="under_filters",
        choices=["day_k9_park", "high_line"],
        default=None,
        help="Pre-game under-bias filter(s) to apply. Repeat flag to combine. "
        "day_k9_park: day games in pitcher parks with K9>=14. "
        "high_line: total line >= 11.0. None = no filter (bet all games).",
    )
    sim_p.add_argument("--output", default=None, help="CSV path for per-bet detail")
    sim_p.add_argument("--db", default="data/mlb.db")

    # -- structural simulation (no model needed)
    ss_p = subparsers.add_parser(
        "simulate-structural",
        help="Flat bet on structural-filter games (UNDER or OVER) — no model required",
    )
    ss_p.add_argument("--start", default="2021-04-01")
    ss_p.add_argument("--end", default="2025-10-01")
    ss_p.add_argument(
        "--filter",
        action="append",
        dest="filters",
        choices=["day_k9_park", "high_line", "hot_wind_out", "summer_hot_wind_out"],
        default=None,
        required=True,
        help=(
            "Structural filter(s) to activate. Repeat to combine. "
            "Under: day_k9_park, high_line. "
            "Over (all months): hot_wind_out. "
            "Over (Jul-Sep only): summer_hot_wind_out."
        ),
    )
    ss_p.add_argument("--flat-bet-pct", type=float, default=0.02)
    ss_p.add_argument("--initial-bankroll", type=float, default=1000.0)
    ss_p.add_argument("--book", default="draftkings")
    ss_p.add_argument("--output", default=None)
    ss_p.add_argument("--db", default="data/mlb.db")

    # -- CLV update
    clv_p = subparsers.add_parser("update-clv", help="Update CLV after market close")
    clv_p.add_argument("--date", default=None)
    clv_p.add_argument("--db", default="data/mlb.db")

    # -- Kalshi-based simulation (full-game and F5)
    ksim_p = subparsers.add_parser("simulate-kalshi", help="Simulation vs Kalshi prices")
    ksim_p.add_argument("--start", default="2025-04-01")
    ksim_p.add_argument("--end", default="2025-10-01")
    ksim_p.add_argument("--target", choices=["fullgame", "f5"], default="fullgame")
    ksim_p.add_argument("--min-edge", type=float, default=MIN_EDGE)
    ksim_p.add_argument("--kelly-mult", type=float, default=KELLY_MULT)
    ksim_p.add_argument("--initial-bankroll", type=float, default=1000.0)
    ksim_p.add_argument("--model", default="hgbr_poisson")
    ksim_p.add_argument("--output", default=None)
    ksim_p.add_argument("--db", default="data/mlb.db")

    # Legacy flat flags for backward compatibility with /backtest skill
    parser.add_argument("--date", default=None)
    parser.add_argument("--simulate", action="store_true")
    parser.add_argument("--simulate-kalshi", action="store_true", dest="simulate_kalshi")
    parser.add_argument("--target", choices=["fullgame", "f5"], default="fullgame")
    parser.add_argument("--start", default="2021-04-01")
    parser.add_argument("--end", default="2024-10-01")
    parser.add_argument("--min-edge", type=float, default=MIN_EDGE)
    parser.add_argument("--kelly-mult", type=float, default=KELLY_MULT)
    parser.add_argument("--initial-bankroll", type=float, default=1000.0)
    parser.add_argument("--model", default="gbr")
    parser.add_argument("--book", default="pinnacle")
    parser.add_argument(
        "--under-filter",
        action="append",
        dest="under_filters",
        choices=["day_k9_park", "high_line"],
        default=None,
    )
    parser.add_argument("--output", default=None)
    parser.add_argument("--db", default="data/mlb.db")

    args = parser.parse_args()

    if args.command == "simulate-kalshi" or getattr(args, "simulate_kalshi", False):
        target = getattr(args, "target", "fullgame")
        start = getattr(args, "start", "2025-04-01")
        end = getattr(args, "end", "2025-10-01")
        summary = simulate_kalshi(
            start=start,
            end=end,
            target=target,
            min_edge=getattr(args, "min_edge", MIN_EDGE),
            kelly_mult=getattr(args, "kelly_mult", KELLY_MULT),
            initial_bankroll=getattr(args, "initial_bankroll", 1000.0),
            model_name=getattr(args, "model", "hgbr_poisson"),
            output_path=args.output,
            db_path=args.db,
        )
        if summary:
            print()
            print("=" * 60)
            print(f"KALSHI SIMULATION ({target.upper()}): {start} to {end}")
            print(f"  Market:            Kalshi {summary['market']}")
            print(f"  Min edge:          ${getattr(args, 'min_edge', MIN_EDGE):.2f}")
            print(f"  Kelly multiplier:  {getattr(args, 'kelly_mult', KELLY_MULT):.2f}x")
            print(f"  Initial bankroll:  ${getattr(args, 'initial_bankroll', 1000.0):,.0f}")
            print("=" * 60)
            print(f"  Games evaluated:   {summary['games_evaluated']:,}")
            print(f"  Bets placed:       {summary['bets_placed']:,}  ({summary['bet_pct']:.1f}%)")
            wins, losses, pushes = summary["wins"], summary["losses"], summary["pushes"]
            print(f"  Record:            {wins}W - {losses}L - {pushes}P")
            print(f"  Win rate:          {summary['win_rate']:.1f}%")
            print(f"  ROI:               {summary['roi']:+.2f}%")
            print(f"  Max drawdown:      -{summary['max_drawdown']:.1f}%")
            print(f"  Sharpe ratio:      {summary['sharpe']:.3f}")
            bf = summary["bankroll_final"]
            bp = summary["bankroll_peak"]
            print(f"  Bankroll final:    ${bf:,.2f}  (peak: ${bp:,.2f})")
            print("=" * 60)
            go = summary["roi"] > 0 and summary["sharpe"] > 0.5
            print("  GO" if go else "  NO-GO")
            print()

    elif args.command == "simulate" or getattr(args, "simulate", False):
        under_filters = getattr(args, "under_filters", None)
        summary = simulate(
            start=getattr(args, "start", "2021-04-01"),
            end=getattr(args, "end", "2024-10-01"),
            min_edge=getattr(args, "min_edge", MIN_EDGE),
            kelly_mult=getattr(args, "kelly_mult", KELLY_MULT),
            initial_bankroll=getattr(args, "initial_bankroll", 1000.0),
            model_name=args.model,
            book=getattr(args, "book", "draftkings"),
            under_filters=under_filters,
            output_path=args.output,
            db_path=args.db,
        )
        if summary:
            params = {
                "start": getattr(args, "start", "2021-04-01"),
                "end": getattr(args, "end", "2024-10-01"),
                "min_edge": getattr(args, "min_edge", MIN_EDGE),
                "kelly_mult": getattr(args, "kelly_mult", KELLY_MULT),
                "initial_bankroll": getattr(args, "initial_bankroll", 1000.0),
                "book": getattr(args, "book", "draftkings"),
                "under_filters": under_filters,
            }
            _print_simulation_report(summary, params)

    elif args.command == "simulate-structural":
        s = simulate_structural(
            start=args.start,
            end=args.end,
            filters=args.filters,
            flat_bet_pct=args.flat_bet_pct,
            initial_bankroll=args.initial_bankroll,
            book=args.book,
            output_path=args.output,
            db_path=args.db,
        )
        if s:
            print()
            print("=" * 60)
            print(f"STRUCTURAL SIMULATION: {args.start} to {args.end}")
            print(f"  Filters:           {', '.join(s['filters'])}")
            print(f"  Flat bet:          {s['flat_bet_pct'] * 100:.1f}% bankroll")
            print(f"  Book:              {args.book}")
            print("=" * 60)
            print(f"  Games evaluated:   {s['games_evaluated']:,}")
            print(f"  Bets placed:       {s['bets_placed']:,}  ({s['bet_pct']:.1f}% of games)")
            print(f"  Record:            {s['wins']}W – {s['losses']}L – {s['pushes']}P")
            print(f"  Win rate:          {s['win_rate']:.1f}%  (break-even: 52.4%)")
            print(f"  ROI:               {s['roi']:+.2f}%")
            print(f"  Bets/year:         {s['bets_per_year']:.0f}")
            print(f"  Sharpe (annual):   {s['sharpe_annualised']:.2f}")
            print(f"  Max drawdown:      -{s['max_drawdown']:.1f}%")
            print(f"  Final bankroll:    ${s['bankroll_final']:,.2f}")
            print("=" * 60)
            go = s["roi"] > 0 and s["sharpe_annualised"] >= 1.0
            if go:
                print("  GO — proceed to paper trading")
            else:
                print(f"  NO-GO — ROI {s['roi']:+.2f}%, Sharpe {s['sharpe_annualised']:.2f} < 1.0")
            print()

    elif args.command == "update-clv":
        n = update_clv(date=args.date, db_path=args.db)
        print(f"Updated CLV for {n} predictions")

    elif args.command == "daily" or (args.command is None and args.date):
        df = run_daily(
            date=args.date,
            db_path=args.db,
            model_name=args.model,
        )
        if df.empty:
            print(f"No priced predictions for {args.date or 'today'}")
        else:
            bets = df[df["bet_side"] != "PASS"]
            print(f"\n{len(df)} games priced — {len(bets)} bets recommended\n")
            if not bets.empty:
                print(
                    bets[
                        [
                            "home_team",
                            "away_team",
                            "line",
                            "over_prob",
                            "edge",
                            "bet_side",
                            "kelly_fraction",
                        ]
                    ].to_string(index=False)
                )
    else:
        parser.print_help()
