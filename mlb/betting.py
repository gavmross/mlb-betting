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
    sp_era_max: float | None = None,
    output_path: str | None = None,
    db_path: str = "data/mlb.db",
) -> dict[str, Any]:
    """
    Walk-forward betting simulation over historical data.

    Uses SBR closing lines (Pinnacle by default) as the market benchmark since
    Kalshi data starts April 2025.  Converts American odds to fair devigged
    probabilities, applies the same EV/Kelly/filter logic as run_daily().

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
        Sportsbook to use for odds (default 'pinnacle' — sharpest line).
    use_negbinom : bool
        Use NegBinom convolution when dispersion_alpha available.
    sp_era_max : float or None
        If set, skip games where the average starting-pitcher ERA (home + away)
        exceeds this threshold.  E.g. 3.75 restricts bets to quality pitching
        matchups where the model has demonstrated higher accuracy.
        None = no filter (default).
    output_path : str or None
        If set, write per-bet detail to this CSV path.
    db_path : str

    Returns
    -------
    dict
        Summary statistics: roi, win_rate, bets_placed, sharpe,
        max_drawdown, avg_clv, bankroll_final.
    """
    with get_conn(db_path) as conn:
        # Join predictions → games → sportsbook_odds for the simulation window
        rows = conn.execute(
            """
            SELECT p.game_id,
                   p.model_name, p.model_version,
                   p.lambda_home, p.lambda_away,
                   p.dispersion_alpha,
                   p.over_prob AS stored_over_prob,
                   g.date, g.home_team, g.away_team,
                   g.home_score, g.away_score,
                   o.total_close,
                   o.over_odds_close, o.under_odds_close,
                   ph.era_season AS home_sp_era,
                   pa.era_season AS away_sp_era
            FROM predictions p
            JOIN games g ON p.game_id = g.game_id
            JOIN sportsbook_odds o
                ON o.date        = g.date
               AND o.home_team   = g.home_team
               AND o.book        = ?
            LEFT JOIN pitchers ph
                ON ph.game_id = p.game_id
               AND ph.team = g.home_team
               AND ph.is_starter = 1
            LEFT JOIN pitchers pa
                ON pa.game_id = p.game_id
               AND pa.team = g.away_team
               AND pa.is_starter = 1
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

        # SP ERA filter — skip weak-pitching matchups where model accuracy is poor
        if sp_era_max is not None:
            home_era = row["home_sp_era"]
            away_era = row["away_sp_era"]
            if (
                home_era is not None
                and away_era is not None
                and (float(home_era) + float(away_era)) / 2.0 > sp_era_max
            ):
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

        h_era = row["home_sp_era"]
        a_era = row["away_sp_era"]
        sp_era_avg = (
            round((float(h_era) + float(a_era)) / 2.0, 2)
            if h_era is not None and a_era is not None
            else None
        )

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
                "sp_era_avg": sp_era_avg,
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

    # Per-bet ROI for Sharpe
    bet_rois = df_bets["pnl"].values / df_bets["stake"].values
    sharpe = float(np.mean(bet_rois) / np.std(bet_rois)) if np.std(bet_rois) > 0 else 0.0

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
        "sharpe": round(sharpe, 3),
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
    if "sp_era_max" in params:
        print(f"  SP ERA filter:     avg ERA <= {params['sp_era_max']:.2f}")
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
    print(f"  Sharpe ratio:      {summary['sharpe']:.3f}")
    print(f"  Average CLV:       {summary['avg_clv']:+.4f}")
    print()
    print(
        f"  Final bankroll:    ${summary['bankroll_final']:,.2f}"
        f"  (peak: ${summary['bankroll_peak']:,.2f})"
    )
    print("=" * 60)

    # Go/no-go decision
    print()
    go = summary["roi"] > 0 and summary["sharpe"] > 0.5 and summary["avg_clv"] >= 0
    if go:
        print("  GO: ROI positive, Sharpe > 0.5, CLV >= 0")
        print("    Model is ready for paper trading.")
    else:
        reasons = []
        if summary["roi"] <= 0:
            reasons.append(f"ROI {summary['roi']:+.2f}% <= 0")
        if summary["sharpe"] <= 0.5:
            reasons.append(f"Sharpe {summary['sharpe']:.3f} <= 0.5")
        if summary["avg_clv"] < 0:
            reasons.append(f"avg CLV {summary['avg_clv']:+.4f} < 0")
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
        "--sp-era-max", type=float, default=None,
        help="Only bet games where avg SP ERA (home+away)/2 <= this value. "
             "Recommended: 3.75. None = no filter.",
    )
    sim_p.add_argument("--output", default=None, help="CSV path for per-bet detail")
    sim_p.add_argument("--db", default="data/mlb.db")

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
    parser.add_argument("--sp-era-max", type=float, default=None, dest="sp_era_max")
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
        sp_era_max = getattr(args, "sp_era_max", None)
        summary = simulate(
            start=getattr(args, "start", "2021-04-01"),
            end=getattr(args, "end", "2024-10-01"),
            min_edge=getattr(args, "min_edge", MIN_EDGE),
            kelly_mult=getattr(args, "kelly_mult", KELLY_MULT),
            initial_bankroll=getattr(args, "initial_bankroll", 1000.0),
            model_name=args.model,
            book=getattr(args, "book", "pinnacle"),
            sp_era_max=sp_era_max,
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
                "book": getattr(args, "book", "pinnacle"),
            }
            if sp_era_max is not None:
                params["sp_era_max"] = sp_era_max
            _print_simulation_report(summary, params)

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
