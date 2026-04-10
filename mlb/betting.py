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
) -> pd.DataFrame:
    """
    Price today's games and write bet recommendations to the predictions table.

    Reads lambda_home, lambda_away from the predictions table (written by
    mlb/model.py --predict), fetches live Kalshi and Polymarket mid-prices,
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

    Returns
    -------
    pd.DataFrame
        Priced predictions for the date, including PASS rows.
    """
    if date is None:
        date = date_type.today().isoformat()

    with get_conn(db_path) as conn:
        rows = conn.execute(
            """
            SELECT p.id, p.game_id, p.model_name, p.model_version,
                   p.lambda_home, p.lambda_away, p.dispersion_alpha,
                   p.kalshi_ticker, p.kalshi_mid_price, p.polymarket_mid_price,
                   k.line, k.open_interest,
                   g.home_team, g.away_team, g.game_time_et
            FROM predictions p
            JOIN games g ON p.game_id = g.game_id
            LEFT JOIN kalshi_markets k
                ON k.ticker = p.kalshi_ticker
                AND k.snapshot_ts = (
                    SELECT MAX(snapshot_ts)
                    FROM kalshi_markets
                    WHERE ticker = p.kalshi_ticker
                )
            WHERE g.date = ?
              AND p.model_name LIKE ?
            ORDER BY g.game_time_et
            """,
            (date, f"{model_name}%"),
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
    model_name: str = "gbr",
    book: str = "pinnacle",
    use_negbinom: bool = True,
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
                   g.date, g.home_team, g.away_team,
                   g.home_score, g.away_score,
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
              AND p.lambda_home  IS NOT NULL
              AND p.lambda_away  IS NOT NULL
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
        lam_h = float(row["lambda_home"])
        lam_a = float(row["lambda_away"])
        alpha = row["dispersion_alpha"]
        line = float(row["total_close"])
        over_odds = int(row["over_odds_close"])
        under_odds = int(row["under_odds_close"])
        actual_total = int(row["home_score"]) + int(row["away_score"])

        # Fair devigged price for the over
        raw_over = american_to_price(over_odds)
        raw_under = american_to_price(under_odds)
        fair_over, _ = devig_prices(raw_over, raw_under)

        # Model P(over) against closing line
        if use_negbinom and alpha is not None and alpha > 0:
            over_prob = p_over_negbinom(lam_h, lam_a, float(alpha), line)
        else:
            over_prob = p_over_poisson(lam_h, lam_a, line)

        ev_result = compute_ev(over_prob, fair_over, min_edge=min_edge)
        bet_side = ev_result["bet_side"]

        # Position limit per date
        n_open = positions_by_date.get(game_date, 0)
        ok, reason = passes_filters(bet_side, None, n_open)
        if not ok:
            continue

        win_prob = over_prob if bet_side == "OVER" else (1.0 - over_prob)
        bet_price = fair_over if bet_side == "OVER" else (1.0 - fair_over)
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
            b = (1.0 / bet_price) - 1.0
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
                "fair_over": round(fair_over, 4),
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
    print(f"  Market benchmark:  SBR {params['book'].capitalize()} closing line")
    print(f"  Min edge:          ${params['min_edge']:.2f}")
    print(f"  Kelly multiplier:  {params['kelly_mult']:.2f}x")
    print(f"  Initial bankroll:  ${params['initial_bankroll']:,.0f}")
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
        print("  ✓ GO: ROI positive, Sharpe > 0.5, CLV ≥ 0")
        print("    Model is ready for paper trading.")
    else:
        reasons = []
        if summary["roi"] <= 0:
            reasons.append(f"ROI {summary['roi']:+.2f}% ≤ 0")
        if summary["sharpe"] <= 0.5:
            reasons.append(f"Sharpe {summary['sharpe']:.3f} ≤ 0.5")
        if summary["avg_clv"] < 0:
            reasons.append(f"avg CLV {summary['avg_clv']:+.4f} < 0")
        print(f"  ✗ NO-GO: {'; '.join(reasons)}")
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
    sim_p.add_argument("--output", default=None, help="CSV path for per-bet detail")
    sim_p.add_argument("--db", default="data/mlb.db")

    # -- CLV update
    clv_p = subparsers.add_parser("update-clv", help="Update CLV after market close")
    clv_p.add_argument("--date", default=None)
    clv_p.add_argument("--db", default="data/mlb.db")

    # Legacy flat flags for backward compatibility with /backtest skill
    parser.add_argument("--date", default=None)
    parser.add_argument("--simulate", action="store_true")
    parser.add_argument("--start", default="2021-04-01")
    parser.add_argument("--end", default="2024-10-01")
    parser.add_argument("--min-edge", type=float, default=MIN_EDGE)
    parser.add_argument("--kelly-mult", type=float, default=KELLY_MULT)
    parser.add_argument("--initial-bankroll", type=float, default=1000.0)
    parser.add_argument("--model", default="gbr")
    parser.add_argument("--book", default="pinnacle")
    parser.add_argument("--output", default=None)
    parser.add_argument("--db", default="data/mlb.db")

    args = parser.parse_args()

    if args.command == "simulate" or getattr(args, "simulate", False):
        summary = simulate(
            start=getattr(args, "start", "2021-04-01"),
            end=getattr(args, "end", "2024-10-01"),
            min_edge=getattr(args, "min_edge", MIN_EDGE),
            kelly_mult=getattr(args, "kelly_mult", KELLY_MULT),
            initial_bankroll=getattr(args, "initial_bankroll", 1000.0),
            model_name=args.model,
            book=getattr(args, "book", "pinnacle"),
            output_path=args.output,
            db_path=args.db,
        )
        if summary:
            _print_simulation_report(
                summary,
                {
                    "start": getattr(args, "start", "2021-04-01"),
                    "end": getattr(args, "end", "2024-10-01"),
                    "min_edge": getattr(args, "min_edge", MIN_EDGE),
                    "kelly_mult": getattr(args, "kelly_mult", KELLY_MULT),
                    "initial_bankroll": getattr(args, "initial_bankroll", 1000.0),
                    "book": getattr(args, "book", "pinnacle"),
                },
            )

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
