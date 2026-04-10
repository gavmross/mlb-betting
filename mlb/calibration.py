"""
Probability calibration for MLB run totals.

Converts model-predicted λ_home and λ_away into P(over/under a line) via
Poisson convolution. Provides an optional NegBinom upgrade when overdispersion
is confirmed.

Functions
---------
p_over_poisson(lam_home, lam_away, line)
    Vectorised Poisson convolution — primary probability method.

p_over_negbinom(mu_home, mu_away, alpha, line)
    NegBinom convolution — use when dispersion > 1.2 across walk-forward folds.

calibrate_predictions(df, model_artefact, line_col)
    Attach over_prob / under_prob to a predictions DataFrame.

overdispersion_report(y_true, lambda_pred)
    Full dispersion diagnostic with go/no-go recommendation.

Usage
-----
    python -m mlb.calibration --check-dispersion --start 2022-04-07 --end 2024-10-01
"""

from __future__ import annotations

import argparse
import logging
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import nbinom, poisson

logger = logging.getLogger(__name__)

MAX_RUNS: int = 30  # P(team > 30 runs) ≈ 0 — safe truncation
DISPERSION_THRESHOLD: float = 1.2  # above this → recommend NegBinom upgrade


# ── Poisson convolution ───────────────────────────────────────────────────────


def p_over_poisson(
    lam_home: float,
    lam_away: float,
    line: float,
    max_runs: int = MAX_RUNS,
) -> float:
    """
    P(home_runs + away_runs > line) under independent Poisson assumption.

    Uses vectorised meshgrid — significantly faster than nested loops.

    Parameters
    ----------
    lam_home : float
        Expected home runs (λ from model).
    lam_away : float
        Expected away runs (λ from model).
    line : float
        Total line (e.g. 8.5).
    max_runs : int
        Upper truncation. P(team > max_runs) ≈ 0.

    Returns
    -------
    float
        P(total > line), in (0, 1).
    """
    h = np.arange(max_runs + 1)
    a = np.arange(max_runs + 1)
    H, A = np.meshgrid(h, a)
    joint = poisson.pmf(H, lam_home) * poisson.pmf(A, lam_away)
    return float(joint[line < H + A].sum())


def p_under_poisson(
    lam_home: float,
    lam_away: float,
    line: float,
    max_runs: int = MAX_RUNS,
) -> float:
    """
    P(home_runs + away_runs < line) under independent Poisson assumption.

    Note: P(exactly = line) is neither over nor under (push on whole-number lines).

    Parameters
    ----------
    lam_home : float
    lam_away : float
    line : float
    max_runs : int

    Returns
    -------
    float
    """
    h = np.arange(max_runs + 1)
    a = np.arange(max_runs + 1)
    H, A = np.meshgrid(h, a)
    joint = poisson.pmf(H, lam_home) * poisson.pmf(A, lam_away)
    return float(joint[line > H + A].sum())


def p_exact_poisson(
    lam_home: float,
    lam_away: float,
    total: int,
    max_runs: int = MAX_RUNS,
) -> float:
    """
    P(home_runs + away_runs == total) under independent Poisson assumption.

    Parameters
    ----------
    lam_home : float
    lam_away : float
    total : int
    max_runs : int

    Returns
    -------
    float
    """
    h = np.arange(max_runs + 1)
    a = np.arange(max_runs + 1)
    H, A = np.meshgrid(h, a)
    joint = poisson.pmf(H, lam_home) * poisson.pmf(A, lam_away)
    return float(joint[total == H + A].sum())


# ── NegBinom convolution ──────────────────────────────────────────────────────


def p_over_negbinom(
    mu_home: float,
    mu_away: float,
    alpha: float,
    line: float,
    max_runs: int = MAX_RUNS,
) -> float:
    """
    P(home_runs + away_runs > line) under independent Negative Binomial assumption.

    NegBinom parametrisation: mean=μ, variance=μ + α·μ².
    Maps to scipy.stats.nbinom(n=1/α, p=n/(n+μ)).

    Use this when dispersion ratio > 1.2 (confirmed overdispersion).

    Parameters
    ----------
    mu_home : float
        Expected home runs.
    mu_away : float
        Expected away runs.
    alpha : float
        Dispersion parameter (1/α = r in NegBinom). Estimated from data.
    line : float
    max_runs : int

    Returns
    -------
    float
    """
    n = 1.0 / alpha
    p_h = n / (n + mu_home)
    p_a = n / (n + mu_away)

    h = np.arange(max_runs + 1)
    a = np.arange(max_runs + 1)
    H, A = np.meshgrid(h, a)
    joint = nbinom.pmf(H, n, p_h) * nbinom.pmf(A, n, p_a)
    return float(joint[line < H + A].sum())


def estimate_alpha(
    y_true: np.ndarray,
    lambda_pred: np.ndarray,
) -> float:
    """
    Estimate NegBinom dispersion parameter alpha from residuals.

    Method: Method of moments — matches var(y)/mean(y) to 1 + alpha*mean(y).

    Parameters
    ----------
    y_true : np.ndarray
        Observed run counts.
    lambda_pred : np.ndarray
        Predicted λ from Poisson model.

    Returns
    -------
    float
        Estimated alpha (> 0). Returns 0.0 if Poisson fit is adequate.
    """
    y = np.asarray(y_true, dtype=float)
    lam = np.asarray(lambda_pred, dtype=float)
    mean_lam = float(np.mean(lam))
    if mean_lam <= 0:
        return 0.0
    dispersion = float(np.var(y) / mean_lam)
    # From var/mean = 1 + alpha*mean → alpha = (var/mean - 1) / mean
    alpha = max(0.0, (dispersion - 1.0) / mean_lam)
    return round(alpha, 6)


# ── Overdispersion diagnostic ─────────────────────────────────────────────────


def overdispersion_report(
    y_true: np.ndarray,
    lambda_pred: np.ndarray,
    label: str = "",
) -> dict[str, Any]:
    """
    Full overdispersion diagnostic.

    Parameters
    ----------
    y_true : np.ndarray
        Observed run counts.
    lambda_pred : np.ndarray
        Predicted λ from Poisson model.
    label : str
        Label for logging (e.g. 'home_runs').

    Returns
    -------
    dict
        Keys: mean_lambda, variance, dispersion_ratio, alpha, negbinom_recommended.
    """
    y = np.asarray(y_true, dtype=float)
    lam = np.asarray(lambda_pred, dtype=float)

    mean_lam = float(np.mean(lam))
    variance = float(np.var(y))
    dispersion = variance / mean_lam if mean_lam > 0 else float("nan")
    alpha = estimate_alpha(y, lam)
    recommended = dispersion > DISPERSION_THRESHOLD

    result: dict[str, Any] = {
        "label": label,
        "n": len(y),
        "mean_lambda": round(mean_lam, 4),
        "mean_y": round(float(np.mean(y)), 4),
        "variance": round(variance, 4),
        "dispersion_ratio": round(dispersion, 4),
        "alpha": alpha,
        "negbinom_recommended": recommended,
    }

    level = logging.WARNING if recommended else logging.INFO
    logger.log(
        level,
        "%s dispersion=%.3f (mean_λ=%.3f var=%.3f alpha=%.4f) → %s",
        label or "target",
        dispersion,
        mean_lam,
        variance,
        alpha,
        "NEGBINOM RECOMMENDED" if recommended else "Poisson OK",
    )

    return result


# ── Batch calibration ─────────────────────────────────────────────────────────


def calibrate_predictions(
    df: pd.DataFrame,
    line_col: str = "total_line_close",
    use_negbinom: bool = False,
    alpha: float = 0.0,
) -> pd.DataFrame:
    """
    Compute P(over) and P(under) for each row in a predictions DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Must have columns: lambda_home, lambda_away, and line_col.
    line_col : str
        Column containing the total line (e.g. 'total_line_close').
    use_negbinom : bool
        If True, use NegBinom convolution instead of Poisson.
    alpha : float
        NegBinom dispersion parameter. Required when use_negbinom=True.

    Returns
    -------
    pd.DataFrame
        Input DataFrame with over_prob and under_prob columns added.
    """
    if line_col not in df.columns:
        raise ValueError(f"Line column '{line_col}' not found in DataFrame")
    if "lambda_home" not in df.columns or "lambda_away" not in df.columns:
        raise ValueError("DataFrame must have lambda_home and lambda_away columns")

    df = df.copy()
    over_probs = []
    under_probs = []

    for _, row in df.iterrows():
        lam_h = float(row["lambda_home"])
        lam_a = float(row["lambda_away"])
        line = row[line_col]

        if pd.isna(line) or pd.isna(lam_h) or pd.isna(lam_a):
            over_probs.append(float("nan"))
            under_probs.append(float("nan"))
            continue

        line = float(line)

        if use_negbinom and alpha > 0:
            p_over = p_over_negbinom(lam_h, lam_a, alpha, line)
        else:
            p_over = p_over_poisson(lam_h, lam_a, line)

        p_under = p_under_poisson(lam_h, lam_a, line)
        over_probs.append(p_over)
        under_probs.append(p_under)

    df["over_prob"] = over_probs
    df["under_prob"] = under_probs

    return df


# ── Vectorised batch for performance ─────────────────────────────────────────


def calibrate_batch(
    lambda_home: np.ndarray,
    lambda_away: np.ndarray,
    lines: np.ndarray,
    max_runs: int = MAX_RUNS,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Vectorised batch calibration for many games at once.

    Builds the joint PMF once per unique (lam_home, lam_away) pair, then
    applies the line mask. More efficient than calling p_over_poisson per row.

    Parameters
    ----------
    lambda_home : np.ndarray  shape (N,)
    lambda_away : np.ndarray  shape (N,)
    lines : np.ndarray        shape (N,)
    max_runs : int

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (over_probs, under_probs), each shape (N,).
    """
    n = len(lambda_home)
    over_probs = np.full(n, float("nan"))
    under_probs = np.full(n, float("nan"))

    h = np.arange(max_runs + 1)
    a = np.arange(max_runs + 1)
    H, A = np.meshgrid(h, a)  # (max_runs+1, max_runs+1)
    totals = H + A             # total runs grid

    for i in range(n):
        lh = float(lambda_home[i])
        la = float(lambda_away[i])
        line = float(lines[i])
        if np.isnan(lh) or np.isnan(la) or np.isnan(line):
            continue
        joint = poisson.pmf(H, lh) * poisson.pmf(A, la)
        over_probs[i] = float(joint[totals > line].sum())
        under_probs[i] = float(joint[totals < line].sum())

    return over_probs, under_probs


# ── Calibration verification ──────────────────────────────────────────────────


def verify_convolution_sums_to_one(
    lam_home: float = 4.5,
    lam_away: float = 4.0,
    line: float = 8.5,
    max_runs: int = MAX_RUNS,
    atol: float = 1e-6,
) -> bool:
    """
    Sanity check: P(over) + P(under) + P(push) must equal 1.0.

    On half-line (e.g. 8.5), P(push) = 0 since total is always integer.
    On whole-line (e.g. 8.0), P(push) > 0.

    Parameters
    ----------
    lam_home : float
    lam_away : float
    line : float
    max_runs : int
    atol : float

    Returns
    -------
    bool
        True if sum ≈ 1.0, False otherwise.
    """
    p_over = p_over_poisson(lam_home, lam_away, line, max_runs)
    p_under = p_under_poisson(lam_home, lam_away, line, max_runs)

    # P(push) is non-zero only on integer lines
    p_push = (
        p_exact_poisson(lam_home, lam_away, int(line), max_runs) if line == int(line) else 0.0
    )

    total = p_over + p_under + p_push
    ok = abs(total - 1.0) < atol

    if not ok:
        logger.warning(
            "Convolution sum check failed: p_over=%.6f p_under=%.6f p_push=%.6f "
            "total=%.6f (expected 1.0, atol=%.2e)",
            p_over, p_under, p_push, total, atol,
        )
    return ok


# ── CLI ───────────────────────────────────────────────────────────────────────


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="MLB calibration checks")
    parser.add_argument(
        "--check-dispersion",
        action="store_true",
        help="Run overdispersion check on walk-forward OOF predictions",
    )
    parser.add_argument("--start", default="2022-04-07")
    parser.add_argument("--end", default="2024-10-01")
    parser.add_argument("--model", choices=["glm", "gbr"], default="gbr")
    args = parser.parse_args()

    if args.check_dispersion:
        from mlb.features import build_features
        from mlb.model import _prepare_xy, make_gbr_poisson, make_poisson_glm

        print(f"Loading features {args.start} to {args.end}...")
        df = build_features(start_date=args.start, end_date=args.end)

        print("Running walk-forward CV to collect OOF predictions...")
        # Use smaller n_splits for speed here
        from sklearn.model_selection import TimeSeriesSplit

        df_sorted = df.sort_values("date").reset_index(drop=True)
        X_all, y_home_all, y_away_all = _prepare_xy(df_sorted)

        tscv = TimeSeriesSplit(n_splits=3, gap=162)
        oof_home_true, oof_home_pred = [], []
        oof_away_true, oof_away_pred = [], []

        for train_idx, test_idx in tscv.split(X_all):
            X_tr = X_all.iloc[train_idx]
            X_te = X_all.iloc[test_idx].copy()
            for col in X_te.columns:
                if X_te[col].isna().any():
                    med = X_tr[col].median()
                    X_te[col] = X_te[col].fillna(0.0 if np.isnan(med) else med)

            m_h = make_gbr_poisson() if args.model == "gbr" else make_poisson_glm()
            m_a = make_gbr_poisson() if args.model == "gbr" else make_poisson_glm()
            m_h.fit(X_tr, y_home_all.iloc[train_idx])
            m_a.fit(X_tr, y_away_all.iloc[train_idx])

            oof_home_true.extend(y_home_all.iloc[test_idx].tolist())
            oof_home_pred.extend(np.clip(m_h.predict(X_te), 0.01, 30.0).tolist())
            oof_away_true.extend(y_away_all.iloc[test_idx].tolist())
            oof_away_pred.extend(np.clip(m_a.predict(X_te), 0.01, 30.0).tolist())

        print("\n=== Overdispersion Report ===")
        r_h = overdispersion_report(np.array(oof_home_true), np.array(oof_home_pred), "home_runs")
        r_a = overdispersion_report(np.array(oof_away_true), np.array(oof_away_pred), "away_runs")

        for r in (r_h, r_a):
            print(
                f"  {r['label']}: dispersion={r['dispersion_ratio']:.3f}  "
                f"alpha={r['alpha']:.4f}  "
                f"→ {'UPGRADE TO NEGBINOM' if r['negbinom_recommended'] else 'Poisson OK'}"
            )

        # Convolution sum check
        print("\n=== Convolution Sanity Check ===")
        ok = verify_convolution_sums_to_one(4.5, 4.0, 8.5)
        print(f"  P(over) + P(under) + P(push) ≈ 1.0: {'PASS' if ok else 'FAIL'}")
