"""
MLB run-scoring prediction model.

Two-target Poisson approach:
  - Model 1 (baseline):  PoissonRegressor for home_runs + away_runs
  - Model 2 (primary):   HistGradientBoostingRegressor(loss='poisson') for home_runs + away_runs

Note: HistGradientBoostingRegressor is used instead of GradientBoostingRegressor because
loss='poisson' is only available on the histogram-based variant (sklearn >= 0.23).
HGBR also handles NaN natively, eliminating the need for explicit imputation on the GBR path.

Walk-forward cross-validation via TimeSeriesSplit(n_splits=5, gap=162).
Never KFold, never shuffle=True.

Usage
-----
    python -m mlb.model --train
    python -m mlb.model --backtest --n-splits 5
    python -m mlb.model --predict --date 2024-06-15
"""

from __future__ import annotations

import argparse
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import d2_tweedie_score, mean_poisson_deviance
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from mlb.db import get_conn
from mlb.features import FEATURE_COLS, build_features

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

MODEL_DIR = Path("data/models")
MODEL_VERSION = "1.0.0"

# Features that contain genuine pre-game market signal (allowed as inputs)
MARKET_FEATURES = {"total_line_open", "total_line_close", "line_movement"}

# Feature columns that go into the model (subset of FEATURE_COLS)
# Exclude total_line_close from training features if doing pure model comparison;
# include it when building the full production model.
TRAIN_FEATURES: list[str] = FEATURE_COLS  # all 49 features including market line


# ── Model factory ─────────────────────────────────────────────────────────────


def make_poisson_glm(alpha: float = 1.0) -> Pipeline:
    """
    Baseline Poisson GLM.

    Parameters
    ----------
    alpha : float
        L2 regularisation strength.

    Returns
    -------
    sklearn.pipeline.Pipeline
    """
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", PoissonRegressor(alpha=alpha, max_iter=500, warm_start=False)),
        ]
    )


def make_gbr_poisson(
    max_iter: int = 300,
    max_depth: int = 4,
    learning_rate: float = 0.05,
    min_samples_leaf: int = 20,
    l2_regularization: float = 0.1,
    random_state: int = 42,
) -> HistGradientBoostingRegressor:
    """
    Primary model: HistGradientBoostingRegressor with Poisson loss.

    Uses the histogram-based variant which supports loss='poisson' and
    handles NaN natively (no explicit imputation required).

    Parameters
    ----------
    max_iter : int
        Number of boosting iterations.
    max_depth : int
    learning_rate : float
    min_samples_leaf : int
    l2_regularization : float
    random_state : int

    Returns
    -------
    HistGradientBoostingRegressor
    """
    return HistGradientBoostingRegressor(
        loss="poisson",
        max_iter=max_iter,
        max_depth=max_depth,
        learning_rate=learning_rate,
        min_samples_leaf=min_samples_leaf,
        l2_regularization=l2_regularization,
        random_state=random_state,
    )


# ── Data preparation ──────────────────────────────────────────────────────────


def _prepare_xy(
    df: pd.DataFrame,
    feature_cols: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Extract feature matrix and two target vectors from a feature DataFrame.

    Rows with null targets are dropped. Null features are filled with column median.

    Parameters
    ----------
    df : pd.DataFrame
        Output of build_features().
    feature_cols : list[str] or None
        Columns to use as features. Defaults to TRAIN_FEATURES.

    Returns
    -------
    tuple[pd.DataFrame, pd.Series, pd.Series]
        (X, y_home, y_away) — all same index.
    """
    if feature_cols is None:
        feature_cols = TRAIN_FEATURES

    present = [c for c in feature_cols if c in df.columns]
    missing_cols = set(feature_cols) - set(present)
    if missing_cols:
        logger.warning("Feature columns missing from DataFrame: %s", missing_cols)

    sub = df.dropna(subset=["home_runs", "away_runs"]).copy()
    X = sub[present].copy()

    # Fill nulls with column median (0 for all-null columns like precip_prob in history)
    for col in X.columns:
        if X[col].isna().any():
            med = X[col].median()
            X[col] = X[col].fillna(0.0 if np.isnan(med) else med)

    y_home = sub["home_runs"].astype(float)
    y_away = sub["away_runs"].astype(float)

    return X, y_home, y_away


def _fill_test_nulls(X_train: pd.DataFrame, X_test: pd.DataFrame) -> pd.DataFrame:
    """
    Fill nulls in X_test using medians computed from X_train.

    Parameters
    ----------
    X_train : pd.DataFrame
    X_test : pd.DataFrame

    Returns
    -------
    pd.DataFrame
        X_test with nulls filled.
    """
    X_test = X_test.copy()
    for col in X_test.columns:
        if X_test[col].isna().any():
            med = X_train[col].median()
            X_test[col] = X_test[col].fillna(0.0 if np.isnan(med) else med)
    return X_test


# ── Overdispersion check ──────────────────────────────────────────────────────


def check_overdispersion(y_true: np.ndarray, lambda_pred: np.ndarray) -> float:
    """
    Compute dispersion ratio: var(residuals) / mean(lambda_pred).

    Parameters
    ----------
    y_true : np.ndarray
    lambda_pred : np.ndarray

    Returns
    -------
    float
        Dispersion ratio. > 1.2 → consider NegBinom upgrade.
    """
    residuals = np.asarray(y_true, dtype=float) - np.asarray(lambda_pred, dtype=float)
    return float(residuals.var() / np.mean(lambda_pred))


# ── Walk-forward cross-validation ─────────────────────────────────────────────


def walk_forward_cv(
    df: pd.DataFrame,
    n_splits: int = 5,
    gap: int = 162,
    model_type: str = "gbr",
    feature_cols: list[str] | None = None,
) -> dict[str, Any]:
    """
    Walk-forward cross-validation for both home_runs and away_runs models.

    Parameters
    ----------
    df : pd.DataFrame
        Feature DataFrame sorted by date ascending.
    n_splits : int
        Number of CV folds.
    gap : int
        Minimum gap (rows) between train and test sets. 162 ≈ one season.
    model_type : str
        'glm' for PoissonRegressor, 'gbr' for GradientBoostingRegressor.
    feature_cols : list[str] or None

    Returns
    -------
    dict
        Keys: fold_results (list of per-fold metrics), summary (aggregate metrics).
    """
    df_sorted = df.sort_values("date").reset_index(drop=True)
    X_all, y_home_all, y_away_all = _prepare_xy(df_sorted, feature_cols)

    tscv = TimeSeriesSplit(n_splits=n_splits, gap=gap)
    fold_results = []

    for fold_idx, (train_idx, test_idx) in enumerate(tscv.split(X_all)):
        X_train = X_all.iloc[train_idx]
        X_test_raw = X_all.iloc[test_idx]
        X_test = _fill_test_nulls(X_train, X_test_raw)

        y_home_train = y_home_all.iloc[train_idx]
        y_home_test = y_home_all.iloc[test_idx]
        y_away_train = y_away_all.iloc[train_idx]
        y_away_test = y_away_all.iloc[test_idx]

        train_dates = df_sorted.iloc[train_idx]["date"]
        test_dates = df_sorted.iloc[test_idx]["date"]

        # Fit models
        if model_type == "glm":
            m_home = make_poisson_glm()
            m_away = make_poisson_glm()
        else:
            m_home = make_gbr_poisson()
            m_away = make_gbr_poisson()

        m_home.fit(X_train, y_home_train)
        m_away.fit(X_train, y_away_train)

        lam_home = m_home.predict(X_test)
        lam_away = m_away.predict(X_test)

        # Clip to valid Poisson range
        lam_home = np.clip(lam_home, 0.01, 30.0)
        lam_away = np.clip(lam_away, 0.01, 30.0)

        # Metrics
        dev_home = mean_poisson_deviance(y_home_test, lam_home)
        dev_away = mean_poisson_deviance(y_away_test, lam_away)
        d2_home = d2_tweedie_score(y_home_test, lam_home, power=1)
        d2_away = d2_tweedie_score(y_away_test, lam_away, power=1)
        disp_home = check_overdispersion(y_home_test.values, lam_home)
        disp_away = check_overdispersion(y_away_test.values, lam_away)

        result = {
            "fold": fold_idx + 1,
            "train_start": train_dates.min(),
            "train_end": train_dates.max(),
            "test_start": test_dates.min(),
            "test_end": test_dates.max(),
            "n_train": len(train_idx),
            "n_test": len(test_idx),
            "dev_home": round(dev_home, 4),
            "dev_away": round(dev_away, 4),
            "d2_home": round(d2_home, 4),
            "d2_away": round(d2_away, 4),
            "disp_home": round(disp_home, 3),
            "disp_away": round(disp_away, 3),
        }
        fold_results.append(result)

        logger.info(
            "Fold %d | train %s→%s | test %s→%s | "
            "dev_home=%.4f d2_home=%.4f | dev_away=%.4f d2_away=%.4f | "
            "disp_home=%.3f disp_away=%.3f",
            fold_idx + 1,
            train_dates.min(), train_dates.max(),
            test_dates.min(), test_dates.max(),
            dev_home, d2_home, dev_away, d2_away,
            disp_home, disp_away,
        )

    # Aggregate
    devs_home = [r["dev_home"] for r in fold_results]
    devs_away = [r["dev_away"] for r in fold_results]
    d2s_home = [r["d2_home"] for r in fold_results]
    d2s_away = [r["d2_away"] for r in fold_results]
    disps_home = [r["disp_home"] for r in fold_results]
    disps_away = [r["disp_away"] for r in fold_results]

    summary = {
        "model_type": model_type,
        "n_splits": n_splits,
        "mean_dev_home": round(float(np.mean(devs_home)), 4),
        "mean_dev_away": round(float(np.mean(devs_away)), 4),
        "mean_d2_home": round(float(np.mean(d2s_home)), 4),
        "mean_d2_away": round(float(np.mean(d2s_away)), 4),
        "mean_disp_home": round(float(np.mean(disps_home)), 3),
        "mean_disp_away": round(float(np.mean(disps_away)), 3),
        "negbinom_recommended": float(np.mean(disps_home)) > 1.2
            or float(np.mean(disps_away)) > 1.2,
    }

    logger.info(
        "CV summary (%s): dev_home=%.4f dev_away=%.4f "
        "d2_home=%.4f d2_away=%.4f disp_home=%.3f disp_away=%.3f",
        model_type,
        summary["mean_dev_home"], summary["mean_dev_away"],
        summary["mean_d2_home"], summary["mean_d2_away"],
        summary["mean_disp_home"], summary["mean_disp_away"],
    )
    if summary["negbinom_recommended"]:
        logger.warning(
            "Overdispersion > 1.2 detected — consider NegBinom upgrade in calibration.py"
        )

    return {"fold_results": fold_results, "summary": summary}


# ── Train final model ─────────────────────────────────────────────────────────


def train(
    start_date: str = "2022-04-07",
    end_date: str = "2024-10-01",
    model_type: str = "gbr",
    db_path: str = "data/mlb.db",
    save: bool = True,
) -> dict[str, Any]:
    """
    Train final models on the full date range and optionally save to disk.

    Parameters
    ----------
    start_date : str
    end_date : str
    model_type : str
        'glm' or 'gbr'.
    db_path : str
    save : bool
        If True, save artefact to data/models/.

    Returns
    -------
    dict
        Artefact dict with fitted models and metadata.
    """
    logger.info("Loading features %s to %s", start_date, end_date)
    df = build_features(start_date=start_date, end_date=end_date, db_path=db_path)

    X, y_home, y_away = _prepare_xy(df)

    logger.info("Training %s on %d games", model_type, len(X))

    if model_type == "glm":
        m_home = make_poisson_glm()
        m_away = make_poisson_glm()
    else:
        m_home = make_gbr_poisson()
        m_away = make_gbr_poisson()

    m_home.fit(X, y_home)
    m_away.fit(X, y_away)

    lam_home_train = np.clip(m_home.predict(X), 0.01, 30.0)
    lam_away_train = np.clip(m_away.predict(X), 0.01, 30.0)

    train_dev_home = mean_poisson_deviance(y_home, lam_home_train)
    train_dev_away = mean_poisson_deviance(y_away, lam_away_train)

    artefact: dict[str, Any] = {
        "model_home": m_home,
        "model_away": m_away,
        "feature_names": list(X.columns),
        "model_type": model_type,
        "train_start": start_date,
        "train_end": end_date,
        "n_train": len(X),
        "train_dev_home": round(train_dev_home, 4),
        "train_dev_away": round(train_dev_away, 4),
        "version": MODEL_VERSION,
        "trained_at": datetime.now(UTC).isoformat(),
    }

    if save:
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        path = MODEL_DIR / f"{model_type}_poisson_v{MODEL_VERSION}.pkl"
        joblib.dump(artefact, path)
        logger.info("Saved model artefact to %s", path)

    logger.info(
        "Train complete: dev_home=%.4f dev_away=%.4f",
        train_dev_home, train_dev_away,
    )
    return artefact


# ── Predict ───────────────────────────────────────────────────────────────────


def predict(
    date: str,
    artefact: dict[str, Any] | None = None,
    model_type: str = "gbr",
    db_path: str = "data/mlb.db",
) -> pd.DataFrame:
    """
    Generate lambda predictions for all games on a given date.

    Parameters
    ----------
    date : str
        Game date in YYYY-MM-DD format.
    artefact : dict or None
        Pre-loaded model artefact. If None, loaded from disk.
    model_type : str
        'glm' or 'gbr' — used to locate artefact file if artefact is None.
    db_path : str

    Returns
    -------
    pd.DataFrame
        One row per game with columns: game_id, date, home_team, away_team,
        lambda_home, lambda_away, predicted_total_runs.
    """
    if artefact is None:
        path = MODEL_DIR / f"{model_type}_poisson_v{MODEL_VERSION}.pkl"
        if not path.exists():
            raise FileNotFoundError(
                f"No model artefact at {path} — run --train first"
            )
        artefact = joblib.load(path)

    df = build_features(start_date=date, end_date=date, db_path=db_path)
    if df.empty:
        logger.warning("No games found for date %s", date)
        return pd.DataFrame()

    feature_names = artefact["feature_names"]
    present = [c for c in feature_names if c in df.columns]
    X = df[present].copy()
    for col in X.columns:
        if X[col].isna().any():
            X[col] = X[col].fillna(X[col].median())

    lam_home = np.clip(artefact["model_home"].predict(X), 0.01, 30.0)
    lam_away = np.clip(artefact["model_away"].predict(X), 0.01, 30.0)

    result = df[["game_id", "date", "home_team", "away_team"]].copy()
    result["lambda_home"] = lam_home
    result["lambda_away"] = lam_away
    result["predicted_total_runs"] = lam_home + lam_away

    return result.reset_index(drop=True)


def write_predictions(
    preds: pd.DataFrame,
    model_type: str,
    db_path: str = "data/mlb.db",
) -> int:
    """
    Write lambda predictions to the predictions table.

    Parameters
    ----------
    preds : pd.DataFrame
        Output of predict().
    model_type : str
    db_path : str

    Returns
    -------
    int
        Number of rows upserted.
    """
    if preds.empty:
        return 0

    now = datetime.now(UTC).isoformat()
    rows = 0
    with get_conn(db_path) as conn:
        for _, row in preds.iterrows():
            conn.execute(
                """INSERT OR REPLACE INTO predictions
                   (game_id, model_name, model_version, predicted_at,
                    lambda_home, lambda_away, predicted_total_runs)
                   VALUES (?,?,?,?,?,?,?)""",
                (
                    row["game_id"],
                    model_type,
                    MODEL_VERSION,
                    now,
                    round(float(row["lambda_home"]), 4),
                    round(float(row["lambda_away"]), 4),
                    round(float(row["predicted_total_runs"]), 4),
                ),
            )
            rows += 1
    logger.info("Wrote %d predictions for model=%s", rows, model_type)
    return rows


# ── Feature importance ────────────────────────────────────────────────────────


def feature_importance(
    artefact: dict[str, Any],
    X_sample: pd.DataFrame | None = None,
    y_home_sample: pd.Series | None = None,
    y_away_sample: pd.Series | None = None,
) -> pd.DataFrame:
    """
    Return feature importances for the GBR model.

    HistGradientBoostingRegressor does not expose .feature_importances_ directly.
    When X_sample + y_*_sample are provided, permutation importance is computed.
    Otherwise raises ValueError for HGBR.

    Parameters
    ----------
    artefact : dict
    X_sample : pd.DataFrame, optional
        Held-out sample for permutation importance (HGBR only).
    y_home_sample : pd.Series, optional
    y_away_sample : pd.Series, optional

    Returns
    -------
    pd.DataFrame
        Columns: feature, importance_home, importance_away, importance_mean.
        Sorted by importance_mean descending.
    """
    if artefact["model_type"] != "gbr":
        raise ValueError("Feature importance only available for GBR models")

    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.inspection import permutation_importance

    m_home = artefact["model_home"]
    m_away = artefact["model_away"]
    names = artefact["feature_names"]

    if isinstance(m_home, HistGradientBoostingRegressor):
        if X_sample is None or y_home_sample is None or y_away_sample is None:
            raise ValueError(
                "HistGradientBoostingRegressor requires X_sample, y_home_sample, "
                "and y_away_sample for permutation importance. "
                "Consider using SHAP for model-agnostic importance instead."
            )
        r_home = permutation_importance(
            m_home, X_sample, y_home_sample, n_repeats=5, random_state=42
        )
        r_away = permutation_importance(
            m_away, X_sample, y_away_sample, n_repeats=5, random_state=42
        )
        imp_home = r_home.importances_mean
        imp_away = r_away.importances_mean
        # Clip negative values to 0 (noise)
        imp_home = np.clip(imp_home, 0.0, None)
        imp_away = np.clip(imp_away, 0.0, None)
    else:
        imp_home = m_home.feature_importances_
        imp_away = m_away.feature_importances_

    df = pd.DataFrame(
        {
            "feature": names,
            "importance_home": imp_home,
            "importance_away": imp_away,
            "importance_mean": (imp_home + imp_away) / 2,
        }
    ).sort_values("importance_mean", ascending=False)

    return df.reset_index(drop=True)


# ── CLI ───────────────────────────────────────────────────────────────────────


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="MLB Poisson run model")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--train", action="store_true", help="Train final model")
    group.add_argument("--backtest", action="store_true", help="Walk-forward CV")
    group.add_argument("--predict", action="store_true", help="Predict for --date")

    parser.add_argument("--model", choices=["glm", "gbr"], default="gbr")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--start", default="2022-04-07")
    parser.add_argument("--end", default="2024-10-01")
    parser.add_argument("--date", default=None)
    parser.add_argument("--no-save", action="store_true")

    args = parser.parse_args()

    if args.train:
        train(
            start_date=args.start,
            end_date=args.end,
            model_type=args.model,
            save=not args.no_save,
        )

    elif args.backtest:
        df = build_features(start_date=args.start, end_date=args.end)
        results = walk_forward_cv(df, n_splits=args.n_splits, model_type=args.model)
        print("\n=== Walk-Forward CV Results ===")
        for fold in results["fold_results"]:
            print(
                f"Fold {fold['fold']} | "
                f"train {fold['train_start']}→{fold['train_end']} | "
                f"test  {fold['test_start']}→{fold['test_end']} | "
                f"dev_home={fold['dev_home']:.4f} d2_home={fold['d2_home']:.4f} | "
                f"dev_away={fold['dev_away']:.4f} d2_away={fold['d2_away']:.4f} | "
                f"disp_home={fold['disp_home']:.3f} disp_away={fold['disp_away']:.3f}"
            )
        s = results["summary"]
        print(f"\nMean: dev_home={s['mean_dev_home']:.4f} dev_away={s['mean_dev_away']:.4f} "
              f"d2_home={s['mean_d2_home']:.4f} d2_away={s['mean_d2_away']:.4f}")
        if s["negbinom_recommended"]:
            print("*** NegBinom upgrade recommended (mean dispersion > 1.2) ***")

    elif args.predict:
        date = args.date or datetime.now().strftime("%Y-%m-%d")
        preds = predict(date=date, model_type=args.model)
        if preds.empty:
            print(f"No games found for {date}")
        else:
            write_predictions(preds, model_type=args.model)
            print(preds[["home_team", "away_team", "lambda_home",
                          "lambda_away", "predicted_total_runs"]].to_string(index=False))
