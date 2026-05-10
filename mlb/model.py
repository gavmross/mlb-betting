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
import statsmodels.api as sm
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import d2_tweedie_score, mean_poisson_deviance
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from mlb.db import get_conn
from mlb.features import FEATURE_COLS, build_features, build_predict_features

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


# ── NegBinom GLM wrapper ──────────────────────────────────────────────────────


class NegBinomGLMWrapper:
    """
    Sklearn-compatible wrapper around statsmodels NegBinom GLM.

    Handles scaling internally so the calling code has a uniform fit/predict
    interface identical to PoissonRegressor and HistGradientBoostingRegressor.

    Parameters
    ----------
    alpha : float
        NegBinom dispersion parameter.  var(y) = μ + alpha·μ².
        Estimate with :func:`mlb.calibration.estimate_alpha` before fitting.
    """

    def __init__(self, alpha: float = 1.0) -> None:
        self.alpha = alpha
        self._scaler: StandardScaler = StandardScaler()
        self._model: Any | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> NegBinomGLMWrapper:
        """
        Fit the NegBinom GLM.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix (nulls filled before calling).
        y : pd.Series
            Target run counts.

        Returns
        -------
        NegBinomGLMWrapper
            self
        """
        X_sc = self._scaler.fit_transform(X.fillna(0.0))
        X_sm = sm.add_constant(X_sc, has_constant="add")
        self._model = sm.GLM(
            y.values,
            X_sm,
            family=sm.families.NegativeBinomial(alpha=self.alpha),
        ).fit(disp=False)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict expected run counts (μ).

        Parameters
        ----------
        X : pd.DataFrame

        Returns
        -------
        np.ndarray
            Predicted μ values, clipped to [0.01, 30].
        """
        if self._model is None:
            raise RuntimeError("Call fit() before predict()")
        X_sc = self._scaler.transform(X.fillna(0.0))
        X_sm = sm.add_constant(X_sc, has_constant="add")
        return np.clip(self._model.predict(X_sm), 0.01, 30.0)


def make_negbinom_glm(alpha: float = 1.0) -> NegBinomGLMWrapper:
    """
    Negative Binomial GLM — use when walk-forward dispersion ratio > 1.2.

    Alpha should be estimated from training data via
    :func:`mlb.calibration.estimate_alpha` before each fold.

    Parameters
    ----------
    alpha : float
        Dispersion parameter.

    Returns
    -------
    NegBinomGLMWrapper
    """
    return NegBinomGLMWrapper(alpha=alpha)


# ── Data preparation ──────────────────────────────────────────────────────────


_TARGET_COLS: dict[str, tuple[str, str]] = {
    "fullgame": ("home_runs", "away_runs"),
    "f5": ("f5_home_score", "f5_away_score"),
}


def _prepare_xy(
    df: pd.DataFrame,
    feature_cols: list[str] | None = None,
    target: str = "fullgame",
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
    target : str
        ``'fullgame'`` uses home_runs/away_runs.
        ``'f5'`` uses f5_home_score/f5_away_score.

    Returns
    -------
    tuple[pd.DataFrame, pd.Series, pd.Series]
        (X, y_home, y_away) — all same index.
    """
    if feature_cols is None:
        feature_cols = TRAIN_FEATURES

    if target not in _TARGET_COLS:
        raise ValueError(f"target must be one of {list(_TARGET_COLS.keys())}")
    home_col, away_col = _TARGET_COLS[target]

    present = [c for c in feature_cols if c in df.columns]
    missing_cols = set(feature_cols) - set(present)
    if missing_cols:
        logger.warning("Feature columns missing from DataFrame: %s", missing_cols)

    sub = df.dropna(subset=[home_col, away_col]).copy()
    if sub.empty:
        logger.warning("No rows with non-null targets (%s, %s)", home_col, away_col)
        return pd.DataFrame(columns=present), pd.Series(dtype=float), pd.Series(dtype=float)

    X = sub[present].copy()

    # Fill nulls with column median (0 for all-null columns like precip_prob in history)
    for col in X.columns:
        if X[col].isna().any():
            med = X[col].median()
            X[col] = X[col].fillna(0.0 if np.isnan(med) else med)

    y_home = sub[home_col].astype(float)
    y_away = sub[away_col].astype(float)

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
    target: str = "fullgame",
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
        'glm' for PoissonRegressor, 'gbr' for HistGradientBoostingRegressor,
        'negbinom' for NegBinomGLMWrapper (requires statsmodels).
        For 'negbinom', alpha is estimated per fold from Poisson GLM residuals.
    feature_cols : list[str] or None
    target : str
        ``'fullgame'`` (home_runs/away_runs) or ``'f5'`` (f5_home_score/f5_away_score).

    Returns
    -------
    dict
        Keys: fold_results (list of per-fold metrics), summary (aggregate metrics).
    """
    df_sorted = df.sort_values("date").reset_index(drop=True)
    X_all, y_home_all, y_away_all = _prepare_xy(df_sorted, feature_cols, target=target)

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
            m_home.fit(X_train, y_home_train)
            m_away.fit(X_train, y_away_train)
        elif model_type == "negbinom":
            from mlb.calibration import estimate_alpha

            # Estimate alpha per fold from Poisson GLM residuals on training data
            glm_alpha_h = make_poisson_glm()
            glm_alpha_a = make_poisson_glm()
            glm_alpha_h.fit(X_train, y_home_train)
            glm_alpha_a.fit(X_train, y_away_train)
            alpha_h = estimate_alpha(y_home_train.values, glm_alpha_h.predict(X_train))
            alpha_a = estimate_alpha(y_away_train.values, glm_alpha_a.predict(X_train))
            m_home = make_negbinom_glm(alpha=alpha_h)
            m_away = make_negbinom_glm(alpha=alpha_a)
            m_home.fit(X_train, y_home_train)
            m_away.fit(X_train, y_away_train)
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

        result: dict[str, Any] = {
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
        if model_type == "negbinom":
            result["alpha_home"] = round(alpha_h, 6)
            result["alpha_away"] = round(alpha_a, 6)
        fold_results.append(result)

        logger.info(
            "Fold %d | train %s→%s | test %s→%s | "
            "dev_home=%.4f d2_home=%.4f | dev_away=%.4f d2_away=%.4f | "
            "disp_home=%.3f disp_away=%.3f",
            fold_idx + 1,
            train_dates.min(),
            train_dates.max(),
            test_dates.min(),
            test_dates.max(),
            dev_home,
            d2_home,
            dev_away,
            d2_away,
            disp_home,
            disp_away,
        )

    # Aggregate
    devs_home = [r["dev_home"] for r in fold_results]
    devs_away = [r["dev_away"] for r in fold_results]
    d2s_home = [r["d2_home"] for r in fold_results]
    d2s_away = [r["d2_away"] for r in fold_results]
    disps_home = [r["disp_home"] for r in fold_results]
    disps_away = [r["disp_away"] for r in fold_results]

    summary: dict[str, Any] = {
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
    if model_type == "negbinom":
        summary["mean_alpha_home"] = round(
            float(np.mean([r["alpha_home"] for r in fold_results])), 6
        )
        summary["mean_alpha_away"] = round(
            float(np.mean([r["alpha_away"] for r in fold_results])), 6
        )

    logger.info(
        "CV summary (%s): dev_home=%.4f dev_away=%.4f "
        "d2_home=%.4f d2_away=%.4f disp_home=%.3f disp_away=%.3f",
        model_type,
        summary["mean_dev_home"],
        summary["mean_dev_away"],
        summary["mean_d2_home"],
        summary["mean_d2_away"],
        summary["mean_disp_home"],
        summary["mean_disp_away"],
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
    target: str = "fullgame",
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
    target : str
        ``'fullgame'`` to predict home_runs/away_runs (KXMLBTOTAL).
        ``'f5'`` to predict f5_home_score/f5_away_score (KXMLBF5TOTAL).
    db_path : str
    save : bool
        If True, save artefact to data/models/.

    Returns
    -------
    dict
        Artefact dict with fitted models and metadata.
    """
    logger.info("Loading features %s to %s (target=%s)", start_date, end_date, target)
    df = build_features(start_date=start_date, end_date=end_date, db_path=db_path)

    X, y_home, y_away = _prepare_xy(df, target=target)
    if X.empty:
        raise ValueError(f"No training data for target='{target}' in {start_date}..{end_date}")

    logger.info("Training %s (target=%s) on %d games", model_type, target, len(X))

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
        "target": target,
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
        suffix = "f5_" if target == "f5" else ""
        path = MODEL_DIR / f"{suffix}{model_type}_poisson_v{MODEL_VERSION}.pkl"
        joblib.dump(artefact, path)
        logger.info("Saved model artefact to %s", path)

    logger.info(
        "Train complete (target=%s): dev_home=%.4f dev_away=%.4f",
        target,
        train_dev_home,
        train_dev_away,
    )
    return artefact


# ── Predict ───────────────────────────────────────────────────────────────────


def predict(
    date: str,
    artefact: dict[str, Any] | None = None,
    model_type: str = "gbr",
    target: str = "fullgame",
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
    target : str
        ``'fullgame'`` or ``'f5'`` — selects which artefact file to load.
    db_path : str

    Returns
    -------
    pd.DataFrame
        One row per game with columns: game_id, date, home_team, away_team,
        lambda_home, lambda_away, predicted_total_runs.
        lambda values are F5 expected runs when target='f5'.
    """
    if artefact is None:
        suffix = "f5_" if target == "f5" else ""
        path = MODEL_DIR / f"{suffix}{model_type}_poisson_v{MODEL_VERSION}.pkl"
        if not path.exists():
            raise FileNotFoundError(f"No model artefact at {path} — run --train first")
        artefact = joblib.load(path)

    # Try upcoming (Preview/Scheduled) games first; fall back to Final games on date
    df = build_predict_features(date=date, db_path=db_path)
    if df.empty:
        logger.info("No scheduled games found for %s — trying Final games", date)
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


def batch_predict(
    start_date: str,
    end_date: str,
    model_type: str = "gbr",
    target: str = "fullgame",
    db_path: str = "data/mlb.db",
) -> int:
    """
    Generate and store predictions for all Final games in a date range.

    Builds the feature matrix once for the full window, applies the model
    in a single pass, and writes all predictions to the predictions table.
    Significantly faster than calling predict() day by day.

    Parameters
    ----------
    start_date, end_date : str
        YYYY-MM-DD range (inclusive).
    model_type : str
    target : str
        'fullgame' or 'f5'.
    db_path : str

    Returns
    -------
    int
        Number of predictions written.
    """
    suffix = "f5_" if target == "f5" else ""
    path = MODEL_DIR / f"{suffix}{model_type}_poisson_v{MODEL_VERSION}.pkl"
    if not path.exists():
        raise FileNotFoundError(f"No model artefact at {path} — run --train first")
    artefact = joblib.load(path)

    df = build_features(start_date=start_date, end_date=end_date, db_path=db_path)
    if df.empty:
        logger.warning("No games found for %s to %s", start_date, end_date)
        return 0

    if target == "f5":
        df = df.dropna(subset=["f5_home_score", "f5_away_score"])
    else:
        df = df.dropna(subset=["home_runs", "away_runs"])

    feature_names = artefact["feature_names"]
    present = [c for c in feature_names if c in df.columns]
    X = df[present].copy()
    for col in X.columns:
        if X[col].isna().any():
            X[col] = X[col].fillna(X[col].median())

    lam_home = np.clip(artefact["model_home"].predict(X), 0.01, 30.0)
    lam_away = np.clip(artefact["model_away"].predict(X), 0.01, 30.0)

    preds = df[["game_id"]].copy()
    preds["lambda_home"] = lam_home
    preds["lambda_away"] = lam_away
    preds["predicted_total_runs"] = lam_home + lam_away

    stored_name = f"f5_{model_type}" if target == "f5" else model_type
    n = write_predictions(preds, model_type=stored_name, db_path=db_path)
    logger.info("batch_predict: wrote %d rows (%s, %s)", n, stored_name, target)
    return n


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


# ── LightGBM direct binary classifier ────────────────────────────────────────


def make_lgbm_binary(
    n_estimators: int = 500,
    learning_rate: float = 0.03,
    max_depth: int = 4,
    num_leaves: int = 31,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    min_child_samples: int = 30,
    random_state: int = 42,
) -> Any:
    """
    LightGBM binary classifier for direct P(over) prediction.

    Trained on (features + total_line_close) → binary target (total_runs > line).
    Eliminates the two-model λ chain and Poisson convolution step.

    Parameters
    ----------
    n_estimators : int
    learning_rate : float
    max_depth : int
    num_leaves : int
    subsample : float
    colsample_bytree : float
    min_child_samples : int
    random_state : int

    Returns
    -------
    LGBMClassifier
    """
    import lightgbm as lgb

    return lgb.LGBMClassifier(
        objective="binary",
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        num_leaves=num_leaves,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        min_child_samples=min_child_samples,
        random_state=random_state,
        verbose=-1,
    )


def _prepare_xy_binary(
    df: pd.DataFrame,
    feature_cols: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Build feature matrix and binary target for the LightGBM classifier.

    Binary target: (total_runs > total_line_close).
    Drops rows where total_runs or total_line_close is null.

    Parameters
    ----------
    df : pd.DataFrame
        Output of build_features().
    feature_cols : list[str] or None

    Returns
    -------
    tuple[pd.DataFrame, pd.Series]
        (X, y_binary) — same index.
    """
    if feature_cols is None:
        feature_cols = TRAIN_FEATURES

    sub = df.dropna(subset=["total_runs", "total_line_close"]).copy()
    if sub.empty:
        return pd.DataFrame(columns=feature_cols), pd.Series(dtype=int)

    sub["_y"] = (sub["total_runs"] > sub["total_line_close"]).astype(int)

    present = [c for c in feature_cols if c in sub.columns]
    X = sub[present].copy()
    for col in X.columns:
        if X[col].isna().any():
            med = X[col].median()
            X[col] = X[col].fillna(0.0 if np.isnan(med) else med)

    return X, sub["_y"]


def walk_forward_cv_binary(
    df: pd.DataFrame,
    n_splits: int = 5,
    gap: int = 162,
    feature_cols: list[str] | None = None,
) -> dict[str, Any]:
    """
    Walk-forward CV for the binary LightGBM model.

    Collects out-of-fold (OOF) probability predictions used to fit the
    isotonic calibrator in train_binary().

    Parameters
    ----------
    df : pd.DataFrame
    n_splits : int
    gap : int
    feature_cols : list[str] or None

    Returns
    -------
    dict
        Keys: fold_results (list), oof_probs (np.ndarray),
        oof_labels (np.ndarray), summary (dict).
    """
    from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

    df_sorted = df.sort_values("date").reset_index(drop=True)
    X_all, y_all = _prepare_xy_binary(df_sorted, feature_cols)

    if X_all.empty:
        raise ValueError("No binary training rows (need total_runs + total_line_close non-null)")

    tscv = TimeSeriesSplit(n_splits=n_splits, gap=gap)
    fold_results: list[dict[str, Any]] = []
    oof_probs = np.full(len(y_all), np.nan)
    oof_labels = y_all.values.copy()
    # Parallel arrays for writing OOF predictions back to DB
    oof_game_ids: list[Any] = [None] * len(y_all)
    oof_lines = np.full(len(y_all), np.nan)
    x_index = X_all.index  # actual df_sorted label indices (may skip nulls)

    for fold_idx, (train_idx, test_idx) in enumerate(tscv.split(X_all)):
        X_train = X_all.iloc[train_idx]
        X_test = _fill_test_nulls(X_train, X_all.iloc[test_idx])
        y_train = y_all.iloc[train_idx]
        y_test = y_all.iloc[test_idx]

        # Use label indices to access df_sorted correctly (X_all may be a subset)
        train_labels = x_index[train_idx]
        test_labels = x_index[test_idx]
        train_dates = df_sorted.loc[train_labels, "date"]
        test_dates = df_sorted.loc[test_labels, "date"]

        model = make_lgbm_binary()
        model.fit(X_train, y_train)
        probs = model.predict_proba(X_test)[:, 1]
        oof_probs[test_idx] = probs

        # Track game_id and line for each OOF prediction
        for pos, lbl in zip(test_idx, test_labels):
            oof_game_ids[pos] = df_sorted.loc[lbl, "game_id"]
            line_val = df_sorted.loc[lbl, "total_line_close"] if "total_line_close" in df_sorted.columns else np.nan
            oof_lines[pos] = float(line_val) if line_val is not None and not (isinstance(line_val, float) and np.isnan(line_val)) else np.nan

        ll = log_loss(y_test, probs)
        brier = brier_score_loss(y_test, probs)
        auc = roc_auc_score(y_test, probs)
        over_rate = float(y_test.mean())
        mean_pred = float(probs.mean())

        result: dict[str, Any] = {
            "fold": fold_idx + 1,
            "train_start": train_dates.min(),
            "train_end": train_dates.max(),
            "test_start": test_dates.min(),
            "test_end": test_dates.max(),
            "n_train": len(train_idx),
            "n_test": len(test_idx),
            "log_loss": round(ll, 4),
            "brier": round(brier, 4),
            "auc": round(auc, 4),
            "over_rate": round(over_rate, 4),
            "mean_pred": round(mean_pred, 4),
            "bias": round(mean_pred - over_rate, 4),
        }
        fold_results.append(result)

        logger.info(
            "Fold %d | train %s->%s | test %s->%s | "
            "log_loss=%.4f brier=%.4f auc=%.4f bias=%+.4f",
            fold_idx + 1,
            train_dates.min(),
            train_dates.max(),
            test_dates.min(),
            test_dates.max(),
            ll,
            brier,
            auc,
            mean_pred - over_rate,
        )

    valid_mask = ~np.isnan(oof_probs)
    oof_probs_v = oof_probs[valid_mask]
    oof_labels_v = oof_labels[valid_mask]

    summary: dict[str, Any] = {
        "model_type": "lgbm_binary",
        "n_splits": n_splits,
        "mean_log_loss": round(float(np.mean([r["log_loss"] for r in fold_results])), 4),
        "mean_brier": round(float(np.mean([r["brier"] for r in fold_results])), 4),
        "mean_auc": round(float(np.mean([r["auc"] for r in fold_results])), 4),
        "mean_bias": round(float(np.mean([r["bias"] for r in fold_results])), 4),
        "oof_log_loss": round(float(log_loss(oof_labels_v, oof_probs_v)), 4),
        "oof_brier": round(float(brier_score_loss(oof_labels_v, oof_probs_v)), 4),
        "oof_auc": round(float(roc_auc_score(oof_labels_v, oof_probs_v)), 4),
    }

    logger.info(
        "CV summary (lgbm_binary): oof_log_loss=%.4f oof_brier=%.4f oof_auc=%.4f mean_bias=%+.4f",
        summary["oof_log_loss"],
        summary["oof_brier"],
        summary["oof_auc"],
        summary["mean_bias"],
    )

    oof_game_ids_v = [oof_game_ids[i] for i in range(len(y_all)) if valid_mask[i]]
    oof_lines_v = oof_lines[valid_mask]

    return {
        "fold_results": fold_results,
        "summary": summary,
        "oof_probs": oof_probs_v,
        "oof_labels": oof_labels_v,
        "oof_game_ids": oof_game_ids_v,
        "oof_lines": oof_lines_v,
    }


def train_binary(
    start_date: str = "2021-04-01",
    end_date: str = "2024-10-01",
    db_path: str = "data/mlb.db",
    n_splits: int = 5,
    save: bool = True,
) -> dict[str, Any]:
    """
    Train the LightGBM binary model with isotonic calibration on OOF predictions.

    Requires rows where both total_runs and total_line_close (SBR closing
    line) are non-null — in practice 2021-present. Walk-forward CV generates
    OOF predictions; IsotonicRegression corrects any systematic probability
    bias before the final model is trained on all data.

    Parameters
    ----------
    start_date : str
    end_date : str
    db_path : str
    n_splits : int
    save : bool

    Returns
    -------
    dict
        Artefact: model, calibrator, feature_names, cv_metrics, metadata.
    """
    from sklearn.isotonic import IsotonicRegression

    logger.info("Loading features %s to %s (lgbm_binary)", start_date, end_date)
    df = build_features(start_date=start_date, end_date=end_date, db_path=db_path)

    cv_results = walk_forward_cv_binary(df, n_splits=n_splits)
    oof_probs = cv_results["oof_probs"]
    oof_labels = cv_results["oof_labels"]

    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(oof_probs, oof_labels)
    logger.info("Fitted isotonic calibrator on %d OOF samples", len(oof_probs))

    # Write calibrated OOF predictions to DB so simulate() can backtest them.
    # These are genuinely out-of-sample within each walk-forward fold.
    cal_oof = np.clip(calibrator.predict(oof_probs), 0.001, 0.999)
    oof_preds = pd.DataFrame(
        {
            "game_id": cv_results["oof_game_ids"],
            "over_prob": cal_oof,
            "line": cv_results["oof_lines"],
        }
    ).dropna(subset=["game_id"])
    n_oof = write_binary_predictions(oof_preds, db_path=db_path)
    logger.info("Wrote %d calibrated OOF predictions to predictions table", n_oof)

    X_all, y_all = _prepare_xy_binary(df)
    if X_all.empty:
        raise ValueError("No training data with non-null total_runs + total_line_close")

    logger.info("Training final LightGBM binary on %d games", len(X_all))
    final_model = make_lgbm_binary()
    final_model.fit(X_all, y_all)

    artefact: dict[str, Any] = {
        "model": final_model,
        "calibrator": calibrator,
        "feature_names": list(X_all.columns),
        "model_type": "lgbm_binary",
        "train_start": start_date,
        "train_end": end_date,
        "n_train": len(X_all),
        "cv_metrics": cv_results["summary"],
        "version": MODEL_VERSION,
        "trained_at": datetime.now(UTC).isoformat(),
    }

    if save:
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        path = MODEL_DIR / f"lgbm_binary_v{MODEL_VERSION}.pkl"
        joblib.dump(artefact, path)
        logger.info("Saved binary artefact to %s", path)

    logger.info(
        "Binary train complete: n=%d oof_log_loss=%.4f oof_auc=%.4f oof_predictions=%d",
        len(X_all),
        cv_results["summary"]["oof_log_loss"],
        cv_results["summary"]["oof_auc"],
        n_oof,
    )
    return artefact


def batch_predict_binary(
    start_date: str,
    end_date: str,
    db_path: str = "data/mlb.db",
) -> int:
    """
    Generate calibrated binary predictions for all Final games in a date range.

    Loads lgbm_binary_v{VERSION}.pkl, builds features, applies the model and
    isotonic calibrator, and writes over_prob to the predictions table.
    lambda_home and lambda_away are stored as 0.0 (sentinel indicating no
    Poisson λ was computed).

    Parameters
    ----------
    start_date : str
    end_date : str
    db_path : str

    Returns
    -------
    int
        Number of predictions written.
    """
    path = MODEL_DIR / f"lgbm_binary_v{MODEL_VERSION}.pkl"
    if not path.exists():
        raise FileNotFoundError(f"No binary artefact at {path} — run --train --model lgbm_binary first")
    artefact = joblib.load(path)

    df = build_features(start_date=start_date, end_date=end_date, db_path=db_path)
    if df.empty:
        return 0

    df = df.dropna(subset=["total_runs"])

    feature_names = artefact["feature_names"]
    present = [c for c in feature_names if c in df.columns]
    X = df[present].copy()
    for col in X.columns:
        if X[col].isna().any():
            med = X[col].median()
            X[col] = X[col].fillna(0.0 if np.isnan(med) else med)

    raw_probs = artefact["model"].predict_proba(X)[:, 1]
    cal_probs = np.clip(artefact["calibrator"].predict(raw_probs), 0.001, 0.999)

    line_col = "total_line_close" if "total_line_close" in df.columns else None

    preds = df[["game_id"]].copy()
    preds["over_prob"] = cal_probs
    preds["line"] = df[line_col].values if line_col else np.nan

    n = write_binary_predictions(preds, db_path=db_path)
    logger.info("batch_predict_binary: wrote %d rows (lgbm_binary)", n)
    return n


def write_binary_predictions(
    preds: pd.DataFrame,
    db_path: str = "data/mlb.db",
) -> int:
    """
    Write binary model predictions to the predictions table.

    Stores over_prob and under_prob directly from the calibrated classifier.
    lambda_home and lambda_away are set to 0.0 as sentinels (no Poisson λ).

    Parameters
    ----------
    preds : pd.DataFrame
        Must contain: game_id, over_prob, line.
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
            over_p = float(row["over_prob"])
            under_p = 1.0 - over_p
            line_raw = row.get("line")
            line_val = float(line_raw) if line_raw is not None and not np.isnan(float(line_raw)) else None

            conn.execute(
                """INSERT OR REPLACE INTO predictions
                   (game_id, model_name, model_version, predicted_at,
                    lambda_home, lambda_away, predicted_total_runs,
                    over_prob, under_prob, line)
                   VALUES (?,?,?,?,?,?,?,?,?,?)""",
                (
                    row["game_id"],
                    "lgbm_binary",
                    MODEL_VERSION,
                    now,
                    0.0,
                    0.0,
                    0.0,
                    round(over_p, 4),
                    round(under_p, 4),
                    line_val,
                ),
            )
            rows += 1
    logger.info("Wrote %d binary predictions (lgbm_binary)", rows)
    return rows


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
    group.add_argument(
        "--predict-range",
        action="store_true",
        help="Batch predict for --start to --end (one pass, fast)",
    )

    parser.add_argument("--model", choices=["glm", "gbr", "lgbm_binary"], default="gbr")
    parser.add_argument(
        "--target",
        choices=["fullgame", "f5"],
        default="fullgame",
        help="Target: 'fullgame' (KXMLBTOTAL) or 'f5' (KXMLBF5TOTAL)",
    )
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--start", default="2022-04-07")
    parser.add_argument("--end", default="2024-10-01")
    parser.add_argument("--date", default=None)
    parser.add_argument("--no-save", action="store_true")

    args = parser.parse_args()

    if args.train:
        if args.model == "lgbm_binary":
            train_binary(
                start_date=args.start,
                end_date=args.end,
                save=not args.no_save,
            )
        else:
            train(
                start_date=args.start,
                end_date=args.end,
                model_type=args.model,
                target=args.target,
                save=not args.no_save,
            )

    elif args.backtest:
        if args.model == "lgbm_binary":
            df = build_features(start_date=args.start, end_date=args.end)
            results = walk_forward_cv_binary(df, n_splits=args.n_splits)
            print("\n=== Walk-Forward CV Results (lgbm_binary) ===")
            for fold in results["fold_results"]:
                print(
                    f"Fold {fold['fold']} | "
                    f"train {fold['train_start']} -> {fold['train_end']} | "
                    f"test  {fold['test_start']} -> {fold['test_end']} | "
                    f"log_loss={fold['log_loss']:.4f} brier={fold['brier']:.4f} "
                    f"auc={fold['auc']:.4f} bias={fold['bias']:+.4f}"
                )
            s = results["summary"]
            print(
                f"\nOOF: log_loss={s['oof_log_loss']:.4f} brier={s['oof_brier']:.4f} "
                f"auc={s['oof_auc']:.4f} mean_bias={s['mean_bias']:+.4f}"
            )
        else:
            df = build_features(start_date=args.start, end_date=args.end)
            results = walk_forward_cv(
                df, n_splits=args.n_splits, model_type=args.model, target=args.target
            )
            print(f"\n=== Walk-Forward CV Results (target={args.target}) ===")
            for fold in results["fold_results"]:
                print(
                    f"Fold {fold['fold']} | "
                    f"train {fold['train_start']} -> {fold['train_end']} | "
                    f"test  {fold['test_start']} -> {fold['test_end']} | "
                    f"dev_home={fold['dev_home']:.4f} d2_home={fold['d2_home']:.4f} | "
                    f"dev_away={fold['dev_away']:.4f} d2_away={fold['d2_away']:.4f} | "
                    f"disp_home={fold['disp_home']:.3f} disp_away={fold['disp_away']:.3f}"
                )
            s = results["summary"]
            print(
                f"\nMean: dev_home={s['mean_dev_home']:.4f} dev_away={s['mean_dev_away']:.4f} "
                f"d2_home={s['mean_d2_home']:.4f} d2_away={s['mean_d2_away']:.4f}"
            )
            if s["negbinom_recommended"]:
                print("*** NegBinom upgrade recommended (mean dispersion > 1.2) ***")

    elif args.predict_range:
        if args.model == "lgbm_binary":
            n = batch_predict_binary(
                start_date=args.start,
                end_date=args.end,
            )
            print(f"Wrote {n} binary predictions for {args.start} to {args.end}")
        else:
            n = batch_predict(
                start_date=args.start,
                end_date=args.end,
                model_type=args.model,
                target=args.target,
            )
            print(f"Wrote {n} predictions ({args.target}) for {args.start} to {args.end}")

    elif args.predict:
        date = args.date or datetime.now().strftime("%Y-%m-%d")
        preds = predict(date=date, model_type=args.model, target=args.target)
        if preds.empty:
            print(f"No games found for {date}")
        else:
            stored_name = f"f5_{args.model}" if args.target == "f5" else args.model
            write_predictions(preds, model_type=stored_name)
            print(
                preds[
                    ["home_team", "away_team", "lambda_home", "lambda_away", "predicted_total_runs"]
                ].to_string(index=False)
            )
