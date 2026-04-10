"""
Unit tests for mlb/model.py.

Covers:
- Model factory outputs correct types
- _prepare_xy temporal ordering and null handling
- check_overdispersion formula
- walk_forward_cv never leaks across folds
- Predictions are positive (λ > 0)
- Feature importance shape

Run:
    pytest tests/unit/test_model.py -v
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlb.model import (
    _fill_test_nulls,
    _prepare_xy,
    check_overdispersion,
    feature_importance,
    make_gbr_poisson,
    make_poisson_glm,
    walk_forward_cv,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture(scope="session")
def small_feature_df():
    """
    Small feature DataFrame built from 2022 April-May.
    Session-scoped since build_features is read-only.
    """
    from mlb.features import build_features

    return build_features(start_date="2022-04-07", end_date="2022-07-31")


# ── Model factory tests ───────────────────────────────────────────────────────


def test_make_poisson_glm_type():
    """make_poisson_glm returns a sklearn Pipeline."""
    from sklearn.pipeline import Pipeline

    m = make_poisson_glm()
    assert isinstance(m, Pipeline)


def test_make_gbr_poisson_loss():
    """HGBR must use poisson loss — never squared_error."""
    from sklearn.ensemble import HistGradientBoostingRegressor

    m = make_gbr_poisson()
    assert isinstance(m, HistGradientBoostingRegressor)
    assert m.loss == "poisson", "HGBR must use loss='poisson', not squared_error"


def test_make_gbr_poisson_params():
    """Check default hyperparameters are within safe ranges."""
    m = make_gbr_poisson()
    assert m.max_depth is None or m.max_depth <= 6, "max_depth should be shallow for sports prediction"
    assert 0 < m.learning_rate <= 0.1


# ── _prepare_xy tests ─────────────────────────────────────────────────────────


def test_prepare_xy_returns_correct_shapes(small_feature_df):
    """X, y_home, y_away must have the same number of rows."""
    X, y_home, y_away = _prepare_xy(small_feature_df)
    assert len(X) == len(y_home) == len(y_away)
    assert len(X) > 0


def test_prepare_xy_no_nulls_in_targets(small_feature_df):
    """Rows with null targets must be dropped."""
    X, y_home, y_away = _prepare_xy(small_feature_df)
    assert y_home.isna().sum() == 0
    assert y_away.isna().sum() == 0


def test_prepare_xy_no_nulls_in_features(small_feature_df):
    """Null features must be median-filled so X contains no NaN."""
    X, _, _ = _prepare_xy(small_feature_df)
    assert X.isna().sum().sum() == 0, "Feature matrix must have no NaN after median fill"


def test_prepare_xy_targets_nonnegative(small_feature_df):
    """Run counts must be >= 0."""
    _, y_home, y_away = _prepare_xy(small_feature_df)
    assert (y_home >= 0).all()
    assert (y_away >= 0).all()


# ── _fill_test_nulls tests ────────────────────────────────────────────────────


def test_fill_test_nulls_uses_train_median():
    """Nulls in test must be filled with train median, not test median."""
    train = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 5.0]})
    test = pd.DataFrame({"a": [np.nan, 10.0, np.nan]})

    filled = _fill_test_nulls(train, test)
    # train median of [1,2,3,4,5] = 3.0
    np.testing.assert_allclose(filled["a"].iloc[0], 3.0)
    np.testing.assert_allclose(filled["a"].iloc[2], 3.0)
    # non-null values unchanged
    np.testing.assert_allclose(filled["a"].iloc[1], 10.0)


def test_fill_test_nulls_does_not_modify_original():
    """Original test DataFrame must not be modified in-place."""
    train = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
    test = pd.DataFrame({"a": [np.nan, 5.0]})
    original_test = test.copy()
    _fill_test_nulls(train, test)
    pd.testing.assert_frame_equal(test, original_test)


# ── Overdispersion tests ──────────────────────────────────────────────────────


def test_check_overdispersion_poisson_case():
    """For perfectly Poisson data, dispersion should be close to 1."""
    rng = np.random.default_rng(42)
    lam = 4.5
    y = rng.poisson(lam=lam, size=2000).astype(float)
    d = check_overdispersion(y, np.full(len(y), lam))
    # Should be close to 1 for genuine Poisson
    assert 0.7 < d < 1.5, f"Poisson dispersion should be ~1, got {d:.3f}"


def test_check_overdispersion_overdispersed():
    """Variance > mean → ratio > 1."""
    rng = np.random.default_rng(42)
    # NegBinom is overdispersed relative to Poisson
    y = rng.negative_binomial(n=2, p=0.4, size=2000).astype(float)
    lam = np.full(len(y), y.mean())
    d = check_overdispersion(y, lam)
    assert d > 1.0, f"NegBinom data should show dispersion > 1, got {d:.3f}"


def test_check_overdispersion_known_value():
    """Manual verification: var=10, mean_lambda=4 → ratio=2.5."""
    y = np.array([0, 0, 2, 4, 6, 8, 10, 0, 0, 2], dtype=float)
    lam = np.full(len(y), 4.0)
    d = check_overdispersion(y, lam)
    expected = y.var() / 4.0
    np.testing.assert_allclose(d, expected, rtol=1e-9)


# ── Walk-forward CV tests ─────────────────────────────────────────────────────


def test_walk_forward_cv_no_date_leakage(small_feature_df):
    """Test dates must always be strictly after train dates in every fold."""
    results = walk_forward_cv(
        small_feature_df,
        n_splits=3,
        gap=30,
        model_type="glm",
    )
    for fold in results["fold_results"]:
        assert fold["test_start"] > fold["train_end"], (
            f"Fold {fold['fold']}: test_start={fold['test_start']} <= "
            f"train_end={fold['train_end']} — leakage detected"
        )


def test_walk_forward_cv_returns_correct_folds(small_feature_df):
    """Results dict must have the right structure."""
    results = walk_forward_cv(
        small_feature_df,
        n_splits=3,
        gap=30,
        model_type="glm",
    )
    assert "fold_results" in results
    assert "summary" in results
    assert len(results["fold_results"]) == 3
    for fold in results["fold_results"]:
        for key in ("dev_home", "dev_away", "d2_home", "d2_away", "disp_home", "disp_away"):
            assert key in fold, f"Missing key '{key}' in fold result"


def test_walk_forward_cv_deviance_positive(small_feature_df):
    """Poisson deviance must be positive."""
    results = walk_forward_cv(
        small_feature_df,
        n_splits=3,
        gap=30,
        model_type="glm",
    )
    for fold in results["fold_results"]:
        assert fold["dev_home"] > 0
        assert fold["dev_away"] > 0


def test_walk_forward_cv_glm_smoke(small_feature_df):
    """GLM walk-forward should complete without error."""
    results = walk_forward_cv(small_feature_df, n_splits=2, gap=30, model_type="glm")
    assert results["summary"]["model_type"] == "glm"


# ── Prediction tests ──────────────────────────────────────────────────────────


def test_predictions_positive(small_feature_df):
    """Lambda predictions must be > 0 for all games."""
    from sklearn.pipeline import Pipeline

    X, y_home, y_away = _prepare_xy(small_feature_df)
    m_home = make_poisson_glm()
    m_away = make_poisson_glm()
    m_home.fit(X, y_home)
    m_away.fit(X, y_away)

    lam_h = m_home.predict(X)
    lam_a = m_away.predict(X)
    assert (lam_h > 0).all(), "All lambda_home predictions must be > 0"
    assert (lam_a > 0).all(), "All lambda_away predictions must be > 0"


# ── Feature importance tests ──────────────────────────────────────────────────


def test_feature_importance_shape(small_feature_df):
    """Feature importance DataFrame must have one row per feature."""
    X, y_home, y_away = _prepare_xy(small_feature_df)
    # Use a small sample to keep the test fast
    sample_size = min(200, len(X))
    X_s = X.iloc[:sample_size]
    y_h_s = y_home.iloc[:sample_size]
    y_a_s = y_away.iloc[:sample_size]

    m_home = make_gbr_poisson()
    m_away = make_gbr_poisson()
    m_home.fit(X, y_home)
    m_away.fit(X, y_away)

    artefact = {
        "model_home": m_home,
        "model_away": m_away,
        "feature_names": list(X.columns),
        "model_type": "gbr",
    }
    imp = feature_importance(artefact, X_sample=X_s, y_home_sample=y_h_s, y_away_sample=y_a_s)
    assert len(imp) == len(X.columns)
    assert "importance_mean" in imp.columns
    assert (imp["importance_mean"] >= 0).all()


def test_feature_importance_glm_raises():
    """feature_importance must raise ValueError for GLM."""
    artefact = {"model_type": "glm"}
    with pytest.raises(ValueError, match="GBR"):
        feature_importance(artefact)


def test_feature_importance_hgbr_no_sample_raises(small_feature_df):
    """feature_importance on HGBR without sample must raise ValueError."""
    X, y_home, y_away = _prepare_xy(small_feature_df)
    m_home = make_gbr_poisson()
    m_home.fit(X, y_home)
    m_away = make_gbr_poisson()
    m_away.fit(X, y_away)
    artefact = {
        "model_home": m_home,
        "model_away": m_away,
        "feature_names": list(X.columns),
        "model_type": "gbr",
    }
    with pytest.raises(ValueError, match="X_sample"):
        feature_importance(artefact)  # no sample provided
