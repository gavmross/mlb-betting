"""
Unit tests for mlb/calibration.py.

Every math function has a correctness test.
Key invariants verified:
  1. P(over) + P(under) + P(push) = 1.0
  2. High λ → high P(over low line)
  3. NegBinom with alpha→0 converges to Poisson
  4. estimate_alpha: Poisson data → alpha ≈ 0
  5. calibrate_batch matches row-by-row p_over_poisson

Run:
    pytest tests/unit/test_calibration.py -v
"""

from __future__ import annotations

import numpy as np
import pytest

from mlb.calibration import (
    DISPERSION_THRESHOLD,
    calibrate_batch,
    calibrate_predictions,
    estimate_alpha,
    overdispersion_report,
    p_exact_poisson,
    p_over_negbinom,
    p_over_poisson,
    p_under_poisson,
    verify_convolution_sums_to_one,
)

# ── P(over) Poisson ───────────────────────────────────────────────────────────


def test_p_over_high_lambdas_low_line():
    """With λ_home=6, λ_away=5, P(over 4.5) should be very high (>0.95)."""
    p = p_over_poisson(6.0, 5.0, 4.5)
    assert p > 0.95, f"Expected >0.95, got {p:.4f}"


def test_p_over_low_lambdas_high_line():
    """With λ_home=2, λ_away=2, P(over 12.5) should be very low (<0.01)."""
    p = p_over_poisson(2.0, 2.0, 12.5)
    assert p < 0.01, f"Expected <0.01, got {p:.4f}"


def test_p_over_in_range():
    """P(over) must be in (0, 1) for any reasonable inputs."""
    for lam_h, lam_a, line in [(4.5, 4.0, 8.5), (3.0, 3.0, 7.5), (5.5, 4.5, 9.0)]:
        p = p_over_poisson(lam_h, lam_a, line)
        assert 0.0 < p < 1.0, f"P(over) out of range: {p:.4f} for λ=({lam_h},{lam_a}) line={line}"


def test_p_over_symmetric():
    """P(over line | λh, λa) == P(over line | λa, λh) — symmetric in lambdas."""
    p1 = p_over_poisson(4.5, 4.0, 8.5)
    p2 = p_over_poisson(4.0, 4.5, 8.5)
    np.testing.assert_allclose(p1, p2, rtol=1e-9)


def test_p_over_half_line_vs_whole_line():
    """P(over 8.5) > P(over 9.0) for same lambdas — half-line is easier to go over."""
    p_half = p_over_poisson(4.5, 4.0, 8.5)
    p_whole = p_over_poisson(4.5, 4.0, 9.0)
    assert p_half > p_whole


# ── P(under) Poisson ─────────────────────────────────────────────────────────


def test_p_under_low_lambdas_high_line():
    """With low λ, P(under high line) should be very high."""
    # λ_home=2, λ_away=2 → expected total ≈ 4; P(under 14.5) ≈ 1
    p = p_under_poisson(2.0, 2.0, 14.5)
    assert p > 0.98, f"Expected >0.98, got {p:.4f}"


def test_p_under_complementary_half_line():
    """On a half-line: P(over) + P(under) ≈ 1.0 (P(push) = 0)."""
    for lam_h, lam_a, line in [(4.5, 4.0, 8.5), (3.0, 3.5, 7.5)]:
        p_over = p_over_poisson(lam_h, lam_a, line)
        p_under = p_under_poisson(lam_h, lam_a, line)
        np.testing.assert_allclose(
            p_over + p_under,
            1.0,
            atol=1e-5,
            err_msg=f"Over + under should sum to 1 on half-line {line}",
        )


# ── Sum-to-one invariant ──────────────────────────────────────────────────────


def test_convolution_sums_to_one_half_line():
    """P(over 8.5) + P(under 8.5) + P(push 8.5) == 1.0 (push=0 on half-line)."""
    ok = verify_convolution_sums_to_one(4.5, 4.0, 8.5)
    assert ok, "Convolution sum-to-one check failed for half-line"


def test_convolution_sums_to_one_whole_line():
    """P(over 9) + P(under 9) + P(push 9) == 1.0."""
    ok = verify_convolution_sums_to_one(4.5, 4.0, 9.0)
    assert ok, "Convolution sum-to-one check failed for whole-line"


def test_p_exact_push_probability():
    """P(push) on whole-number line must be positive and < 1."""
    p = p_exact_poisson(4.5, 4.0, 9)
    assert 0.0 < p < 1.0


def test_p_exact_impossible():
    """P(total == -1) = 0 (impossible run count)."""
    p = p_exact_poisson(4.5, 4.0, -1)
    np.testing.assert_allclose(p, 0.0, atol=1e-10)


# ── NegBinom convolution ──────────────────────────────────────────────────────


def test_p_over_negbinom_in_range():
    """P(over) via NegBinom must be in (0, 1)."""
    p = p_over_negbinom(4.5, 4.0, alpha=0.2, line=8.5)
    assert 0.0 < p < 1.0


def test_p_over_negbinom_converges_to_poisson():
    """As alpha → 0, NegBinom P(over) should converge to Poisson P(over)."""
    lam_h, lam_a, line = 4.5, 4.0, 8.5
    p_pois = p_over_poisson(lam_h, lam_a, line)
    # Very small alpha (near-Poisson)
    p_nb = p_over_negbinom(lam_h, lam_a, alpha=1e-6, line=line)
    np.testing.assert_allclose(p_nb, p_pois, atol=0.002)


def test_p_over_negbinom_higher_than_poisson():
    """Overdispersion → fatter tails → higher P(over high line) than Poisson."""
    lam_h, lam_a, line = 4.5, 4.0, 14.5  # high line — tails matter
    p_pois = p_over_poisson(lam_h, lam_a, line)
    p_nb = p_over_negbinom(lam_h, lam_a, alpha=0.3, line=line)
    assert p_nb > p_pois, (
        f"NegBinom with large alpha should give higher P(over high line): "
        f"Poisson={p_pois:.5f}, NegBinom={p_nb:.5f}"
    )


# ── estimate_alpha ────────────────────────────────────────────────────────────


def test_estimate_alpha_poisson_data():
    """Genuine Poisson data should yield alpha close to 0."""
    rng = np.random.default_rng(42)
    lam = 4.5
    y = rng.poisson(lam=lam, size=5000).astype(float)
    lam_pred = np.full(len(y), lam)
    alpha = estimate_alpha(y, lam_pred)
    # alpha should be close to 0 for Poisson data
    assert alpha < 0.1, f"Poisson data should yield alpha ≈ 0, got {alpha:.4f}"


def test_estimate_alpha_overdispersed():
    """Overdispersed data should yield alpha > 0."""
    rng = np.random.default_rng(42)
    # NegBinom with known alpha=0.3, mu=4.5
    # scipy: nbinom(n, p) where n=1/alpha, p=n/(n+mu)
    true_alpha = 0.3
    mu = 4.5
    n = 1.0 / true_alpha  # = 3.333
    p = n / (n + mu)
    y = rng.negative_binomial(n=int(round(n)), p=p, size=5000).astype(float)
    lam_pred = np.full(len(y), y.mean())
    alpha = estimate_alpha(y, lam_pred)
    assert alpha > 0.05, f"Overdispersed NegBinom data should yield alpha > 0.05, got {alpha:.4f}"


def test_estimate_alpha_nonnegative():
    """estimate_alpha must never return negative values."""
    rng = np.random.default_rng(0)
    # Underdispersed-ish data (Binomial — less spread than Poisson)
    y = rng.binomial(n=10, p=0.4, size=1000).astype(float)
    alpha = estimate_alpha(y, np.full(len(y), 4.0))
    assert alpha >= 0.0


# ── overdispersion_report ─────────────────────────────────────────────────────


def test_overdispersion_report_keys():
    """Report must contain all required keys."""
    rng = np.random.default_rng(42)
    y = rng.poisson(4.5, size=500).astype(float)
    r = overdispersion_report(y, np.full(len(y), 4.5), label="test")
    required = {"label", "n", "mean_lambda", "mean_y", "variance",
                "dispersion_ratio", "alpha", "negbinom_recommended"}
    assert required.issubset(set(r.keys()))


def test_overdispersion_report_threshold():
    """dispersion > 1.2 → negbinom_recommended = True."""
    # Create highly overdispersed data
    rng = np.random.default_rng(42)
    y = rng.negative_binomial(2, 0.3, size=2000).astype(float)
    r = overdispersion_report(y, np.full(len(y), y.mean()), label="overdispersed")
    assert r["negbinom_recommended"] is True
    assert r["dispersion_ratio"] > DISPERSION_THRESHOLD


# ── calibrate_batch ───────────────────────────────────────────────────────────


def test_calibrate_batch_matches_row_by_row():
    """calibrate_batch must match p_over_poisson called per row."""
    rng = np.random.default_rng(42)
    n = 10
    lam_h = rng.uniform(3.0, 6.0, n)
    lam_a = rng.uniform(3.0, 6.0, n)
    lines = np.full(n, 8.5)

    over_batch, under_batch = calibrate_batch(lam_h, lam_a, lines)

    for i in range(n):
        expected_over = p_over_poisson(lam_h[i], lam_a[i], 8.5)
        expected_under = p_under_poisson(lam_h[i], lam_a[i], 8.5)
        np.testing.assert_allclose(over_batch[i], expected_over, rtol=1e-9)
        np.testing.assert_allclose(under_batch[i], expected_under, rtol=1e-9)


def test_calibrate_batch_nan_handling():
    """NaN inputs should produce NaN outputs without raising."""
    lam_h = np.array([4.5, np.nan, 4.5])
    lam_a = np.array([4.0, 4.0, np.nan])
    lines = np.array([8.5, 8.5, 8.5])

    over_probs, under_probs = calibrate_batch(lam_h, lam_a, lines)

    assert not np.isnan(over_probs[0])
    assert np.isnan(over_probs[1])
    assert np.isnan(over_probs[2])


# ── calibrate_predictions ────────────────────────────────────────────────────


def test_calibrate_predictions_adds_columns():
    """calibrate_predictions must add over_prob and under_prob columns."""
    import pandas as pd

    df = pd.DataFrame({
        "game_id": ["g1", "g2"],
        "lambda_home": [4.5, 3.5],
        "lambda_away": [4.0, 4.5],
        "total_line_close": [8.5, 8.5],
    })
    result = calibrate_predictions(df)
    assert "over_prob" in result.columns
    assert "under_prob" in result.columns
    assert result["over_prob"].notna().all()


def test_calibrate_predictions_missing_line_col_raises():
    """calibrate_predictions must raise ValueError for missing line column."""
    import pandas as pd

    df = pd.DataFrame({
        "lambda_home": [4.5],
        "lambda_away": [4.0],
    })
    with pytest.raises(ValueError, match="total_line_close"):
        calibrate_predictions(df)
