"""
Unit tests for mlb/betting.py — EV, Kelly, CLV, and helper math.

All tests use known analytical values so they catch formula regressions.
Float comparisons use np.testing.assert_allclose(rtol=1e-6, atol=1e-4).
"""

import numpy as np
import pytest

from mlb.betting import (
    american_to_price,
    compute_clv,
    compute_ev,
    devig_prices,
    get_consensus,
    kelly_bet,
    passes_filters,
)


# ── compute_ev ────────────────────────────────────────────────────────────────


class TestComputeEv:
    def test_over_edge_above_threshold(self):
        """over_prob > price by enough → bet_side = OVER."""
        result = compute_ev(over_prob=0.60, kalshi_over_price=0.50)
        assert result["bet_side"] == "OVER"
        # ev_over = 0.60 * 0.50 - 0.40 * 0.50 = 0.10
        np.testing.assert_allclose(result["ev_over"], 0.10, rtol=1e-6)
        np.testing.assert_allclose(result["edge"], 0.10, rtol=1e-6)

    def test_under_edge_above_threshold(self):
        """over_prob < price → UNDER has edge."""
        result = compute_ev(over_prob=0.40, kalshi_over_price=0.50)
        assert result["bet_side"] == "UNDER"
        # ev_under = 0.60 * 0.50 - 0.40 * 0.50 = 0.10
        np.testing.assert_allclose(result["ev_under"], 0.10, rtol=1e-6)

    def test_no_edge_returns_pass(self):
        """No edge on either side → PASS."""
        result = compute_ev(over_prob=0.50, kalshi_over_price=0.50)
        assert result["bet_side"] == "PASS"
        np.testing.assert_allclose(result["ev_over"], 0.0, atol=1e-9)

    def test_small_edge_below_min_returns_pass(self):
        """Edge of $0.02 below default $0.03 threshold → PASS."""
        # over_prob = 0.52, price = 0.50 → edge = 0.02
        result = compute_ev(over_prob=0.52, kalshi_over_price=0.50)
        assert result["bet_side"] == "PASS"

    def test_custom_min_edge(self):
        """Custom lower threshold allows bet on smaller edge."""
        result = compute_ev(over_prob=0.52, kalshi_over_price=0.50, min_edge=0.01)
        assert result["bet_side"] == "OVER"

    def test_ev_formula_symmetry(self):
        """EV should be anti-symmetric: ev_over(p, q) = -ev_under(p, q)."""
        result = compute_ev(over_prob=0.65, kalshi_over_price=0.55)
        np.testing.assert_allclose(result["ev_over"], -result["ev_under"], atol=1e-9)

    def test_ev_extreme_probability(self):
        """Very high model probability against even price → large EV."""
        result = compute_ev(over_prob=0.90, kalshi_over_price=0.50)
        # ev_over = 0.90 * 0.50 - 0.10 * 0.50 = 0.40
        np.testing.assert_allclose(result["ev_over"], 0.40, rtol=1e-6)


# ── kelly_bet ─────────────────────────────────────────────────────────────────


class TestKellyBet:
    def test_known_value_no_cap(self):
        """
        win_prob=0.55, price=0.50 → b=1.0, f*=(0.55*1 - 0.45)/1 = 0.10
        Fractional 0.25x → 0.025 → not capped.
        """
        result = kelly_bet(win_prob=0.55, kalshi_price=0.50)
        np.testing.assert_allclose(result, 0.025, rtol=1e-6)

    def test_known_value_hits_cap(self):
        """
        win_prob=0.70, price=0.50 → b=1.0, f*=0.40
        Fractional 0.25x → 0.10 → capped at 0.05.
        """
        result = kelly_bet(win_prob=0.70, kalshi_price=0.50)
        np.testing.assert_allclose(result, 0.05, rtol=1e-6)

    def test_no_edge_returns_zero(self):
        """Breakeven bet → Kelly = 0."""
        result = kelly_bet(win_prob=0.50, kalshi_price=0.50)
        assert result == 0.0

    def test_negative_edge_returns_zero(self):
        """Negative EV → Kelly clamps to 0, never negative."""
        result = kelly_bet(win_prob=0.40, kalshi_price=0.50)
        assert result == 0.0

    def test_custom_kelly_mult(self):
        """Full kelly (mult=1.0) on breakeven+ bet."""
        # win_prob=0.60, price=0.50 → f* = 0.20
        result = kelly_bet(win_prob=0.60, kalshi_price=0.50, kelly_mult=1.0)
        np.testing.assert_allclose(result, 0.05, rtol=1e-6)  # capped at 0.05

    def test_custom_max_pct(self):
        """Custom cap applies correctly."""
        result = kelly_bet(win_prob=0.80, kalshi_price=0.50, max_pct=0.10)
        np.testing.assert_allclose(result, 0.10, rtol=1e-6)

    def test_invalid_price_zero_returns_zero(self):
        """Price of 0 is invalid — return 0 safely."""
        result = kelly_bet(win_prob=0.60, kalshi_price=0.0)
        assert result == 0.0

    def test_invalid_price_one_returns_zero(self):
        """Price of 1 is invalid — return 0 safely."""
        result = kelly_bet(win_prob=0.60, kalshi_price=1.0)
        assert result == 0.0

    def test_result_never_exceeds_max_pct(self):
        """Regardless of probability, output never exceeds max_pct."""
        for p in [0.6, 0.7, 0.8, 0.9, 0.99]:
            result = kelly_bet(win_prob=p, kalshi_price=0.50)
            assert result <= 0.05, f"p={p} → kelly={result} exceeds max_pct"


# ── compute_clv ───────────────────────────────────────────────────────────────


class TestComputeClv:
    def test_over_positive_clv(self):
        """Closing higher than entry → positive CLV for OVER."""
        clv = compute_clv(entry=0.48, closing=0.52, side="OVER")
        np.testing.assert_allclose(clv, 0.04, rtol=1e-6)

    def test_over_negative_clv(self):
        """Closing lower than entry → negative CLV for OVER."""
        clv = compute_clv(entry=0.54, closing=0.50, side="OVER")
        np.testing.assert_allclose(clv, -0.04, rtol=1e-6)

    def test_under_positive_clv(self):
        """For UNDER at entry=0.52, closing=0.50 → under closing was 0.50 → CLV=0.02."""
        # entry=0.52 for UNDER means we paid 1-0.52=0.48 for the under
        # closing under price = 1-0.50=0.50 → under got more expensive → CLV = entry - (1-closing)
        clv = compute_clv(entry=0.52, closing=0.50, side="UNDER")
        # 0.52 - (1 - 0.50) = 0.52 - 0.50 = 0.02
        np.testing.assert_allclose(clv, 0.02, rtol=1e-6)

    def test_zero_clv_on_flat_close(self):
        """Entry equals fair close → CLV = 0."""
        clv = compute_clv(entry=0.50, closing=0.50, side="OVER")
        np.testing.assert_allclose(clv, 0.0, atol=1e-9)

    def test_unknown_side_returns_zero(self):
        """Unrecognised side → 0 (safe default)."""
        clv = compute_clv(entry=0.50, closing=0.55, side="PUSH")
        assert clv == 0.0


# ── american_to_price ─────────────────────────────────────────────────────────


class TestAmericanToPrice:
    def test_negative_110(self):
        """−110 is the standard juice line; 110/210 ≈ 0.5238."""
        result = american_to_price(-110)
        np.testing.assert_allclose(result, 110 / 210, rtol=1e-6)

    def test_positive_110(self):
        """+110 → 100/210 ≈ 0.4762."""
        result = american_to_price(110)
        np.testing.assert_allclose(result, 100 / 210, rtol=1e-6)

    def test_even_money(self):
        """+100 (even money) → 0.50."""
        result = american_to_price(100)
        np.testing.assert_allclose(result, 0.50, rtol=1e-6)

    def test_negative_115(self):
        """-115 → 115/215."""
        result = american_to_price(-115)
        np.testing.assert_allclose(result, 115 / 215, rtol=1e-6)

    def test_heavy_favourite(self):
        """-200 → 200/300 = 0.6667."""
        result = american_to_price(-200)
        np.testing.assert_allclose(result, 200 / 300, rtol=1e-6)


# ── devig_prices ──────────────────────────────────────────────────────────────


class TestDevigPrices:
    def test_symmetric_vig_removes_to_half(self):
        """Symmetric -110/-110 book → fair price 0.50 each."""
        raw_over = american_to_price(-110)
        raw_under = american_to_price(-110)
        fair_over, fair_under = devig_prices(raw_over, raw_under)
        np.testing.assert_allclose(fair_over, 0.50, rtol=1e-4)
        np.testing.assert_allclose(fair_under, 0.50, rtol=1e-4)

    def test_devigged_prices_sum_to_one(self):
        """Fair prices always sum to 1.0."""
        for over_odds, under_odds in [(-110, -110), (-115, +105), (-120, +100)]:
            fair_over, fair_under = devig_prices(
                american_to_price(over_odds), american_to_price(under_odds)
            )
            np.testing.assert_allclose(fair_over + fair_under, 1.0, atol=1e-9)

    def test_asymmetric_book_shifts_fair_price(self):
        """When over is favoured (-115 vs +105), fair over > 0.50."""
        fair_over, fair_under = devig_prices(
            american_to_price(-115), american_to_price(105)
        )
        assert fair_over > 0.50
        assert fair_under < 0.50


# ── get_consensus ─────────────────────────────────────────────────────────────


class TestGetConsensus:
    def test_kalshi_cheaper(self):
        """kalshi_mid < poly_mid → kalshi is cheap for OVER."""
        result = get_consensus(kalshi_mid=0.45, poly_mid=0.50)
        assert result["kalshi_is_cheap_for_over"] is True
        np.testing.assert_allclose(result["spread"], 0.05, rtol=1e-6)

    def test_kalshi_dearer(self):
        """kalshi_mid > poly_mid → kalshi is not cheap."""
        result = get_consensus(kalshi_mid=0.55, poly_mid=0.50)
        assert result["kalshi_is_cheap_for_over"] is False

    def test_spread_always_positive(self):
        """Spread is absolute value regardless of direction."""
        r1 = get_consensus(0.45, 0.50)
        r2 = get_consensus(0.50, 0.45)
        np.testing.assert_allclose(r1["spread"], r2["spread"], rtol=1e-6)

    def test_market_avg(self):
        """Market average is the midpoint of the two prices."""
        result = get_consensus(0.48, 0.52)
        np.testing.assert_allclose(result["market_avg"], 0.50, rtol=1e-6)


# ── passes_filters ────────────────────────────────────────────────────────────


class TestPassesFilters:
    def test_pass_bet_side_rejected(self):
        ok, _ = passes_filters("PASS", open_interest=5000, current_positions=0)
        assert ok is False

    def test_too_many_positions(self):
        ok, reason = passes_filters("OVER", open_interest=5000, current_positions=3)
        assert ok is False
        assert "max" in reason

    def test_low_open_interest(self):
        ok, reason = passes_filters("OVER", open_interest=500, current_positions=0)
        assert ok is False
        assert "open_interest" in reason

    def test_none_open_interest_passes(self):
        """None open interest is allowed (Kalshi data may be missing)."""
        ok, _ = passes_filters("OVER", open_interest=None, current_positions=0)
        assert ok is True

    def test_valid_bet_passes(self):
        ok, reason = passes_filters("UNDER", open_interest=2000, current_positions=1)
        assert ok is True
        assert reason == ""
