# Math & Betting Rules
**Active when:** editing mlb/model.py, mlb/calibration.py, mlb/betting.py,
or any file containing probability, EV, or Kelly calculations

---

## Poisson Model — Non-Negotiable

Run scoring is count data. These rules are absolute:

- NEVER use `loss='squared_error'` for the run prediction models
- NEVER use Normal CDF to derive P(over) — use Poisson convolution only
- Models must output λ (expected runs) — a positive float
- Train two separate model instances: one for home_runs, one for away_runs

```python
# CORRECT model setup
from sklearn.linear_model import PoissonRegressor
from sklearn.ensemble import GradientBoostingRegressor

model_home = PoissonRegressor(alpha=1.0)          # or GBR(loss='poisson')
model_away = PoissonRegressor(alpha=1.0)

# WRONG
from sklearn.linear_model import Ridge            # wrong distribution assumption
model = Ridge()                                   # predicts total, not per-team λ
```

## Poisson Convolution

P(over) must be computed via joint Poisson probability mass:

```python
from scipy.stats import poisson
import numpy as np

def p_over_vectorised(lam_home: float, lam_away: float,
                      line: float, max_runs: int = 30) -> float:
    h = np.arange(max_runs + 1)
    a = np.arange(max_runs + 1)
    H, A = np.meshgrid(h, a)
    joint = poisson.pmf(H, lam_home) * poisson.pmf(A, lam_away)
    return float(joint[H + A > line].sum())
```

max_runs=30 is sufficient (P(team scores >30) ≈ 0).

## Overdispersion Check

After fitting PoissonRegressor, always check:

```python
dispersion = residuals.var() / lambda_pred.mean()
# If dispersion > 1.2 across walk-forward folds → upgrade to NegBinom
```

Negative Binomial upgrade requires statsmodels — see calibration.py.

## Walk-Forward CV

```python
from sklearn.model_selection import TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits=5, gap=162)
# gap=162 prevents season-boundary leakage
# NEVER use KFold, StratifiedKFold, or shuffle=True on time series data
```

## Evaluation Metrics

```python
from sklearn.metrics import mean_poisson_deviance, d2_tweedie_score, log_loss

# Primary regression metric (not MAE/RMSE)
deviance = mean_poisson_deviance(y_true, lambda_pred)   # lower = better
d2       = d2_tweedie_score(y_true, lambda_pred, power=1)  # higher = better

# Primary probability metric (after convolution)
ll = log_loss(over_outcomes_binary, over_probs)  # lower = better

# Primary end-to-end metric
# ROI in betting simulation — the only metric that truly matters
```

## Expected Value

```python
def compute_ev(over_prob: float, kalshi_over_price: float) -> dict:
    """
    over_prob: P(total > line), from Poisson convolution
    kalshi_over_price: Kalshi YES mid-price in dollars (0–1)
    """
    ev_over  = over_prob * (1 - kalshi_over_price) - (1 - over_prob) * kalshi_over_price
    ev_under = (1 - over_prob) * kalshi_over_price - over_prob * (1 - kalshi_over_price)
    # Only bet when EV > $0.03 — minimum edge threshold
    if ev_over > 0.03:    bet_side = 'OVER'
    elif ev_under > 0.03: bet_side = 'UNDER'
    else:                 bet_side = 'PASS'
    return {'ev_over': ev_over, 'ev_under': ev_under,
            'edge': over_prob - kalshi_over_price, 'bet_side': bet_side}
```

## Kelly Criterion

```python
def kelly_bet(win_prob: float, kalshi_price: float,
              kelly_mult: float = 0.25, max_pct: float = 0.05) -> float:
    """
    win_prob: P(bet wins) — over_prob if OVER, (1 - over_prob) if UNDER
    kelly_mult: ALWAYS 0.25x — never bet full Kelly
    max_pct: NEVER exceed 5% of bankroll on any single game
    """
    b = (1 / kalshi_price) - 1
    full_kelly = max(0.0, (win_prob * b - (1 - win_prob)) / b)
    return min(full_kelly * kelly_mult, max_pct)
```

## CLV

```python
def compute_clv(entry: float, closing: float, side: str) -> float:
    """Positive CLV = we beat the closing price = evidence of edge."""
    return closing - entry if side == 'OVER' else entry - (1 - closing)
```

Track CLV independently of P&L. Sustained positive CLV over 200+ bets
is a more reliable signal of edge than short-run P&L variance.

## Position Constraints

These are hard limits — never override them:
- Min edge: $0.03 vs Kalshi mid-price
- Min Kalshi open interest: $1,000
- Max single position: 5% of bankroll
- Max simultaneous open positions: 3 games
