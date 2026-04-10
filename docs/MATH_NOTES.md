# Math Notes

Derivations and mathematical foundations for the MLB total runs
prediction and betting system. Claude Code must read this file when
implementing mlb/model.py, mlb/calibration.py, and mlb/betting.py.

---

## 1. Why Poisson for Run Scoring

Baseball runs scored per team per game are count data — discrete,
non-negative integers. The appropriate distribution for count data
is Poisson (or Negative Binomial if overdispersed).

A Poisson random variable X ~ Poisson(λ) has:
- Mean: E[X] = λ
- Variance: Var[X] = λ
- PMF: P(X = k) = e^(-λ) × λ^k / k!

The Poisson GLM uses a log link function:
```
log(λ) = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ
λ = exp(Xβ)
```

This guarantees λ > 0 regardless of feature values (no negative
run predictions possible), which is a key advantage over linear
regression with squared error loss.

**Poisson deviance** is the correct loss function:
```
D(y, λ) = 2 × Σ [ y×log(y/λ) - (y - λ) ]
```

sklearn's `PoissonRegressor` minimises half the Poisson deviance.
`GradientBoostingRegressor(loss='poisson')` minimises the same objective.

---

## 2. Two-Target Model Structure

We model each team's runs independently:

```
home_runs ~ Poisson(λ_home)    where λ_home = exp(X_home @ β_home)
away_runs ~ Poisson(λ_away)    where λ_away = exp(X_away @ β_away)
```

Two separate model instances are trained — one for home runs, one
for away runs. The models share the same feature set but learn
different coefficients.

**Why not predict total_runs directly?**

1. Predicting per-team λ lets us price ANY over/under line via
   convolution — not just the posted line
2. The independence assumption (home and away runs are independent
   Poisson processes) is approximately valid in baseball since
   each half-inning is scored independently
3. Per-team predictions open the door to exact-score and
   alternative-total markets later

---

## 3. Poisson Convolution — Deriving P(over)

Given λ_home and λ_away, we want P(total_runs > line).

Since home_runs and away_runs are independent:

```
P(total > line) = Σ_h Σ_a  P(H=h) × P(A=a)   for all h+a > line
               = Σ_h Σ_a  Poisson(h|λ_home) × Poisson(a|λ_away)
                           where h + a > line
```

In code:
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

**Why max_runs=30?**
P(team scores > 30 runs) is astronomically small. The highest team
run total in modern MLB history is 30 (Texas Rangers, 2007).
Truncating at 30 introduces negligible error while keeping computation fast.

**Verification check:**
```python
# Sum of all joint probabilities should ≈ 1.0
assert abs(joint.sum() - 1.0) < 1e-6
# P(over) + P(under_or_push) should ≈ 1.0
p_over = p_over_vectorised(5.0, 4.5, 8.5)
p_under_or_push = 1 - p_over
assert 0 < p_over < 1
```

---

## 4. Overdispersion and Negative Binomial

Poisson assumes mean = variance. If run scoring is overdispersed
(variance > mean), Poisson underestimates tail probabilities —
exactly where betting lines often sit.

**Overdispersion test:**
```python
dispersion_ratio = residuals.var() / lambda_pred.mean()
# If > 1.2 consistently across walk-forward folds: upgrade to NegBinom
```

**Negative Binomial** adds a dispersion parameter α:
```
X ~ NegBinom(μ, α)
E[X]   = μ
Var[X] = μ + α×μ²
```

As α → 0, NegBinom → Poisson. α is estimated from data.

The NegBinom PMF (scipy parameterisation with n=1/α, p=n/(n+μ)):
```python
from scipy.stats import nbinom

def p_over_negbinom(mu_home: float, mu_away: float,
                    alpha: float, line: float, max_runs: int = 30) -> float:
    n = 1.0 / alpha
    p_h = n / (n + mu_home)
    p_a = n / (n + mu_away)
    H = np.arange(max_runs + 1)
    A = np.arange(max_runs + 1)
    HH, AA = np.meshgrid(H, A)
    joint = nbinom.pmf(HH, n, p_h) * nbinom.pmf(AA, n, p_a)
    return float(joint[HH + AA > line].sum())
```

---

## 5. Model Evaluation Metrics

### Poisson Deviance (primary regression metric)
```
D(y, λ̂) = 2 × Σ [ y×log(y/λ̂) - (y - λ̂) ]
```
Lower is better. A null model predicting the mean gives a baseline
deviance. The D² score (analogous to R²) measures improvement:
```
D² = 1 - D(y, λ̂) / D(y, ȳ)
```
Target: D² > 0.05 (sports prediction has limited signal).

### Log-Loss on P(over) (primary probability metric)
After computing P(over) via convolution, evaluate against actual
over/under outcomes (binary):
```
LogLoss = -1/N × Σ [ y×log(p) + (1-y)×log(1-p) ]
```
where y=1 if total > line, y=0 if total ≤ line.
Target: LogLoss < 0.693 (0.693 = random model predicting 0.5 always).

### ROI (primary end-to-end metric)
```
ROI = (final_bankroll - initial_bankroll) / initial_bankroll × 100
```
A model with good Poisson deviance but zero betting ROI is not useful.
ROI is the only metric that accounts for the full pipeline including
line selection, Kelly sizing, and edge threshold filtering.

---

## 6. Expected Value

For a Kalshi YES (OVER) bet:
- You pay `price` dollars per contract
- If over hits: you receive $1.00 (net profit = 1 - price)
- If under hits: you lose `price` dollars

```
EV_over  = P(over) × (1 - price) - P(under) × price
EV_under = P(under) × price       - P(over)  × (1 - price)
```

where `price` is the Kalshi mid-price for the OVER side (0–1).

**Removing Kalshi vig:**
Kalshi prices have a bid-ask spread. The mid-price approximates
the no-vig probability:
```
mid_price = (yes_bid + yes_ask) / 2
```

For traditional sportsbooks with American odds, the no-vig
implied probability is derived as:
```python
def american_to_implied_prob(american_odds: int) -> float:
    if american_odds > 0:
        return 100 / (american_odds + 100)
    else:
        return abs(american_odds) / (abs(american_odds) + 100)

def remove_vig(prob_over: float, prob_under: float) -> tuple[float, float]:
    """Normalise two implied probs to sum to 1.0."""
    total = prob_over + prob_under
    return prob_over / total, prob_under / total
```

**Edge:**
```
edge = P(over)_model - mid_price_kalshi
```
Positive edge = our model thinks over is more likely than Kalshi does.
Only bet when |edge| > 0.03 (3 cents minimum).

---

## 7. Kelly Criterion

The Kelly criterion maximises the expected logarithm of wealth,
which maximises long-run compounding growth.

**Derivation:**

Let p = probability of winning, q = 1-p, b = net profit per unit risked.
For a Kalshi bet: b = (1 / price) - 1

After N bets with fraction f of bankroll:
```
Bankroll = W₀ × (1 + f×b)^wins × (1 - f)^losses
```

Taking log and maximising over f:
```
d/df [ p×log(1 + f×b) + q×log(1 - f) ] = 0
p×b / (1 + f×b) - q / (1 - f) = 0
```

Solving:
```
f* = (p×b - q) / b  =  (p×b - (1-p)) / b
```

In code:
```python
def kelly_fraction(p: float, price: float) -> float:
    """
    p:     calibrated probability of winning the bet
    price: Kalshi mid-price of the bet side (0–1)
    """
    b = (1.0 / price) - 1.0   # net profit per unit risked
    q = 1.0 - p
    f_star = (p * b - q) / b
    return max(0.0, f_star)    # Kelly is always >= 0 when edge > 0
```

**Fractional Kelly (always 0.25x):**

Full Kelly is theoretically optimal but extremely volatile in practice.
The model's probability estimate is never perfectly calibrated, and
overbetting leads to ruin. We always apply 0.25x fractional Kelly:

```python
bet_size = min(kelly_fraction(p, price) * 0.25, 0.05)
# Hard cap at 5% of bankroll regardless of Kelly output
```

**Verification:**
For p=0.60, price=0.50:
- b = (1/0.50) - 1 = 1.0
- f* = (0.60×1.0 - 0.40) / 1.0 = 0.20
- Fractional (0.25x): 0.20 × 0.25 = 0.05
- Capped at 5%: 0.05 ✓

For p=0.50, price=0.50 (no edge):
- f* = (0.50×1.0 - 0.50) / 1.0 = 0.0 ✓

For p=0.45, price=0.50 (negative edge):
- f* = (0.45×1.0 - 0.55) / 1.0 = -0.10 → max(0, -0.10) = 0.0 ✓

---

## 8. Closing Line Value (CLV)

CLV measures whether we obtained a better price than where the
market eventually settled (the closing price).

```
CLV_over  = closing_over_price  - entry_over_price
CLV_under = entry_over_price    - closing_over_price
            (equivalently: entry_under_price - closing_under_price)
```

Positive CLV = we got a better price than the close = evidence of edge.

**Why CLV matters more than short-run P&L:**

P&L over a small sample is dominated by variance. Even a model with
real edge will have losing stretches of 50+ bets due to randomness.
CLV is a leading indicator — if we consistently enter at better
prices than where the market closes, that is evidence of an
informational advantage, independent of short-run outcomes.

Rule of thumb: over 200+ bets, sustained positive average CLV (>$0.01)
is strong evidence of real edge. Sustained negative CLV despite
positive P&L suggests lucky variance that will revert.

**Relationship to sharp money:**
The closing line is the most efficient price — it has absorbed all
sharp betting action. Beating it means we acted on information before
the sharps fully priced it in.

---

## 9. Elo Rating System

Team Elo ratings are used as a feature (team strength signal).

**Update formula:**
```
expected  = 1 / (1 + 10^((elo_opponent - elo_self) / 400))
new_elo   = old_elo + K × (actual - expected)
```

where actual = 1 for win, 0 for loss, K = 20.

**Zero-sum property:**
The total Elo across all teams must be constant after each game
(winner gains exactly what loser loses):
```
Δelo_winner = K × (1 - expected_winner)
Δelo_loser  = K × (0 - expected_loser) = -K × (1 - expected_winner)
Δelo_winner + Δelo_loser = 0  ✓
```

Verification after every batch update:
```python
np.testing.assert_allclose(
    elo_before.sum(), elo_after.sum(),
    rtol=1e-6, atol=1e-4
)
```

**Initialisation:** all teams start at 1500 at the beginning of the
2015 season. At the start of each subsequent season, apply regression
to mean: `elo_new = 0.7 × elo_end_of_prior_season + 0.3 × 1500`.
This reflects the uncertainty introduced by roster changes and
prevents ratings from diverging too far from 1500 over time.
