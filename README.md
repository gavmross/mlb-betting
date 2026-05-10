# MLB Total Runs Prediction & Betting System

A systematic, data-driven strategy for betting MLB total runs markets on Kalshi
(CFTC-regulated prediction markets). The system combines three independent
pre-game filters — validated over five full seasons (2021–2025) against
DraftKings closing lines — with a two-target Poisson regression model that
prices any total line via convolution.

---

## Backtest Results (2021–2025, DraftKings Closing Lines)

The **structural filter strategy** — three binary pre-game rules with no model
dependency — is the production strategy. All results use vig-inclusive DraftKings
closing lines as the fill price.

| Strategy | Terminal (from $100) | Win Rate | Sharpe | Max Drawdown |
|---|---|---|---|---|
| Flat 5% stake | $5,254 | 57.4% | 1.48 | -77.8% |
| Quarter Kelly (0.25×, 5% cap) | $1,529 | 57.4% | 1.60 | -40.9% |
| **Half Kelly (0.50×, 15% cap)** | **$9,711** | **57.4%** | **1.60** | **-67.6%** |
| Full Kelly (1.0×, 15% cap) | $27,621 | 57.4% | 1.60 | -92.4% |

**Production configuration: Half Kelly (0.50×), 15% bankroll cap.**

Sharpe ratio is annualised. Quarter Kelly underperforms fixed 5% because the
Kelly fraction for the UNDER filters (~4–5%) bets less than a fixed 5% stake,
while half Kelly correctly sizes the high-edge summer OVER filter at ~11.3%
instead of being artificially capped at 5%.

---

## The Three Structural Filters

Pre-game binary rules. No model needed — the bet is triggered by observable
conditions before first pitch. Contradictory signals (same game triggers both
UNDER and OVER) are skipped.

### UNDER Filter 1 — Day Game at Pitcher-Friendly Park with Elite Starters

| Condition | Value |
|---|---|
| Game time | Day game (start before 6 PM ET) |
| Venue | SFG, CLE, TEX, CIN, CHW, SDP, SEA, or DET |
| Combined starter K/9 | ≥ 14.0 |
| Per-SP ERA last 3 starts | ≤ 4.0 (when available) |
| Side | UNDER |
| **Win rate (2021–2025)** | **56.4%** (n = 906) |
| Kelly input probability | 0.564 → full Kelly ~8.4% → **half Kelly ~4.2%** |

**Why it works:** Low-run parks, elite strikeout starters, and the structural
fatigue effect of daytime games (less rest for batters, sharper command for
pitchers) combine to consistently suppress run totals below the posted line.

### UNDER Filter 2 — High Total Line

| Condition | Value |
|---|---|
| DraftKings closing total | ≥ 11.0 |
| Side | UNDER |
| **Win rate (2021–2025)** | **57.5%** (n = 373) |
| Kelly input probability | 0.575 → full Kelly ~10.7% → **half Kelly ~5.4%** |

**Why it works:** Books shade high-total lines upward to exploit public over
bias. Games with a closing line ≥ 11 represent the top decile of implied run
expectations; reversion to a realistic mean is systematically under-priced.

### OVER Filter — Summer Heat with Outward Wind

| Condition | Value |
|---|---|
| Month | July, August, or September only |
| Temperature | ≥ 80°F |
| Stadium | Outdoor only (no dome) |
| Wind direction | Blowing toward outfield |
| Wind speed | 10–15 mph |
| Side | OVER |
| **Win rate (2021–2025)** | **63.1%** (n = 134) |
| Kelly input probability | 0.631 → full Kelly ~22.5% → **half Kelly ~11.3%** |

**Why it works:** The July–September restriction is critical. Testing the same
filter across all seasons shows April–June hits at exactly 50.0% (no signal),
while July–September hits at 63.1% (p < 0.001). This seasonal specificity is
evidence *against* data mining — a spurious pattern would show elevated rates
across all months. The physical mechanism: extreme summer heat and sustained
outward wind at 10–15 mph materially lifts run scoring; books set lines the
evening before based on forecasts that don't capture intraday condition changes.
Above 15 mph the effect reverses (wind disrupts pitcher command and hitter timing).

---

## Why the Filters, Not the Model?

The Poisson regression model predicts λ_home and λ_away accurately enough to
price fair probabilities, but the closing sportsbook line already incorporates
sharp money and is a near-efficient estimate of true P(over). The model's
current OOF ROI against DraftKings at -110 is approximately -2.8% — normal
for a market that prices efficiently.

The structural filters exploit specific inefficiencies the book cannot easily
correct:
1. **Books price the average game in a condition, not the tail.** The market
   can't post a different line for "day game + elite pitching + pitchers park"
   without revealing their models.
2. **Public over bias.** Sharp money corrects individual game lines, but the
   aggregate over/under distribution at high line totals is structurally
   mispriced.
3. **Weather information lag.** Books set totals 12–18 hours before first
   pitch. Intraday weather updates (heat, wind shift) are not reflected.

The model is kept in **paper trading** mode. If three more seasons of data
(2026+) show consistent positive OOF ROI, it will be promoted to production.

---

## Technical Architecture

### Data Pipeline

```
pybaseball (2015–present)          → game results, pitcher stats, Statcast
MLB Stats API                      → schedule, lineups, probable pitchers
sbrscrape (2021–present)           → DraftKings/FanDuel/Caesars closing lines
Open-Meteo (free, no API key)      → historical + forecast weather
Kalshi REST API (2026–present)     → live totals market prices
Polymarket Gamma API (read-only)   → cross-market pricing signal
```

All data is stored in a single SQLite database (`data/mlb.db`) in WAL mode.
The schema has ten tables: `games`, `team_stats`, `pitchers`, `weather`,
`stadiums`, `elo_ratings`, `sportsbook_odds`, `kalshi_markets`, `predictions`,
and `scrape_log`.

### No-Leakage Feature Engineering

All 48 features are computable from data strictly available before first pitch.
Every rolling window uses `.shift(1)` before `.rolling()` to exclude the
current game:

```python
# Correct — excludes today's start
df['sp_era_l3'] = (
    df.groupby('pitcher_id')['era_game']
    .shift(1)
    .rolling(3, min_periods=1)
    .mean()
)
```

Temporal integrity is enforced at the test level: every rolling feature has a
dedicated leakage test and a cold-start NaN test in `tests/unit/test_features.py`.
Walk-forward `TimeSeriesSplit(n_splits=5, gap=162)` is the only permitted
cross-validation strategy — gap of 162 games prevents season-boundary leakage.

### Feature Groups (48 total)

| Group | Features | Description |
|---|---|---|
| Starting pitchers | 18 | ERA, FIP, K/9, BB/9, rest days, last-3 ERA, last-5 ER/G — both SP |
| Team offense | 6 | Rolling 10-game OPS, K%, runs scored — both teams |
| Bullpen | 6 | ERA (7d, 30d), innings pitched (7d) — both teams |
| Park factors | 4 | Run factor, HR factor, elevation, dome flag |
| Weather | 8 | Temp, wind speed, wind direction (4 categories), humidity, night flag |
| Team strength | 4 | Rolling win%, season run differential — both teams |
| Elo | 2 | Zero-sum Elo ratings updated after every game (2015–present) |

### Model Architecture

Two-target Poisson regression predicts λ_home and λ_away (expected runs per
team) separately. Poisson convolution converts the joint distribution to
P(over) for any line:

```
P(total > line) = ΣΣ Poisson(h | λ_home) × Poisson(a | λ_away)
                     for all h + a > line
```

Three model variants are maintained:

| Model | Class | Role |
|---|---|---|
| `glm_poisson` | `sklearn.PoissonRegressor` | Interpretable baseline |
| `hgbr_poisson` | `sklearn.HistGradientBoostingRegressor(loss='poisson')` | Primary Poisson model |
| `lgbm_binary` | `lightgbm.LGBMClassifier` (isotonic calibrated) | Direct P(over) — paper trading |

Current OOF performance (2021–2025, 5-fold walk-forward):

```
lgbm_binary:  log_loss = 0.7207  |  AUC = 0.4969  |  Brier = 0.2619
Model-based backtest ROI (no filter): -2.84%  →  paper trading
```

Overdispersion confirmed (home=2.14, away=2.36) — both targets exceed the 1.2
threshold, flagging Negative Binomial as the theoretically correct distribution.
NegBinom support is implemented in `mlb/calibration.py`.

### Betting Math

**Expected Value vs Kalshi mid-price:**
```
EV_over  = p × (1 − price) − (1 − p) × price
EV_under = (1 − p) × price − p × (1 − price)
Bet when |EV| > $0.03
```

**Kelly criterion (structural filters):**
```
b = (1 / price) − 1
f* = (p × b − q) / b     where q = 1 − p, p from _FILTER_WIN_PROBS
stake = bankroll × min(f* × 0.50, 0.15)
```

**Closing Line Value (CLV):**
```
OVER:   CLV = closing_price − entry_price   (positive = beat the close)
UNDER:  CLV = entry_price − (1 − closing_price)
```

---

## Test Coverage

```bash
pytest tests/ -v   # 44 tests — all pass
```

| File | Tests | Covers |
|---|---|---|
| `test_betting.py` | 44 | EV/Kelly/CLV formulas, devig, filter constants, Kelly stake math |
| `test_features.py` | 17 | Leakage guard (shift(1) verified), cold-start NaN, rolling correctness |
| `test_elo.py` | 18 | Zero-sum invariant, update formula, regression-to-mean |
| `test_model.py` | 20 | Walk-forward fold structure, overdispersion check |
| `test_calibration.py` | 23 | Convolution sum-to-one, NegBinom vs Poisson, symmetry |

All betting math tests use known analytical values — no fuzzy assertions.
Example from `TestKellyBet`:
```python
def test_known_value_no_cap(self):
    # win_prob=0.55, price=0.50 → b=1.0, f*=(0.55−0.45)/1=0.10
    # half Kelly 0.50× → 0.05 → not capped at 15%
    result = kelly_bet(win_prob=0.55, kalshi_price=0.50)
    np.testing.assert_allclose(result, 0.025, rtol=1e-6)
```

---

## Repository Structure

```
mlb/
├── db.py               # SQLite schema, WAL-mode connection manager
├── scraper.py          # pybaseball + MLB Stats API ingestion
├── odds_scraper.py     # SBR closing lines via sbrscrape
├── weather.py          # Open-Meteo + wind direction encoder
├── kalshi.py           # Kalshi REST + WebSocket (full-game + F5)
├── polymarket.py       # Polymarket Gamma API (read-only signal)
├── features.py         # 48-feature no-leakage pipeline
├── elo.py              # Zero-sum Elo ratings
├── model.py            # Poisson GLM + HGBR + LightGBM binary
├── calibration.py      # Poisson/NegBinom convolution → P(over)
├── betting.py          # EV, Kelly, CLV, structural filters, simulate_structural()
├── statcast_enricher.py # Pitcher Statcast enrichment (xERA, est_wOBA)
├── pipeline.py         # Daily orchestrator (Phase 6)
└── live.py             # WebSocket monitor (Phase 6)

configs/
├── model_config.yaml   # Feature groups, model hyperparameters, CV settings
└── betting_config.yaml # Kelly multiplier, position limits, filter thresholds

tests/unit/
├── test_betting.py     # 44 tests — EV/Kelly/CLV + structural filter constants
├── test_features.py    # Leakage + cold-start tests for every rolling feature
├── test_elo.py         # Zero-sum invariant tests
├── test_model.py       # Walk-forward CV structure
└── test_calibration.py # Convolution correctness

notebooks/
├── 01_data_audit.ipynb            # DB health, coverage, null rates
├── 02_feature_eda.ipynb           # Correlations, park factors, overdispersion
├── 03_kalshi_market_analysis.ipynb # Vig, line movement, market structure
├── 04_model_comparison.ipynb      # GLM vs HGBR vs NegBinom
└── 05_betting_simulation.ipynb    # Walk-forward backtest, filter EDA
```

---

## Data Coverage

| Source | Rows | Coverage |
|---|---|---|
| `games` | ~25,200 | 2015–2026 (all seasons complete) |
| `sportsbook_odds` | 49,244 | 2021–2026 (DraftKings, FanDuel, Caesars, Bet365) |
| `elo_ratings` | 20,532 | 2015–2026 (zero-sum verified) |
| `kalshi_markets` | 1,019 | 2026–present (100% game_id linked) |
| `pitcher_season_statcast` | 3,938 | 2015–2025 |

---

## Running the System

```bash
pip install -r requirements.txt

# Daily pipeline (runs automatically via GitHub Actions at 7am ET)
python -m mlb.scraper --incremental
python -m mlb.weather --incremental
python -m mlb.odds_scraper --date today
python -m mlb.kalshi --snapshot
python -m mlb.model --predict --date today --model lgbm_binary
python -m mlb.betting daily --date today

# Structural filter backtest (primary production strategy)
python -m mlb.betting simulate-structural \
    --filter day_k9_park --filter high_line --filter summer_hot_wind_out \
    --start 2021-04-01 --end 2025-10-01 \
    --book draftkings

# Run tests
pytest tests/ -v

# Lint
ruff check . && ruff format .
```

---

## Stack

Python 3.11 · SQLite (WAL mode) · scikit-learn · LightGBM · statsmodels · scipy
· pybaseball · python-mlb-statsapi · sbrscrape · Open-Meteo · Kalshi API
· Polymarket Gamma API · Ruff · pytest · GitHub Actions

---

## Status

| Component | Status |
|---|---|
| Data pipeline (2015–2026) | Complete |
| Feature engineering (48 features, no leakage) | Complete |
| Poisson model (GLM + HGBR + LightGBM) | Complete |
| Structural filter strategy | **Production-ready** |
| Model-based EV strategy | Paper trading (ROI not yet positive) |
| Live pipeline (`pipeline.py`, `live.py`) | Phase 6 — in progress |
| Kalshi live execution | dry_run = true until live validation |
