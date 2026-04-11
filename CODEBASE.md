# MLB Betting System — Codebase Guide

A complete walkthrough of every file, how they connect, and how to operate
the system from scratch.

---

## What This System Does

Predicts total runs scored in MLB games and identifies positive expected-value
bets on Kalshi totals markets (over/under on total runs per game).

The core loop:

```
Historical game data  →  Feature engineering  →  Two-target Poisson model
                                                          ↓
                                          λ_home + λ_away (expected runs)
                                                          ↓
                                     Poisson/NegBinom convolution → P(over)
                                                          ↓
                                    EV vs Kalshi mid-price → Kelly stake
```

Bets are placed on Kalshi. Polymarket is used as a read-only cross-market
pricing signal only.

---

## Repository Layout

```
mlb-betting/
│
├── mlb/                    # all production Python — one module per concern
│   ├── db.py               # schema, WAL-mode connection context manager
│   ├── scraper.py          # game schedule + box scores (MLB Stats API)
│   ├── odds_scraper.py     # opening/closing totals lines (SBR / sbrscrape)
│   ├── weather.py          # game-day weather (Open-Meteo)
│   ├── kalshi.py           # Kalshi market snapshots + backfill
│   ├── polymarket.py       # Polymarket read-only price signal
│   ├── features.py         # no-leakage feature engineering pipeline
│   ├── elo.py              # team Elo ratings (zero-sum, per-season)
│   ├── model.py            # Poisson GLM + HGBR + NegBinom GLM, walk-forward CV
│   ├── calibration.py      # Poisson/NegBinom convolution → P(over/under)
│   └── betting.py          # EV, Kelly, CLV, daily pricer, backtest simulator
│
├── notebooks/              # research & analysis (not imported by production code)
│   ├── 01_data_audit.ipynb          # DB health, row counts, coverage gaps
│   ├── 02_feature_eda.ipynb         # feature correlations, over/under bias
│   ├── 03_kalshi_market_analysis.ipynb  # market structure, vig, line movement
│   └── 04_model_comparison.ipynb   # GLM vs HGBR vs NegBinom, calibration curves
│
├── tests/
│   ├── conftest.py                  # shared fixtures (DB connections)
│   └── unit/
│       ├── test_features.py         # leakage tests for every rolling feature
│       ├── test_elo.py              # zero-sum invariant, update math
│       ├── test_model.py            # CV structure, overdispersion check
│       ├── test_calibration.py      # convolution correctness, sum-to-one
│       └── test_betting.py          # EV, Kelly, CLV formulas
│
├── configs/
│   ├── model_config.yaml   # feature groups, model hyperparameters, CV settings
│   └── betting_config.yaml # Kelly multiplier, position limits, edge threshold
│
├── docs/
│   ├── ARCHITECTURE.md     # full spec (schema, feature list, model design)
│   ├── DATA_LEAKAGE.md     # leakage rules with worked examples
│   └── MATH_NOTES.md       # EV, Kelly, Poisson convolution derivations
│
├── .claude/                # Claude Code automation (not application code)
│   ├── settings.json       # hook wiring (SessionStart, PreToolUse, PostToolUse)
│   ├── rules/              # auto-loaded coding rules per file context
│   ├── agents/             # subagent definitions (data-engineer, ml-engineer, etc.)
│   ├── skills/             # slash-command workflows (/run-pipeline, /backtest, etc.)
│   └── hooks/              # lifecycle scripts (session start, pre/post tool use)
│
├── .github/workflows/
│   └── update.yml          # daily pipeline: 7am ET, scrape → predict → price
│
├── data/                   # gitignored — never committed
│   ├── mlb.db              # SQLite database (WAL mode)
│   ├── raw/                # cached API responses (sbr/, kalshi/)
│   └── models/             # serialised model artefacts (.pkl)
│
├── pyproject.toml          # build config, Ruff settings, pytest settings
├── requirements.txt        # pinned dependencies
├── CLAUDE.md               # master context file for Claude Code sessions
└── MEMORY.md               # session state tracker (current phase, progress)
```

---

## The Database (`mlb/db.py`)

Single SQLite file at `data/mlb.db`, always opened in WAL mode.

**Key function:**
```python
with get_conn() as conn:   # yields a WAL-mode connection, commits on exit
    rows = conn.execute("SELECT ...").fetchall()
```

**Tables at a glance:**

| Table | What it holds | Key columns |
|---|---|---|
| `games` | One row per game | `game_id`, `date`, `home_team`, `away_team`, `home_score`, `away_score`, `total_runs` |
| `team_stats` | Batting stats per team per game | `ops`, `woba`, `wrc_plus`, `k_pct`, `era_starter`, `bullpen_era_7d` |
| `pitchers` | Per-pitcher game log | `pitcher_id`, `ip`, `era_season`, `fip_season`, `k9_season`, `days_rest` |
| `weather` | Game-day weather snapshot | `temp_f`, `wind_speed_mph`, `wind_dir_label`, `precip_prob`, `is_dome` |
| `stadiums` | Static park data | `latitude`, `longitude`, `orientation_deg`, `park_run_factor`, `is_dome` |
| `elo_ratings` | Team Elo per date | `team`, `date`, `elo`, `season` |
| `sportsbook_odds` | SBR closing lines | `total_open`, `total_close`, `over_odds_close`, `book` |
| `kalshi_markets` | Kalshi price snapshots | `ticker`, `line`, `yes_bid`, `yes_ask`, `mid_price`, `open_interest` |
| `predictions` | Model outputs + bet recs | `lambda_home`, `lambda_away`, `over_prob`, `edge`, `ev`, `kelly_fraction`, `bet_side` |
| `scrape_log` | Audit trail for every run | `source`, `rows_inserted`, `status`, `error_msg` |

All writes use `INSERT OR IGNORE` or `INSERT OR REPLACE` — never raw `INSERT`.
Schema changes go through the `/db-migrate` skill (backup → alter → verify).

---

## Data Pipeline Modules

### `mlb/scraper.py` — Game Data

Pulls schedule and box scores from the MLB Stats API via `mlbstatsapi`.

```bash
python -m mlb.scraper --start 2022-04-07 --end 2024-10-01   # backfill
python -m mlb.scraper --incremental                          # new games only
```

Writes to `games`, `team_stats`, `pitchers`, `scrape_log`.
Rate limit: 1 second between API calls. Raw responses cached in `data/raw/`.

**Coverage:** 2015–present via MLB Stats API.

---

### `mlb/odds_scraper.py` — Sportsbook Lines

Pulls opening and closing totals lines from SportsBooksReview via `sbrscrape`.

```bash
python -m mlb.odds_scraper --start 2022-04-07 --end 2024-10-01
python -m mlb.odds_scraper --date today
```

Writes to `sportsbook_odds`, `scrape_log`.
Rate limit: 1.5 seconds between requests. Raw responses cached in `data/raw/sbr/`.

**Available books:** bet365, betmgm, caesars, draftkings, fanduel.
Note: Pinnacle left the US market; the consensus of these five books is used
as the sharpest available benchmark. **Coverage:** 2021–present.

---

### `mlb/weather.py` — Game-Day Weather

Fetches hourly weather from Open-Meteo (no API key required) and extracts
the observation or forecast closest to first-pitch time.

```bash
python -m mlb.weather --incremental
python -m mlb.weather --start 2022-04-07 --end 2024-10-01
```

Writes to `weather`, `scrape_log`.

Key function: `encode_wind(wind_deg, cf_orientation)` converts meteorological
wind direction into one of `'in'`, `'out'`, `'cross_left'`, `'cross_right'`
relative to the stadium's home-plate-to-CF orientation. Wind blowing out
boosts run scoring; wind blowing in suppresses it.

Dome stadiums skip weather fetching entirely — `is_dome = 1` on the `stadiums`
table and all weather features are set to neutral values.

---

### `mlb/kalshi.py` — Kalshi Markets

Authenticates with RSA key pair and snapshots KXMLB totals markets.

```bash
python -m mlb.kalshi --snapshot            # today's open markets
python -m mlb.kalshi --start 2025-04-01    # backfill historical
```

Writes to `kalshi_markets`, `scrape_log`.
Credentials: `KALSHI_KEY_ID` env var + `~/.kalshi/private_key.pem`.
**Coverage:** April 2025 onward (Kalshi MLB data didn't exist before this).

---

### `mlb/polymarket.py` — Polymarket Signal

Read-only. Fetches over/under prices from the Polymarket Gamma API (no auth).

```bash
python -m mlb.polymarket --snapshot
```

Writes `polymarket_mid_price` into matching rows of the `predictions` table.
Used only as a cross-market signal — if Kalshi mid-price < Polymarket mid-price
for the over, Kalshi is the cheaper entry.

---

### `mlb/elo.py` — Team Elo Ratings

Computes rolling Elo ratings for all 30 MLB teams.

```bash
python -m mlb.elo --start-season 2022 --end-season 2024
python -m mlb.elo --reset    # recompute from scratch
```

Writes to `elo_ratings`, `scrape_log`.

**Algorithm:**
- Starting Elo: 1500 for all teams
- K-factor: 20
- Off-season regression: `elo = 0.70 × elo_end + 0.30 × 1500`
- Zero-sum invariant: sum of all ratings is constant after every game update

Elo is used as a team-strength feature (`elo_home`, `elo_away`) in the model.

---

## Feature Engineering (`mlb/features.py`)

The entire pipeline is a single function:

```python
df = build_features(start_date='2022-04-07', end_date='2024-10-01')
# Returns one row per game with all features + targets (home_runs, away_runs)
```

**The leakage rule — non-negotiable:**
Every rolling stat uses `.shift(1)` before `.rolling()` to exclude the current
game. Sorting by `(entity, date)` before groupby-rolling is also required.

```python
# Correct
df['home_ops_10d'] = (
    df.groupby('team')['ops']
    .shift(1)           # exclude today
    .rolling(10, min_periods=3)
    .mean()
)
```

**Feature groups (49 total):**

| Group | Features | Source table |
|---|---|---|
| A — Starting pitchers | `sp_siera_season`, `sp_fip_season`, `sp_era_l3`, `sp_k9_season`, `sp_bb9_season`, `sp_days_rest`, `sp_velocity_l3`, combined sums | `pitchers` |
| B — Team offense | `ops_10d`, `woba_10d`, `wrc_plus`, `k_pct_10d`, combined sums | `team_stats` |
| C — Bullpen | `bullpen_era_7d`, `bullpen_era_30d`, `bullpen_ip_7d` | `team_stats` |
| D — Park factors | `park_run_factor`, `park_hr_factor`, `park_elevation_ft`, `is_dome` | `stadiums` |
| E — Weather | `temp_f`, `wind_speed_mph`, `wind_dir_out/in/cross`, `precip_prob`, `is_night_game` | `weather` |
| F — Market signal | `total_line_open`, `line_movement` | `sportsbook_odds` |
| G — Team strength | `elo_home`, `elo_away`, `win_pct_10d`, `run_diff_pg_season` | `elo_ratings`, `games` |

**Targets:** `home_runs` (λ_home) and `away_runs` (λ_away) — modelled separately.

---

## Modelling (`mlb/model.py` + `mlb/calibration.py`)

### Why Two Targets?

We predict `home_runs` and `away_runs` as separate Poisson-distributed counts
rather than predicting `total_runs` directly. This lets us:
- Respect the discrete, count nature of the data
- Price any line, not just the posted one
- Use the correct Poisson convolution to get P(over)

### Three Models

| Model | Class | When to use |
|---|---|---|
| `PoissonGLM` | `sklearn.linear_model.PoissonRegressor` | Baseline; interpretable coefficients |
| `HGBR(poisson)` | `sklearn.ensemble.HistGradientBoostingRegressor(loss='poisson')` | Primary; handles non-linearity and NaN natively |
| `NegBinomGLM` | `NegBinomGLMWrapper` (statsmodels under the hood) | Upgrade path when dispersion > 1.2 |

**Overdispersion** (confirmed from EDA): MLB run scoring has `var(y) >> mean(λ)`.
The dispersion ratio was measured at 2.14 (home) and 2.36 (away) — both well
above the 1.2 threshold — so the NegBinom upgrade is required for production.

### Walk-Forward CV

```python
from sklearn.model_selection import TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits=5, gap=162)   # gap = one full season
```

`KFold` is banned. `gap=162` prevents any season-boundary leakage. Only data
strictly before the test window is used for training.

### From λ to P(over)

```python
from mlb.calibration import p_over_negbinom

p_over = p_over_negbinom(
    mu_home=4.3,   # λ from home model
    mu_away=4.1,   # λ from away model
    alpha=0.22,    # dispersion parameter (estimated per fold)
    line=8.5,
)
# → e.g. 0.487 meaning the model gives 48.7% chance total exceeds 8.5
```

The convolution sums `P(h) × P(a)` over all `(h, a)` combinations where
`h + a > line` using a 31×31 grid (0–30 runs per team; P(>30) ≈ 0).

### CLI

```bash
python -m mlb.model --train                   # train on full dataset, save artefact
python -m mlb.model --backtest --n-splits 5   # walk-forward CV with metrics
python -m mlb.model --predict --date today    # write λ_home/λ_away to predictions table
```

---

## Betting Engine (`mlb/betting.py`)

### Core Math

**Expected Value:**
```
EV_over  = p_over × (1 − price) − (1 − p_over) × price
EV_under = (1 − p_over) × price − p_over × (1 − price)
Bet when EV > $0.03 (min edge threshold)
```

**Kelly Criterion:**
```
b = (1 / price) − 1
f* = (p × b − q) / b    where q = 1 − p
Bet size = f* × 0.25    (fractional Kelly, always 0.25×)
Hard cap: never bet more than 5% of bankroll
```

**Closing Line Value (CLV):**
```
OVER:   CLV = closing_price − entry_price   (positive = we beat the close)
UNDER:  CLV = entry_price − (1 − closing_price)
```
Sustained positive CLV over 200+ bets is the strongest evidence of real edge.

### Position Limits

- Minimum edge: $0.03 vs Kalshi mid-price
- Minimum Kalshi open interest: $1,000
- Maximum simultaneous positions: 3 games at once
- Maximum single-game bet: 5% of bankroll

### American Odds Conversion

SBR uses American odds (e.g. −110). These are converted to Kalshi-equivalent
implied probabilities and then devigged before use:

```python
raw_over  = american_to_price(-110)   # → 0.5238
raw_under = american_to_price(-110)   # → 0.5238
fair_over, fair_under = devig_prices(raw_over, raw_under)  # → (0.50, 0.50)
```

### CLI

```bash
python -m mlb.betting daily --date today        # price today, write to predictions
python -m mlb.betting simulate \
    --start 2021-04-01 --end 2024-10-01 \
    --min-edge 0.03 --kelly-mult 0.25 \
    --initial-bankroll 1000                     # historical simulation
python -m mlb.betting simulate --min-edge 0.02 --output sensitivity_002.csv
python -m mlb.betting update-clv --date 2025-04-09
```

---

## Daily Pipeline

The intended daily flow (also what GitHub Actions runs at 7am ET):

```
1. python -m mlb.scraper --incremental       → update games/team_stats/pitchers
2. python -m mlb.weather --incremental       → fetch forecast for today's games
3. python -m mlb.odds_scraper --date today   → fetch today's opening lines
4. python -m mlb.kalshi --snapshot           → snapshot Kalshi prices
5. python -m mlb.polymarket --snapshot       → snapshot Polymarket prices
6. python -m mlb.model --predict --date today → write λ to predictions table
7. python -m mlb.betting daily --date today  → compute EV/Kelly, write bets
```

Steps 1–7 are automated in `.github/workflows/update.yml`.

---

## Tests (`tests/`)

```bash
pytest tests/ -v                          # full suite
pytest tests/unit/test_features.py -v    # run after any feature change
```

**Test counts (last run):**

| File | Tests | Covers |
|---|---|---|
| `test_features.py` | 17 | Leakage guard, cold-start NaN, shift(1) correctness |
| `test_elo.py` | 18 | Zero-sum invariant, update formula, regression-to-mean |
| `test_model.py` | 20 | CV structure, fold boundaries, overdispersion check |
| `test_calibration.py` | 23 | Convolution sum-to-one, NegBinom vs Poisson, symmetry |
| `test_betting.py` | 38 | EV formula, Kelly known values, CLV sign conventions, devig |

Every new rolling feature requires two new tests: a **leakage test** (verifies
`shift(1)` is applied) and a **cold-start test** (verifies NaN for first game).

Every new math function requires at least one **correctness test** with a
manually-computed known value.

---

## Configuration

### `configs/model_config.yaml`

Controls which feature groups are active, model hyperparameters, and CV settings.
Set `enabled: false` under a feature group to exclude it during experiments.
The `overdispersion.threshold: 1.2` setting controls when NegBinom is flagged.

### `configs/betting_config.yaml`

Controls all position sizing and filter parameters. Key settings:

```yaml
sizing:
  kelly_multiplier: 0.25    # NEVER change to 1.0
  max_bet_pct: 0.05         # 5% bankroll cap
  max_open_positions: 3

filters:
  min_edge: 0.03            # $0.03 vs Kalshi mid
  min_open_interest: 1000.0

kalshi:
  dry_run: true             # stays true until backtesting confirms edge
```

---

## Notebooks

The notebooks are research and analysis tools. They do not contain production
logic — all reusable code lives in `mlb/`. Run them top-to-bottom with a
populated database.

| Notebook | Purpose | Key output |
|---|---|---|
| `01_data_audit.ipynb` | DB health check, coverage gaps, data quality | Row counts, null rates, date coverage |
| `02_feature_eda.ipynb` | Feature correlation with actual scoring | Park factor r=0.71, market ceiling r=0.22; overdispersion confirmed (2.1–2.4) |
| `03_kalshi_market_analysis.ipynb` | Market structure, vig analysis, line movement | Vig ~5.6%, under bias ~2%, Kalshi preferred over SBR books |
| `04_model_comparison.ipynb` | GLM vs HGBR vs NegBinom, calibration curves | Three-model comparison table, final model selection |

Notebook 05 (`05_betting_simulation.ipynb`) is the next step — walk-forward
betting simulation with sensitivity analysis and go/no-go decision.

---

## Key Rules (Summary)

**Data integrity:**
- Every rolling feature: `.shift(1)` before `.rolling()` — no exceptions
- Sort by `(entity, date)` before groupby-rolling
- All DB writes: `INSERT OR IGNORE` or `INSERT OR REPLACE`

**Model:**
- Never use `loss='squared_error'` — run scoring is count data
- Never use `norm.cdf` for P(over) — use Poisson/NegBinom convolution
- Walk-forward `TimeSeriesSplit` only — never `KFold` or `shuffle=True`

**Betting:**
- Always 0.25× fractional Kelly — never full Kelly
- Hard cap: 5% of bankroll per game
- Only bet when EV > $0.03 vs Kalshi mid-price
- `dry_run: true` in `betting_config.yaml` until backtest confirms positive ROI

**Code style:**
- Ruff for linting and formatting (`ruff check . && ruff format .`)
- Type hints on all public functions
- NumPy-style docstrings on all public functions
- No bare `except:` — always catch specific exception types

---

## From Scratch Setup

```bash
# 1. Install
pip install -r requirements.txt

# 2. Initialise schema
python -m mlb.db

# 3. Backfill historical data (takes ~30 min)
python -m mlb.scraper --start 2022-04-07 --end 2024-10-01
python -m mlb.odds_scraper --start 2022-04-07 --end 2024-10-01
python -m mlb.weather --start 2022-04-07 --end 2024-10-01

# 4. Compute Elo ratings
python -m mlb.elo --start-season 2022 --end-season 2024

# 5. Train model
python -m mlb.model --train

# 6. Run notebooks in order (01 → 04) to validate data and select model

# 7. Run betting simulation
python -m mlb.betting simulate --start 2021-04-01 --end 2024-10-01

# 8. If ROI > 0 and Sharpe > 0.5: set dry_run: false in configs/betting_config.yaml
#    and run the daily pipeline
python -m mlb.scraper --incremental
python -m mlb.model --predict --date today
python -m mlb.betting daily --date today
```
