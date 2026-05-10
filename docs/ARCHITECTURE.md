# MLB Total Runs Betting System — Architecture Reference
**Last updated:** 2026-05-09
**Status:** Implementation complete through Phase 5 (Betting Engine)

---

## Regulatory Context

MLB signed an exclusive prediction market partnership with Polymarket (March 2026).
Kalshi still operates MLB totals markets and the API works normally. We pull
Polymarket pricing as a read-only cross-market signal only. All bets are placed
on Kalshi.

**Backtesting data gap:** Kalshi MLB data starts April 2025. For 2021–2024 we
use SBR closing totals lines as the market benchmark. For 2015–2020 we train the
regression model on pybaseball data but cannot run the betting simulation (no
market line available). SBR scraping via sbrscrape covers 2021–present.

---

## Repository Structure

```
mlb-betting/
├── CLAUDE.md                        # Master context — always loaded
├── MEMORY.md                        # Session state — update each session
├── docs/
│   ├── ARCHITECTURE.md              # This file
│   ├── DATA_LEAKAGE.md              # Leakage rules and worked examples
│   └── MATH_NOTES.md                # EV, Kelly, Poisson convolution derivations
├── .mcp.json                        # Project-scoped MCP servers
├── .claude/
│   ├── settings.json                # Hook lifecycle config (SessionStart, PreToolUse, PostToolUse)
│   ├── rules/                       # Scoped rules — auto-loaded by directory context
│   │   ├── data.md                  # shift(1), DB writes, rate limits, Elo zero-sum
│   │   ├── math.md                  # Poisson-only, convolution, EV, Kelly, CLV, position limits
│   │   ├── python.md                # Ruff, type hints, NumPy docstrings, logging, SQLite pattern
│   │   ├── testing.md               # Fixture scopes, required leakage tests, math tests, Elo test
│   │   └── git.md                   # Branch naming, Conventional Commits, gitignore, pre-commit
│   ├── agents/                      # Subagent definitions — auto-delegated by task context
│   │   ├── data-engineer.md         # Scraping, ETL, DB writes, feature engineering, weather fetching
│   │   ├── ml-engineer.md           # Poisson models, walk-forward CV, convolution, evaluation
│   │   ├── stats-reviewer.md        # EV/Kelly/CLV/Poisson math auditor — outputs PASS/FAIL/WARN
│   │   └── code-reviewer.md         # Ruff, pytest, DB safety, type hints, PR checklist
│   ├── skills/                      # Slash command workflows
│   │   ├── run-pipeline/SKILL.md    # /run-pipeline — 9-step daily pipeline
│   │   ├── backtest/SKILL.md        # /backtest — walk-forward + simulation + sensitivity table
│   │   ├── add-feature/SKILL.md     # /add-feature — safe feature addition with leakage test
│   │   └── db-migrate/SKILL.md      # /db-migrate — backup-first migration workflow
│   └── hooks/                       # Lifecycle automation (fire automatically)
│       ├── pre_tool_use.py          # Blocks dangerous DB ops; warns on squared_error/norm.cdf/leakage
│       ├── post_tool_use.py         # Auto-Ruff after .py edits; auto leakage tests after features.py
│       └── session_start.py         # Prints git branch, DB row counts, today's bets, MEMORY.md
├── mlb/
│   ├── __init__.py
│   ├── db.py
│   ├── scraper.py
│   ├── odds_scraper.py
│   ├── weather.py
│   ├── kalshi.py
│   ├── polymarket.py
│   ├── features.py
│   ├── elo.py
│   ├── model.py
│   ├── calibration.py
│   ├── betting.py
│   ├── pipeline.py
│   └── live.py
├── data/
│   ├── mlb.db
│   ├── raw/
│   │   ├── sbr/
│   │   └── kalshi/
│   └── models/
├── notebooks/
│   ├── 01_data_audit.ipynb
│   ├── 02_feature_eda.ipynb
│   ├── 03_kalshi_market_analysis.ipynb
│   ├── 04_model_comparison.ipynb
│   └── 05_betting_simulation.ipynb
├── tests/
│   ├── conftest.py
│   ├── unit/
│   │   ├── test_features.py
│   │   ├── test_model.py
│   │   ├── test_calibration.py
│   │   ├── test_betting.py
│   │   └── test_elo.py
│   └── integration/
│       ├── test_pipeline.py
│       └── test_temporal.py
├── configs/
│   ├── model_config.yaml
│   └── betting_config.yaml
├── .github/workflows/
│   └── update.yml
├── pyproject.toml
└── requirements.txt
```

---

## Database Schema

### `games`
```sql
CREATE TABLE games (
    game_id         TEXT PRIMARY KEY,
    date            TEXT NOT NULL,
    season          INTEGER NOT NULL,
    home_team       TEXT NOT NULL,
    away_team       TEXT NOT NULL,
    home_score      INTEGER,
    away_score      INTEGER,
    total_runs      INTEGER,            -- TARGET VARIABLE (full game): home + away runs
    f5_home_score   INTEGER,            -- F5 target: home runs in innings 1-5
    f5_away_score   INTEGER,            -- F5 target: away runs in innings 1-5
    f5_total_runs   INTEGER,            -- F5 target: f5_home + f5_away (KXMLBF5TOTAL)
    venue           TEXT,
    game_time_et    TEXT,
    status          TEXT DEFAULT 'scheduled',
    created_at      TEXT DEFAULT (datetime('now')),
    updated_at      TEXT DEFAULT (datetime('now'))
);
-- Schema change 2026-05-09: Added f5_home_score, f5_away_score, f5_total_runs
-- for First 5 Innings model target (KXMLBF5TOTAL betting market)
CREATE INDEX idx_games_date ON games(date);
CREATE INDEX idx_games_season ON games(season);
CREATE INDEX idx_games_home ON games(home_team, date);
CREATE INDEX idx_games_away ON games(away_team, date);
```

### `team_stats`
```sql
CREATE TABLE team_stats (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    game_id         TEXT NOT NULL REFERENCES games(game_id),
    team            TEXT NOT NULL,
    is_home         INTEGER NOT NULL,
    runs            INTEGER,
    hits            INTEGER,
    errors          INTEGER,
    ops             REAL,
    obp             REAL,
    slg             REAL,
    woba            REAL,
    wrc_plus        REAL,
    k_pct           REAL,
    bb_pct          REAL,
    era_starter     REAL,
    fip_starter     REAL,
    xfip_starter    REAL,
    siera_starter   REAL,
    k9_starter      REAL,
    bb9_starter     REAL,
    days_rest_sp    INTEGER,
    bullpen_era_7d  REAL,
    bullpen_era_30d REAL,
    bullpen_ip_7d   REAL,
    UNIQUE(game_id, team)
);
```

### `pitchers`
```sql
CREATE TABLE pitchers (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    game_id         TEXT NOT NULL REFERENCES games(game_id),
    pitcher_id      INTEGER NOT NULL,
    pitcher_name    TEXT NOT NULL,
    team            TEXT NOT NULL,
    is_starter      INTEGER NOT NULL,
    ip              REAL,
    er              INTEGER,
    era_season      REAL,
    fip_season      REAL,
    xfip_season     REAL,
    siera_season    REAL,
    era_l3          REAL,
    k9_season       REAL,
    bb9_season      REAL,
    hr9_season      REAL,
    velocity_avg    REAL,
    days_rest       INTEGER,
    UNIQUE(game_id, pitcher_id)
);
```

### `weather`
```sql
CREATE TABLE weather (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    game_id         TEXT NOT NULL REFERENCES games(game_id),
    snapshot_type   TEXT NOT NULL,      -- 'historical' or 'forecast'
    fetched_at      TEXT NOT NULL,
    temp_f          REAL,
    wind_speed_mph  REAL,
    wind_dir_deg    REAL,
    wind_dir_label  TEXT,               -- 'out', 'in', 'cross_left', 'cross_right'
    precip_prob     REAL,
    humidity        REAL,
    is_dome         INTEGER DEFAULT 0,
    UNIQUE(game_id, snapshot_type)
);
```

### `stadiums` — static, populated once at setup
```sql
CREATE TABLE stadiums (
    team            TEXT PRIMARY KEY,
    stadium_name    TEXT,
    latitude        REAL,
    longitude       REAL,
    orientation_deg REAL,   -- home plate to CF, meteorological degrees
    elevation_ft    REAL,
    is_dome         INTEGER DEFAULT 0,
    park_run_factor REAL,   -- FanGraphs 3yr regressed (Coors=1.26, Petco=0.89)
    park_hr_factor  REAL
);
```

### `elo_ratings`
```sql
CREATE TABLE elo_ratings (
    id      INTEGER PRIMARY KEY AUTOINCREMENT,
    team    TEXT NOT NULL,
    date    TEXT NOT NULL,
    elo     REAL NOT NULL,
    season  INTEGER NOT NULL,
    UNIQUE(team, date)
);
```

### `sportsbook_odds`
```sql
CREATE TABLE sportsbook_odds (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    game_id          TEXT,
    date             TEXT NOT NULL,
    home_team        TEXT NOT NULL,
    away_team        TEXT NOT NULL,
    book             TEXT NOT NULL,
    total_open       REAL,              -- opening over/under line
    total_close      REAL,              -- closing line — PRIMARY BENCHMARK
    over_odds_open   INTEGER,
    under_odds_open  INTEGER,
    over_odds_close  INTEGER,
    under_odds_close INTEGER,
    home_ml_open     INTEGER,
    away_ml_open     INTEGER,
    home_ml_close    INTEGER,
    away_ml_close    INTEGER,
    source           TEXT DEFAULT 'sbr',
    UNIQUE(date, home_team, away_team, book)
);
-- Coverage: 2021–present via sbrscrape
-- total_close (Pinnacle) is the primary field for the betting simulation
```

### `kalshi_markets`
```sql
CREATE TABLE kalshi_markets (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    game_id         TEXT,
    ticker          TEXT NOT NULL,
    event_ticker    TEXT NOT NULL,
    market_type     TEXT NOT NULL,      -- 'total_over' or 'total_under'
    line            REAL,               -- e.g. 8.5
    date            TEXT NOT NULL,
    snapshot_ts     TEXT NOT NULL,
    yes_bid         REAL,
    yes_ask         REAL,
    mid_price       REAL,
    volume          REAL,
    open_interest   REAL,
    status          TEXT,
    result          TEXT,
    UNIQUE(ticker, snapshot_ts)
);
```

### `predictions`
```sql
CREATE TABLE predictions (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    game_id                 TEXT NOT NULL REFERENCES games(game_id),
    model_name              TEXT NOT NULL,
    model_version           TEXT NOT NULL,
    predicted_at            TEXT NOT NULL,
    lambda_home             REAL NOT NULL,   -- predicted expected runs, home team
    lambda_away             REAL NOT NULL,   -- predicted expected runs, away team
    predicted_total_runs    REAL,            -- lambda_home + lambda_away (convenience)
    dispersion_alpha        REAL,            -- NegBinom dispersion if upgraded from Poisson
    line                    REAL,
    over_prob               REAL,            -- P(total > line) via Poisson convolution
    under_prob              REAL,
    kalshi_ticker           TEXT,
    kalshi_mid_price        REAL,
    polymarket_mid_price    REAL,
    edge                    REAL,
    ev                      REAL,
    kelly_fraction          REAL,
    recommended_bet         REAL,            -- 0.25x fractional Kelly
    bet_side                TEXT,            -- 'OVER', 'UNDER', 'PASS'
    closing_kalshi_price    REAL,
    closing_total_line      REAL,
    clv                     REAL,
    UNIQUE(game_id, model_name, model_version)
);
```

### `scrape_log`
```sql
CREATE TABLE scrape_log (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    run_at           TEXT DEFAULT (datetime('now')),
    source           TEXT NOT NULL,
    season           INTEGER,
    date_range_start TEXT,
    date_range_end   TEXT,
    rows_inserted    INTEGER DEFAULT 0,
    rows_updated     INTEGER DEFAULT 0,
    status           TEXT DEFAULT 'success',
    error_msg        TEXT
);
```

---

## Data Sources

| Source | What | Method | Cost | Coverage |
|---|---|---|---|---|
| **pybaseball** | Game stats, Statcast, FanGraphs | pip package | Free | 2015–present |
| **MLB Stats API** | Schedule, lineups, probable pitchers | pip package | Free | 2010–present |
| **sbrscrape** | Totals + ML odds per sportsbook | pip package | Free | 2021–present |
| **Open-Meteo** | Historical + forecast weather | REST, no key | Free | 1940–present |
| **Kalshi REST** | Live + historical prices | REST + RSA auth | Free | 2026–present |
| **Polymarket Gamma** | MLB prices, read-only | REST, no auth | Free | 2026–present |

**Note on SBR books:** sbrscrape captures DraftKings, FanDuel, Caesars, Bet365, BetMGM.
Pinnacle exited the US market and is not available. Use DraftKings as the simulation benchmark
(most liquid US book). Kalshi totals data starts at 2026 Opening Day (2026-03-31).

---

## Feature Engineering

### Target Variables
We model **two separate targets** — `home_runs` and `away_runs` — rather than
`total_runs` directly. Each model predicts λ (expected runs) for one team.
We then convolve the two Poisson distributions to compute P(over) for any line.

```
Model predicts:  λ_home, λ_away  (expected runs per team)
Convolution:     P(h + a > line) = ΣΣ Poisson(h|λ_home) × Poisson(a|λ_away)
                                     where h + a > line
```

This approach is superior to predicting `total_runs` directly because:
- It respects the discrete, non-negative, count nature of the data
- It lets us price any line, not just the posted one
- It opens the door to exact-score markets later
- No separate calibration step — the model output IS already the right parameter

### No-Leakage Rule (Absolute)
```python
# CORRECT — today excluded via shift(1)
df['sp_siera_l5'] = (
    df.groupby('pitcher_id')['siera_game']
    .shift(1)
    .rolling(5, min_periods=2)
    .mean()
)

# WRONG — leaks today's data into the feature
df['sp_siera_l5'] = df.groupby('pitcher_id')['siera_game'].rolling(5).mean()
```

### Feature Groups (53 total — matches `FEATURE_COLS` in `mlb/features.py`)

**A: Starting Pitchers — both SPs**
```
home_sp_era_season   away_sp_era_season   # season ERA (capped at 13.5 = 3× avg)
home_sp_fip_season   away_sp_fip_season   # fielding-independent pitching
home_sp_k9_season    away_sp_k9_season    # strikeout rate (suppresses runs)
home_sp_bb9_season   away_sp_bb9_season   # walk rate
home_sp_days_rest    away_sp_days_rest    # rest days before start
home_sp_era_l3       away_sp_era_l3       # ERA over last 3 starts (recent form)
home_sp_er_pg_l5     away_sp_er_pg_l5     # earned runs per start, last 5 (raw momentum)
sp_fip_combined      sp_k9_combined       # sum of both SPs (total run suppression)
sp_era_l3_combined   sp_er_pg_l5_combined
```

**B: Team Offense — both teams**
```
home_ops_10d     away_ops_10d       # rolling 10-game OPS
home_k_pct_10d   away_k_pct_10d    # rolling strikeout rate (fewer baserunners = fewer runs)
home_runs_10d    away_runs_10d     # rolling 10-game avg runs scored (momentum)
```

**C: Bullpen — both teams**
```
home_bullpen_era_7d    away_bullpen_era_7d
home_bullpen_era_30d   away_bullpen_era_30d
home_bullpen_ip_7d     away_bullpen_ip_7d    # innings pitched (fatigue proxy)
```

**D: Park Factors — critical for run totals**
```
park_run_factor    # Coors=1.26, Petco=0.89 (FanGraphs 3yr regressed)
park_hr_factor
park_elevation_ft  # altitude — ball carries farther at Coors
is_dome
```

**E: Weather — game-day, from Open-Meteo**
```
temp_f               # cold suppresses scoring
wind_speed_mph
wind_dir_out         # blowing toward OF (boosts runs)
wind_dir_in          # blowing in from OF (suppresses runs)
wind_dir_cross_right
wind_dir_cross_left
precip_prob
humidity
is_night_game
```

**F: Market Signal — SBR closing lines (2021+)**
```
total_line_open    # opening total line
total_line_close   # closing total line (primary market consensus)
line_movement      # close − open (sharp money proxy)
```

**F2: Kalshi Cross-Market Signal (F5 vs full-game pricing divergence)**
```
kalshi_fullgame_line   # Kalshi full-game over/under line
kalshi_f5_line         # Kalshi First 5 Innings over/under line
f5_ratio               # f5_line / fullgame_line — encodes expected run distribution
                       # ~0.5 = uniform; >0.5 = front-loaded; <0.5 = back-loaded
```

**G: Team Strength**
```
home_win_pct_10d          away_win_pct_10d          # rolling 10-game win %
home_run_diff_pg_season   away_run_diff_pg_season   # season run differential per game
```

**Elo**
```
elo_home   elo_away   # zero-sum Elo ratings updated after every game
```

### Wind Direction Encoding
```python
def encode_wind(wind_deg: float, cf_orientation: float) -> str:
    relative = (wind_deg - cf_orientation + 360) % 360
    if 315 <= relative or relative < 45:   return 'in'
    elif 135 <= relative < 225:            return 'out'
    elif 45 <= relative < 135:             return 'cross_right'
    else:                                  return 'cross_left'
```

---

## Model Architecture

### Why Poisson GLM, Not Ridge/GBR with Squared Error

Run scoring is count data. Squared-error regression assumes Normal residuals,
which is wrong for counts — it assigns probability mass to negative values, is
symmetric when run distributions are right-skewed, and optimizes the wrong
objective. Instead we use models that natively speak Poisson.

The pipeline for both models:
```
Features → model → λ_home, λ_away → Poisson convolution → P(over)
```

### Model 1: Poisson GLM (Baseline)
```python
from sklearn.linear_model import PoissonRegressor

# Link function: log(λ) = Xβ  →  λ = exp(Xβ)
# Guarantees λ > 0 (no negative run predictions)
# Directly outputs λ — exactly what the Poisson convolution needs
# alpha is L2 regularization (equivalent to Ridge penalty)

poisson_glm = PoissonRegressor(alpha=1.0, max_iter=300)
```

Train two instances — one for home runs, one for away runs:
```python
glm_home = PoissonRegressor(alpha=1.0, max_iter=300)
glm_away = PoissonRegressor(alpha=1.0, max_iter=300)

glm_home.fit(X_train, y_home_train)
glm_away.fit(X_train, y_away_train)

lambda_home = glm_home.predict(X_test)   # expected home runs
lambda_away = glm_away.predict(X_test)   # expected away runs
```

### Model 2: HistGradientBoostingRegressor with Poisson Loss (Primary Poisson model, `hgbr_poisson`)
```python
from sklearn.ensemble import HistGradientBoostingRegressor

# Histogram-based GBR — faster than GradientBoostingRegressor, handles NaN natively
hgbr_home = HistGradientBoostingRegressor(
    loss='poisson',         # optimizes Poisson deviance, not squared error
    max_iter=300,
    max_depth=4,
    learning_rate=0.05,
    min_samples_leaf=20,
    l2_regularization=0.1,
    random_state=42,
)
hgbr_away = HistGradientBoostingRegressor(loss='poisson', max_iter=300,
    max_depth=4, learning_rate=0.05, min_samples_leaf=20, random_state=42)
```

Output is λ (expected runs) — same interface as PoissonRegressor.

### Model 3: LightGBM Binary Classifier (Primary Betting Model, `lgbm_binary`)
```python
import lightgbm as lgb

# Trained directly on over/under outcome (binary classification)
# Target: 1 if actual_total > total_line_close, 0 otherwise
# Outputs calibrated P(over) after isotonic calibration on OOF predictions
lgbm = lgb.LGBMClassifier(
    objective='binary',
    n_estimators=500,
    learning_rate=0.03,
    max_depth=4,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_samples=30,
    reg_alpha=0.1,
    reg_lambda=0.1,
    random_state=42,
)
```

Walk-forward CV produces OOF predicted probabilities; an isotonic calibrator is fit on those OOF
samples and applied to all predictions. Artefact saved as `data/models/lgbm_binary_v1.0.0.pkl`.

**Current CV performance (2022–2024, 5 folds):**
```
oof_log_loss = 0.7239  |  oof_auc = 0.5076  |  oof_brier = 0.2635
```

### Walk-Forward CV
```python
from sklearn.model_selection import TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits=5, gap=162)
# gap=162 prevents leakage at season boundaries
# Never use KFold or shuffle=True
```

### Poisson Convolution — P(over)
```python
from scipy.stats import poisson
import numpy as np

def p_over_poisson(lambda_home: float, lambda_away: float, line: float,
                   max_runs: int = 30) -> float:
    """
    P(home_runs + away_runs > line) under independent Poisson assumption.
    max_runs=30 is a safe practical ceiling (P(>30 runs) ≈ 0).
    """
    p_over = 0.0
    for h in range(max_runs + 1):
        for a in range(max_runs + 1):
            if h + a > line:
                p_over += poisson.pmf(h, lambda_home) * poisson.pmf(a, lambda_away)
    return p_over

def p_over_vectorised(lambda_home: float, lambda_away: float,
                      line: float, max_runs: int = 30) -> float:
    """Vectorised version — significantly faster for batch prediction."""
    h = np.arange(max_runs + 1)
    a = np.arange(max_runs + 1)
    H, A = np.meshgrid(h, a)
    joint = poisson.pmf(H, lambda_home) * poisson.pmf(A, lambda_away)
    return joint[H + A > line].sum()
```

### Overdispersion Check — When to Upgrade to Negative Binomial
After fitting PoissonRegressor, check whether variance > mean in residuals:

```python
def check_overdispersion(y_true: np.ndarray, lambda_pred: np.ndarray) -> float:
    """
    Dispersion ratio: var(residuals) / mean(lambda_pred).
    > 1.2 consistently across walk-forward folds → upgrade to NegBinom.
    """
    residuals = y_true - lambda_pred
    return residuals.var() / lambda_pred.mean()
```

If dispersion > 1.2, switch baseline to Negative Binomial GLM:
```python
import statsmodels.api as sm

# Negative Binomial relaxes the Poisson mean=variance constraint
# alpha: dispersion parameter estimated from data
nb_model = sm.GLM(
    y_train,
    sm.add_constant(X_train),
    family=sm.families.NegativeBinomial(alpha=1.0)
).fit()

# Updated convolution using NegBinom instead of Poisson
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
    return joint[HH + AA > line].sum()
```

### Evaluation Metrics
```
Poisson deviance (lower = better, use instead of MAE/RMSE):
  from sklearn.metrics import mean_poisson_deviance
  deviance = mean_poisson_deviance(y_true, lambda_pred)

D² score (analogous to R² for Poisson, 1.0 = perfect):
  from sklearn.metrics import d2_tweedie_score
  d2 = d2_tweedie_score(y_true, lambda_pred, power=1)

Log-loss on P(over) after convolution (primary probability metric):
  from sklearn.metrics import log_loss
  ll = log_loss(over_outcomes, over_probs)

Betting simulation (primary end-to-end metric):
  ROI, CLV, win rate, max drawdown, Sharpe ratio

Model comparison summary table (from Notebook 4):
  Model           | Deviance | D²   | Log-loss | ROI backtest
  PoissonGLM      |  x.xx    | x.xx |  x.xxx   |   x.x%
  GBR(poisson)    |  x.xx    | x.xx |  x.xxx   |   x.x%
  NegBinomGLM (*) |  x.xx    | x.xx |  x.xxx   |   x.x%
  (* only if overdispersion confirmed)
```

---

## Betting Engine

Two independent strategy layers exist in `mlb/betting.py`:

1. **Structural filters** (`simulate_structural`) — no model, bet on pre-game conditions.
   These are the **primary production strategy**, validated 2021–2025.
2. **Model-based EV** (`simulate`, `run_daily`) — uses λ_home/λ_away from Poisson models.
   Currently in paper-trading until ROI turns consistently positive.

---

### Structural Filters (Primary Production Strategy)

Pre-game binary rules with no model dependency. Uses SBR closing lines as the market.
Contradictory signals (game triggers both UNDER and OVER) are skipped.

**Backtest 2021–2025, DraftKings, half Kelly 0.5x + 15% cap, $100 start:**
```
$100 → $9,711  |  Win rate 57.4%  |  Sharpe 1.60  |  Max drawdown -67.6%
```

**UNDER Filters**

| Filter | Condition | EDA Win% | n |
|---|---|---|---|
| `day_k9_park` | Day game at SFG/CLE/TEX/CIN/CHW/SDP/SEA/DET + K9 combined ≥ 14.0 + era_l3 ≤ 4.0 | 56.4% | 906 |
| `high_line` | Closing line ≥ 11.0 | 57.5% | 373 |

**OVER Filter**

| Filter | Condition | EDA Win% | n |
|---|---|---|---|
| `summer_hot_wind_out` | Jul–Sep + temp ≥ 80°F + outdoor + wind "out" 10–14 mph | 63.1% | 134 |

Wind cap at 15 mph: above that, gusts disrupt pitcher command and the effect reverses.
July–September restriction removes April–June where the signal is absent (50.0% hit rate).

**Per-filter Kelly input win rates** (`_FILTER_WIN_PROBS` in betting.py):
```python
_FILTER_WIN_PROBS = {
    "day_k9_park":          0.564,
    "high_line":            0.575,
    "hot_wind_out":         0.560,   # all-season variant (weaker)
    "summer_hot_wind_out":  0.631,   # production variant
}
```

**CLI:**
```bash
python -m mlb.betting simulate-structural \
    --filter day_k9_park --filter high_line --filter summer_hot_wind_out \
    --start 2021-04-01 --end 2025-10-01 \
    --book draftkings
```

---

### Bet Sizing — Half Kelly (0.5x), 15% Cap

Validated against flat and quarter/full Kelly variants. Selected configuration:

```python
# sizing='quarter_kelly' parameter with kelly_mult=0.50, kelly_cap=0.15
win_prob = _FILTER_WIN_PROBS[triggered_filter]
b        = (1 / raw_price) - 1          # net odds at DK closing price (vig-inclusive)
full_k   = max(0.0, (win_prob * b - (1 - win_prob)) / b)
stake    = bankroll * min(full_k * 0.50, 0.15)
```

Typical stakes at -110 DK odds:
- `day_k9_park`: full Kelly 8.4% → half Kelly **~4.2%**
- `high_line`: full Kelly 10.7% → half Kelly **~5.4%**
- `summer_hot_wind_out`: full Kelly 22.5% → half Kelly **~11.3%** (below 15% cap)

Growth comparison ($100 → 4.5 years):
```
Flat  5%  → $5,254  Sharpe 1.48  DD -77.8%  (reference)
Qtr Kelly → $1,529  Sharpe 1.60  DD -40.9%  (too conservative)
Half Kelly 15% cap → $9,711  Sharpe 1.60  DD -67.6%  ← production
Full Kelly 15% cap → $27,621 Sharpe 1.60  DD -92.4%  (too aggressive)
```

---

### Model-Based EV Calculation

```python
def compute_ev(over_prob: float, kalshi_over_price: float) -> dict:
    ev_over  = over_prob * (1 - kalshi_over_price) - (1 - over_prob) * kalshi_over_price
    ev_under = (1 - over_prob) * kalshi_over_price - over_prob * (1 - kalshi_over_price)
    if ev_over > 0.03:    bet_side = 'OVER'
    elif ev_under > 0.03: bet_side = 'UNDER'
    else:                 bet_side = 'PASS'
    return {'ev_over': ev_over, 'ev_under': ev_under,
            'edge': over_prob - kalshi_over_price, 'bet_side': bet_side}
```

### Kelly (Model-Based Path)
```python
def kelly_bet(win_prob: float, kalshi_price: float,
              kelly_mult: float = 0.50, max_pct: float = 0.15) -> float:
    b = (1 / kalshi_price) - 1
    full_kelly = max(0.0, (win_prob * b - (1 - win_prob)) / b)
    return min(full_kelly * kelly_mult, max_pct)
```

### CLV
```python
def compute_clv(entry: float, closing: float, side: str) -> float:
    """Positive = beat the closing price. Sustained positive CLV = real edge."""
    return closing - entry if side == 'OVER' else entry - (1 - closing)
```

### Position Constraints
- Min edge: $0.03 vs Kalshi mid
- Min open interest: $1,000
- Max simultaneous positions: 3

### Cross-Market Signal
```python
def get_consensus(kalshi_mid: float, poly_mid: float) -> dict:
    spread = abs(kalshi_mid - poly_mid)
    # spread > 0.04 means one platform is lagging
    return {'spread': spread, 'kalshi_is_cheap_for_over': kalshi_mid < poly_mid}
```

---

## Data Source Notes

### sbrscrape
```python
from sbrscrape import Scoreboard
# game['total'] = {'draftkings': 8.5, 'pinnacle': 8.5, ...}
# game['over_odds'] = {'draftkings': -110, 'pinnacle': -107}
# Use Pinnacle closing line as sharpest benchmark
# Rate limit: 1.5s between requests
```

### Open-Meteo
```python
# Free, no API key, historical from 1940
# Endpoint: https://archive-api.open-meteo.com/v1/archive
# Live/forecast: https://api.open-meteo.com/v1/forecast
# Return hourly data, extract hour closest to first pitch
```

### Kalshi
```python
# Auth: RSA key pair
# Private key: ~/.kalshi/private_key.pem
# Key ID env var: KALSHI_KEY_ID
# Library: pip install pykalshi
# Totals tickers: KXMLB-{YYMONDD}-{TEAM}-T{LINE*10}
```

### Polymarket
```python
# No auth needed. Gamma API: https://gamma-api.polymarket.com/markets
# US access confirmed (Polymarket US, CFTC-regulated, 2026)
```

---

## Claude Code Infrastructure

### How the .claude/ Files Work Together

```
Session opens
    ↓
session_start.py fires   → prints git status, DB counts, MEMORY.md, today's bets
    ↓
CLAUDE.md loads          → critical rules, commands, pointers to everything below
    ↓
Task begins
    ↓
Claude routes to agent   → data-engineer / ml-engineer / stats-reviewer / code-reviewer
    ↓
Scoped rules load        → data.md / math.md / python.md / testing.md / git.md
    ↓
User runs /skill         → run-pipeline / backtest / add-feature / db-migrate
    ↓
Every Bash / edit        → pre_tool_use.py checks for dangerous ops + leakage patterns
Every .py edit           → post_tool_use.py auto-Ruff; leakage tests if features.py
```

### Rules Reference

| File | Path | Auto-activates for |
|---|---|---|
| Data integrity | `.claude/rules/data.md` | `mlb/scraper.py`, `mlb/features.py`, `mlb/elo.py`, `data/` |
| Math & betting | `.claude/rules/math.md` | `mlb/model.py`, `mlb/calibration.py`, `mlb/betting.py` |
| Python standards | `.claude/rules/python.md` | Any `.py` file in `mlb/` |
| Testing | `.claude/rules/testing.md` | Any file in `tests/` |
| Git workflow | `.claude/rules/git.md` | Git commands, PR creation |

### Agents Reference

| Agent | Path | Invoke for |
|---|---|---|
| `data-engineer` | `.claude/agents/data-engineer.md` | Scraping, ETL, DB writes, feature engineering, weather |
| `ml-engineer` | `.claude/agents/ml-engineer.md` | Model training, CV, Poisson convolution, evaluation |
| `stats-reviewer` | `.claude/agents/stats-reviewer.md` | Auditing EV/Kelly/CLV/Poisson math — outputs PASS/FAIL/WARN |
| `code-reviewer` | `.claude/agents/code-reviewer.md` | Code quality, Ruff, pytest, DB safety, PR review |

Explicit invocation example: *"Use the stats-reviewer to audit mlb/betting.py"*

### Skills Reference

| Command | Path | Executes |
|---|---|---|
| `/run-pipeline` | `.claude/skills/run-pipeline/SKILL.md` | DB health → scrape → weather → odds → features → predict → price → recommend |
| `/backtest` | `.claude/skills/backtest/SKILL.md` | Walk-forward CV → betting simulation → sensitivity table → go/no-go decision |
| `/add-feature` | `.claude/skills/add-feature/SKILL.md` | Leakage check → implement with shift(1) → write tests → register in config |
| `/db-migrate` | `.claude/skills/db-migrate/SKILL.md` | Backup → apply migration → verify counts → update code + docs |

### Hooks Reference

| Hook | Path | Trigger | Action |
|---|---|---|---|
| SessionStart | `.claude/hooks/session_start.py` | Every new session | Git branch, DB row counts, today's bets, MEMORY.md |
| PreToolUse | `.claude/hooks/pre_tool_use.py` | Before every Bash or file edit | Block destructive DB ops; warn on `squared_error`, `norm.cdf`, `.rolling()` without `.shift(1)` |
| PostToolUse | `.claude/hooks/post_tool_use.py` | After every `.py` file edit | Auto-run Ruff; auto-run leakage tests when `features.py` touched |

Hook wiring: `.claude/settings.json`

---


```json
{
  "mcpServers": {
    "sqlite": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-sqlite", "--db-path", "./data/mlb.db"]
    },
    "github": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-github"],
      "env": {"GITHUB_PERSONAL_ACCESS_TOKEN": "${GITHUB_TOKEN}"}
    },
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "C:\\Users\\YOU\\mlb-betting"]
    },
    "context7": {
      "command": "npx",
      "args": ["-y", "@upstash/context7-mcp@latest"]
    }
  }
}
```

```powershell
# Install (run once)
claude mcp add github --scope user -- npx -y @modelcontextprotocol/server-github
claude mcp add context7 --scope user -- npx -y @upstash/context7-mcp@latest
claude mcp add sqlite --scope project -- npx -y @modelcontextprotocol/server-sqlite --db-path ./data/mlb.db
claude mcp add filesystem --scope project -- npx -y @modelcontextprotocol/server-filesystem C:\Users\YOU\mlb-betting
```

---

## Implementation Sequence

```
Step 1   pyproject.toml + requirements.txt
         (pybaseball, MLB-StatsAPI, sbrscrape, openmeteo-requests, pykalshi,
          scipy, scikit-learn, statsmodels, joblib, optuna, shap,
          pytest, ruff, jupyter, plotly)
Step 2   mlb/db.py — schema + WAL mode
Step 3   .claude/ infrastructure — rules, agents, skills, hooks
Step 4   data/stadiums.py — coordinates, orientations, park factors
Step 5   mlb/scraper.py — pybaseball 2015–2024 + MLB Stats API
Step 6   mlb/odds_scraper.py — sbrscrape backfill 2021–2024
Step 7   mlb/weather.py — Open-Meteo + wind encoder + backfill
Step 8   mlb/kalshi.py — REST + WebSocket
Step 9   mlb/polymarket.py — Gamma API read-only
Step 10  Initial data load + validation
Step 11  mlb/features.py — full no-leakage pipeline
Step 12  mlb/elo.py
Step 13  tests/unit/
Step 14  notebooks/01_data_audit.ipynb
Step 15  notebooks/02_feature_eda.ipynb
Step 16  mlb/model.py — PoissonRegressor + GBR(loss='poisson'), two targets
         (home_runs + away_runs), walk-forward TimeSeriesSplit
Step 17  mlb/calibration.py — Poisson convolution, vectorised P(over),
         overdispersion check, NegBinom upgrade path
Step 18  notebooks/03_kalshi_market_analysis.ipynb
Step 19  notebooks/04_model_comparison.ipynb
Step 20  mlb/betting.py — EV, Kelly, CLV
Step 21  notebooks/05_betting_simulation.ipynb
Step 22  mlb/pipeline.py — daily orchestrator
Step 23  mlb/live.py — WebSocket monitor
Step 24  .github/workflows/update.yml
Step 25  Integration tests + README
```

---

## Open Research Questions (Resolve in Notebooks)

1. **Kalshi totals liquidity** — Verify in Notebook 3. Fallback to moneyline
   market if totals are too thin.
2. **Ensemble vs winner-take-all** — Decide after Notebook 4.
3. **Min edge threshold** — Start $0.03, test $0.02 and $0.05 in Notebook 5.
4. **Over bias** — Public bets overs; books shade totals slightly high. Check
   if unders have structural edge in SBR data.
5. **Weather by park** — Zero weather features for dome stadiums. Measure
   importance by park exposure in Notebook 2.
6. **Is there actual edge?** — Only Notebook 5 answers this. Live trading
   waits for positive backtested ROI.
