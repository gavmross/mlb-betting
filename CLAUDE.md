# MLB Total Runs Prediction & Betting System

## Project Goal
Predict total runs scored in MLB games using two-target Poisson regression
(λ_home + λ_away). Convert to P(over/under) via Poisson convolution. Identify
positive EV bets on Kalshi totals markets. Execute fractional Kelly-sized
positions. Use Polymarket as a cross-market pricing signal.

**Full architecture and schema:** @docs/ARCHITECTURE.md
**Session memory / current state:** @MEMORY.md (create if missing)

---

## Stack
- **Language:** Python 3.11+, Windows native
- **Database:** SQLite WAL mode — `data/mlb.db`
- **ML:** scikit-learn (PoissonRegressor + GradientBoostingRegressor), statsmodels (NegBinom if needed), scipy
- **Data:** pybaseball, python-mlb-statsapi, sbrscrape, Open-Meteo, Kalshi API, Polymarket API
- **Betting:** Kalshi totals markets (over/under total runs)
- **Quality:** Ruff, pytest, NumPy docstrings, type hints required
- **Automation:** GitHub Actions (daily pipeline)

---

## Critical Domain Rules — Non-Negotiable

### Temporal Integrity (Highest Priority)
- Every rolling feature MUST use `.shift(1)` before `.rolling()` — no exceptions
- Features must be computable using ONLY data available before first pitch
- Walk-forward (`TimeSeriesSplit`) is the ONLY valid CV strategy — never KFold
- After any feature change: `pytest tests/unit/test_features.py -v`
- Full rules: @.claude/rules/data.md

### Model: Two-Target Poisson Approach
- Predict **two separate targets**: `home_runs` (λ_home) and `away_runs` (λ_away)
- Models: `PoissonRegressor` (baseline) + `GradientBoostingRegressor(loss='poisson')` (primary)
- Never use `loss='squared_error'` — run scoring is count data, not Normal
- Convert λ_home + λ_away → P(over) via Poisson convolution (not Normal CDF)
- Check overdispersion after fitting: `var(residuals) / mean(λ)` — if > 1.2, upgrade to NegBinom
- Primary eval metric: ROI in betting simulation. Secondary: Poisson deviance, D², log-loss
- Full rules: @.claude/rules/math.md

### Betting Math
- EV on OVER: `(p_over * (1 - kalshi_price)) - ((1 - p_over) * kalshi_price)`
- Kelly: `f* = (p * b - q) / b` where `b = (1 / kalshi_price) - 1`
- Always apply 0.25x fractional Kelly — never full Kelly
- Hard cap: never bet more than 5% of bankroll on a single game
- Only bet when edge > $0.03 vs Kalshi mid-price
- Skip if Kalshi open interest < $1,000

### Database Safety
- Never DROP or TRUNCATE in production
- All writes use INSERT OR IGNORE or INSERT OR REPLACE
- WAL mode on every connection: `PRAGMA journal_mode=WAL`
- Schema changes require a note in @docs/ARCHITECTURE.md

---

## Commands

```bash
# Install
pip install -r requirements.txt

# Daily pipeline
python -m mlb.pipeline --date today

# Incremental data update
python -m mlb.scraper --incremental
python -m mlb.weather --incremental

# Train
python -m mlb.model --train --walk-forward

# Backtest
python -m mlb.model --backtest --n-splits 5

# Tests
pytest tests/ -v
pytest tests/unit/test_features.py -v    # run after every feature change

# Code quality
ruff check . && ruff format .

# DB
sqlite3 data/mlb.db ".schema"
sqlite3 data/mlb.db "SELECT name, COUNT(*) FROM sqlite_master WHERE type='table'"

# Live monitor
python -m mlb.live --watch
```

---

## Python Standards
- Ruff replaces Black + Flake8 + isort — run before every commit
- Type hints on all public functions
- NumPy-style docstrings on all public functions
- No bare `except:` — catch specific exceptions
- Float comparisons in tests: `np.testing.assert_allclose(rtol=1e-6, atol=1e-4)`

## Testing
- Session-scoped fixtures for read-only DB access
- Function-scoped for any mutable state
- Every new feature → leakage test in `tests/unit/test_features.py`
- Every new math function → correctness test

## Git
- Trunk-based: branch per feature (`feature/`, `fix/`, `data/`)
- Conventional Commits: `feat:`, `fix:`, `data:`, `test:`, `docs:`
- Never commit `data/mlb.db`, `data/raw/`, or `data/models/`

---

## Scoped Rules
Auto-loaded by Claude when working in their relevant directories.

| File | Activates when editing | Covers |
|---|---|---|
| @.claude/rules/data.md | scraper, features, elo, any data/ file | shift(1), DB writes, rate limits, Elo zero-sum |
| @.claude/rules/math.md | model, calibration, betting | Poisson-only, convolution, EV, Kelly, CLV |
| @.claude/rules/python.md | any .py file in mlb/ | Ruff, type hints, docstrings, logging, SQLite pattern |
| @.claude/rules/testing.md | any file in tests/ | fixture scopes, required leakage tests, math tests |
| @.claude/rules/git.md | git operations | branch naming, Conventional Commits, gitignore |

## Agents
Auto-delegated by task context. Invoke explicitly: "use the stats-reviewer to audit betting.py"

| Agent | File | Handles |
|---|---|---|
| `data-engineer` | @.claude/agents/data-engineer.md | Scraping, ETL, DB writes, feature engineering, weather |
| `ml-engineer` | @.claude/agents/ml-engineer.md | Model training, walk-forward CV, Poisson convolution, evaluation |
| `stats-reviewer` | @.claude/agents/stats-reviewer.md | EV/Kelly/CLV audit, Poisson correctness, PASS/FAIL report |
| `code-reviewer` | @.claude/agents/code-reviewer.md | Ruff, pytest, DB safety, type hints, PR review |

## Skills
Slash commands. Type `/skill-name` in Claude Code to execute the full workflow.

| Command | File | Does |
|---|---|---|
| `/run-pipeline` | @.claude/skills/run-pipeline/SKILL.md | 9-step daily pipeline: scrape → weather → features → predict → price → recommend |
| `/backtest` | @.claude/skills/backtest/SKILL.md | Walk-forward CV + betting simulation + sensitivity table + go/no-go decision |
| `/add-feature` | @.claude/skills/add-feature/SKILL.md | Safe feature addition: leakage check → implement → test → register |
| `/db-migrate` | @.claude/skills/db-migrate/SKILL.md | Backup → apply migration → verify → update code + docs |

## Hooks
Fire automatically — no invocation needed.

| Hook | File | Fires when | Does |
|---|---|---|---|
| SessionStart | @.claude/hooks/session_start.py | Every new session | Prints git branch, DB row counts, today's bets, MEMORY.md state |
| PreToolUse | @.claude/hooks/pre_tool_use.py | Before every Bash or file edit | Blocks destructive DB ops; warns on `squared_error`, `norm.cdf`, rolling without shift |
| PostToolUse | @.claude/hooks/post_tool_use.py | After every Python file edit | Auto-runs Ruff; auto-runs leakage tests if features.py was touched |

Hook lifecycle config: @.claude/settings.json

---

## Implementation Phases

```
Phase 0: Foundation           repo, schema, pyproject.toml             [ ]
Phase 1: Data Infrastructure  scraper, odds, weather, Kalshi, Poly     [ ]
Phase 2: Feature Engineering  no-leakage pipeline, Elo                 [ ]
Phase 3: Research Notebooks   market analysis, EDA, model compare      [ ]
Phase 4: Model Layer          PoissonGLM + GBR(poisson), walk-forward, convolution [ ]
Phase 5: Betting Engine       EV, Kelly, CLV, backtest simulation      [ ]
Phase 6: Live Pipeline        daily orchestrator, WebSocket monitor    [ ]
```

Update MEMORY.md with current phase, what's done, and blockers at end of each session.
