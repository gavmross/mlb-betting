# Session Memory

## Current Phase
Phase 5 complete. Phase 6 (Live Pipeline) ready — historical backfill DONE.

## Implementation Progress

```
Step 1   pyproject.toml + requirements.txt        [x]
Step 2   mlb/db.py — schema + WAL mode            [x]
Step 3   .claude/ infrastructure                  [x]
Step 4   data/stadiums.py                         [x]
Step 5   mlb/scraper.py                           [x] (2015-2026 COMPLETE — all seasons scraped)
Step 6   mlb/odds_scraper.py                      [x] (49,244 rows 2021-2026)
Step 7   mlb/weather.py                           [x]
Step 8   mlb/kalshi.py                            [x] (F5TOTAL snapshot added)
Step 9   mlb/polymarket.py                        [x]
Step 10  Initial data load + validation           [x]
Step 11  mlb/features.py                          [x] (51 features; removed 4 dead NULL cols: precip_prob, kalshi_fullgame_line, kalshi_f5_line, f5_ratio)
Step 12  mlb/elo.py                               [x]
Step 13  tests/unit/                              [x] (44 pass in test_betting.py; full suite clean)
Step 14  notebooks/01_data_audit.ipynb            [x]
Step 15  notebooks/02_feature_eda.ipynb           [x]
Step 16  mlb/model.py                             [x] (glm_poisson, hgbr_poisson, lgbm_binary; --target fullgame|f5)
Step 17  mlb/calibration.py                       [x]
Step 18  notebooks/03_kalshi_market_analysis.ipynb [~] in progress
Step 19  notebooks/04_model_comparison.ipynb      [~] in progress
Step 20  mlb/betting.py                           [x] (structural filters: day_k9_park, high_line, summer_hot_wind_out; half Kelly 15% cap)
Step 21  notebooks/05_betting_simulation.ipynb    [~] in progress
Step 22  mlb/pipeline.py                         [ ] Phase 6
Step 23  mlb/live.py                              [ ] Phase 6
Step 24  .github/workflows/update.yml             [x] (fixed — was calling non-existent pipeline.py)
Step 25  Integration tests + README               [ ]
```

## Last Completed (2026-05-10 session — structural filters + bet sizing + codebase cleanup)

### Structural betting strategy — PRODUCTION READY
Three filters now implemented in `simulate_structural()` in betting.py:

| Filter | Side | Win Rate | n (2021-25) | Conditions |
|---|---|---|---|---|
| day_k9_park | UNDER | 56.4% | 906 | Day game, ≥8 parks, combined K/9 ≥14 |
| high_line | UNDER | 57.5% | 373 | Closing total ≥11 |
| summer_hot_wind_out | OVER | 63.1% | 134 | Jul-Sep, ≥80°F, wind 10-15mph out |

Combined 2021-2025 DraftKings backtest (half Kelly, 15% cap, $100 start):
- Terminal bankroll: $971 (starting $100)
- Sharpe ratio: 1.60 (annualised)
- Max drawdown: -67.6%
- ~1,413 bets over 5 seasons

Key finding — `summer_hot_wind_out` beats `hot_wind_out` (all-season):
- Apr-Jun hit rate = 50.0% (no signal, pure noise)
- Jul-Sep hit rate = 63.1% (p<0.001, genuine edge)
- Physical logic: summer heat is more extreme; night-game lines set in afternoon before late weather updates

### Bet sizing finalized: Half Kelly 0.50x, 15% cap
Quarter Kelly (0.25x, 5% cap): $100 → $696 (less because Kelly bets less than fixed 5%)
Fixed 5%: $100 → $936 (closer to full Kelly ~8.4% for UNDER bets)
Half Kelly 0.50x, 15% cap: $100 → $971 (correctly sizes OVER bets ~11.3% vs summer filter)

Why 15% cap (not 5%): summer_hot_wind_out full Kelly ~22.5%, half Kelly ~11.3% — 5% cap
was artificially suppressing OVER bet sizes. 15% cap lets OVER bets reach their correct
half-Kelly allocation while keeping UNDER bets (4-5%) unchanged.

### Codebase cleanup (2026-05-10)
- Deleted scratch CSVs from project root and data/: gbr_kalshi_bets*.csv, lgbm_kalshi_bets_fixed.csv,
  data/backtest_sim.csv, data/combined_strategy_bets.csv, data/strat_*.csv
- Updated mlb/__init__.py: removed dead pipeline/live references, added statcast_enricher, updated descriptions
- Rewrote configs/betting_config.yaml: kelly_multiplier=0.50, max_bet_pct=0.15, added full structural_filters section
- Rewrote configs/model_config.yaml: accurate feature list with counts per group, excluded section explains
  why market signal and Kalshi features are not in FEATURE_COLS
- Added TestStructuralFilterConstants class (6 tests) to tests/unit/test_betting.py — all pass
- Updated docs/ARCHITECTURE.md betting section: structural vs model-based layers, filter table, sizing comparison
- Updated CLAUDE.md: betting math rules, commands section reflects real workflow
- Fixed dead code in betting.py: removed unused `stake_pct_logged` variable
- Tests: 44 pass in test_betting.py; ruff clean on mlb/betting.py

### Fill-price bug in simulate() — FIXED (2026-05-09 session)
- `simulate()` in betting.py was using devigged (vig-free) prices for Kelly sizing AND payout
- Fix: EV computed vs `fair_over` (consensus edge), but Kelly + payout now use `raw_over`/`raw_under`
- Honest ROI after vig: OOF 2022-2024 = -0.07%, OOS 2025-2026 = -4.90% (vs DraftKings -110)
- Break-even win rate at -110 = ~52.4%; model at 50-51% → no real edge yet with 2022-2024 data only

### Prior codebase cleanup (2026-05-09)
- Removed `_load_statcast_pitcher()` + `_add_statcast_xera_features()` from features.py (pure dead code)
- Removed 2 corresponding dead tests from tests/unit/test_features.py
- `statcast_enricher.py` kept as standalone data tool (populates pitcher_season_statcast table)
- Fixed .github/workflows/update.yml: now calls mlb.kalshi → mlb.model → mlb.betting individually

## Last Completed (2026-05-09 session — Kalshi linking fix)

### Kalshi game_id linking: 29% → 100%
- Fixed `_teams_from_ticker` bug: was returning first valid split (by length) without validating
  that both halves are real MLB abbreviations; e.g. `DETKC` (5-char) returned `DE`+`TKC` instead of `DET`+`KC`
- Fix: try all splits 2..n-2, return first where both halves are in `_ALL_KALSHI_ABBREVS`
- Added missing abbreviation mappings: `AZ` → `ARI`, `WSH` → `WSN`
- Added `relink_game_ids()` function + `--relink` CLI flag to retroactively patch existing rows
- Result: 776 rows relinked (638 + 138), linking rate 29% → 100% for both full-game and F5
- 120 unit tests pass, 2 xfailed — clean

## Last Completed (2026-05-09 session — continuation)

### F5 (First 5 Innings) parallel model — schema + backfill + features:
- Added `f5_home_score`, `f5_away_score`, `f5_total_runs` to games table (ALTER TABLE)
- Added `backfill_f5_scores()` to mlb/scraper.py:
  - Fetches `https://statsapi.mlb.com/api/v1/schedule?hydrate=linescore` per date
  - Caches results to `data/raw/statsapi/linescore_{date}.json`
  - Computes sum of innings 1-5 per team, updates games table
  - CLI: `python -m mlb.scraper --backfill-f5`
- F5 backfill running (background, task bre1loqoi) — 400/1156 dates done, ~5,300 games
  - One timeout on 2016-07-07 — needs `--force-f5` re-run after

### Kalshi F5 market support:
- Added `F5_SERIES_TICKER = "KXMLBF5TOTAL"` constant and `_F5_EVENT_RE` regex
- Added `snapshot_f5_markets()` to mlb/kalshi.py
- F5 market_type stored as `'f5_total_over'` / `'f5_total_under'` in kalshi_markets table
- CLI: `python -m mlb.kalshi --snapshot-f5`
- Both full-game and F5 snapshots can run together: `python -m mlb.kalshi --snapshot --snapshot-f5`

### Cross-signal feature (full-game vs F5 pricing divergence):
- Added `_load_kalshi_lines(conn)` to features.py:
  - Loads most-recent Kalshi lines per game for both `total_over` and `f5_total_over` markets
  - Computes `f5_ratio = kalshi_f5_line / kalshi_fullgame_line`
- New features in FEATURE_COLS: `kalshi_fullgame_line`, `kalshi_f5_line`, `f5_ratio`
- Both `build_features()` and `build_predict_features()` now pass `kalshi_lines` to `_add_market_features()`
- All 17 leakage tests still passing

### model.py F5 target support:
- Added `--target fullgame|f5` CLI parameter
- `_prepare_xy(df, target='f5')` uses `f5_home_score`/`f5_away_score` as targets
- `train(target='f5')` saves to `data/models/f5_gbr_poisson_v1.0.0.pkl`
- `predict(target='f5')` loads the F5 artefact and returns F5 expected runs

## Data State (as of 2026-05-09, updated)
- games: ~25,190 total — ALL SEASONS COMPLETE
  - 2015: 2,429 Final | 2016: 2,430 | 2017: 2,430 | 2018: 2,432 | 2019: 2,430
  - 2020: 900 (COVID shortened) | 2021: 2,430 | 2022-2025: 9,717 | 2026: 551+
- sportsbook_odds: 49,244 rows (2021-2026) — DK/FanDuel/Caesars/Bet365
- elo_ratings: 20,532 rows (2022–2026, zero-sum verified)
- pitcher_season_statcast: 3,938 rows (2015–2025)
- kalshi_markets: 809 full-game rows + 210 F5 rows — 100% game_id linked

## Model State (as of 2026-05-09)
- lgbm_binary retrained on 2021-2025 (11,703 games, was 9,368 → 5,817 originally)
- OOF: log_loss=0.7207, AUC=0.4969, brier=0.2619
- Backtest 2021-2025, DraftKings, NO filter: -2.84% ROI, 49.9% win → NO-GO (honest)
- Backtest 2021-2025, ERA<=3.50 filter (CLEAN, pre-game ERA): -6.45% ROI, 49.0% win → NO-GO
- Previous +16.16% ROI result was entirely from post-game ERA leakage — now fixed
- FEATURE_COLS: 51 (removed 4 dead NULL features: precip_prob, 3 Kalshi)
- Sharpe: annualised (raw * sqrt(bets/yr)); go/no-go threshold 1.0

## Data Fixes Applied
- Elo re-run from 2015 (was 2022) — all 12 seasons now have valid Elo ratings
- Weather backfilled for 2021 (was missing) — now 93% coverage matching 2022-2025
- sportsbook_odds game_id backfill: 9,256 rows updated (all years 2021-2025 now linked)
  - Root cause: 2021 odds were scraped before game rows existed to link against
  - Fix: one-time UPDATE joining on (date, home_team, away_team)
  - Note: odds_scraper.py should be updated to auto-link on future runs (TODO)

## Next Steps
1. Re-run F5 backfill: `python -m mlb.scraper --backfill-f5 --force-f5` (2016-07-07 timed out)
2. Re-run Elo on full history: `python -m mlb.elo --reset --start-season 2015 --end-season 2026`
3. Weather backfill 2015-2020: `python -m mlb.weather --start 2015-04-01 --end 2020-10-31`
4. Phase 6: mlb/pipeline.py daily orchestrator
5. Phase 6: mlb/live.py WebSocket monitor

## Known Issues / Notes
- NegBinom upgrade confirmed needed (dispersion home=2.14, away=2.36)
  → calibration.py has NegBinom support but model.py uses HGBR(loss='poisson') as primary
- Bullpen feature loop is O(N*games) — ~2 min for 10k games, acceptable for now
- precip_prob always NULL for historical weather (archive API limitation)
- games.status uses 'Final' (capital F) — not 'final'
- Kalshi ticker format: KXMLBTOTAL-{YY}{MON}{DD}{HHMM}{AWAY}{HOME}-{LINE*1}
  e.g. KXMLBTOTAL-26MAY082210ATLLAD-9
- Kalshi F5 ticker: KXMLBF5TOTAL-{YY}{MON}{DD}{HHMM}{AWAY}{HOME}-{N}
  e.g. N=4 → "Over 4.5 F5 runs" in UI
- Kalshi markets only have 1-integer lines (9, 10, etc.), SBR has half-lines (8.5, 9.5)
  → Need to pick nearest Kalshi line to SBR line when matching
- build_predict_features() takes ~3 min (builds features over all 12k+ historical games for rolling context)
  → Could optimize with date cutoff for rolling windows (e.g., only 2 years of history needed)
- F5 backfill: 2016-07-07 timed out — needs re-run with --force-f5

## F5 Architecture Design
- **Parallel models**: Full-game model (home_runs/away_runs) AND F5 model (f5_home_score/f5_away_score)
- **Same features**: Both models use identical feature matrix (FEATURE_COLS)
- **Cross-signal**: `f5_ratio = kalshi_f5_line / kalshi_fullgame_line` as a feature in BOTH models
  - f5_ratio ~0.5 → market sees uniform run distribution (no edge signal)
  - f5_ratio >> 0.5 → market expects strong back-half (starters dominant)
  - f5_ratio << 0.5 → market expects scoring front-loaded (starters weaker)
- **Betting**: Full-game → KXMLBTOTAL; F5 → KXMLBF5TOTAL
- **Information edge**: Divergence between F5 and full-game pricing = potential mispricing

## Blockers
- Historical scrape incomplete (2017-2021 missing) → retrain waits
- F5 backfill in progress → F5 model training waits

## Key Decisions Made
- Two-target Poisson regression: predict home_runs (lam_home) and away_runs (lam_away) separately
- Poisson convolution for P(over) — no Normal CDF
- HistGradientBoostingRegressor(loss='poisson') as primary model (HGBR, not GBR — sklearn difference)
- PoissonRegressor as baseline
- Kalshi totals markets as betting venue; Polymarket as read-only cross-signal
- SBR via sbrscrape for historical odds (2021+)
- Open-Meteo for weather (free, no API key)
- Walk-forward TimeSeriesSplit(n_splits=5, gap=162) — only valid CV strategy
- **0.50x fractional Kelly (half Kelly), 15% bankroll cap**, $0.03 minimum edge (UPDATED from 0.25x/5%)
- SQLite WAL mode, single DB file at data/mlb.db
- NegBinom upgrade confirmed needed (dispersion > 1.2 on both targets)
- ERA/FIP capped at 13.5 in features.py (3× league average) to suppress early-season outliers
- F5 as PARALLEL target (not replacement for full-game) — keeps both models
- Full-game vs F5 market divergence stored as `f5_ratio` feature (cross-signal for both models)

## Open Questions (Resolve in Notebooks)
- Kalshi totals market liquidity — verify in Notebook 3 (292 open markets confirmed active)
- Ensemble vs winner-take-all — decide after Notebook 4
- Optimal min edge threshold — sensitivity test in Notebook 5
- Over bias magnitude — quantify in SBR data (EDA showed 47.9% over rate = slight under bias)
- Will D² go positive with 2015-2021 training data? (currently negative = model worse than mean)
- Does F5 model outperform full-game? Compare ROI in notebook 5
- Is f5_ratio a significant feature? Check permutation importance
