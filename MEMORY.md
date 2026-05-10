# Session Memory

## Current Phase
Phase 5 complete. Phase 6 (Live Pipeline) pending historical data backfill.

## Implementation Progress

```
Step 1   pyproject.toml + requirements.txt        [x]
Step 2   mlb/db.py — schema + WAL mode            [x]
Step 3   .claude/ infrastructure                  [x]
Step 4   data/stadiums.py                         [x]
Step 5   mlb/scraper.py                           [x] (2015-2017 in progress, 2022-2026 done)
Step 6   mlb/odds_scraper.py                      [x] (49,244 rows 2021-2026)
Step 7   mlb/weather.py                           [x]
Step 8   mlb/kalshi.py                            [x] (F5TOTAL snapshot added)
Step 9   mlb/polymarket.py                        [x]
Step 10  Initial data load + validation           [x]
Step 11  mlb/features.py                          [x] (53 features; statcast dead code removed)
Step 12  mlb/elo.py                               [x]
Step 13  tests/unit/                              [x] (20 pass, 2 xfailed)
Step 14  notebooks/01_data_audit.ipynb            [x]
Step 15  notebooks/02_feature_eda.ipynb           [x]
Step 16  mlb/model.py                             [x] (glm_poisson, hgbr_poisson, lgbm_binary; --target fullgame|f5)
Step 17  mlb/calibration.py                       [x]
Step 18  notebooks/03_kalshi_market_analysis.ipynb [~] in progress
Step 19  notebooks/04_model_comparison.ipynb      [~] in progress
Step 20  mlb/betting.py                           [x] (fill-price bug fixed; vig-inclusive payout)
Step 21  notebooks/05_betting_simulation.ipynb    [~] in progress
Step 22  mlb/pipeline.py                         [ ] Phase 6
Step 23  mlb/live.py                              [ ] Phase 6
Step 24  .github/workflows/update.yml             [x] (fixed — was calling non-existent pipeline.py)
Step 25  Integration tests + README               [ ]
```

## Last Completed (2026-05-09 session — housecleaning + fill-price fix)

### Fill-price bug in simulate() — FIXED
- `simulate()` in betting.py was using devigged (vig-free) prices for Kelly sizing AND payout
- Fix: EV computed vs `fair_over` (consensus edge), but Kelly + payout now use `raw_over`/`raw_under`
- Honest ROI after vig: OOF 2022-2024 = -0.07%, OOS 2025-2026 = -4.90% (vs DraftKings -110)
- Break-even win rate at -110 = ~52.4%; model at 50-51% → no real edge yet with 2022-2024 data only

### Codebase housecleaning
- Removed `_load_statcast_pitcher()` + `_add_statcast_xera_features()` from features.py
  (computed prior-season xERA/est_wOBA/luck columns but none were in FEATURE_COLS → pure dead code)
- Removed 2 corresponding dead tests from tests/unit/test_features.py (test_statcast_xera_*)
- `statcast_enricher.py` kept as standalone data tool (populates pitcher_season_statcast table)
- Updated docs/ARCHITECTURE.md: feature list now matches actual FEATURE_COLS (53 features),
  model section updated (GBR → HGBR, added lgbm_binary), data sources table corrected (no Pinnacle)
- Updated CLAUDE.md: commands section now shows real daily workflow; phase status accurate
- Fixed .github/workflows/update.yml: was calling mlb.pipeline (doesn't exist); now calls
  mlb.kalshi → mlb.model → mlb.betting individually
- Unit tests: 20 pass, 2 xfailed (cross-season cold-start) — clean

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

## Data State (as of 2026-05-09)
- games: ~15,800 total
  - 2015: 2,429 Final (COMPLETE)
  - 2016: 2,430 Final (COMPLETE)
  - 2017: ~791 Final (in progress ~33%)
  - 2018-2021: 0 (not yet scraped)
  - 2022-2025: 9,717 Final (COMPLETE)
  - 2026: 551 Final + ~15 Preview
  - f5_total_runs: ~5,300+ filled (backfill running)
- team_stats: ~30,000+ rows
- pitchers: ~85,000+ rows
- sportsbook_odds: 49,244 rows (2021-2026, near-complete)
- weather: backfilled through available history
- elo_ratings: 20,532 rows (2022–2026, zero-sum verified)
- pitcher_season_statcast: 3,938 rows (2015–2025)
- kalshi_markets: 809 full-game rows + 210 F5 rows — 100% game_id linked (fixed)

## After Historical Scrape Completes (TODO)
1. Re-run F5 backfill for skipped dates: `python -m mlb.scraper --backfill-f5 --force-f5`
   - Known skipped: 2016-07-07 (timeout)
2. Re-run Elo: `python -m mlb.elo --reset --start-season 2015 --end-season 2026`
3. Weather backfill: `python -m mlb.weather --start 2015-04-01 --end 2021-10-31`
4. Retrain full-game model on 2015-2025: `python -m mlb.model --train --start 2015-04-01 --end 2025-10-01`
5. Train F5 model on 2015-2025: `python -m mlb.model --train --target f5 --start 2015-04-01 --end 2025-10-01`
6. Run notebook 04 for model comparison with larger training set
7. Run F5 backtest to compare F5 ROI vs full-game ROI

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
- 0.25x fractional Kelly, 5% bankroll cap, $0.03 minimum edge
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
