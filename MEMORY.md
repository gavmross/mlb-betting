# Session Memory

## Current Phase
Phase 1 — Data Infrastructure (wrapping up)

## Implementation Progress

```
Step 1   pyproject.toml + requirements.txt        [ ]
Step 2   mlb/db.py — schema + WAL mode            [x]
Step 3   .claude/ infrastructure                  [x] (pre-built)
Step 4   data/stadiums.py                         [x]
Step 5   mlb/scraper.py                           [x] (2022 full; 2023/2024 ~Jul, finishing)
Step 6   mlb/odds_scraper.py                      [x]
Step 7   mlb/weather.py                           [x]
Step 8   mlb/kalshi.py                            [x]
Step 9   mlb/polymarket.py                        [x]
Step 10  Initial data load + validation           [x]
Step 11  mlb/features.py                          [x]
Step 12  mlb/elo.py                               [x]
Step 13  tests/unit/                              [x] (test_features.py 17 pass, test_elo.py 18 pass)
Step 14  notebooks/01_data_audit.ipynb            [x]
Step 15  notebooks/02_feature_eda.ipynb           [x]
Step 16  mlb/model.py                             [x]
Step 17  mlb/calibration.py                       [x]
Step 18  notebooks/03_kalshi_market_analysis.ipynb [ ]
Step 19  notebooks/04_model_comparison.ipynb      [ ]
Step 20  mlb/betting.py                           [ ]
Step 21  notebooks/05_betting_simulation.ipynb    [ ]
Step 22  mlb/pipeline.py                          [ ]
Step 23  mlb/live.py                              [ ]
Step 24  .github/workflows/update.yml             [ ]
Step 25  Integration tests + README               [ ]
```

## Last Completed
Step 17 — mlb/calibration.py + tests/unit/test_calibration.py (23 tests pass).
Step 16 — mlb/model.py + tests/unit/test_model.py (20 tests pass).
Note: HistGradientBoostingRegressor(loss='poisson') used instead of GBR — loss='poisson' is HGBR-only in sklearn.
Step 15 — notebooks/02_feature_eda.ipynb executed.
Key findings: NegBinom confirmed (dispersion 2.1–2.4); market line r=0.189 is ceiling;
park factors top non-market signal (r=0.13); SP ERA weak (r=0.028); 47.9% over rate (slight under bias).
- games: 7,290 (2022: 2430, 2023: 2430, 2024: 2430 — all complete)
- team_stats: 14,444 | pitchers: 61,591 | odds: 10,393 | weather: ~1,006 (backfilling)
- Elo: 14,574 rows recomputed on full data, zero-sum holds

## Known Issues / Notes
- 2023 scraper: data through 2023-07-09, finishing rest (2023-07-10 → 2023-10-01) in background
- 2024 scraper: data through 2024-07-06, finishing rest (2024-07-07 → 2024-10-01) in background
- After scrapers finish: re-run `python -m mlb.elo --reset` and `python -m mlb.weather --incremental`
- Also run odds backfill for 2023-10-01+ and 2024-04-02+
- precip_prob always NULL for historical weather (archive API limitation)
- Bullpen feature loop is slow O(N*games) — acceptable for now, optimize later
- OVERDISPERSION CONFIRMED: home_score dispersion=2.14, away_score=2.36 (both >> 1.2)
  → NegBinom upgrade will be needed in calibration.py
- Park factor correlation with actual scoring: r=0.708 (strong signal)
- Shell CWD hook fix: stub at notebooks/.claude/hooks/pre_tool_use.py (exits 0) — keep it

## In Progress
- 2023 + 2024 game scraper (Jul → Oct) running in background task bawct6ggb

## Blockers
- None

## Key Decisions Made
- Two-target Poisson regression: predict home_runs (λ_home) and away_runs (λ_away) separately
- Poisson convolution for P(over) — no Normal CDF
- GradientBoostingRegressor(loss='poisson') as primary model
- PoissonRegressor as baseline
- Kalshi totals markets as betting venue; Polymarket as read-only cross-signal
- SBR via sbrscrape for historical odds (2021+)
- Open-Meteo for weather (free, no API key)
- Walk-forward TimeSeriesSplit(n_splits=5, gap=162) — only valid CV strategy
- 0.25x fractional Kelly, 5% bankroll cap, $0.03 minimum edge
- SQLite WAL mode, single DB file at data/mlb.db
- NegBinom upgrade confirmed needed (dispersion > 1.2 on both targets)

## Open Questions (Resolve in Notebooks)
- Kalshi totals market liquidity — verify in Notebook 3
- Ensemble vs winner-take-all — decide after Notebook 4
- Optimal min edge threshold — sensitivity test in Notebook 5
- Over bias magnitude — quantify in SBR data
