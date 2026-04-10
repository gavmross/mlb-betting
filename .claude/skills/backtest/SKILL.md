---
name: backtest
description: >
  Run walk-forward backtest and betting simulation on historical data.
  Use when the user says "backtest", "evaluate the model", "simulate
  betting", "what's the ROI", or "does the model have edge".
---

# Walk-Forward Backtest & Betting Simulation

## Step 1 — Confirm Data Coverage
```bash
sqlite3 data/mlb.db "
SELECT season, COUNT(*) as games,
       SUM(CASE WHEN total_runs IS NOT NULL THEN 1 ELSE 0 END) as with_result
FROM games
GROUP BY season ORDER BY season"
```
Confirm 2015–2024 games present with results. Note seasons with sparse data.

```bash
sqlite3 data/mlb.db "
SELECT MIN(date), MAX(date), COUNT(*) FROM sportsbook_odds"
```
Confirm SBR odds coverage. Betting simulation only runs for dates with odds.

## Step 2 — Run Walk-Forward CV
```bash
python -m mlb.model --backtest --n-splits 5
```

This trains on all data before each fold and evaluates on the held-out fold.
Expected output per fold:
```
Fold 1: train 2015-2019 | test 2020 | deviance=X.XX | D²=X.XX | log_loss=X.XXX
Fold 2: train 2015-2020 | test 2021 | ...
...
Fold 5: train 2015-2023 | test 2024 | ...
```

## Step 3 — Overdispersion Check
After the backtest, print the overdispersion ratio per fold:
```bash
python -m mlb.calibration --overdispersion-check
```
If mean ratio > 1.2 — recommend NegBinom upgrade and add to open questions list.

## Step 4 — Betting Simulation
```bash
python -m mlb.betting --simulate \
  --start 2021-04-01 \
  --end 2024-10-01 \
  --min-edge 0.03 \
  --kelly-mult 0.25 \
  --initial-bankroll 1000
```

Expected output:
```
Betting Simulation: 2021-2024 vs SBR Pinnacle closing line
Games evaluated:    N
Bets placed:        M  (X.X% of games)
Win rate:           XX.X%
Total ROI:          +X.X%
Max drawdown:       -X.X%
Sharpe ratio:       X.XX
Average CLV:        +$0.0XX
```

## Step 5 — Sensitivity Analysis
Test different edge thresholds to find the optimal filter:
```bash
python -m mlb.betting --simulate --min-edge 0.02 --output sensitivity_002.csv
python -m mlb.betting --simulate --min-edge 0.03 --output sensitivity_003.csv
python -m mlb.betting --simulate --min-edge 0.05 --output sensitivity_005.csv
```

## Step 6 — Report

Summarise findings in a table:
```
Edge Threshold | Bets | Win Rate | ROI   | Sharpe | Max DD
$0.02          | XXX  | XX.X%    | +X.X% | X.XX   | -X.X%
$0.03          | XXX  | XX.X%    | +X.X% | X.XX   | -X.X%
$0.05          | XXX  | XX.X%    | +X.X% | X.XX   | -X.X%
```

Decision rule: "Model is ready for paper trading" only if:
- ROI > 0% at $0.03 threshold
- Average CLV is positive
- Sharpe ratio > 0.5

If criteria not met — document which open research questions remain and
what additional work is needed before live trading.
