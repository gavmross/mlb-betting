---
name: run-pipeline
description: >
  Run the full MLB daily pipeline: scrape new data, fetch weather, engineer
  features, generate predictions, price against Kalshi, output bet
  recommendations. Use when the user says "run pipeline", "update data",
  "generate predictions", or "what should I bet today".
---

# Run Full MLB Pipeline

Execute these steps in order. Stop and report if any step fails.

## Step 1 — DB Health Check
```bash
sqlite3 data/mlb.db "SELECT name, COUNT(*) as rows FROM (
  SELECT 'games' as name, COUNT(*) as cnt FROM games
  UNION ALL SELECT 'predictions', COUNT(*) FROM predictions
  UNION ALL SELECT 'weather', COUNT(*) FROM weather
  UNION ALL SELECT 'sportsbook_odds', COUNT(*) FROM sportsbook_odds
) GROUP BY name"
```
Expected: all tables exist and have rows. If any table is empty, stop and alert.

## Step 2 — Incremental Game Scrape
```bash
python -m mlb.scraper --incremental
```
Expected output: "Scraped N new games for YYYY-MM-DD". If N=0 and today is a
game day, investigate. If the season hasn't started or it's off-season, N=0 is normal.

## Step 3 — Fetch Weather for Today's Games
```bash
python -m mlb.weather --incremental
```
Fetches Open-Meteo forecast for today's scheduled games. Expected: weather row
per scheduled game.

## Step 4 — Update Odds
```bash
python -m mlb.odds_scraper --date today
```
Pulls today's lines from SBR. Expected: odds rows for today's games.

## Step 5 — Engineer Features
```bash
python -m mlb.features --date today
```
After completion, verify no leakage:
```bash
pytest tests/unit/test_features.py -v -q
```
If any leakage test fails — STOP. Do not proceed to model prediction.

## Step 6 — Generate Predictions
```bash
python -m mlb.model --predict --date today
```
Expected: lambda_home, lambda_away, over_prob for each game written to predictions table.

## Step 7 — Fetch Kalshi & Polymarket Prices
```bash
python -m mlb.kalshi --snapshot --date today
python -m mlb.polymarket --snapshot --date today
```
Expected: mid-prices for all open MLB totals markets.

## Step 8 — Compute EV and Bet Recommendations
```bash
python -m mlb.betting --date today
```
Expected: bet_side, edge, ev, recommended_bet written to predictions table.

## Step 9 — Output Summary
Query and print today's bet recommendations:
```bash
sqlite3 data/mlb.db "
SELECT g.home_team, g.away_team, g.game_time_et,
       p.predicted_total_runs, p.line, p.over_prob,
       p.kalshi_mid_price, p.edge, p.bet_side, p.recommended_bet
FROM predictions p
JOIN games g ON p.game_id = g.game_id
WHERE g.date = date('now')
  AND p.bet_side != 'PASS'
ORDER BY p.edge DESC"
```

Report: N games today, M bets recommended, top edge = X.
