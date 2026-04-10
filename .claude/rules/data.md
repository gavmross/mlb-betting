# Data Integrity Rules
**Active when:** editing mlb/scraper.py, mlb/odds_scraper.py, mlb/weather.py,
mlb/features.py, mlb/elo.py, or any file in data/

---

## Temporal Integrity — Non-Negotiable

Every rolling feature MUST exclude the current game via `.shift(1)`:

```python
# CORRECT
df['sp_siera_l5'] = (
    df.groupby('pitcher_id')['siera_game']
    .shift(1)                          # exclude today
    .rolling(5, min_periods=2)
    .mean()
)

# WRONG — leaks today's data
df['sp_siera_l5'] = df.groupby('pitcher_id')['siera_game'].rolling(5).mean()
```

Rules:
- Sort by (team/pitcher, date) BEFORE any groupby-rolling operation
- Features must be computable from data strictly available before first pitch
- Pitcher stats for game N must use stats through game N-1 only
- Train/test splits must split by date — never randomly
- After any feature addition: `pytest tests/unit/test_features.py -v`

## Two-Target Structure

Features are built twice — once with the home team as the "focal" team,
once with the away team. The model trains separate instances for home_runs
and away_runs. Features must work correctly for both orientations.

## Database Writes

- All DB writes use INSERT OR IGNORE or INSERT OR REPLACE — never raw INSERT
- Log every scrape run to the scrape_log table with: source, date_range, rows_inserted, status
- Rate limit external API calls:
  - pybaseball: 1 second minimum between calls
  - sbrscrape: 1.5 seconds minimum between date requests
  - Open-Meteo: no strict limit but batch requests where possible
  - Kalshi REST: respect 429 responses with exponential backoff

## Elo System

- Elo must be zero-sum within each season
- Verify after every batch update:
  ```python
  np.testing.assert_allclose(total_elo_before, total_elo_after, rtol=1e-6, atol=1e-4)
  ```
- Apply off-season regression to mean (revert ~30% toward 1500) between seasons
- No inactivity decay during season (teams play every day)

## Raw Data Caching

- Cache all raw API responses to data/raw/ before parsing
- Never re-fetch if a valid cache entry exists for that date
- Cache filenames: {source}_{date}.json (e.g. pybaseball_2024-06-15.json)
