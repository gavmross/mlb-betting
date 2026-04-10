"""
MLB Total Runs Prediction & Betting System

Modules
-------
db          : SQLite connection, schema creation, WAL mode
scraper     : pybaseball + MLB Stats API data ingestion
odds_scraper: SBR odds via sbrscrape
weather     : Open-Meteo weather fetcher + wind direction encoder
kalshi      : Kalshi REST + WebSocket client
polymarket  : Polymarket Gamma API read-only price fetcher
features    : Feature engineering pipeline (no-leakage)
elo         : Team Elo rating system
model       : PoissonRegressor + GBR(poisson) + walk-forward CV
calibration : Poisson convolution, overdispersion check
betting     : EV, Kelly criterion, CLV, cross-market signal
pipeline    : Daily orchestrator
live        : Real-time Kalshi WebSocket monitor
"""
