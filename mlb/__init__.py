"""
MLB Total Runs Prediction & Betting System

Modules
-------
db               : SQLite connection, schema creation, WAL mode
scraper          : pybaseball + MLB Stats API data ingestion
odds_scraper     : SBR odds via sbrscrape
weather          : Open-Meteo weather fetcher + wind direction encoder
kalshi           : Kalshi REST + WebSocket client
polymarket       : Polymarket Gamma API read-only price fetcher
features         : Feature engineering pipeline (no-leakage)
elo              : Team Elo rating system
model            : PoissonRegressor + HGBR(poisson) + LightGBM binary + walk-forward CV
calibration      : Poisson convolution, overdispersion check, NegBinom upgrade path
betting          : EV, Kelly criterion, CLV, structural filters, simulate_structural
statcast_enricher: Pitcher season Statcast enrichment (xERA, est_wOBA, luck delta)
"""
