"""
Database connection and schema management.

Provides a WAL-mode SQLite connection context manager and a one-time
schema creation function. All other modules import ``get_conn`` for DB access.
"""

import logging
import sqlite3
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path

logger = logging.getLogger(__name__)

DB_PATH = "data/mlb.db"

# ── DDL ───────────────────────────────────────────────────────────────────────

_SCHEMA_SQL = """
PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS games (
    game_id         TEXT PRIMARY KEY,
    date            TEXT NOT NULL,
    season          INTEGER NOT NULL,
    home_team       TEXT NOT NULL,
    away_team       TEXT NOT NULL,
    home_score      INTEGER,
    away_score      INTEGER,
    total_runs      INTEGER,
    f5_home_score   INTEGER,
    f5_away_score   INTEGER,
    f5_total_runs   INTEGER,
    venue           TEXT,
    game_time_et    TEXT,
    status          TEXT DEFAULT 'scheduled',
    created_at      TEXT DEFAULT (datetime('now')),
    updated_at      TEXT DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_games_date   ON games(date);
CREATE INDEX IF NOT EXISTS idx_games_season ON games(season);
CREATE INDEX IF NOT EXISTS idx_games_home   ON games(home_team, date);
CREATE INDEX IF NOT EXISTS idx_games_away   ON games(away_team, date);

CREATE TABLE IF NOT EXISTS team_stats (
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

CREATE TABLE IF NOT EXISTS pitchers (
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

CREATE TABLE IF NOT EXISTS weather (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    game_id         TEXT NOT NULL REFERENCES games(game_id),
    snapshot_type   TEXT NOT NULL,
    fetched_at      TEXT NOT NULL,
    temp_f          REAL,
    wind_speed_mph  REAL,
    wind_dir_deg    REAL,
    wind_dir_label  TEXT,
    precip_prob     REAL,
    humidity        REAL,
    is_dome         INTEGER DEFAULT 0,
    UNIQUE(game_id, snapshot_type)
);

CREATE TABLE IF NOT EXISTS stadiums (
    team            TEXT PRIMARY KEY,
    stadium_name    TEXT,
    latitude        REAL,
    longitude       REAL,
    orientation_deg REAL,
    elevation_ft    REAL,
    is_dome         INTEGER DEFAULT 0,
    park_run_factor REAL,
    park_hr_factor  REAL
);

CREATE TABLE IF NOT EXISTS elo_ratings (
    id      INTEGER PRIMARY KEY AUTOINCREMENT,
    team    TEXT NOT NULL,
    date    TEXT NOT NULL,
    elo     REAL NOT NULL,
    season  INTEGER NOT NULL,
    UNIQUE(team, date)
);

CREATE TABLE IF NOT EXISTS sportsbook_odds (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    game_id          TEXT,
    date             TEXT NOT NULL,
    home_team        TEXT NOT NULL,
    away_team        TEXT NOT NULL,
    book             TEXT NOT NULL,
    total_open       REAL,
    total_close      REAL,
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

CREATE TABLE IF NOT EXISTS kalshi_markets (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    game_id         TEXT,
    ticker          TEXT NOT NULL,
    event_ticker    TEXT NOT NULL,
    market_type     TEXT NOT NULL,
    line            REAL,
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

CREATE TABLE IF NOT EXISTS predictions (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    game_id                 TEXT NOT NULL REFERENCES games(game_id),
    model_name              TEXT NOT NULL,
    model_version           TEXT NOT NULL,
    predicted_at            TEXT NOT NULL,
    lambda_home             REAL NOT NULL,
    lambda_away             REAL NOT NULL,
    predicted_total_runs    REAL,
    dispersion_alpha        REAL,
    line                    REAL,
    over_prob               REAL,
    under_prob              REAL,
    kalshi_ticker           TEXT,
    kalshi_mid_price        REAL,
    polymarket_mid_price    REAL,
    edge                    REAL,
    ev                      REAL,
    kelly_fraction          REAL,
    recommended_bet         REAL,
    bet_side                TEXT,
    closing_kalshi_price    REAL,
    closing_total_line      REAL,
    clv                     REAL,
    UNIQUE(game_id, model_name, model_version)
);

CREATE TABLE IF NOT EXISTS scrape_log (
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
"""


# ── Connection ────────────────────────────────────────────────────────────────


@contextmanager
def get_conn(db_path: str = DB_PATH) -> Generator[sqlite3.Connection, None, None]:
    """
    Yield a WAL-mode SQLite connection with row_factory set.

    Commits on clean exit, rolls back on exception.

    Parameters
    ----------
    db_path : str, optional
        Path to the SQLite database file. Defaults to ``data/mlb.db``.

    Yields
    ------
    sqlite3.Connection
        Open connection with WAL mode and ``sqlite3.Row`` row factory.
    """
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# ── Schema ────────────────────────────────────────────────────────────────────


def create_schema(conn: sqlite3.Connection) -> None:
    """
    Apply the full schema DDL to an open connection.

    Idempotent — uses ``CREATE TABLE IF NOT EXISTS`` throughout.
    Safe to call on an existing database.

    Parameters
    ----------
    conn : sqlite3.Connection
        Open SQLite connection (WAL mode recommended).
    """
    conn.executescript(_SCHEMA_SQL)
    logger.info("Schema created / verified")


def init_db(db_path: str = DB_PATH) -> None:
    """
    Ensure the database file and schema exist.

    Creates the ``data/`` directory if needed, then applies the schema.

    Parameters
    ----------
    db_path : str, optional
        Path to the SQLite database file. Defaults to ``data/mlb.db``.
    """
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    with get_conn(db_path) as conn:
        create_schema(conn)
    logger.info("Database initialised at %s", db_path)


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    init_db()
