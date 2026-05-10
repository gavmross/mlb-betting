"""
Kalshi MLB totals market client.

Snapshots open markets and backfills historical settled markets for the KXMLB
series into the ``kalshi_markets`` table.

Authentication
--------------
Set env var ``KALSHI_KEY_ID`` and place RSA private key at
``~/.kalshi/private_key.pem``.

Writes to
---------
- ``kalshi_markets`` — one row per (ticker, snapshot_ts)
- ``scrape_log``     — one row per run

Usage
-----
Snapshot today's open markets::

    python -m mlb.kalshi --snapshot

Backfill historical::

    python -m mlb.kalshi --start 2025-04-01 --end 2025-10-01
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import time as _time
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import requests
from pykalshi import KalshiClient
from pykalshi.enums import MarketStatus
from pykalshi.exceptions import AuthenticationError, KalshiAPIError

from mlb.db import get_conn

logger = logging.getLogger(__name__)

SERIES_TICKER = "KXMLBTOTAL"
F5_SERIES_TICKER = "KXMLBF5TOTAL"

# Historical API base — no authentication required
HIST_BASE = "https://external-api.kalshi.com/trade-api/v2"
# MLB season runs April–October; approximate all games as EDT (UTC-4)
_EDT_OFFSET_H = 4

# Month abbreviation → zero-padded month number (Kalshi uses APR, MAY, etc.)
_MONTH_MAP: dict[str, str] = {
    "JAN": "01",
    "FEB": "02",
    "MAR": "03",
    "APR": "04",
    "MAY": "05",
    "JUN": "06",
    "JUL": "07",
    "AUG": "08",
    "SEP": "09",
    "OCT": "10",
    "NOV": "11",
    "DEC": "12",
}

# Regex to match Kalshi MLB totals event tickers.
# Live format (with time):       KXMLBTOTAL-26APR102210AZPHI
# Historical format (no time):   KXMLBTOTAL-25OCT17MILLAD
# The HHMM group is made optional with (?:\d{4})? so both formats match.
_EVENT_RE = re.compile(
    r"KXMLBTOTAL-(\d{2})([A-Z]{3})(\d{2})(?:\d{4})?([A-Z]+)",
    re.IGNORECASE,
)

# Regex to match Kalshi F5 total runs event tickers.
# Same dual-format support as _EVENT_RE.
_F5_EVENT_RE = re.compile(
    r"KXMLBF5TOTAL-(\d{2})([A-Z]{3})(\d{2})(?:\d{4})?([A-Z]+)",
    re.IGNORECASE,
)

# Regex to extract line from ticker, e.g. KXMLBTOTAL-26APR101840AZPHI-9 → 9.0
# YES side = P(total >= line), equivalent to P(over line-0.5) in standard betting.
_LINE_RE = re.compile(r"-(\d+)$", re.IGNORECASE)

# Regex to parse "Away vs Home Total Runs?" from market title
_TITLE_TEAMS_RE = re.compile(r"^([A-Za-z ]+) vs ([A-Za-z ]+) Total", re.IGNORECASE)

# Regex to detect UNDER side in ticker/title (YES = over in new format, kept for safety)
_UNDER_RE = re.compile(r"under|UNDER|_U_|-U-", re.IGNORECASE)

# Regex to extract the combined team code from event ticker.
# e.g. KXMLBTOTAL-26MAY091610HOUCIN  → group(1)='HOUCIN'
# e.g. KXMLBF5TOTAL-26MAY091610HOUCIN → group(1)='HOUCIN'
_TEAMS_FROM_TICKER_RE = re.compile(
    r"KXMLB(?:F5)?TOTAL-\d{2}[A-Z]{3}\d{2}(?:\d{4})?([A-Z]+)",
    re.IGNORECASE,
)

# Kalshi 2-3 char abbreviation → DB abbreviation (where different)
# Missing entries = same abbreviation used in both systems
_KALSHI_TO_DB: dict[str, str] = {
    "ATH": "OAK",  # Sacramento Athletics (née Oakland) → our DB still uses OAK through 2024
    "AZ": "ARI",   # Arizona Diamondbacks — Kalshi uses 'AZ', DB uses 'ARI'
    "WAS": "WSN",
    "WSH": "WSN",  # Washington Nationals — Kalshi uses both 'WAS' and 'WSH'
    "KC": "KCR",
    "TB": "TBR",
    "SF": "SFG",
    "SD": "SDP",
    "CWS": "CHW",
}

# All valid Kalshi team abbreviations (both native and DB forms).
# Used to validate ticker splits — only accept a split where both halves are known.
# Must stay in sync with _KALSHI_TO_DB keys + all DB team abbreviations.
_ALL_KALSHI_ABBREVS: frozenset[str] = frozenset(_KALSHI_TO_DB.keys()) | frozenset(
    [
        "ARI", "ATL", "BAL", "BOS", "CHC", "CHW", "CIN", "CLE", "COL",
        "DET", "HOU", "KCR", "LAA", "LAD", "MIA", "MIL", "MIN", "NYM",
        "NYY", "OAK", "PHI", "PIT", "SDP", "SEA", "SFG", "STL", "TBR",
        "TEX", "TOR", "WSN",
    ]
)
# Note: _KALSHI_TO_DB keys (AZ, ARI, WAS, WSH, KC, TB, SF, SD, CWS, ATH) are
# automatically included via frozenset union above.


# ── Auth ──────────────────────────────────────────────────────────────────────


def _build_client() -> KalshiClient:
    """
    Instantiate KalshiClient from env var + key file.

    Returns
    -------
    KalshiClient
        Authenticated client.

    Raises
    ------
    KeyError
        If ``KALSHI_KEY_ID`` is not set.
    FileNotFoundError
        If private key file does not exist.
    """
    key_id = os.environ["KALSHI_KEY_ID"]
    candidates = [
        Path(os.environ.get("KALSHI_KEY_PATH", "")) if os.environ.get("KALSHI_KEY_PATH") else None,
        Path(".kalshi/private-key.pem"),
        Path(".kalshi/private_key.pem"),
        Path("~/.kalshi/private_key.pem").expanduser(),
        Path("~/.kalshi/private-key.pem").expanduser(),
    ]
    key_path = next((p for p in candidates if p and p.exists()), None)
    if key_path is None:
        raise FileNotFoundError(
            "Kalshi private key not found. Checked: .kalshi/private-key.pem, "
            "~/.kalshi/private_key.pem. Set KALSHI_KEY_PATH env var to override."
        )
    return KalshiClient(api_key_id=key_id, private_key_path=str(key_path))


# ── Parsing helpers ───────────────────────────────────────────────────────────


def _safe_dollars(val: str | None) -> float | None:
    """Convert Kalshi dollar string ('0.47') to float, or None."""
    if val is None:
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def _safe_volume(val: str | None) -> float | None:
    """Convert Kalshi fixed-point volume string to float, or None."""
    return _safe_dollars(val)


def _parse_event_ticker(event_ticker: str) -> dict | None:
    """
    Extract game date from a Kalshi MLB totals or F5 totals event ticker.

    Parameters
    ----------
    event_ticker : str
        e.g. ``'KXMLBTOTAL-26APR101840AZPHI'`` or
        ``'KXMLBF5TOTAL-26APR101840AZPHI'``

    Returns
    -------
    dict or None
        Keys: ``date`` (YYYY-MM-DD), ``is_f5`` (bool).
        None if pattern does not match either series.
    """
    m = _F5_EVENT_RE.match(event_ticker)
    is_f5 = True
    if not m:
        m = _EVENT_RE.match(event_ticker)
        is_f5 = False
    if not m:
        return None
    yy, mon, dd = m.group(1), m.group(2).upper(), m.group(3)
    month = _MONTH_MAP.get(mon)
    if not month:
        return None
    game_date = f"20{yy}-{month}-{dd}"
    return {"date": game_date, "is_f5": is_f5}


def _parse_line(ticker: str) -> float | None:
    """
    Extract the total line from a Kalshi totals ticker.

    e.g. ``'KXMLBTOTAL-26APR101840AZPHI-9'`` → ``9.0``

    The YES side represents P(total >= line), equivalent to
    P(over line-0.5) in standard half-run betting.

    Parameters
    ----------
    ticker : str
        Full Kalshi market ticker.

    Returns
    -------
    float or None
    """
    m = _LINE_RE.search(ticker)
    if not m:
        return None
    return float(m.group(1))


def _parse_market_type(
    ticker: str, title: str | None, subtitle: str | None, is_f5: bool = False
) -> str:
    """
    Determine market type string.

    Returns ``'f5_total_over'`` / ``'f5_total_under'`` for F5 markets,
    ``'total_over'`` / ``'total_under'`` for full-game markets.

    Parameters
    ----------
    ticker : str
    title : str or None
    subtitle : str or None
    is_f5 : bool
        True if this is a First 5 Innings market.

    Returns
    -------
    str
    """
    text = " ".join(filter(None, [ticker, title, subtitle]))
    prefix = "f5_" if is_f5 else ""
    if _UNDER_RE.search(text):
        return f"{prefix}total_under"
    return f"{prefix}total_over"


def _mid_price(bid: float | None, ask: float | None) -> float | None:
    """Compute mid-price from bid and ask. Returns single side if one is None."""
    if bid is not None and ask is not None:
        return (bid + ask) / 2.0
    return bid if bid is not None else ask


# ── Market → DB row ───────────────────────────────────────────────────────────


def _market_to_row(market, snapshot_ts: str) -> dict | None:
    """
    Convert a pykalshi Market object to a ``kalshi_markets`` DB row.

    Parameters
    ----------
    market : pykalshi._sync.markets.Market
    snapshot_ts : str
        ISO 8601 UTC timestamp for this snapshot.

    Returns
    -------
    dict or None
        Row dict ready for insertion, or None if parsing fails.
    """
    event_info = _parse_event_ticker(market.event_ticker or "")
    if not event_info:
        logger.debug("Could not parse event ticker: %s", market.event_ticker)
        return None

    line = _parse_line(market.ticker)
    market_type = _parse_market_type(
        market.ticker,
        getattr(market, "title", None),
        getattr(market, "subtitle", None),
        is_f5=event_info.get("is_f5", False),
    )

    bid = _safe_dollars(market.yes_bid_dollars)
    ask = _safe_dollars(market.yes_ask_dollars)

    return {
        "game_id": None,  # resolved separately
        "ticker": market.ticker,
        "event_ticker": market.event_ticker,
        "market_type": market_type,
        "line": line,
        "date": event_info["date"],
        "snapshot_ts": snapshot_ts,
        "yes_bid": bid,
        "yes_ask": ask,
        "mid_price": _mid_price(bid, ask),
        "volume": _safe_volume(getattr(market, "volume_fp", None)),
        "open_interest": _safe_volume(getattr(market, "open_interest_fp", None)),
        "status": str(market.status.value) if market.status else None,
        "result": getattr(market, "result", None),
        "_title": getattr(market, "title", None),  # temp key for game_id linking
    }


# ── DB helpers ────────────────────────────────────────────────────────────────


def _insert_row(conn, row: dict) -> bool:
    """
    Insert a single kalshi_markets row.

    Parameters
    ----------
    conn : sqlite3.Connection
    row : dict
        Row dict from ``_market_to_row`` (``_title`` key removed before insert).

    Returns
    -------
    bool
        True if inserted (not ignored as duplicate).
    """
    # Strip temp key used for game_id linking
    insert_row = {k: v for k, v in row.items() if k != "_title"}

    cursor = conn.execute(
        """INSERT OR IGNORE INTO kalshi_markets
           (game_id, ticker, event_ticker, market_type, line,
            date, snapshot_ts, yes_bid, yes_ask, mid_price,
            volume, open_interest, status, result)
           VALUES (:game_id, :ticker, :event_ticker, :market_type, :line,
                   :date, :snapshot_ts, :yes_bid, :yes_ask, :mid_price,
                   :volume, :open_interest, :status, :result)""",
        insert_row,
    )
    return cursor.rowcount > 0


def _teams_from_ticker(ticker: str) -> tuple[str, str] | None:
    """
    Extract (away_abbr, home_abbr) from a Kalshi ticker by splitting the team code.

    The combined team code (e.g. 'HOUCIN') is split at the midpoint.
    Handles 3+3, 2+3, and 3+2 char splits by trying midpoints.

    Parameters
    ----------
    ticker : str

    Returns
    -------
    tuple[str, str] or None
        (away, home) abbreviations mapped to DB conventions, or None.
    """
    m = _TEAMS_FROM_TICKER_RE.search(ticker)
    if not m:
        return None
    code = m.group(1).upper()
    n = len(code)
    # Try all valid splits; prefer midpoint, then midpoint±1, then rest.
    # Only accept splits where both halves are recognised MLB abbreviations.
    midpoint = n // 2
    candidates = [midpoint, midpoint + 1, midpoint - 1] + list(range(2, n - 1))
    seen: set[int] = set()
    for split in candidates:
        if split in seen or not (1 < split < n - 1):
            continue
        seen.add(split)
        away_raw = code[:split]
        home_raw = code[split:]
        if away_raw in _ALL_KALSHI_ABBREVS and home_raw in _ALL_KALSHI_ABBREVS:
            away = _KALSHI_TO_DB.get(away_raw, away_raw)
            home = _KALSHI_TO_DB.get(home_raw, home_raw)
            return away, home
    return None


def _resolve_game_id(conn, game_date: str, title: str | None, ticker: str = "") -> str | None:
    """
    Look up game_id by date. Tries ticker-based team extraction first,
    falls back to title keyword matching.

    Parameters
    ----------
    conn : sqlite3.Connection
    game_date : str
        YYYY-MM-DD
    title : str or None
        Kalshi market title, e.g. ``'Arizona vs Philadelphia Total Runs?'``
    ticker : str
        Full Kalshi ticker for direct team extraction.

    Returns
    -------
    str or None
    """
    # Primary: parse teams directly from ticker
    if ticker:
        teams = _teams_from_ticker(ticker)
        if teams:
            away_abbr, home_abbr = teams
            row = conn.execute(
                """SELECT game_id FROM games
                   WHERE date = ? AND home_team = ? AND away_team = ?
                   LIMIT 1""",
                (game_date, home_abbr, away_abbr),
            ).fetchone()
            if row:
                return row["game_id"]

    # Fallback: fuzzy title matching via stadiums table
    if not title:
        return None
    m = _TITLE_TEAMS_RE.match(title)
    if not m:
        return None
    away_name = m.group(1).strip()
    home_name = m.group(2).strip()
    row = conn.execute(
        """SELECT g.game_id FROM games g
           JOIN stadiums sh ON sh.team = g.home_team
           JOIN stadiums sa ON sa.team = g.away_team
           WHERE g.date = ?
             AND (LOWER(sh.stadium_name) LIKE '%' || LOWER(?) || '%'
                  OR LOWER(?) LIKE '%' || LOWER(g.home_team) || '%')
             AND (LOWER(sa.stadium_name) LIKE '%' || LOWER(?) || '%'
                  OR LOWER(?) LIKE '%' || LOWER(g.away_team) || '%')
           LIMIT 1""",
        (game_date, home_name, home_name, away_name, away_name),
    ).fetchone()
    return row["game_id"] if row else None


def relink_game_ids(db_path: str = "data/mlb.db") -> int:
    """
    Retroactively resolve game_id for kalshi_markets rows where it is NULL.

    Useful after improving ``_teams_from_ticker`` or adding new abbreviation
    mappings.  Iterates unlinked rows, re-runs resolution, and patches the DB.

    Parameters
    ----------
    db_path : str

    Returns
    -------
    int
        Number of rows updated.
    """
    updated = 0
    with get_conn(db_path) as conn:
        rows = conn.execute(
            "SELECT id, ticker, date FROM kalshi_markets WHERE game_id IS NULL"
        ).fetchall()
        for row in rows:
            game_id = _resolve_game_id(
                conn, row["date"], title=None, ticker=row["ticker"]
            )
            if game_id:
                conn.execute(
                    "UPDATE kalshi_markets SET game_id = ? WHERE id = ?",
                    (game_id, row["id"]),
                )
                updated += 1
    logger.info("relink_game_ids: patched %d rows", updated)
    return updated


# ── Snapshot ──────────────────────────────────────────────────────────────────


def snapshot_open_markets(db_path: str = "data/mlb.db") -> int:
    """
    Fetch all open KXMLB markets and snapshot prices into ``kalshi_markets``.

    Parameters
    ----------
    db_path : str
        Path to SQLite database.

    Returns
    -------
    int
        Number of rows inserted.
    """
    try:
        client = _build_client()
    except (KeyError, FileNotFoundError) as exc:
        logger.error("Kalshi auth failed: %s", exc)
        return 0

    snapshot_ts = datetime.now(tz=UTC).isoformat()

    try:
        markets = client.get_markets(
            series_ticker=SERIES_TICKER,
            status=MarketStatus.OPEN,
            fetch_all=True,
        )
    except (KalshiAPIError, AuthenticationError) as exc:
        logger.error("Kalshi API error fetching open markets: %s", exc)
        return 0

    logger.info("Kalshi snapshot: %d open KXMLB markets", len(markets))

    rows = []
    for market in markets:
        row = _market_to_row(market, snapshot_ts)
        if row:
            rows.append(row)

    inserted = 0
    with get_conn(db_path) as conn:
        for row in rows:
            # Best-effort game_id resolution
            game_id = _resolve_game_id(
                conn, row["date"], row["_title"], ticker=row.get("ticker", "")
            )
            row["game_id"] = game_id
            if _insert_row(conn, row):
                inserted += 1

        conn.execute(
            """INSERT INTO scrape_log
               (source, rows_inserted, status)
               VALUES (?,?,?)""",
            ("kalshi_snapshot", inserted, "success"),
        )

    logger.info("Kalshi snapshot: inserted %d rows", inserted)
    return inserted


def snapshot_f5_markets(db_path: str = "data/mlb.db") -> int:
    """
    Fetch all open KXMLBF5TOTAL markets and snapshot prices into ``kalshi_markets``.

    F5 markets use ``market_type='f5_total_over'`` / ``'f5_total_under'``.

    Parameters
    ----------
    db_path : str
        Path to SQLite database.

    Returns
    -------
    int
        Number of rows inserted.
    """
    try:
        client = _build_client()
    except (KeyError, FileNotFoundError) as exc:
        logger.error("Kalshi auth failed: %s", exc)
        return 0

    snapshot_ts = datetime.now(tz=UTC).isoformat()

    try:
        markets = client.get_markets(
            series_ticker=F5_SERIES_TICKER,
            status=MarketStatus.OPEN,
            fetch_all=True,
        )
    except (KalshiAPIError, AuthenticationError) as exc:
        logger.error("Kalshi API error fetching F5 open markets: %s", exc)
        return 0

    logger.info("Kalshi F5 snapshot: %d open markets", len(markets))

    rows = []
    for market in markets:
        row = _market_to_row(market, snapshot_ts)
        if row:
            rows.append(row)

    inserted = 0
    with get_conn(db_path) as conn:
        for row in rows:
            game_id = _resolve_game_id(
                conn, row["date"], row["_title"], ticker=row.get("ticker", "")
            )
            row["game_id"] = game_id
            if _insert_row(conn, row):
                inserted += 1

        conn.execute(
            """INSERT INTO scrape_log
               (source, rows_inserted, status)
               VALUES (?,?,?)""",
            ("kalshi_f5_snapshot", inserted, "success"),
        )

    logger.info("Kalshi F5 snapshot: inserted %d rows", inserted)
    return inserted


# ── Historical fetch ──────────────────────────────────────────────────────────


def fetch_markets_for_date(
    date_str: str,
    db_path: str = "data/mlb.db",
    include_f5: bool = True,
) -> int:
    """
    Fetch all settled KXMLB (and optionally KXMLBF5TOTAL) markets for a date.

    Pulls all markets (no status filter) for both series and filters to those
    whose parsed date matches ``date_str``. Retrieves final prices and results
    for backtesting.

    Parameters
    ----------
    date_str : str
        Date in ``YYYY-MM-DD`` format.
    db_path : str
        Path to SQLite database.
    include_f5 : bool
        If True (default), also fetch F5 (KXMLBF5TOTAL) markets.

    Returns
    -------
    int
        Number of rows inserted.
    """
    try:
        client = _build_client()
    except (KeyError, FileNotFoundError) as exc:
        logger.error("Kalshi auth failed: %s", exc)
        return 0

    snapshot_ts = datetime.now(tz=UTC).isoformat()
    all_markets = []

    series_to_fetch = [SERIES_TICKER]
    if include_f5:
        series_to_fetch.append(F5_SERIES_TICKER)

    for series in series_to_fetch:
        try:
            markets = client.get_markets(series_ticker=series, fetch_all=True)
            all_markets.extend(markets)
        except (KalshiAPIError, AuthenticationError) as exc:
            logger.error("Kalshi API error fetching %s for %s: %s", series, date_str, exc)

    inserted = 0
    with get_conn(db_path) as conn:
        for market in all_markets:
            row = _market_to_row(market, snapshot_ts)
            if row is None or row["date"] != date_str:
                continue
            game_id = _resolve_game_id(
                conn, row["date"], row["_title"], ticker=row.get("ticker", "")
            )
            row["game_id"] = game_id
            if _insert_row(conn, row):
                inserted += 1

    logger.debug("Kalshi historical %s: inserted %d rows", date_str, inserted)
    return inserted


# ── Orchestration ─────────────────────────────────────────────────────────────


def _date_range(start: date, end: date) -> list[str]:
    days = (end - start).days + 1
    return [(start + timedelta(days=i)).isoformat() for i in range(days)]


def run(
    start_date: str,
    end_date: str,
    incremental: bool = True,
    db_path: str = "data/mlb.db",
) -> None:
    """
    Backfill historical Kalshi market data for a date range.

    Parameters
    ----------
    start_date : str
        First date, ``YYYY-MM-DD``.
    end_date : str
        Last date, ``YYYY-MM-DD``.
    incremental : bool
        If True, skip dates already in ``kalshi_markets``.
    db_path : str
        Path to SQLite database.
    """
    start = date.fromisoformat(start_date)
    end = date.fromisoformat(end_date)
    all_dates = _date_range(start, end)

    if incremental:
        with get_conn(db_path) as conn:
            latest = conn.execute("SELECT MAX(date) FROM kalshi_markets").fetchone()[0]
        if latest:
            resume = (date.fromisoformat(latest) + timedelta(days=1)).isoformat()
            all_dates = [d for d in all_dates if d >= resume]
            logger.info("Incremental: resuming from %s", resume)

    total_inserted = 0
    errors: list[str] = []

    for date_str in all_dates:
        with get_conn(db_path) as conn:
            already = conn.execute(
                "SELECT 1 FROM kalshi_markets WHERE date = ? LIMIT 1",
                (date_str,),
            ).fetchone()
        if already:
            continue

        try:
            n = fetch_markets_for_date(date_str, db_path)
        except Exception as exc:
            logger.error("Kalshi fetch failed for %s: %s", date_str, exc)
            errors.append(f"{date_str}: {exc}")
            continue

        total_inserted += n
        if n:
            logger.info("%s — inserted %d Kalshi rows", date_str, n)

    with get_conn(db_path) as conn:
        conn.execute(
            """INSERT INTO scrape_log
               (source, date_range_start, date_range_end,
                rows_inserted, status, error_msg)
               VALUES (?,?,?,?,?,?)""",
            (
                "kalshi_backfill",
                start_date,
                end_date,
                total_inserted,
                "success" if not errors else "partial",
                "; ".join(errors[:10]) if errors else None,
            ),
        )

    logger.info(
        "Done — dates=%d inserted=%d errors=%d",
        len(all_dates),
        total_inserted,
        len(errors),
    )


# ── CLI ───────────────────────────────────────────────────────────────────────

# ── Historical API (no auth required) ────────────────────────────────────────


def fetch_historical_cutoff(session: requests.Session | None = None) -> dict:
    """
    Return the cutoff timestamps that separate live from historical data.

    Markets settled before ``market_settled_ts`` must be accessed via
    ``/historical/markets`` and ``/historical/markets/{ticker}/candlesticks``.

    Returns
    -------
    dict
        Keys: market_settled_ts, trades_created_ts, orders_updated_ts (ISO 8601).
    """
    s = session or requests.Session()
    r = s.get(f"{HIST_BASE}/historical/cutoff", timeout=15)
    r.raise_for_status()
    return r.json()


def fetch_all_historical_markets(
    series_ticker: str,
    session: requests.Session | None = None,
    page_size: int = 1000,
) -> list[dict]:
    """
    Paginate through all settled historical markets for a series ticker.

    Parameters
    ----------
    series_ticker : str
        e.g. ``'KXMLBTOTAL'`` or ``'KXMLBF5TOTAL'``
    session : requests.Session, optional
    page_size : int
        Records per page (max 1000).

    Returns
    -------
    list[dict]
        All market dicts returned by the API.
    """
    s = session or requests.Session()
    all_markets: list[dict] = []
    cursor: str | None = None

    while True:
        params: dict = {"series_ticker": series_ticker, "limit": page_size}
        if cursor:
            params["cursor"] = cursor

        r = s.get(f"{HIST_BASE}/historical/markets", params=params, timeout=30)
        r.raise_for_status()
        data = r.json()

        batch = data.get("markets", [])
        all_markets.extend(batch)
        cursor = data.get("cursor") or None

        logger.debug(
            "Historical markets page: got %d, total so far %d",
            len(batch),
            len(all_markets),
        )
        if not cursor or not batch:
            break

    return all_markets


def fetch_market_candlesticks_hist(
    ticker: str,
    start_ts: int,
    end_ts: int,
    period_interval: int = 60,
    session: requests.Session | None = None,
    max_retries: int = 5,
) -> list[dict]:
    """
    Fetch candlestick (OHLC) data for a settled market.

    Retries with exponential backoff on 429 (rate-limit) responses.

    Parameters
    ----------
    ticker : str
        Full Kalshi market ticker.
    start_ts : int
        Unix timestamp (seconds) — inclusive start.
    end_ts : int
        Unix timestamp (seconds) — inclusive end.
    period_interval : int
        Candle width in minutes.  Valid values: 1, 60, 1440.
    session : requests.Session, optional
    max_retries : int
        Number of retry attempts on 429 before raising.

    Returns
    -------
    list[dict]
        Candlestick objects.  Empty list if ticker not found (404).
    """
    s = session or requests.Session()
    url = f"{HIST_BASE}/historical/markets/{ticker}/candlesticks"
    params = {"start_ts": start_ts, "end_ts": end_ts, "period_interval": period_interval}

    wait = 2.0
    for attempt in range(max_retries + 1):
        r = s.get(url, params=params, timeout=30)
        if r.status_code == 404:
            return []
        if r.status_code == 429:
            if attempt == max_retries:
                r.raise_for_status()
            logger.debug("429 on %s, waiting %.1fs (attempt %d)", ticker, wait, attempt + 1)
            _time.sleep(wait)
            wait = min(wait * 2, 60.0)
            continue
        r.raise_for_status()
        return r.json().get("candlesticks", [])

    return []  # unreachable but satisfies type checker


def _game_ts_utc(game_date: str, game_time_et: str | None) -> int:
    """
    Convert a game date + ET time string to a UTC Unix timestamp.

    Uses the America/New_York timezone so DST transitions (EDT→EST in
    November playoff games) are handled correctly.

    Parameters
    ----------
    game_date : str
        YYYY-MM-DD
    game_time_et : str or None
        HH:MM (24-hour ET).  Falls back to ``19:10`` (7:10 PM) if None/invalid.

    Returns
    -------
    int
        Unix timestamp (seconds, UTC).
    """
    time_str = game_time_et or "19:10"
    try:
        hh, mm = map(int, time_str.split(":"))
    except ValueError:
        hh, mm = 19, 10

    try:
        year = int(game_date[:4])
        month = int(game_date[5:7])
        day = int(game_date[8:10])
    except (ValueError, IndexError):
        return int(datetime.now(UTC).timestamp())

    try:
        from zoneinfo import ZoneInfo
        et_tz = ZoneInfo("America/New_York")
        dt_et = datetime(year, month, day, hh, mm, tzinfo=et_tz)
        return int(dt_et.timestamp())
    except Exception:
        # Fallback: approximate with fixed offset (EDT = UTC-4 for most of season)
        dt_naive = datetime(year, month, day, hh, mm)
        offset_h = 5 if month in (11, 12, 1, 2, 3) else 4
        return int((dt_naive + timedelta(hours=offset_h)).timestamp())


def backfill_pregame_prices(
    series_ticker: str = SERIES_TICKER,
    start_date: str = "2025-04-01",
    end_date: str | None = None,
    db_path: str = "data/mlb.db",
    delay: float = 0.5,
) -> int:
    """
    Backfill pre-game Kalshi prices from the historical candlestick API.

    For each settled market in the series, fetches 60-minute candles
    covering the 3 hours before first pitch and stores the last candle's
    close bid/ask as a proper pre-game snapshot in ``kalshi_markets``.

    The historical endpoint requires no authentication — RSA key not needed.

    Parameters
    ----------
    series_ticker : str
        ``'KXMLBTOTAL'`` (full-game) or ``'KXMLBF5TOTAL'`` (F5).
    start_date : str
        Only process games on or after this date.
    end_date : str or None
        Only process games on or before this date.  Defaults to today.
    db_path : str
    delay : float
        Sleep between candlestick requests (seconds).

    Returns
    -------
    int
        Number of new pre-game price rows inserted.
    """
    end_date = end_date or date.today().isoformat()
    is_f5 = "F5" in series_ticker.upper()

    session = requests.Session()
    session.headers.update({"Accept": "application/json"})

    logger.info("Fetching all historical %s markets...", series_ticker)
    try:
        all_markets = fetch_all_historical_markets(series_ticker, session=session)
    except requests.RequestException as exc:
        logger.error("Failed to fetch historical markets: %s", exc)
        return 0

    logger.info("Found %d total historical %s markets", len(all_markets), series_ticker)

    # Filter to our date window using the event ticker
    in_window: list[dict] = []
    for mkt in all_markets:
        event_ticker = mkt.get("event_ticker", "")
        info = _parse_event_ticker(event_ticker)
        if info and start_date <= info["date"] <= end_date:
            mkt["_game_date"] = info["date"]
            in_window.append(mkt)

    logger.info(
        "%d markets in date window %s to %s", len(in_window), start_date, end_date
    )

    # Load game times from DB for accurate first-pitch timestamps
    with get_conn(db_path) as conn:
        game_time_rows = conn.execute(
            "SELECT game_id, date, home_team, away_team, game_time_et FROM games "
            "WHERE date BETWEEN ? AND ?",
            (start_date, end_date),
        ).fetchall()

    # Map (date, home, away) → game_time_et for quick lookup
    _game_times: dict[str, str | None] = {}
    _game_id_map: dict[str, str | None] = {}
    for row in game_time_rows:
        key = f"{row['date']}|{row['home_team']}|{row['away_team']}"
        _game_times[key] = row["game_time_et"]
        _game_id_map[key] = row["game_id"]

    # Skip tickers that already have a pre-game price row (mid not near 0 or 1)
    with get_conn(db_path) as conn:
        existing = frozenset(
            r[0]
            for r in conn.execute(
                "SELECT DISTINCT ticker FROM kalshi_markets "
                "WHERE mid_price BETWEEN 0.03 AND 0.97"
            ).fetchall()
        )

    logger.info("%d tickers already have pre-game prices — skipping", len(existing))

    inserted = 0
    skipped = 0
    errors = 0

    for i, mkt in enumerate(in_window, 1):
        ticker = mkt.get("ticker", "")
        event_ticker = mkt.get("event_ticker", "")
        game_date = mkt["_game_date"]

        if ticker in existing:
            skipped += 1
            continue

        line = _parse_line(ticker)
        if line is None:
            skipped += 1
            continue

        market_type = _parse_market_type(
            ticker, mkt.get("title"), mkt.get("subtitle"), is_f5=is_f5
        )

        # Resolve game_id and game_time_et
        with get_conn(db_path) as conn:
            game_id = _resolve_game_id(conn, game_date, mkt.get("title"), ticker)

        game_time_et: str | None = None
        if game_id:
            for key, gid in _game_id_map.items():
                if gid == game_id:
                    game_time_et = _game_times.get(key)
                    break

        # Compute UTC Unix timestamps bracketing first pitch
        game_ts = _game_ts_utc(game_date, game_time_et)
        fetch_start = game_ts - 3 * 3600  # 3 hours before
        fetch_end = game_ts              # up to first pitch

        # Fetch 60-minute candles
        try:
            candles = fetch_market_candlesticks_hist(
                ticker, fetch_start, fetch_end, period_interval=60, session=session
            )
        except requests.RequestException as exc:
            logger.warning("Candlestick fetch failed for %s: %s", ticker, exc)
            errors += 1
            continue
        finally:
            _time.sleep(delay)

        if not candles:
            skipped += 1
            continue

        # Pick the latest candle ending at or before first pitch
        valid = [c for c in candles if c.get("end_period_ts", 0) <= game_ts]
        if not valid:
            valid = candles
        candle = max(valid, key=lambda c: c.get("end_period_ts", 0))

        # Extract close prices from the bid/ask OHLC objects
        bid_close = _safe_dollars(
            (candle.get("yes_bid") or {}).get("close")
        )
        ask_close = _safe_dollars(
            (candle.get("yes_ask") or {}).get("close")
        )
        mid = _mid_price(bid_close, ask_close)
        if mid is None:
            skipped += 1
            continue

        # Also grab trade price if available (cross-check)
        price_close = _safe_dollars(
            (candle.get("price") or {}).get("close")
        )
        if mid is None and price_close is not None:
            mid = price_close

        end_period_ts = candle.get("end_period_ts", fetch_end)
        snapshot_ts = datetime.fromtimestamp(end_period_ts, UTC).isoformat()

        volume = _safe_volume(candle.get("volume"))
        oi = _safe_volume(candle.get("open_interest"))

        row: dict = {
            "game_id": game_id,
            "ticker": ticker,
            "event_ticker": event_ticker,
            "market_type": market_type,
            "line": line,
            "date": game_date,
            "snapshot_ts": snapshot_ts,
            "yes_bid": bid_close,
            "yes_ask": ask_close,
            "mid_price": mid,
            "volume": volume,
            "open_interest": oi,
            "status": mkt.get("status"),
            "result": mkt.get("result"),
        }

        with get_conn(db_path) as conn:
            ok = _insert_row(conn, row)
            if ok:
                inserted += 1
            else:
                skipped += 1

        if i % 100 == 0:
            logger.info(
                "backfill_pregame_prices: %d/%d processed, inserted=%d skipped=%d errors=%d",
                i,
                len(in_window),
                inserted,
                skipped,
                errors,
            )

    logger.info(
        "backfill_pregame_prices complete: inserted=%d skipped=%d errors=%d "
        "(series=%s, %s to %s)",
        inserted,
        skipped,
        errors,
        series_ticker,
        start_date,
        end_date,
    )
    return inserted


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="Kalshi MLB market fetcher")
    parser.add_argument(
        "--snapshot",
        action="store_true",
        help="Snapshot currently open full-game markets (for today's games)",
    )
    parser.add_argument(
        "--snapshot-f5",
        action="store_true",
        help="Snapshot currently open F5 (first 5 innings) markets",
    )
    parser.add_argument("--start", help="Backfill start date YYYY-MM-DD")
    parser.add_argument(
        "--end",
        default=date.today().isoformat(),
        help="Backfill end date YYYY-MM-DD (default: today)",
    )
    parser.add_argument(
        "--incremental",
        action="store_true",
        default=True,
        help="Skip dates already in kalshi_markets",
    )
    parser.add_argument(
        "--no-incremental",
        dest="incremental",
        action="store_false",
        help="Re-fetch all dates regardless of existing rows",
    )
    parser.add_argument(
        "--relink",
        action="store_true",
        help="Retroactively resolve game_id for rows where it is NULL",
    )
    parser.add_argument(
        "--backfill-prices",
        action="store_true",
        help="Backfill pre-game prices for full-game markets via historical candlestick API",
    )
    parser.add_argument(
        "--backfill-f5-prices",
        action="store_true",
        help="Backfill pre-game prices for F5 markets via historical candlestick API",
    )
    args = parser.parse_args()

    if args.relink:
        n = relink_game_ids()
        print(f"Relinked {n} Kalshi market rows")
    elif args.backfill_prices or args.backfill_f5_prices:
        if args.backfill_prices:
            start = args.start or "2025-04-01"
            n = backfill_pregame_prices(
                series_ticker=SERIES_TICKER,
                start_date=start,
                end_date=args.end,
            )
            print(f"Backfilled {n} pre-game full-game price rows ({start} to {args.end})")
        if args.backfill_f5_prices:
            start = args.start or "2025-04-01"
            n_f5 = backfill_pregame_prices(
                series_ticker=F5_SERIES_TICKER,
                start_date=start,
                end_date=args.end,
            )
            print(f"Backfilled {n_f5} pre-game F5 price rows ({start} to {args.end})")
    elif args.snapshot or args.snapshot_f5:
        if args.snapshot:
            n = snapshot_open_markets()
            print(f"Snapshotted {n} open full-game Kalshi markets")
        if args.snapshot_f5:
            n_f5 = snapshot_f5_markets()
            print(f"Snapshotted {n_f5} open F5 Kalshi markets")
    elif args.start:
        run(start_date=args.start, end_date=args.end, incremental=args.incremental)
    else:
        parser.error("Provide --snapshot, --backfill-prices, --relink, or --start DATE")
