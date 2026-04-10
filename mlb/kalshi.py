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
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from pykalshi import KalshiClient
from pykalshi.enums import MarketStatus
from pykalshi.exceptions import AuthenticationError, KalshiAPIError

from mlb.db import get_conn

logger = logging.getLogger(__name__)

SERIES_TICKER = "KXMLBTOTAL"

# Month abbreviation → zero-padded month number (Kalshi uses APR, MAY, etc.)
_MONTH_MAP: dict[str, str] = {
    "JAN": "01", "FEB": "02", "MAR": "03", "APR": "04",
    "MAY": "05", "JUN": "06", "JUL": "07", "AUG": "08",
    "SEP": "09", "OCT": "10", "NOV": "11", "DEC": "12",
}

# Regex to match Kalshi MLB totals event tickers.
# Format: KXMLBTOTAL-26APR101840AZPHI
#   26=year, APR=month, 10=day, 1840=game time, AZPHI=away+home abbrevs
_EVENT_RE = re.compile(
    r"KXMLBTOTAL-(\d{2})([A-Z]{3})(\d{2})\d{4}([A-Z]+)",
    re.IGNORECASE,
)

# Regex to extract line from ticker, e.g. KXMLBTOTAL-26APR101840AZPHI-9 → 9.0
# YES side = P(total >= line), equivalent to P(over line-0.5) in standard betting.
_LINE_RE = re.compile(r"-(\d+)$", re.IGNORECASE)

# Regex to parse "Away vs Home Total Runs?" from market title
_TITLE_TEAMS_RE = re.compile(r"^([A-Za-z ]+) vs ([A-Za-z ]+) Total", re.IGNORECASE)

# Regex to detect UNDER side in ticker/title (YES = over in new format, kept for safety)
_UNDER_RE = re.compile(r"under|UNDER|_U_|-U-", re.IGNORECASE)


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
    Extract game date from a Kalshi MLB totals event ticker.

    Parameters
    ----------
    event_ticker : str
        e.g. ``'KXMLBTOTAL-26APR101840AZPHI'``

    Returns
    -------
    dict or None
        Keys: ``date`` (YYYY-MM-DD).
        None if pattern does not match.
    """
    m = _EVENT_RE.match(event_ticker)
    if not m:
        return None
    yy, mon, dd = m.group(1), m.group(2).upper(), m.group(3)
    month = _MONTH_MAP.get(mon)
    if not month:
        return None
    game_date = f"20{yy}-{month}-{dd}"
    return {"date": game_date}


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


def _parse_market_type(ticker: str, title: str | None, subtitle: str | None) -> str:
    """
    Determine whether this is a 'total_over' or 'total_under' market.

    Kalshi typically lists OVER as the YES side. Checks ticker and title
    for 'under' keyword; defaults to 'total_over'.

    Parameters
    ----------
    ticker : str
    title : str or None
    subtitle : str or None

    Returns
    -------
    str
        ``'total_over'`` or ``'total_under'``.
    """
    text = " ".join(filter(None, [ticker, title, subtitle]))
    if _UNDER_RE.search(text):
        return "total_under"
    return "total_over"


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


def _resolve_game_id(conn, game_date: str, title: str | None) -> str | None:
    """
    Look up game_id by date, using team city/name keywords from the market title.

    Parses ``'Arizona vs Philadelphia Total Runs?'`` → searches games on
    ``game_date`` where home_team or away_team contains 'Arizona' or
    'Philadelphia' as a substring (case-insensitive).

    Parameters
    ----------
    conn : sqlite3.Connection
    game_date : str
        YYYY-MM-DD
    title : str or None
        Kalshi market title, e.g. ``'Arizona vs Philadelphia Total Runs?'``

    Returns
    -------
    str or None
    """
    if not title:
        return None
    m = _TITLE_TEAMS_RE.match(title)
    if not m:
        return None
    away_name = m.group(1).strip()
    home_name = m.group(2).strip()
    # Search using stadium/city keyword contained in home_team abbreviation
    # mapping is imprecise — fall back to searching all games on date and
    # matching by partial city name stored in the stadiums table.
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

    snapshot_ts = datetime.now(tz=timezone.utc).isoformat()

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
            game_id = _resolve_game_id(conn, row["date"], row["_title"])
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


# ── Historical fetch ──────────────────────────────────────────────────────────


def fetch_markets_for_date(
    date_str: str,
    db_path: str = "data/mlb.db",
) -> int:
    """
    Fetch all settled KXMLB markets for a specific date and store them.

    Pulls all markets (no status filter) and filters to those whose parsed
    date matches ``date_str``. This retrieves final prices and results for
    backtesting.

    Parameters
    ----------
    date_str : str
        Date in ``YYYY-MM-DD`` format.
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

    snapshot_ts = datetime.now(tz=timezone.utc).isoformat()

    try:
        markets = client.get_markets(
            series_ticker=SERIES_TICKER,
            fetch_all=True,
        )
    except (KalshiAPIError, AuthenticationError) as exc:
        logger.error("Kalshi API error fetching markets for %s: %s", date_str, exc)
        return 0

    inserted = 0
    with get_conn(db_path) as conn:
        for market in markets:
            row = _market_to_row(market, snapshot_ts)
            if row is None or row["date"] != date_str:
                continue
            game_id = _resolve_game_id(conn, row["date"], row["_title"])
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
            latest = conn.execute(
                "SELECT MAX(date) FROM kalshi_markets"
            ).fetchone()[0]
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
        help="Snapshot currently open markets (for today's games)",
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
    args = parser.parse_args()

    if args.snapshot:
        n = snapshot_open_markets()
        print(f"Snapshotted {n} open Kalshi markets")
    elif args.start:
        run(start_date=args.start, end_date=args.end, incremental=args.incremental)
    else:
        parser.error("Provide --snapshot or --start DATE")
