"""
Polymarket MLB market client — read-only cross-market signal.

Fetches YES prices for MLB totals (over/under runs) from the Polymarket
Gamma API and stores them as a secondary pricing signal alongside Kalshi
prices in the ``predictions`` table.

Polymarket is a read-only signal. All actual bets are placed on Kalshi.
A spread > $0.04 between Kalshi and Polymarket mid-prices may indicate
one exchange is lagging — favour the cheaper side.

Data source
-----------
Gamma API: https://gamma-api.polymarket.com/markets (no auth required)
CLOB  API: https://clob.polymarket.com/markets      (fallback, no auth)

Writes to
---------
- ``predictions.polymarket_mid_price`` — updated in-place when a match
  is found for an existing prediction row
- ``scrape_log`` — one row per snapshot run

Notes
-----
- Polymarket coverage: April 2025 onward (MLB exclusive deal March 2026)
- Market matching is best-effort; unmatched predictions stay NULL
- Mid-price = (bestBid + bestAsk) / 2 when both present, else lastTradePrice
"""

from __future__ import annotations

import json
import logging
import re
import time
import urllib.error
import urllib.request
from datetime import date, datetime, timezone
from typing import Any

from mlb.db import get_conn

logger = logging.getLogger(__name__)

GAMMA_URL = "https://gamma-api.polymarket.com"
CLOB_URL = "https://clob.polymarket.com"

RATE_LIMIT_S: float = 0.5

# Team name fragments that appear in Polymarket questions → canonical abbr
# Polymarket uses full city/nickname names in questions
_TEAM_KEYWORDS: dict[str, str] = {
    "yankees": "NYY", "mets": "NYM", "red sox": "BOS", "blue jays": "TOR",
    "orioles": "BAL", "rays": "TBR", "guardians": "CLE", "white sox": "CHW",
    "twins": "MIN", "tigers": "DET", "royals": "KCR", "astros": "HOU",
    "rangers": "TEX", "mariners": "SEA", "athletics": "OAK", "angels": "LAA",
    "phillies": "PHI", "braves": "ATL", "marlins": "MIA", "nationals": "WSN",
    "dodgers": "LAD", "giants": "SFG", "padres": "SDP", "rockies": "COL",
    "diamondbacks": "ARI", "cubs": "CHC", "cardinals": "STL", "brewers": "MIL",
    "pirates": "PIT", "reds": "CIN",
}

# Regex to extract a runs line from a market question, e.g. "8.5 runs" / "over 9"
_LINE_RE = re.compile(r"(\d+(?:\.\d+)?)\s*runs?", re.IGNORECASE)
_OVER_THRESHOLD_RE = re.compile(
    r"(?:over|more than|exceed)\s+(\d+(?:\.\d+)?)", re.IGNORECASE
)
_UNDER_THRESHOLD_RE = re.compile(
    r"(?:under|fewer than|less than)\s+(\d+(?:\.\d+)?)", re.IGNORECASE
)


# ── HTTP helpers ──────────────────────────────────────────────────────────────


def _get_json(url: str, timeout: float = 10.0) -> Any:
    """
    GET a URL and return parsed JSON, or None on error.

    Parameters
    ----------
    url : str
    timeout : float

    Returns
    -------
    Any
        Parsed JSON, or None if the request failed.
    """
    req = urllib.request.Request(url, headers={"User-Agent": "mlb-betting/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        logger.warning("Polymarket HTTP %s for %s", exc.code, url)
        return None
    except urllib.error.URLError as exc:
        logger.warning("Polymarket URL error for %s: %s", url, exc.reason)
        return None
    except (json.JSONDecodeError, ValueError) as exc:
        logger.warning("Polymarket JSON parse error for %s: %s", url, exc)
        return None


# ── Parsing helpers ───────────────────────────────────────────────────────────


def _extract_teams(question: str) -> list[str]:
    """
    Extract canonical team abbreviations mentioned in a market question.

    Parameters
    ----------
    question : str
        Market question text.

    Returns
    -------
    list[str]
        Up to two canonical team abbreviations found.
    """
    q_lower = question.lower()
    found = []
    for keyword, abbr in _TEAM_KEYWORDS.items():
        if keyword in q_lower and abbr not in found:
            found.append(abbr)
    return found


def _extract_line(question: str) -> tuple[float | None, str | None]:
    """
    Extract run line and bet side from a market question.

    Parameters
    ----------
    question : str
        Market question text.

    Returns
    -------
    tuple[float | None, str | None]
        (line, side) where side is ``'over'``, ``'under'``, or None.
    """
    # Try explicit over/under language
    m = _OVER_THRESHOLD_RE.search(question)
    if m:
        return float(m.group(1)), "over"
    m = _UNDER_THRESHOLD_RE.search(question)
    if m:
        return float(m.group(1)), "under"
    # Fall back to bare line number
    m = _LINE_RE.search(question)
    if m:
        return float(m.group(1)), None
    return None, None


def _extract_date(question: str) -> str | None:
    """
    Extract YYYY-MM-DD from a market question if present.

    Parameters
    ----------
    question : str

    Returns
    -------
    str or None
    """
    m = re.search(r"(\d{4}-\d{2}-\d{2})", question)
    return m.group(1) if m else None


def _mid_price(
    best_bid: float | None,
    best_ask: float | None,
    last_trade: float | None,
) -> float | None:
    """
    Compute mid-price from available price fields.

    Parameters
    ----------
    best_bid : float or None
    best_ask : float or None
    last_trade : float or None

    Returns
    -------
    float or None
    """
    if best_bid is not None and best_ask is not None:
        return (best_bid + best_ask) / 2.0
    if best_bid is not None:
        return best_bid
    if best_ask is not None:
        return best_ask
    return last_trade


def _parse_gamma_market(market: dict) -> dict | None:
    """
    Parse a Gamma API market dict into a structured record.

    Parameters
    ----------
    market : dict
        Raw market dict from Gamma API response.

    Returns
    -------
    dict or None
        Keys: question, date, teams, line, side, mid_price, question_id.
        None if not an MLB runs market.
    """
    question = market.get("question", "")
    if not question:
        return None

    # Must mention runs or be over/under
    if not any(kw in question.lower() for kw in ["run", "over", "under", "total"]):
        return None

    teams = _extract_teams(question)
    if not teams:
        return None

    line, side = _extract_line(question)

    game_date = _extract_date(question)
    if not game_date:
        # Try endDateIso
        end = market.get("endDateIso") or market.get("endDate", "")
        game_date = end[:10] if end else None

    bid = market.get("bestBid")
    ask = market.get("bestAsk")
    last = market.get("lastTradePrice")

    try:
        bid_f = float(bid) if bid is not None else None
        ask_f = float(ask) if ask is not None else None
        last_f = float(last) if last is not None else None
    except (TypeError, ValueError):
        bid_f = ask_f = last_f = None

    mid = _mid_price(bid_f, ask_f, last_f)

    # Also check outcomePrices if mid is unavailable
    if mid is None:
        outcome_prices = market.get("outcomePrices")
        if outcome_prices:
            try:
                prices = json.loads(outcome_prices) if isinstance(outcome_prices, str) else outcome_prices
                if prices:
                    mid = float(prices[0])
            except (json.JSONDecodeError, ValueError, IndexError, TypeError):
                pass

    if mid is None:
        return None

    return {
        "question": question,
        "question_id": market.get("questionID") or market.get("id"),
        "date": game_date,
        "teams": teams,
        "line": line,
        "side": side or "over",  # default: YES = over
        "mid_price": mid,
    }


# ── Gamma API fetch ───────────────────────────────────────────────────────────


def fetch_mlb_markets(
    game_date: str | None = None,
    limit: int = 200,
) -> list[dict]:
    """
    Fetch MLB runs markets from the Polymarket Gamma API.

    Parameters
    ----------
    game_date : str or None
        Filter to markets ending on or near this date (YYYY-MM-DD).
        None = all active MLB runs markets.
    limit : int
        Maximum markets to fetch per page.

    Returns
    -------
    list[dict]
        Parsed market records. Empty list if no data available.
    """
    params = f"limit={limit}&active=true"
    if game_date:
        # No native date filter — fetch and post-filter
        params += "&tag=mlb"

    url = f"{GAMMA_URL}/markets?{params}"
    data = _get_json(url)
    if not data or not isinstance(data, list):
        logger.debug("Polymarket Gamma returned no data for %s", game_date or "all")
        return []

    time.sleep(RATE_LIMIT_S)

    parsed = []
    for raw in data:
        record = _parse_gamma_market(raw)
        if record is None:
            continue
        if game_date and record.get("date") != game_date:
            continue
        parsed.append(record)

    logger.debug(
        "Polymarket: %d raw markets → %d MLB runs parsed for %s",
        len(data),
        len(parsed),
        game_date or "all",
    )
    return parsed


# ── Game matching ─────────────────────────────────────────────────────────────


def _match_market_to_game(
    conn,
    market: dict,
) -> str | None:
    """
    Find a game_id matching a Polymarket market's date and teams.

    Parameters
    ----------
    conn : sqlite3.Connection
    market : dict
        Parsed market record.

    Returns
    -------
    str or None
    """
    game_date = market.get("date")
    teams = market.get("teams", [])
    if not game_date or not teams:
        return None

    for team in teams:
        row = conn.execute(
            """SELECT game_id FROM games
               WHERE date = ? AND (home_team = ? OR away_team = ?)
               LIMIT 1""",
            (game_date, team, team),
        ).fetchone()
        if row:
            return row["game_id"]
    return None


# ── Snapshot ──────────────────────────────────────────────────────────────────


def snapshot_prices(
    game_date: str | None = None,
    db_path: str = "data/mlb.db",
) -> int:
    """
    Fetch current Polymarket MLB prices and update ``predictions`` table.

    Only updates rows where ``polymarket_mid_price`` is currently NULL
    (or where a fresh snapshot is warranted for the given date).

    Parameters
    ----------
    game_date : str or None
        Date to snapshot. None = all active markets.
    db_path : str
        Path to SQLite database.

    Returns
    -------
    int
        Number of prediction rows updated.
    """
    markets = fetch_mlb_markets(game_date=game_date)
    if not markets:
        logger.info("Polymarket: no MLB markets found for %s", game_date or "all")
        return 0

    updated = 0
    snapshot_ts = datetime.now(tz=timezone.utc).isoformat()

    with get_conn(db_path) as conn:
        for market in markets:
            game_id = _match_market_to_game(conn, market)
            if game_id is None:
                continue

            line = market.get("line")
            mid = market.get("mid_price")
            if mid is None:
                continue

            # Update predictions row that matches this game + line
            cursor = conn.execute(
                """UPDATE predictions
                   SET polymarket_mid_price = ?
                   WHERE game_id = ?
                     AND ABS(COALESCE(line, -999) - COALESCE(?, -999)) < 0.1
                     AND polymarket_mid_price IS NULL""",
                (mid, game_id, line),
            )
            updated += cursor.rowcount

        conn.execute(
            """INSERT INTO scrape_log
               (source, rows_inserted, status)
               VALUES (?,?,?)""",
            ("polymarket_snapshot", updated, "success"),
        )

    logger.info(
        "Polymarket snapshot: updated %d prediction rows for %s",
        updated,
        game_date or "all",
    )
    return updated


# ── Price lookup ──────────────────────────────────────────────────────────────


def get_price_for_game(
    game_id: str,
    line: float,
    db_path: str = "data/mlb.db",
) -> float | None:
    """
    Return current Polymarket mid-price for a game's totals market.

    First checks the DB for a cached price. If not found, fetches live.

    Parameters
    ----------
    game_id : str
    line : float
        The totals line (e.g. 8.5).
    db_path : str

    Returns
    -------
    float or None
    """
    with get_conn(db_path) as conn:
        row = conn.execute(
            """SELECT g.date, g.home_team, g.away_team
               FROM games g WHERE g.game_id = ?""",
            (game_id,),
        ).fetchone()
        if not row:
            return None

        # Check predictions cache
        cached = conn.execute(
            """SELECT polymarket_mid_price FROM predictions
               WHERE game_id = ? AND ABS(COALESCE(line,-999) - ?) < 0.1
               LIMIT 1""",
            (game_id, line),
        ).fetchone()
        if cached and cached["polymarket_mid_price"] is not None:
            return cached["polymarket_mid_price"]

    # Live fetch
    game_date = row["date"]
    markets = fetch_mlb_markets(game_date=game_date)
    for market in markets:
        mteams = set(market.get("teams", []))
        if not mteams.intersection({row["home_team"], row["away_team"]}):
            continue
        if market.get("line") is not None and abs(market["line"] - line) < 0.1:
            return market.get("mid_price")

    return None


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    import logging as _logging

    _logging.basicConfig(
        level=_logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="Polymarket MLB price fetcher")
    parser.add_argument(
        "--snapshot",
        action="store_true",
        help="Snapshot current Polymarket prices into predictions table",
    )
    parser.add_argument("--date", default=date.today().isoformat(), help="Date YYYY-MM-DD")
    args = parser.parse_args()

    if args.snapshot:
        n = snapshot_prices(game_date=args.date)
        print(f"Updated {n} predictions with Polymarket prices for {args.date}")
    else:
        markets = fetch_mlb_markets(game_date=args.date)
        print(f"Found {len(markets)} Polymarket MLB markets for {args.date}")
        for m in markets:
            print(
                f"  {m['teams']} | line={m['line']} | side={m['side']} | mid={m['mid_price']:.3f}"
            )
