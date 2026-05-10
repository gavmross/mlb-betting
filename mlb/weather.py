"""
Weather fetcher for MLB games.

Pulls hourly weather from Open-Meteo and stores the observation or forecast
closest to first-pitch time for each game.  Wind direction is encoded relative
to the stadium's centre-field orientation so downstream features can distinguish
"blowing out" from "blowing in".

Data sources
------------
- Historical  : https://archive-api.open-meteo.com/v1/archive  (no API key)
- Forecast    : https://api.open-meteo.com/v1/forecast          (no API key)

Writes to
---------
- ``weather``    — one row per (game_id, snapshot_type)
- ``scrape_log`` — one row per run

snapshot_type
-------------
'historical'  — game is completed; we pull the archived observation
'forecast'    — game is upcoming; we pull the latest NWP forecast
"""

from __future__ import annotations

import argparse
import logging
from datetime import UTC, date, datetime, timedelta

import numpy as np
import openmeteo_requests
import requests_cache
from retry_requests import retry

from mlb.db import get_conn

logger = logging.getLogger(__name__)

HISTORICAL_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

HOURLY_VARS = [
    "temperature_2m",         # index 0
    "wind_speed_10m",         # index 1
    "wind_direction_10m",     # index 2
    "precipitation_probability",  # index 3
    "relative_humidity_2m",   # index 4
]

# Default game time when game_time_et is missing (7pm ET is typical evening start)
DEFAULT_GAME_HOUR_ET = 19


# ── Open-Meteo client (cached + retry) ───────────────────────────────────────


def _build_client() -> openmeteo_requests.Client:
    """Return a cached, retry-enabled Open-Meteo client."""
    cache_session = requests_cache.CachedSession(
        "data/raw/openmeteo_cache", expire_after=timedelta(hours=6)
    )
    retry_session = retry(cache_session, retries=3, backoff_factor=0.5)
    return openmeteo_requests.Client(session=retry_session)


# ── Wind direction encoding ───────────────────────────────────────────────────


def encode_wind(wind_deg: float, cf_orientation: float) -> str:
    """
    Classify wind direction relative to a stadium's centre-field bearing.

    Parameters
    ----------
    wind_deg : float
        Meteorological wind direction (degrees, 0=N, 90=E, where wind is
        coming *from*).
    cf_orientation : float
        Bearing from home plate toward centre field (meteorological degrees).

    Returns
    -------
    str
        One of ``'out'``, ``'in'``, ``'cross_right'``, ``'cross_left'``.

    Notes
    -----
    Wind *from* ``wind_deg`` is blowing *toward* ``(wind_deg + 180) % 360``.
    We rotate into the stadium frame so that 0° = straight in from CF,
    180° = straight out toward CF.
    """
    # Angle of wind source relative to CF direction.
    # relative=0  → wind FROM CF direction → blowing IN (suppresses runs)
    # relative=180 → wind FROM behind HP  → blowing OUT (boosts runs)
    relative = (wind_deg - cf_orientation + 360) % 360

    if relative < 45 or relative >= 315:
        return "in"
    elif 135 <= relative < 225:
        return "out"
    elif 45 <= relative < 135:
        return "cross_right"
    else:
        return "cross_left"


# ── Weather extraction ────────────────────────────────────────────────────────


def _extract_hour(
    response,
    game_hour_utc: int,
    timezone_offset_h: int,
) -> dict:
    """
    Extract weather values for the hour closest to first pitch.

    Parameters
    ----------
    response : openmeteo_requests response object
        Single-location API response.
    game_hour_utc : int
        Hour of first pitch in UTC (0-23).
    timezone_offset_h : int
        UTC offset of the API response timezone (used to align indices).

    Returns
    -------
    dict
        Keys: temp_f, wind_speed_mph, wind_dir_deg, precip_prob, humidity.
    """
    h = response.Hourly()
    temps = h.Variables(0).ValuesAsNumpy()
    wind_spd = h.Variables(1).ValuesAsNumpy()
    wind_dir = h.Variables(2).ValuesAsNumpy()
    precip = h.Variables(3).ValuesAsNumpy()
    humidity = h.Variables(4).ValuesAsNumpy()

    # Convert UTC game hour to local response hour index
    local_hour = (game_hour_utc + timezone_offset_h) % 24
    idx = max(0, min(local_hour, len(temps) - 1))

    def _val(arr: np.ndarray, i: int) -> float | None:
        try:
            v = float(arr[i])
            return None if np.isnan(v) else v
        except (IndexError, TypeError, ValueError):
            return None

    return {
        "temp_f": _val(temps, idx),
        "wind_speed_mph": _val(wind_spd, idx),
        "wind_dir_deg": _val(wind_dir, idx),
        "precip_prob": _val(precip, idx),
        "humidity": _val(humidity, idx),
    }


def _game_hour_utc(game_time_et: str | None) -> int:
    """
    Convert an HH:MM ET game time string to UTC hour (summer: ET = UTC-4).

    Parameters
    ----------
    game_time_et : str or None
        e.g. ``'19:10'``.  If None, uses DEFAULT_GAME_HOUR_ET.

    Returns
    -------
    int
        UTC hour (0-23).
    """
    if not game_time_et:
        return (DEFAULT_GAME_HOUR_ET + 4) % 24  # rough ET→UTC
    try:
        hh = int(game_time_et.split(":")[0])
        return (hh + 4) % 24
    except (ValueError, AttributeError):
        return (DEFAULT_GAME_HOUR_ET + 4) % 24


# ── Per-game fetch ────────────────────────────────────────────────────────────


def fetch_weather_for_game(
    client: openmeteo_requests.Client,
    game_id: str,
    game_date: str,
    game_time_et: str | None,
    latitude: float,
    longitude: float,
    cf_orientation: float,
    is_dome: int,
    snapshot_type: str,
) -> dict | None:
    """
    Fetch and parse weather for a single game.

    Parameters
    ----------
    client : openmeteo_requests.Client
        Shared API client.
    game_id : str
        Game primary key.
    game_date : str
        Date in ``YYYY-MM-DD`` format.
    game_time_et : str or None
        ``HH:MM`` first-pitch time (Eastern time).
    latitude, longitude : float
        Stadium coordinates.
    cf_orientation : float
        Home-plate-to-CF bearing (degrees).
    is_dome : int
        1 if the stadium is a fixed dome (weather irrelevant).
    snapshot_type : str
        ``'historical'`` or ``'forecast'``.

    Returns
    -------
    dict or None
        Row ready for insertion into ``weather``, or None on error.
    """
    if is_dome:
        # Dome stadiums — store a null row so we don't re-attempt
        return {
            "game_id": game_id,
            "snapshot_type": snapshot_type,
            "fetched_at": datetime.now(tz=UTC).isoformat(),
            "temp_f": None,
            "wind_speed_mph": None,
            "wind_dir_deg": None,
            "wind_dir_label": None,
            "precip_prob": None,
            "humidity": None,
            "is_dome": 1,
        }

    url = HISTORICAL_URL if snapshot_type == "historical" else FORECAST_URL
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": game_date,
        "end_date": game_date,
        "hourly": HOURLY_VARS,
        "temperature_unit": "fahrenheit",
        "wind_speed_unit": "mph",
        "timezone": "UTC",
    }

    try:
        responses = client.weather_api(url, params=params)
        resp = responses[0]
    except Exception as exc:
        logger.error("Open-Meteo error for game %s (%s): %s", game_id, game_date, exc)
        return None

    game_hour_utc = _game_hour_utc(game_time_et)
    wx = _extract_hour(resp, game_hour_utc, timezone_offset_h=0)

    wind_label: str | None = None
    if wx["wind_dir_deg"] is not None:
        wind_label = encode_wind(wx["wind_dir_deg"], cf_orientation)

    return {
        "game_id": game_id,
        "snapshot_type": snapshot_type,
        "fetched_at": datetime.now(tz=UTC).isoformat(),
        "temp_f": wx["temp_f"],
        "wind_speed_mph": wx["wind_speed_mph"],
        "wind_dir_deg": wx["wind_dir_deg"],
        "wind_dir_label": wind_label,
        "precip_prob": wx["precip_prob"],
        "humidity": wx["humidity"],
        "is_dome": 0,
    }


# ── Orchestration ─────────────────────────────────────────────────────────────


def run(
    incremental: bool = True,
    start_date: str | None = None,
    end_date: str | None = None,
    db_path: str = "data/mlb.db",
) -> None:
    """
    Fetch weather for all games not yet in the ``weather`` table.

    Completed games (status='Final') get ``snapshot_type='historical'``.
    Scheduled games within the next 7 days get ``snapshot_type='forecast'``.

    Parameters
    ----------
    incremental : bool
        If True (default), skips games that already have a weather row.
    start_date : str or None
        Restrict to games on or after this date.  None = no lower bound.
    end_date : str or None
        Restrict to games on or before this date.  None = today + 7 days.
    db_path : str
        Path to the SQLite database.
    """
    client = _build_client()

    # Build query for games needing weather
    conditions = ["g.game_id IS NOT NULL"]
    params: list = []

    if incremental:
        # Only games without a weather row
        conditions.append(
            "g.game_id NOT IN (SELECT game_id FROM weather WHERE game_id IS NOT NULL)"
        )

    if start_date:
        conditions.append("g.date >= ?")
        params.append(start_date)

    if end_date:
        conditions.append("g.date <= ?")
        params.append(end_date)
    else:
        # Don't fetch forecasts more than 7 days out (Open-Meteo limit)
        cutoff = (date.today() + timedelta(days=7)).isoformat()
        conditions.append("g.date <= ?")
        params.append(cutoff)

    # Only regular season: Final or upcoming Scheduled games
    conditions.append("(g.status = 'Final' OR g.status = 'Preview')")

    where = " AND ".join(conditions)
    sql = f"""
        SELECT g.game_id, g.date, g.game_time_et, g.status,
               g.home_team,
               s.latitude, s.longitude, s.orientation_deg, s.is_dome
        FROM games g
        JOIN stadiums s ON s.team = g.home_team
        WHERE {where}
        ORDER BY g.date
    """

    with get_conn(db_path) as conn:
        games = conn.execute(sql, params).fetchall()

    logger.info("Weather fetch: %d games to process", len(games))
    total_inserted = total_skipped = 0
    errors: list[str] = []

    for row in games:
        game_id = row["game_id"]
        game_date = row["date"]
        snapshot_type = "historical" if row["status"] == "Final" else "forecast"

        wx = fetch_weather_for_game(
            client=client,
            game_id=game_id,
            game_date=game_date,
            game_time_et=row["game_time_et"],
            latitude=row["latitude"],
            longitude=row["longitude"],
            cf_orientation=row["orientation_deg"],
            is_dome=row["is_dome"],
            snapshot_type=snapshot_type,
        )

        if wx is None:
            errors.append(game_id)
            total_skipped += 1
            continue

        with get_conn(db_path) as conn:
            conn.execute(
                """INSERT OR IGNORE INTO weather
                   (game_id, snapshot_type, fetched_at, temp_f, wind_speed_mph,
                    wind_dir_deg, wind_dir_label, precip_prob, humidity, is_dome)
                   VALUES (:game_id, :snapshot_type, :fetched_at, :temp_f,
                           :wind_speed_mph, :wind_dir_deg, :wind_dir_label,
                           :precip_prob, :humidity, :is_dome)""",
                wx,
            )
        total_inserted += 1

        if total_inserted % 100 == 0:
            logger.info("Progress: %d weather rows inserted", total_inserted)

    with get_conn(db_path) as conn:
        conn.execute(
            """INSERT INTO scrape_log
               (source, rows_inserted, status, error_msg)
               VALUES (?,?,?,?)""",
            (
                "open_meteo",
                total_inserted,
                "success" if not errors else "partial",
                f"{len(errors)} games failed" if errors else None,
            ),
        )

    logger.info(
        "Done — inserted=%d skipped=%d errors=%d",
        total_inserted,
        total_skipped,
        len(errors),
    )


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="Open-Meteo weather fetcher")
    parser.add_argument("--start", help="Start date YYYY-MM-DD")
    parser.add_argument("--end", help="End date YYYY-MM-DD")
    parser.add_argument(
        "--incremental",
        action="store_true",
        default=True,
        help="Skip games already in weather table (default: true)",
    )
    parser.add_argument(
        "--no-incremental",
        dest="incremental",
        action="store_false",
        help="Re-fetch all games regardless of existing rows",
    )
    args = parser.parse_args()

    run(incremental=args.incremental, start_date=args.start, end_date=args.end)
