"""
common.py — shared helpers for the K-projection pipeline.

Data sources (all free, no API key required):
  - MLB Stats API   https://statsapi.mlb.com/api/v1   (schedule, probable pitchers, boxscores, rosters)
  - Baseball Savant  via the `pybaseball` package        (Statcast pitch-level data for CSW%/SwStr%)
  - Open-Meteo       https://open-meteo.com              (weather forecast/history, no key required)

Everything here is plain `requests` + `pandas`. No network calls happen at import time.
"""
from __future__ import annotations
import datetime as dt
import json
import math
import os
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"

MLB_API = "https://statsapi.mlb.com/api/v1"
MLB_API_V1_1 = "https://statsapi.mlb.com/api/v1.1"
OPEN_METEO = "https://api.open-meteo.com/v1/forecast"

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "whiff-line-hobby-project/1.0"})


def _get(url, params=None, retries=3, backoff=1.5):
    """GET with a couple of retries — free public APIs occasionally hiccup."""
    last_err = None
    for attempt in range(retries):
        try:
            resp = SESSION.get(url, params=params, timeout=20)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:  # noqa: BLE001 - want to retry on anything transient
            last_err = e
            time.sleep(backoff * (attempt + 1))
    raise RuntimeError(f"GET {url} failed after {retries} tries: {last_err}")


# ---------------------------------------------------------------------------
# MLB Stats API
# ---------------------------------------------------------------------------

def get_schedule(date: dt.date) -> list[dict]:
    """Return today's games with probable pitchers, venue, and game time."""
    data = _get(
        f"{MLB_API}/schedule",
        params={
            "sportId": 1,
            "date": date.isoformat(),
            "hydrate": "probablePitcher,linescore,venue,weather",
        },
    )
    games = []
    for date_block in data.get("dates", []):
        for g in date_block.get("games", []):
            games.append(g)
    return games


def get_boxscore(game_pk: int) -> dict:
    """Full boxscore for a completed game (used to grade strikeout props)."""
    return _get(f"{MLB_API}/game/{game_pk}/boxscore")


def get_pitcher_strikeouts_from_boxscore(box: dict, pitcher_id: int) -> int | None:
    """Pull a specific pitcher's strikeout total out of a boxscore payload."""
    for side in ("home", "away"):
        players = box.get("teams", {}).get(side, {}).get("players", {})
        key = f"ID{pitcher_id}"
        if key in players:
            stats = players[key].get("stats", {}).get("pitching", {})
            if "strikeOuts" in stats:
                return int(stats["strikeOuts"])
    return None


def get_team_roster_recent_hitters(team_id: int, season: int) -> list[dict]:
    """Season hitting stats for a team's roster, used to build lineup K% vs handedness."""
    data = _get(
        f"{MLB_API}/teams/{team_id}/roster",
        params={"rosterType": "active"},
    )
    return data.get("roster", [])


def get_player_season_stats(player_id: int, season: int, group: str = "hitting") -> dict:
    data = _get(
        f"{MLB_API}/people/{player_id}/stats",
        params={"stats": "season", "group": group, "season": season},
    )
    stats = data.get("stats", [])
    if stats and stats[0].get("splits"):
        return stats[0]["splits"][0].get("stat", {})
    return {}


# ---------------------------------------------------------------------------
# Open-Meteo (weather — outdoor parks only)
# ---------------------------------------------------------------------------

def get_weather(lat: float, lon: float, date: dt.date) -> dict:
    """Temperature (F) and wind speed (mph) forecast for a park on a given date."""
    try:
        data = _get(
            OPEN_METEO,
            params={
                "latitude": lat,
                "longitude": lon,
                "daily": "temperature_2m_max,windspeed_10m_max",
                "temperature_unit": "fahrenheit",
                "windspeed_unit": "mph",
                "timezone": "America/New_York",
                "start_date": date.isoformat(),
                "end_date": date.isoformat(),
            },
        )
        daily = data.get("daily", {})
        return {
            "temp_f": (daily.get("temperature_2m_max") or [None])[0],
            "wind_mph": (daily.get("windspeed_10m_max") or [None])[0],
        }
    except Exception:
        return {"temp_f": None, "wind_mph": None}


# ---------------------------------------------------------------------------
# Odds / edge math
# ---------------------------------------------------------------------------

def american_to_prob(price: float) -> float:
    """Convert American odds to raw (vig-included) implied probability."""
    if price < 0:
        return -price / (-price + 100)
    return 100 / (price + 100)


def devig_two_way(over_price: float, under_price: float) -> tuple[float, float]:
    """Strip the vig from a two-way market using the multiplicative method."""
    p_over_raw = american_to_prob(over_price)
    p_under_raw = american_to_prob(under_price)
    total = p_over_raw + p_under_raw
    return p_over_raw / total, p_under_raw / total


def kelly_fraction(model_prob: float, price: float, fraction: float = 0.25) -> float:
    """Quarter-Kelly stake as a fraction of bankroll. Returns 0 if no edge."""
    b = (100 / -price) if price < 0 else (price / 100)  # net odds
    q = 1 - model_prob
    edge = model_prob * b - q
    if edge <= 0:
        return 0.0
    full_kelly = edge / b
    return max(0.0, full_kelly * fraction)


def load_json(path: Path, default):
    if not path.exists():
        return default
    with open(path) as f:
        return json.load(f)


def save_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)


def utc_now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
