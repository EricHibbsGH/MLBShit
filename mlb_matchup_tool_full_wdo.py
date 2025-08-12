"""
mlb_matchup_tool
==================

This module provides a simple command‑line tool for comparing a Major League
Baseball (MLB) hitter and pitcher.  It relies on the open MLB Stats API to
retrieve head‑to‑head matchup data, career summaries, and seasonal statistics.

Rather than scraping HTML pages, which can break without warning, the script
uses the official `statsapi.mlb.com` JSON endpoints.  The endpoints are free
to use and do not require an API key.  Because this environment cannot load
external Python packages such as ``pybaseball``, the functions here are
implemented directly with ``requests`` and standard library modules.

The tool accepts a batter name, a pitcher name and optional environmental
information (temperature, wind and altitude).  It looks up player IDs, pulls
their matchup history and career stats, and then applies simple
weather/ballpark adjustments informed by published research:

* A 1995 study found that fly balls at 50 °F travel roughly 16 feet less than
  at 90 °F — a 40 °F swing produces about a 4 % change in distance【627969437292740†L26-L31】.  The
  temperature adjustment implemented here applies a 0.1 % distance change per
  degree Fahrenheit away from a 70 °F baseline.
* Research on Denver’s Coors Field suggests that thin air at an elevation of
  5,280 ft should add roughly 10 % to fly‑ball distance compared to sea level
  【779714290220229†L74-L78】.  The altitude adjustment here applies a 2 % distance change per
  1,000 ft above sea level.

Ballpark dimensions for all current MLB stadiums are embedded from the
Ballparks of Baseball comparison table【720097225716978†L119-L134】.  If the matchup takes
place at a ballpark listed in ``STADIUM_DIMENSIONS`` the script reports its
left‑, center‑ and right‑field fence distances.

Example usage at the command line::

    python mlb_matchup_tool.py --batter "Mike Trout" --pitcher "Justin Verlander" \
        --stadium "Angel Stadium" --temperature 80 --wind_speed 5 --wind_dir out

Running the script will print a summary of past plate appearances between
Trout and Verlander, career numbers for each player, ballpark dimensions and
an estimate of how the environmental factors might influence batted‑ball
distance.

Note: because the MLB Stats API is updated daily, statistics returned by
the tool reflect the current moment and may differ from historical values.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import requests

# Extend the default timeout for network requests.  Some environments
# throttle outbound connections, so a longer timeout reduces failures.
REQUEST_TIMEOUT: int = 20


###############################################
# Constants and configuration
###############################################

# Ballpark dimensions (left, center, right) in feet.  Data are taken from
# Ballparks of Baseball’s comparison table【720097225716978†L119-L134】, which lists each
# current MLB park along with its fence distances.  A few parks have unusual
# shapes; for simplicity we record the down‑the‑line (left and right) and
# straightaway center distances.
STADIUM_DIMENSIONS: Dict[str, Tuple[int, int, int]] = {
    # American League parks
    "American Family Field": (332, 400, 325),
    "Angel Stadium": (330, 396, 330),
    "Busch Stadium": (336, 400, 335),
    "Camden Yards": (337, 406, 320),
    "Chase Field": (330, 407, 335),
    "Citi Field": (335, 405, 330),
    "Citizens Bank Park": (330, 401, 329),
    "Comerica Park": (345, 420, 330),
    "Coors Field": (347, 415, 350),
    "Dodger Stadium": (330, 400, 300),
    "Fenway Park": (310, 420, 302),
    "Globe Life Field": (329, 407, 326),
    "Great American Ball Park": (328, 404, 325),  # table lists R,C,L but reorder to L,C,R
    "Guaranteed Rate Field": (330, 400, 335),
    "Kauffman Stadium": (330, 400, 330),
    "loanDepot Park": (340, 420, 335),
    "Minute Maid Park": (315, 435, 326),
    "Nationals Park": (336, 403, 335),
    "Oracle Park": (339, 399, 309),
    "Petco Park": (336, 396, 322),
    "PNC Park": (325, 399, 320),
    "Progressive Field": (325, 405, 325),
    "Rogers Centre": (328, 400, 328),
    "Steinbrenner Field": (318, 408, 314),  # temporary home park
    "Sutter Health Park": (330, 403, 325),
    "T‑Mobile Park": (331, 405, 327),
    "Target Field": (339, 404, 328),
    "Truist Park": (335, 400, 325),
    "Wrigley Field": (355, 400, 353),
    "Yankee Stadium": (318, 404, 314),
}

# Approximate elevation (ft) above sea level for select ballparks.  Only
# Coors Field has a substantially high altitude; others are near sea level.
STADIUM_ALTITUDE: Dict[str, int] = {
    "Coors Field": 5280,
    # Most other parks lie between sea level and ~500 ft.  Values here are
    # intentionally conservative – the effect on fly‑ball distance is minor
    # relative to Coors Field.  Users may override altitude via command line.
    "Chase Field": 1080,
    "Globe Life Field": 560,
    "Minute Maid Park": 80,
    "Dodger Stadium": 345,
    "Angel Stadium": 160,
    "Truist Park": 1050,
}


@dataclass
class PlayerStats:
    """Aggregate statistics for a hitter against a particular pitcher."""
    at_bats: int = 0
    hits: int = 0
    doubles: int = 0
    triples: int = 0
    home_runs: int = 0
    strikeouts: int = 0
    walks: int = 0
    total_bases: int = 0

    @property
    def batting_average(self) -> float:
        return (self.hits / self.at_bats) if self.at_bats else float('nan')

    @property
    def on_base_percentage(self) -> float:
        # OBP = (H + BB) / PA; plate appearances approximated as AB + BB
        pa = self.at_bats + self.walks
        return ((self.hits + self.walks) / pa) if pa else float('nan')

    @property
    def slugging(self) -> float:
        return (self.total_bases / self.at_bats) if self.at_bats else float('nan')

    @property
    def ops(self) -> float:
        return self.on_base_percentage + self.slugging


def search_player_id(name: str) -> Tuple[int, str]:
    """Look up the MLBAM player ID for a given player name.

    The MLB Stats API search endpoint returns a list of players whose names
    contain the supplied substring.  The first result is returned.  If no
    results are found, a ``ValueError`` is raised.

    Parameters
    ----------
    name : str
        The player's full or partial name (e.g., "Mike Trout").

    Returns
    -------
    tuple
        A tuple of (player_id, full_name).
    """
    url = f"https://statsapi.mlb.com/api/v1/people/search?names={name}"
    resp = requests.get(url, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    people = resp.json().get("people", [])
    if not people:
        raise ValueError(f"No players found for name: {name}")
    player = people[0]
    return int(player["id"]), player["fullName"]


def _aggregate_vs_stats(splits: list) -> PlayerStats:
    """Helper to aggregate a list of vsPlayer splits into a PlayerStats object."""
    stats = PlayerStats()
    for entry in splits:
        stat = entry.get("stat", {})
        stats.at_bats += int(stat.get("atBats", 0))
        stats.hits += int(stat.get("hits", 0))
        stats.doubles += int(stat.get("doubles", 0))
        stats.triples += int(stat.get("triples", 0))
        stats.home_runs += int(stat.get("homeRuns", 0))
        stats.strikeouts += int(stat.get("strikeOuts", 0))
        stats.walks += int(stat.get("baseOnBalls", 0))
        stats.total_bases += int(stat.get("totalBases", 0))
    return stats


def get_head_to_head_stats(batter_id: int, pitcher_id: int) -> PlayerStats:
    """Retrieve all available batting statistics for a hitter vs. a pitcher.

    Calls the MLB Stats API ``vsPlayer`` stat type for hitting.  Splits are
    provided per season; they are aggregated into a single ``PlayerStats``.

    Parameters
    ----------
    batter_id : int
        MLBAM ID for the hitter.
    pitcher_id : int
        MLBAM ID for the pitcher.

    Returns
    -------
    PlayerStats
        Aggregated head‑to‑head stats.
    """
    url = (
        f"https://statsapi.mlb.com/api/v1/people/{batter_id}/stats?"
        f"stats=vsPlayer&opposingPlayerId={pitcher_id}&group=hitting"
    )
    resp = requests.get(url, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    data = resp.json()
    splits = data.get("stats", [{}])[0].get("splits", [])
    return _aggregate_vs_stats(splits)


def get_career_stat_line(player_id: int, group: str) -> Dict[str, str]:
    """Get career slash line (avg, obp, slg, ops) for a player.

    Parameters
    ----------
    player_id : int
        MLBAM ID for the player.
    group : str
        Either ``"hitting"`` or ``"pitching"``.

    Returns
    -------
    dict
        A dictionary containing 'avg', 'obp', 'slg' and 'ops' strings.  If
        values are missing, they default to "NA".
    """
    url = f"https://statsapi.mlb.com/api/v1/people/{player_id}/stats?stats=career&group={group}"
    resp = requests.get(url, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    stats = resp.json().get("stats", [])
    if not stats or not stats[0].get("splits"):
        return {"avg": "NA", "obp": "NA", "slg": "NA", "ops": "NA"}
    stat = stats[0]["splits"][0]["stat"]
    return {
        "avg": stat.get("avg", "NA"),
        "obp": stat.get("obp", "NA"),
        "slg": stat.get("slg", "NA"),
        "ops": stat.get("ops", "NA"),
    }


def compute_environment_factor(
    temperature: float = 70.0,
    wind_speed: float = 0.0,
    wind_dir: str = "none",
    altitude: float = 0.0,
) -> float:
    """Compute a multiplicative factor based on environmental conditions.

    The factor represents the approximate change in fly‑ball distance relative
    to neutral conditions.  A value above 1 implies longer carries (e.g., warm
    temperatures, tail winds or high altitude).  A value below 1 implies shorter
    distances.

    *Temperature:*  Every degree Fahrenheit above 70 °F adds roughly 0.1 % to
    distance; every degree below subtracts 0.1 %.  This is derived from the
    observation that a 40 °F temperature swing (50 °F to 90 °F) changes fly‑ball
    distance by about 16 ft out of 400 ft (~4 %)【627969437292740†L26-L31】.

    *Wind:*  A light tail wind helps carry the ball; a head wind knocks it down.
    Observational estimates suggest that a 5 mph tail wind can add roughly
    18–20 ft to a fly ball (about 5 %) – about 1 % per mph.  We apply 0.1 %
    distance change per mph for simplicity.  Only winds labelled ``"out"`` or
    ``"in"`` modify the factor; other directions are ignored.

    *Altitude:*  Thin air reduces drag.  At Coors Field (5,280 ft), a fly ball
    should travel about 10 % farther than at sea level【779714290220229†L74-L78】.  The
    altitude adjustment used here assumes a 2 % change per 1,000 ft above sea
    level.

    Parameters
    ----------
    temperature : float
        Air temperature in degrees Fahrenheit.
    wind_speed : float
        Wind speed in miles per hour.
    wind_dir : str
        Direction of wind relative to the batter: ``"out"`` (toward the outfield),
        ``"in"`` (toward home plate) or any other value for crosswind/no wind.
    altitude : float
        Ballpark elevation in feet above sea level.

    Returns
    -------
    float
        Multiplicative factor (e.g., 1.05 = +5 %).
    """
    # Temperature adjustment: 0.1 % per degree from a 70 °F baseline
    temp_delta = temperature - 70.0
    temp_factor = 1.0 + (0.001 * temp_delta)

    # Wind adjustment: 0.1 % per mph for tail/head winds
    wind_factor = 1.0
    if wind_dir.lower() == "out":
        wind_factor += 0.001 * wind_speed
    elif wind_dir.lower() == "in":
        wind_factor -= 0.001 * wind_speed

    # Altitude adjustment: 2 % per 1,000 ft above sea level
    altitude_factor = 1.0 + 0.02 * (altitude / 1000.0)

    return temp_factor * wind_factor * altitude_factor


def format_percentage(value: float) -> str:
    """Format a float as a percent with one decimal place."""
    return f"{value * 100:.1f}%"


def compare_matchup(
    batter_name: str,
    pitcher_name: str,
    stadium: Optional[str] = None,
    temperature: Optional[float] = None,
    wind_speed: Optional[float] = None,
    wind_dir: Optional[str] = None,
    altitude: Optional[float] = None,
) -> Dict[str, object]:
    """Compute and return a structured comparison of a batter and pitcher.

    This high‑level function ties together player lookup, head‑to‑head
    aggregation, career stats retrieval and environmental adjustments.

    Parameters
    ----------
    batter_name : str
        Name of the hitter.
    pitcher_name : str
        Name of the pitcher.
    stadium : str, optional
        Name of the ballpark.  If supplied, dimensions and altitude will be
        looked up from the constants above.
    temperature : float, optional
        Game‑time temperature (°F).  Defaults to 70 °F if not provided.
    wind_speed : float, optional
        Wind speed (mph).  Defaults to 0.
    wind_dir : str, optional
        Direction of wind relative to home plate ("out" or "in").
    altitude : float, optional
        Override altitude (ft).  If not given and ``stadium`` is provided,
        altitude will be looked up in ``STADIUM_ALTITUDE``.

    Returns
    -------
    dict
        A nested dictionary containing results that can be printed or
        serialized.
    """
    batter_id, batter_full = search_player_id(batter_name)
    pitcher_id, pitcher_full = search_player_id(pitcher_name)

    h2h_stats = get_head_to_head_stats(batter_id, pitcher_id)
    batter_career = get_career_stat_line(batter_id, "hitting")
    pitcher_career = get_career_stat_line(pitcher_id, "pitching")

    # Ballpark info
    dims = None
    alt = 0.0
    if stadium:
        dims = STADIUM_DIMENSIONS.get(stadium)
        alt = STADIUM_ALTITUDE.get(stadium, 0.0)
    # Allow altitude override
    if altitude is not None:
        alt = altitude

    # Environmental factor
    t = temperature if temperature is not None else 70.0
    ws = wind_speed if wind_speed is not None else 0.0
    wd = wind_dir if wind_dir is not None else "none"
    env_factor = compute_environment_factor(t, ws, wd, alt)

    return {
        "batter": {
            "name": batter_full,
            "career": batter_career,
        },
        "pitcher": {
            "name": pitcher_full,
            "career": pitcher_career,
        },
        "matchup": {
            "at_bats": h2h_stats.at_bats,
            "hits": h2h_stats.hits,
            "doubles": h2h_stats.doubles,
            "triples": h2h_stats.triples,
            "home_runs": h2h_stats.home_runs,
            "strikeouts": h2h_stats.strikeouts,
            "walks": h2h_stats.walks,
            "batting_average": h2h_stats.batting_average,
            "on_base_percentage": h2h_stats.on_base_percentage,
            "slugging": h2h_stats.slugging,
            "ops": h2h_stats.ops,
        },
        "ballpark": {
            "name": stadium,
            "dimensions": dims,
            "altitude": alt,
        },
        "environment": {
            "temperature": t,
            "wind_speed": ws,
            "wind_direction": wd,
            "altitude": alt,
            "distance_factor": env_factor,
            "distance_factor_percent": format_percentage(env_factor - 1.0),
        },
    }


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Compare an MLB hitter and pitcher using head‑to‑head data, career stats "
            "and simple environmental adjustments."
        )
    )
    parser.add_argument("--batter", required=True, help="Name of the hitter (e.g., 'Mike Trout')")
    parser.add_argument("--pitcher", required=True, help="Name of the pitcher (e.g., 'Justin Verlander')")
    parser.add_argument("--stadium", help="Name of the ballpark (e.g., 'Coors Field')")
    parser.add_argument("--temperature", type=float, help="Game temperature in °F (default: 70)")
    parser.add_argument("--wind_speed", type=float, help="Wind speed in mph (default: 0)")
    parser.add_argument(
        "--wind_dir",
        choices=["out", "in", "none"],
        help="Wind direction relative to home plate: 'out' (tail), 'in' (head) or 'none' (default)",
    )
    parser.add_argument("--altitude", type=float, help="Override ballpark altitude in feet")
    parser.add_argument(
        "--json", action="store_true", help="Print raw JSON instead of a formatted summary"
    )
    args = parser.parse_args()

    result = compare_matchup(
        batter_name=args.batter,
        pitcher_name=args.pitcher,
        stadium=args.stadium,
        temperature=args.temperature,
        wind_speed=args.wind_speed,
        wind_dir=args.wind_dir,
        altitude=args.altitude,
    )

    if args.json:
        print(json.dumps(result, indent=2))
        return

    # Nicely format the result for human readers
    print(f"Hitter: {result['batter']['name']}")
    print(f"  Career slash line (AVG/OBP/SLG/OPS): "
          f"{result['batter']['career']['avg']}/"\
          f"{result['batter']['career']['obp']}/"\
          f"{result['batter']['career']['slg']}/"\
          f"{result['batter']['career']['ops']}")
    print(f"Pitcher: {result['pitcher']['name']}")
    print(f"  Career slash line against (AVG/OBP/SLG/OPS): "
          f"{result['pitcher']['career']['avg']}/"\
          f"{result['pitcher']['career']['obp']}/"\
          f"{result['pitcher']['career']['slg']}/"\
          f"{result['pitcher']['career']['ops']}")
    print("\nHead‑to‑Head Matchup:")
    if result['matchup']['at_bats'] == 0:
        print("  No recorded plate appearances between these players.")
    else:
        print(f"  At‑bats: {result['matchup']['at_bats']}")
        print(f"  Hits: {result['matchup']['hits']} "
              f"(2B: {result['matchup']['doubles']}, 3B: {result['matchup']['triples']}, HR: {result['matchup']['home_runs']})")
        print(f"  Walks: {result['matchup']['walks']}, Strikeouts: {result['matchup']['strikeouts']}")
        ba = result['matchup']['batting_average']
        obp = result['matchup']['on_base_percentage']
        slg = result['matchup']['slugging']
        ops = result['matchup']['ops']
        print(f"  Slash line vs pitcher (AVG/OBP/SLG/OPS): "
              f"{ba:.3f}/{obp:.3f}/{slg:.3f}/{ops:.3f}")

    if result['ballpark']['name']:
        dims = result['ballpark']['dimensions']
        print(f"\nBallpark: {result['ballpark']['name']}")
        if dims:
            print(f"  Dimensions (L‑C‑R): {dims[0]}‑{dims[1]}‑{dims[2]} ft")
        if result['ballpark']['altitude']:
            print(f"  Altitude: {result['ballpark']['altitude']} ft")

    env = result['environment']
    # Report environment adjustments only if non‑default
    if (
        env['temperature'] != 70 or env['wind_speed'] != 0 or env['altitude'] != 0
    ):
        print("\nEnvironmental factors:")
        print(f"  Temperature: {env['temperature']} °F")
        if env['wind_speed']:
            print(f"  Wind: {env['wind_speed']} mph ({env['wind_direction']})")
        if env['altitude']:
            print(f"  Altitude override: {env['altitude']} ft")
        print(f"  Estimated change in fly‑ball distance: {env['distance_factor_percent']}")


if __name__ == "__main__":
    main()


# ===============================
# Weather-Driven Hit Odds (NEW)
# ===============================
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import os
import math
import time
import json
import requests
from datetime import datetime, timedelta, timezone

try:
    from flask import Flask, jsonify, request
except Exception:
    # If Flask app exists elsewhere, we assume 'app' is defined.
    pass

# Reuse existing Flask app if present; otherwise create one
try:
    app
except NameError:
    app = Flask(__name__)

STATS_API = "https://statsapi.mlb.com/api/v1"
LIVE_API_FMT = "https://statsapi.mlb.com/api/v1.1/game/{gamePk}/feed/live"

# Optional local data files (park factors and player baselines)
PARK_FACTORS_PATH = os.environ.get("PARK_FACTORS_PATH", "data/park_factors.json")
PLAYER_BASELINES_PATH = os.environ.get("PLAYER_BASELINES_PATH", "data/player_baselines.csv")

# Caps per hit type
CAPS = {"single": 0.10, "double": 0.15, "triple": 0.20, "hr": 0.25}

# Stable weights (documented in model_notes per row)
WEIGHTS = {
    "temp_per_10F": {"single": 0.01, "double": 0.02, "triple": 0.03, "hr": 0.05},
    "wind_per_10mph_out": {"single": 0.00, "double": 0.02, "triple": 0.03, "hr": 0.08},
    "humidity_per_10pct": {"single": 0.00, "double": 0.00, "triple": 0.01, "hr": 0.02},
    "pressure_per_10hpa": {"single": -0.01, "double": -0.02, "triple": -0.03, "hr": -0.04},
    "precip_per_10pct": {"single": -0.01, "double": -0.01, "triple": -0.02, "hr": -0.02},
    "park": {"single": 1.0, "double": 1.0, "triple": 1.0, "hr": 1.0},  # multiplier (applied separately)
}

@dataclass
class Weather:
    temp_f: Optional[float]
    wind_mph: Optional[float]
    wind_direction_deg: Optional[int]
    humidity_pct: Optional[int]
    pressure_hpa: Optional[int]
    precip_chance_pct: Optional[int]

@dataclass
class ParkFactors:
    single: float
    double: float
    triple: float
    hr: float

def _load_park_factors() -> Dict[str, ParkFactors]:
    if not os.path.exists(PARK_FACTORS_PATH):
        return {}
    with open(PARK_FACTORS_PATH, "r", encoding="utf-8") as f:
        raw = json.load(f)
    pf: Dict[str, ParkFactors] = {}
    for stadium, vals in raw.items():
        pf[stadium] = ParkFactors(
            single=float(vals.get("single", 1.0)),
            double=float(vals.get("double", 1.0)),
            triple=float(vals.get("triple", 1.0)),
            hr=float(vals.get("hr", 1.0)),
        )
    return pf

def _load_player_baselines() -> Dict[int, Dict[str, float]]:
    # Expected CSV header: player_id,handedness,vs_throws,prob_1B,prob_2B,prob_3B,prob_HR,source,updated_at
    baselines: Dict[int, Dict[str, float]] = {}
    if not os.path.exists(PLAYER_BASELINES_PATH):
        return baselines
    import csv
    with open(PLAYER_BASELINES_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            try:
                pid = int(r["player_id"])
                vs = r.get("vs_throws","").upper()
                key = f"{pid}:{vs}"
                baselines[key] = {
                    "single": float(r.get("prob_1B","0")),
                    "double": float(r.get("prob_2B","0")),
                    "triple": float(r.get("prob_3B","0")),
                    "hr": float(r.get("prob_HR","0")),
                    "updated_at": r.get("updated_at"),
                    "source": r.get("source", ""),
                }
            except Exception:
                continue
    return baselines

PARK_FACTORS = _load_park_factors()
PLAYER_BASELINES = _load_player_baselines()

def iso_now_z() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00","Z")

def clamp(value: float, min_v: float, max_v: float) -> float:
    return max(min_v, min(value, max_v))

def _cosine_against_cf(wind_dir_deg: Optional[int]) -> float:
    # CF out is 0° reference (blowing out), in degrees; if unknown, return 0 component
    if wind_dir_deg is None:
        return 0.0
    # Map deg such that 0 means out to CF; 180 means in from CF
    # We'll assume 0 is to CF; use cos in radians
    rad = math.radians(wind_dir_deg % 360)
    return math.cos(rad)

def compute_lifts(weather: Weather, park: ParkFactors, roof_status: str) -> Tuple[Dict[str,float], bool, str]:
    notes = []
    uncertain = False
    lifts: Dict[str, float] = {"single":0.0,"double":0.0,"triple":0.0,"hr":0.0}

    if roof_status == "closed":
        notes.append("Roof closed → neutralized weather effects.")
        return lifts, False, "; ".join(notes)

    # Components
    # Temperature vs 70F
    if weather.temp_f is not None:
        temp_delta = (weather.temp_f - 70.0)/10.0
        for k in lifts:
            lifts[k] += WEIGHTS["temp_per_10F"][k] * temp_delta
        notes.append(f"Temp Δ vs 70F: {weather.temp_f:.0f}F → {temp_delta:+.2f} * weights")
    else:
        uncertain = True
        notes.append("Missing temp")

    # Wind toward CF (cosine)
    if weather.wind_mph is not None and weather.wind_direction_deg is not None:
        eff = (weather.wind_mph/10.0) * _cosine_against_cf(weather.wind_direction_deg)
        for k in lifts:
            lifts[k] += WEIGHTS["wind_per_10mph_out"][k] * eff
        notes.append(f"Wind: {weather.wind_mph:.0f} mph @ {weather.wind_direction_deg}° → eff {eff:+.2f}")
    else:
        uncertain = True
        notes.append("Missing wind")

    # Humidity
    if weather.humidity_pct is not None:
        hum = (weather.humidity_pct - 50)/10.0
        for k in lifts:
            lifts[k] += WEIGHTS["humidity_per_10pct"][k] * hum
        notes.append(f"Humidity: {weather.humidity_pct}%")
    else:
        uncertain = True
        notes.append("Missing humidity")

    # Pressure
    if weather.pressure_hpa is not None:
        press = (weather.pressure_hpa - 1013)/10.0
        for k in lifts:
            lifts[k] += WEIGHTS["pressure_per_10hpa"][k] * press
        notes.append(f"Pressure: {weather.pressure_hpa} hPa")
    else:
        uncertain = True
        notes.append("Missing pressure")

    # Precip chance
    if weather.precip_chance_pct is not None:
        prec = weather.precip_chance_pct/10.0
        for k in lifts:
            lifts[k] += WEIGHTS["precip_per_10pct"][k] * prec
        notes.append(f"Precip: {weather.precip_chance_pct}%")
    else:
        uncertain = True
        notes.append("Missing precip")

    # Apply caps
    for k,v in list(lifts.items()):
        lifts[k] = clamp(v, -CAPS[k], CAPS[k])

    # Park factor note (multiplier applied later)
    notes.append(f"Park factors (x): 1B {park.single:.2f}, 2B {park.double:.2f}, 3B {park.triple:.2f}, HR {park.hr:.2f}")

    return lifts, uncertain, "; ".join(notes)

def apply_lifts_and_renorm(base: Dict[str,float], lifts: Dict[str,float], park: ParkFactors) -> Dict[str,float]:
    # Multiplicative lift -> capped previously, then park multiplier
    adj = {}
    for k in ["single","double","triple","hr"]:
        b = base.get(k, 0.0)
        # Guard against absurd baselines; cap at 0.5 per type
        b = clamp(b, 0.0, 0.5)
        adj[k] = b * (1.0 + lifts[k])
    # Park multipliers
    adj["single"] *= park.single
    adj["double"] *= park.double
    adj["triple"] *= park.triple
    adj["hr"] *= park.hr
    # Renormalize if sum > 0.9
    s = sum(adj.values())
    scale = 1.0
    if s > 0.9:
        scale = 0.9 / s
    for k in adj:
        adj[k] *= scale
    adj["no_hit"] = 1.0 - sum(adj.values())
    # Guard range
    for k in adj:
        adj[k] = float(clamp(adj[k], 0.0, 1.0))
    return adj

def _fetch_schedule_today() -> List[Dict]:
    # Use UTC today, we will filter by stadium-local date later
    today = datetime.now(timezone.utc).date().isoformat()
    params = {"sportId":1,"date": today}
    r = requests.get(f"{STATS_API}/schedule", params=params, timeout=20)
    r.raise_for_status()
    data = r.json()
    games = []
    for d in data.get("dates", []):
        for g in d.get("games", []):
            if g.get("status",{}).get("abstractGameCode") in ("P","F","O","C"):  # P: Postponed, F: Final etc.
                # We'll include S for scheduled only
                pass
            games.append(g)
    return games

def _fetch_live(game_pk: int) -> Dict:
    r = requests.get(LIVE_API_FMT.format(gamePk=game_pk), timeout=20)
    r.raise_for_status()
    return r.json()

def _extract_confirmed_lineups(live: Dict) -> Tuple[List[Dict], List[Dict]]:
    box = live.get("liveData", {}).get("boxscore", {}).get("teams", {})
    home = box.get("home", {})
    away = box.get("away", {})
    def team_players(t):
        batting_order = t.get("battingOrder", [])
        players = t.get("players", {})
        starters = []
        if batting_order and len(batting_order) >= 9:
            order_map = {int(o): i+1 for i,o in enumerate(sorted([int(x) for x in batting_order[:9]]))}
            for pid_key, pdata in players.items():
                try:
                    order = pdata.get("battingOrder")
                    if order is None: 
                        continue
                    ord_int = int(order)
                    if ord_int in order_map:
                        starters.append({
                            "player_id": int(pdata.get("person",{}).get("id")),
                            "player_name": pdata.get("person",{}).get("fullName"),
                            "batting_order": order_map[ord_int],
                            "position": pdata.get("position",{}).get("abbreviation"),
                            "bats": pdata.get("stats",{}).get("batting",{}).get("batSide",{}).get("code") or pdata.get("person",{}).get("batSide",{}).get("code")
                        })
                except Exception:
                    continue
        return starters
    return team_players(away), team_players(home)

def _opponent_pitcher_info(live: Dict, home_away: str) -> Tuple[str,str]:
    # 'home_away' indicates the hitter's team location; opposing pitcher is other team probable/starting
    game_data = live.get("gameData", {})
    probable = game_data.get("probablePitchers", {})
    # Fallback to startingPitcher in boxscore if available
    box = live.get("liveData", {}).get("boxscore", {}).get("teams", {})
    opp_key = "home" if home_away=="away" else "away"
    opp_box = box.get(opp_key, {})
    # Try boxscore first for confirmed start
    try:
        sp = opp_box.get("pitchers", [])
        if sp:
            pitcher_id = sp[0]
            players = opp_box.get("players", {})
            pdata = players.get(f"ID{pitcher_id}", {})
            name = pdata.get("person",{}).get("fullName", "")
            throws = pdata.get("person",{}).get("pitchHand",{}).get("code", "")
            if name:
                return name, (throws or "").upper()[:1] or "R"
    except Exception:
        pass
    # Probable
    key = "home" if home_away=="away" else "away"
    p = probable.get(key, {})
    name = p.get("fullName") or p.get("lastFirstName") or ""
    throws = p.get("pitchHand",{}).get("code","")
    return name, (throws or "R").upper()[:1]

def _venue_meta(live: Dict) -> Tuple[str, str, Optional[float], Optional[float], str, str, Optional[str]]:
    gd = live.get("gameData", {})
    venue = gd.get("venue", {})
    stadium = venue.get("name","")
    tz = gd.get("datetime",{}).get("timeZone",{}).get("id") or gd.get("venue",{}).get("timeZone",{}).get("id") or "America/New_York"
    city = venue.get("location",{}).get("city","")
    state = venue.get("location",{}).get("state","")
    roof = venue.get("roofType","unknown").lower()
    coords = venue.get("location",{})
    lat = coords.get("latitude")
    lon = coords.get("longitude")
    return stadium, tz, float(lat) if lat else None, float(lon) if lon else None, city, state, roof

def _game_local_start(g: Dict, tz: str) -> str:
    # Convert gameDate to tz
    gd = g.get("gameDate")
    if not gd:
        return ""
    dt = datetime.fromisoformat(gd.replace("Z","+00:00"))
    try:
        # No pytz, manual offset string
        return dt.astimezone(timezone.utc).astimezone().isoformat()
    except Exception:
        return dt.isoformat()

def _is_postponed(g: Dict) -> bool:
    code = g.get("status",{}).get("abstractGameCode","")
    detailed = g.get("status",{}).get("detailedState","")
    if code in ("P",): return True
    if "Postponed" in detailed or "Suspended" in detailed or "Relocated" in detailed:
        return True
    return False

def _get_open_meteo(lat: float, lon: float, iso_start_local: str) -> Weather:
    # We request hourly forecast for the date of start
    # Derive date/time
    try:
        start_dt = datetime.fromisoformat(iso_start_local)
    except Exception:
        return Weather(None,None,None,None,None,None)
    date_str = start_dt.date().isoformat()
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "temperature_2m,relative_humidity_2m,pressure_msl,precipitation_probability,wind_speed_10m,wind_direction_10m",
        "timezone": "auto",
        "start_date": date_str,
        "end_date": date_str
    }
    r = requests.get("https://api.open-meteo.com/v1/forecast", params=params, timeout=20)
    r.raise_for_status()
    data = r.json()
    hourly = data.get("hourly", {})
    times = hourly.get("time", [])
    if not times:
        return Weather(None,None,None,None,None,None)
    # pick the closest hour to start
    target = start_dt.replace(minute=0, second=0, microsecond=0)
    idx = None
    best = None
    for i, t in enumerate(times):
        dt = datetime.fromisoformat(t)
        diff = abs((dt - target).total_seconds())
        if best is None or diff < best:
            best = diff; idx = i
    if idx is None:
        return Weather(None,None,None,None,None,None)
    try:
        return Weather(
            temp_f = float(hourly["temperature_2m"][idx]) * 9/5 + 32,
            wind_mph = float(hourly["wind_speed_10m"][idx]) * 0.621371,
            wind_direction_deg = int(hourly["wind_direction_10m"][idx]),
            humidity_pct = int(hourly["relative_humidity_2m"][idx]),
            pressure_hpa = int(hourly["pressure_msl"][idx]),
            precip_chance_pct = int(hourly["precipitation_probability"][idx])
        )
    except Exception:
        return Weather(None,None,None,None,None,None)

def _baseline_for(pid: int, vs_throws: str) -> Optional[Dict[str,float]]:
    rec = PLAYER_BASELINES.get(f"{pid}:{vs_throws.upper()}", None) or PLAYER_BASELINES.get(f"{pid}:ANY", None)
    return rec

@app.get("/api/weather_hit_odds")
def api_weather_hit_odds():
    # Build list for "stadium-local today" using MLB schedule + live endpoints
    try:
        games = _fetch_schedule_today()
    except Exception as e:
        return jsonify({"error":"schedule_fetch_failed","detail":str(e)}), 502

    out: List[Dict] = []
    any_uncertain_weather = False

    for g in games:
        if _is_postponed(g):
            continue
        game_pk = g.get("gamePk")
        try:
            live = _fetch_live(game_pk)
        except Exception:
            continue

        # Venue meta + roof
        stadium, tz, lat, lon, city, state, roof = _venue_meta(live)
        roof = roof if roof in ("open","closed","unknown") else "unknown"

        # Doubleheader game number
        game_number = g.get("gameNumber") or 1
        gid = f"{g.get('officialDate','')}-{g.get('teams',{}).get('away',{}).get('team',{}).get('abbreviation','')}-{g.get('teams',{}).get('home',{}).get('team',{}).get('abbreviation','')}-Gm{game_number}"

        # Confirmed lineups
        away_starters, home_starters = _extract_confirmed_lineups(live)
        if not away_starters and not home_starters:
            # No confirmed lineups yet for this game
            continue

        # Opposing pitchers
        away_pitcher_name, away_pitcher_throws = _opponent_pitcher_info(live, home_away="away")
        home_pitcher_name, home_pitcher_throws = _opponent_pitcher_info(live, home_away="home")

        # Local start time
        game_start_local = _game_local_start(g, tz)

        # Park factors
        pf = PARK_FACTORS.get(stadium)
        if not pf:
            # Required feed missing -> skip with data_unavailable reason (game-level)
            continue

        # Weather
        weather = None
        uncertain_weather = False
        if roof == "closed":
            weather = Weather(None,None,None,None,None,None)
        else:
            if lat is not None and lon is not None and game_start_local:
                try:
                    weather = _get_open_meteo(lat, lon, game_start_local)
                    # If any weather element missing -> uncertain
                    if None in (weather.temp_f, weather.wind_mph, weather.wind_direction_deg, weather.humidity_pct, weather.pressure_hpa, weather.precip_chance_pct):
                        uncertain_weather = True
                except Exception:
                    uncertain_weather = True
                    weather = Weather(None,None,None,None,None,None)
            else:
                uncertain_weather = True
                weather = Weather(None,None,None,None,None,None)

        # Helpers
        def build_row(p: Dict, team_side: str, opp_pitcher_name: str, opp_pitcher_throws: str):
            # Baseline must exist (required), otherwise exclude
            baseline = _baseline_for(p["player_id"], opp_pitcher_throws)
            if not baseline: 
                return None, "missing_baseline"
            # Compute lifts
            lifts, more_uncertain, notes = compute_lifts(weather, pf, roof)
            adj = apply_lifts_and_renorm(baseline, lifts, pf)
            # Confidence (simple deterministic score)
            conf = 0.9  # confirmed lineup
            if uncertain_weather or more_uncertain:
                conf *= 0.85
            # baseline recency
            if not baseline.get("updated_at"):
                conf *= 0.9
            # guard
            conf = float(clamp(conf, 0.0, 1.0))

            rec = {
                "player_id": f"mlb-{p['player_id']}",
                "player_name": p["player_name"],
                "team": g.get("teams",{}).get(team_side,{}).get("team",{}).get("abbreviation",""),
                "opponent": g.get("teams",{}).get("home" if team_side=="away" else "away",{}).get("team",{}).get("abbreviation",""),
                "game_id": gid,
                "stadium": stadium,
                "game_start_local": game_start_local,
                "batting_order": p["batting_order"],
                "bats": (p.get("bats") or "").upper()[:1] or "R",
                "opp_pitcher": opp_pitcher_name,
                "opp_pitcher_throws": opp_pitcher_throws,
                "lineup_status": "confirmed",
                "roof_status": roof,
                "weather": {
                    "temp_f": weather.temp_f,
                    "wind_mph": weather.wind_mph,
                    "wind_direction_deg": weather.wind_direction_deg,
                    "humidity_pct": weather.humidity_pct,
                    "pressure_hpa": weather.pressure_hpa,
                    "precip_chance_pct": weather.precip_chance_pct
                },
                "park_factors": {"single": pf.single, "double": pf.double, "triple": pf.triple, "hr": pf.hr},
                "probabilities": adj,
                "confidence": conf,
                "uncertain_weather": bool(uncertain_weather or more_uncertain or roof == "unknown"),
                "last_updated": iso_now_z(),
                "model_notes": notes,
                "data_unavailable": None
            }
            return rec, None

        # Assemble for each starter
        for p in away_starters:
            row, reason = build_row(p, "away", home_pitcher_name, home_pitcher_throws or "R")
            if row: out.append(row)

        for p in home_starters:
            row, reason = build_row(p, "home", away_pitcher_name, away_pitcher_throws or "R")
            if row: out.append(row)

    # Deduplicate by player_id + game_id (keep latest)
    dedup: Dict[Tuple[str,str], Dict] = {}
    for r in out:
        key = (r["player_id"], r["game_id"])
        if key not in dedup or dedup[key]["last_updated"] < r["last_updated"]:
            dedup[key] = r

    # Sort by HR probability desc by default
    rows = sorted(dedup.values(), key=lambda x: x["probabilities"].get("hr",0.0), reverse=True)

    return jsonify(rows)




# ===============================
# Weather-Driven Hit Odds (NEW)
# ===============================
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import os
import math
import json
import requests
from datetime import datetime, timezone
from flask import Flask, jsonify

# Reuse existing app if defined; otherwise create one.
try:
    app  # type: ignore
except NameError:  # pragma: no cover
    app = Flask(__name__)

STATS_API = "https://statsapi.mlb.com/api/v1"
LIVE_API_FMT = "https://statsapi.mlb.com/api/v1.1/game/{gamePk}/feed/live"

PARK_FACTORS_PATH = os.environ.get("PARK_FACTORS_PATH", "data/park_factors.json")
PLAYER_BASELINES_PATH = os.environ.get("PLAYER_BASELINES_PATH", "data/player_baselines.csv")

CAPS = {"single": 0.10, "double": 0.15, "triple": 0.20, "hr": 0.25}

WEIGHTS = {
    "temp_per_10F": {"single": 0.01, "double": 0.02, "triple": 0.03, "hr": 0.05},
    "wind_per_10mph_out": {"single": 0.00, "double": 0.02, "triple": 0.03, "hr": 0.08},
    "humidity_per_10pct": {"single": 0.00, "double": 0.00, "triple": 0.01, "hr": 0.02},
    "pressure_per_10hpa": {"single": -0.01, "double": -0.02, "triple": -0.03, "hr": -0.04},
    "precip_per_10pct": {"single": -0.01, "double": -0.01, "triple": -0.02, "hr": -0.02},
}

@dataclass
class Weather:
    temp_f: Optional[float]
    wind_mph: Optional[float]
    wind_direction_deg: Optional[int]
    humidity_pct: Optional[int]
    pressure_hpa: Optional[int]
    precip_chance_pct: Optional[int]

@dataclass
class ParkFactors:
    single: float
    double: float
    triple: float
    hr: float

def iso_now_z() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00","Z")

def clamp(value: float, min_v: float, max_v: float) -> float:
    return max(min_v, min(value, max_v))

def _load_park_factors() -> Dict[str, ParkFactors]:
    if not os.path.exists(PARK_FACTORS_PATH):
        return {}
    with open(PARK_FACTORS_PATH, "r", encoding="utf-8") as f:
        raw = json.load(f)
    out = {}
    for stadium, vals in raw.items():
        out[stadium] = ParkFactors(
            single=float(vals.get("single", 1.0)),
            double=float(vals.get("double", 1.0)),
            triple=float(vals.get("triple", 1.0)),
            hr=float(vals.get("hr", 1.0)),
        )
    return out

def _load_player_baselines() -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    if not os.path.exists(PLAYER_BASELINES_PATH):
        return out
    import csv
    with open(PLAYER_BASELINES_PATH, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            try:
                pid = int(r["player_id"])
                vs = (r.get("vs_throws") or "ANY").upper()
                out[f"{pid}:{vs}"] = {
                    "single": float(r.get("prob_1B","0")),
                    "double": float(r.get("prob_2B","0")),
                    "triple": float(r.get("prob_3B","0")),
                    "hr": float(r.get("prob_HR","0")),
                    "updated_at": r.get("updated_at",""),
                    "source": r.get("source",""),
                }
            except Exception:
                continue
    return out

PARK_FACTORS_CACHE = _load_park_factors()
PLAYER_BASELINES_CACHE = _load_player_baselines()

def _cosine_against_cf(wind_dir_deg: Optional[int]) -> float:
    if wind_dir_deg is None:
        return 0.0
    return math.cos(math.radians(wind_dir_deg % 360))

def compute_lifts(weather: Weather, park: ParkFactors, roof_status: str):
    notes = []
    uncertain = False
    lifts = {"single":0.0,"double":0.0,"triple":0.0,"hr":0.0}
    if roof_status == "closed":
        notes.append("Roof closed → neutralized weather effects.")
        return lifts, False, "; ".join(notes)

    if weather.temp_f is not None:
        temp_delta = (weather.temp_f - 70.0)/10.0
        for k in lifts: lifts[k] += WEIGHTS["temp_per_10F"][k] * temp_delta
        notes.append(f"Temp Δ vs 70F: {weather.temp_f:.0f}F")
    else:
        uncertain = True; notes.append("Missing temp")

    if weather.wind_mph is not None and weather.wind_direction_deg is not None:
        eff = (weather.wind_mph/10.0) * _cosine_against_cf(weather.wind_direction_deg)
        for k in lifts: lifts[k] += WEIGHTS["wind_per_10mph_out"][k] * eff
        notes.append(f"Wind: {weather.wind_mph:.0f} mph @ {weather.wind_direction_deg}°")
    else:
        uncertain = True; notes.append("Missing wind")

    if weather.humidity_pct is not None:
        hum = (weather.humidity_pct - 50)/10.0
        for k in lifts: lifts[k] += WEIGHTS["humidity_per_10pct"][k] * hum
        notes.append(f"Humidity: {weather.humidity_pct}%")
    else:
        uncertain = True; notes.append("Missing humidity")

    if weather.pressure_hpa is not None:
        press = (weather.pressure_hpa - 1013)/10.0
        for k in lifts: lifts[k] += WEIGHTS["pressure_per_10hpa"][k] * press
        notes.append(f"Pressure: {weather.pressure_hpa} hPa")
    else:
        uncertain = True; notes.append("Missing pressure")

    if weather.precip_chance_pct is not None:
        prec = weather.precip_chance_pct/10.0
        for k in lifts: lifts[k] += WEIGHTS["precip_per_10pct"][k] * prec
        notes.append(f"Precip: {weather.precip_chance_pct}%")
    else:
        uncertain = True; notes.append("Missing precip")

    for k in list(lifts.keys()):
        lo, hi = -CAPS[k], CAPS[k]
        lifts[k] = clamp(lifts[k], lo, hi)

    notes.append(f"Park factors (x): 1B {park.single:.2f}, 2B {park.double:.2f}, 3B {park.triple:.2f}, HR {park.hr:.2f}")
    return lifts, uncertain, "; ".join(notes)

def apply_lifts_and_renorm(base: Dict[str,float], lifts: Dict[str,float], park: ParkFactors):
    adj = {}
    for k in ("single","double","triple","hr"):
        b = clamp(float(base.get(k,0.0)), 0.0, 0.5)
        adj[k] = b * (1.0 + float(lifts.get(k,0.0)))
    adj["single"] *= park.single
    adj["double"] *= park.double
    adj["triple"] *= park.triple
    adj["hr"] *= park.hr
    s = sum(adj.values())
    if s > 0.9:
        scale = 0.9 / s
        for k in adj: adj[k] *= scale
    adj["no_hit"] = 1.0 - sum(adj.values())
    for k in adj: adj[k] = float(clamp(adj[k], 0.0, 1.0))
    return adj

def _fetch_schedule_today() -> List[Dict]:
    today = datetime.now(timezone.utc).date().isoformat()
    r = requests.get(f"{STATS_API}/schedule", params={"sportId":1, "date": today}, timeout=20)
    r.raise_for_status()
    data = r.json()
    games = []
    for d in data.get("dates", []):
        games.extend(d.get("games", []))
    return games

def _fetch_live(game_pk: int) -> Dict:
    r = requests.get(LIVE_API_FMT.format(gamePk=game_pk), timeout=20)
    r.raise_for_status()
    return r.json()

def _extract_confirmed_lineups(live: Dict):
    box = live.get("liveData", {}).get("boxscore", {}).get("teams", {})
    def team_players(t):
        batting_order = t.get("battingOrder", [])
        players = t.get("players", {})
        starters = []
        if batting_order and len(batting_order) >= 9:
            order_map = {int(o): i+1 for i,o in enumerate(sorted([int(x) for x in batting_order[:9]]))}
            for pid_key, pdata in players.items():
                order = pdata.get("battingOrder")
                if order is None: 
                    continue
                ord_int = int(order)
                if ord_int in order_map:
                    starters.append({
                        "player_id": int(pdata.get("person",{}).get("id")),
                        "player_name": pdata.get("person",{}).get("fullName"),
                        "batting_order": order_map[ord_int],
                        "position": pdata.get("position",{}).get("abbreviation"),
                        "bats": pdata.get("stats",{}).get("batting",{}).get("batSide",{}).get("code") or pdata.get("person",{}).get("batSide",{}).get("code")
                    })
        return starters
    home = team_players(box.get("home", {}))
    away = team_players(box.get("away", {}))
    return away, home

def _opponent_pitcher_info(live: Dict, home_away: str):
    gd = live.get("gameData", {})
    probable = gd.get("probablePitchers", {})
    box = live.get("liveData", {}).get("boxscore", {}).get("teams", {})
    opp_key = "home" if home_away=="away" else "away"
    opp_box = box.get(opp_key, {})
    try:
        sp = opp_box.get("pitchers", [])
        if sp:
            pitcher_id = sp[0]
            players = opp_box.get("players", {})
            pdata = players.get(f"ID{pitcher_id}", {})
            name = pdata.get("person",{}).get("fullName", "")
            throws = pdata.get("person",{}).get("pitchHand",{}).get("code", "")
            if name:
                return name, (throws or "").upper()[:1] or "R"
    except Exception:
        pass
    key = "home" if home_away=="away" else "away"
    p = probable.get(key, {})
    name = p.get("fullName") or p.get("lastFirstName") or ""
    throws = p.get("pitchHand",{}).get("code","")
    return name, (throws or "R").upper()[:1]

def _venue_meta(live: Dict):
    gd = live.get("gameData", {})
    venue = gd.get("venue", {})
    stadium = venue.get("name","")
    tz = gd.get("datetime",{}).get("timeZone",{}).get("id") or gd.get("venue",{}).get("timeZone",{}).get("id") or "America/New_York"
    coords = venue.get("location",{})
    lat = coords.get("latitude"); lon = coords.get("longitude")
    roof = (venue.get("roofType") or "unknown").lower()
    return stadium, tz, (float(lat) if lat else None), (float(lon) if lon else None), roof

from zoneinfo import ZoneInfo

def _game_local_start(g: Dict, tz: str) -> str:
    gd = g.get("gameDate")
    if not gd: return ""
    dt_utc = datetime.fromisoformat(gd.replace("Z","+00:00"))
    try: z = ZoneInfo(tz)
    except Exception: z = ZoneInfo("America/New_York")
    return dt_utc.astimezone(z).isoformat()

def _is_today_in_tz(g: Dict, tz: str) -> bool:
    iso = _game_local_start(g, tz)
    if not iso: return False
    dt = datetime.fromisoformat(iso)
    now_tz = datetime.now(ZoneInfo(tz))
    return dt.date() == now_tz.date()

def _is_postponed(g: Dict) -> bool:
    code = g.get("status",{}).get("abstractGameCode","")
    detailed = g.get("status",{}).get("detailedState","")
    if code == "P": return True
    if "Postponed" in detailed or "Suspended" in detailed or "Relocated" in detailed:
        return True
    return False

def _get_open_meteo(lat: float, lon: float, iso_start_local: str) -> Weather:
    try:
        start_dt = datetime.fromisoformat(iso_start_local)
    except Exception:
        return Weather(None,None,None,None,None,None)
    date_str = start_dt.date().isoformat()
    r = requests.get("https://api.open-meteo.com/v1/forecast", params={
        "latitude": lat,
        "longitude": lon,
        "hourly": "temperature_2m,relative_humidity_2m,pressure_msl,precipitation_probability,wind_speed_10m,wind_direction_10m",
        "timezone": "auto",
        "start_date": date_str,
        "end_date": date_str
    }, timeout=20)
    r.raise_for_status()
    data = r.json()
    hourly = data.get("hourly", {})
    times = hourly.get("time", [])
    if not times:
        return Weather(None,None,None,None,None,None)
    target = start_dt.replace(minute=0, second=0, microsecond=0)
    idx, best = None, None
    for i,t in enumerate(times):
        dt = datetime.fromisoformat(t)
        diff = abs((dt-target).total_seconds())
        if best is None or diff<best: best, idx = diff, i
    if idx is None:
        return Weather(None,None,None,None,None,None)
    try:
        return Weather(
            temp_f = float(hourly["temperature_2m"][idx]) * 9/5 + 32,
            wind_mph = float(hourly["wind_speed_10m"][idx]) * 0.621371,
            wind_direction_deg = int(hourly["wind_direction_10m"][idx]),
            humidity_pct = int(hourly["relative_humidity_2m"][idx]),
            pressure_hpa = int(hourly["pressure_msl"][idx]),
            precip_chance_pct = int(hourly["precipitation_probability"][idx])
        )
    except Exception:
        return Weather(None,None,None,None,None,None)

def _baseline_for(pid: int, vs_throws: str) -> Optional[Dict[str,float]]:
    key = f"{pid}:{vs_throws.upper()}"
    return PLAYER_BASELINES_CACHE.get(key) or PLAYER_BASELINES_CACHE.get(f"{pid}:ANY")

@app.get("/api/weather_hit_odds")
def api_weather_hit_odds():
    try:
        games = _fetch_schedule_today()
    except Exception as e:
        return jsonify({"error":"schedule_fetch_failed","detail":str(e)}), 502

    out: List[Dict] = []

    for g in games:
        if _is_postponed(g):
            continue
        try:
            live = _fetch_live(g.get("gamePk"))
        except Exception:
            continue

        stadium, tz, lat, lon, roof = _venue_meta(live)
        if not _is_today_in_tz(g, tz):
            continue

        game_number = g.get("gameNumber") or 1
        gid = f"{g.get('officialDate','')}-{g.get('teams',{}).get('away',{}).get('team',{}).get('abbreviation','')}-{g.get('teams',{}).get('home',{}).get('team',{}).get('abbreviation','')}-Gm{game_number}"

        away_starters, home_starters = _extract_confirmed_lineups(live)
        if not away_starters and not home_starters:
            continue  # Awaiting confirmed lineup

        away_pitcher_name, away_pitcher_throws = _opponent_pitcher_info(live, home_away="away")
        home_pitcher_name, home_pitcher_throws = _opponent_pitcher_info(live, home_away="home")

        game_start_local = _game_local_start(g, tz)

        pf = PARK_FACTORS_CACHE.get(stadium)
        if not pf:
            continue  # Required input missing

        if roof == "closed":
            weather = Weather(None,None,None,None,None,None)
            uncertain_weather = False
        else:
            if lat is not None and lon is not None and game_start_local:
                try:
                    weather = _get_open_meteo(lat, lon, game_start_local)
                    uncertain_weather = any(v is None for v in [
                        weather.temp_f, weather.wind_mph, weather.wind_direction_deg,
                        weather.humidity_pct, weather.pressure_hpa, weather.precip_chance_pct
                    ]) or (roof == "unknown")
                except Exception:
                    weather = Weather(None,None,None,None,None,None)
                    uncertain_weather = True
            else:
                weather = Weather(None,None,None,None,None,None)
                uncertain_weather = True

        def build_row(p: Dict, side: str, opp_name: str, opp_throws: str):
            baseline = _baseline_for(p["player_id"], opp_throws or "R")
            if not baseline:
                return None
            lifts, more_uncertain, notes = compute_lifts(weather, pf, roof)
            adj = apply_lifts_and_renorm(baseline, lifts, pf)
            conf = 0.90
            if uncertain_weather or more_uncertain:
                conf *= 0.85
            if not baseline.get("updated_at"):
                conf *= 0.90
            rec = {
                "player_id": f"mlb-{p['player_id']}",
                "player_name": p["player_name"],
                "team": g.get("teams",{}).get(side,{}).get("team",{}).get("abbreviation",""),
                "opponent": g.get("teams",{}).get("home" if side=="away" else "away",{}).get("team",{}).get("abbreviation",""),
                "game_id": gid,
                "stadium": stadium,
                "game_start_local": game_start_local,
                "batting_order": p["batting_order"],
                "bats": (p.get("bats") or "").upper()[:1] or "R",
                "opp_pitcher": opp_name,
                "opp_pitcher_throws": opp_throws or "R",
                "lineup_status": "confirmed",
                "roof_status": roof,
                "weather": {
                    "temp_f": weather.temp_f,
                    "wind_mph": weather.wind_mph,
                    "wind_direction_deg": weather.wind_direction_deg,
                    "humidity_pct": weather.humidity_pct,
                    "pressure_hpa": weather.pressure_hpa,
                    "precip_chance_pct": weather.precip_chance_pct
                },
                "park_factors": {
                    "single": pf.single, "double": pf.double, "triple": pf.triple, "hr": pf.hr
                },
                "probabilities": adj,
                "confidence": float(clamp(conf, 0.0, 1.0)),
                "uncertain_weather": bool(uncertain_weather or more_uncertain),
                "last_updated": iso_now_z(),
                "model_notes": notes,
                "data_unavailable": None
            }
            return rec

        for p in away_starters:
            row = build_row(p, "away", home_pitcher_name, home_pitcher_throws)
            if row: out.append(row)
        for p in home_starters:
            row = build_row(p, "home", away_pitcher_name, away_pitcher_throws)
            if row: out.append(row)

    dedup = {}
    for r in out:
        key = (r["player_id"], r["game_id"])
        if key not in dedup or dedup[key]["last_updated"] < r["last_updated"]:
            dedup[key] = r

    rows = sorted(dedup.values(), key=lambda x: x["probabilities"].get("hr",0.0), reverse=True)
    return jsonify(rows)
