# synthetic_routes.py
# Generate many realistic-ish flight routes OFFLINE (no APIs).
# Output: synthetic_routes.csv with columns similar to OpenSky:
# timestamp, icao24, callsign, origin_country, latitude, longitude,
# baro_altitude (m), velocity (m/s), heading (deg), vertical_rate (m/s), on_ground (bool)

import math
import random
from typing import Tuple, List
import numpy as np
import pandas as pd
from datetime import datetime, timezone

# ---------------- Config ----------------
SEED = 42
random.seed(SEED); np.random.seed(SEED)

NUM_ROUTES        = 400          # how many routes to create
SAMPLE_EVERY_SEC  = 5             # sampling interval
DURATION_SEC_MIN  = 8*60          # per-route duration range
DURATION_SEC_MAX  = 15*60

# Geographic area (center + radius).
AREA_CENTER_LAT   = 32.0          # e.g., Israel-ish
AREA_CENTER_LON   = 34.8
AREA_RADIUS_KM    = 120.0         # routes kept roughly within this radius

# Cruise altitude & speeds (drawn per route; will vary during phases)
CRUISE_ALT_FT_RANGE = (30000, 38000)   # target cruise level
# Ground speed (knots) targets per phase; jittered per route
CLIMB_GS_KT_RANGE   = (220, 280)
CRUISE_GS_KT_RANGE  = (430, 490)
DESCENT_GS_KT_RANGE = (220, 280)

# Vertical speed (feet per minute) targets per phase; jittered
CLIMB_VS_FPM_RANGE  = (1200, 2500)
DESCENT_VS_FPM_RANGE= (800, 1800)      # magnitude; sign negative in descent

# Smoothing / turn dynamics
HEADING_CHANGE_DEG_PER_MIN = (2.0, 8.0)  # slow wandering
HEADING_NOISE_DEG          = 1.0

# Output CSV
OUT_CSV = "synthetic_routes_10000.csv"

# --------------- Helpers ----------------
LAT_KM = 111.0
def lon_km(lat_deg: float) -> float:
    return LAT_KM * max(math.cos(math.radians(lat_deg)), 1e-6)

def offset_latlon(lat0: float, lon0: float, dx_km: float, dy_km: float) -> Tuple[float, float]:
    """Move from (lat0, lon0) by dx,dy in km (east, north)."""
    dlat = dy_km / LAT_KM
    dlon = dx_km / lon_km(lat0)
    return lat0 + dlat, lon0 + dlon

def rand_in_disc(center_lat: float, center_lon: float, radius_km: float) -> Tuple[float, float]:
    """Random point in disc ~ uniform by radius."""
    r = radius_km * math.sqrt(random.random())
    theta = random.random() * 2*math.pi
    dx, dy = r * math.cos(theta), r * math.sin(theta)
    return offset_latlon(center_lat, center_lon, dx, dy)

def kt_to_mps(knots: float) -> float:
    return knots * 0.514444

def fpm_to_mps(fpm: float) -> float:
    return fpm * 0.00508

def ft_to_m(ft: float) -> float:
    return ft * 0.3048

def now_epoch() -> int:
    return int(datetime.now(timezone.utc).timestamp())

# ------------- Route synthesis ----------
def synthesize_one_route() -> pd.DataFrame:
    # per-route random targets
    cruise_gs_kt   = random.uniform(*CRUISE_GS_KT_RANGE)
    climb_gs_kt    = random.uniform(*CLIMB_GS_KT_RANGE)
    descent_gs_kt  = random.uniform(*DESCENT_GS_KT_RANGE)
    climb_vs_fpm   = random.uniform(*CLIMB_VS_FPM_RANGE)
    descent_vs_fpm = random.uniform(*DESCENT_VS_FPM_RANGE)

    cruise_alt_ft  = random.uniform(*CRUISE_ALT_FT_RANGE)

    # durations (seconds) per phase: climb ~20–30%, cruise ~40–60%, descent ~20–30%
    total_T   = random.randint(DURATION_SEC_MIN, DURATION_SEC_MAX)
    climb_T   = int(total_T * random.uniform(0.20, 0.30))
    descent_T = int(total_T * random.uniform(0.20, 0.30))
    cruise_T  = max(total_T - climb_T - descent_T, SAMPLE_EVERY_SEC)

    # initial position & heading
    lat, lon = rand_in_disc(AREA_CENTER_LAT, AREA_CENTER_LON, AREA_RADIUS_KM*0.5)
    heading_deg = random.uniform(0, 360)

    # small constant turn rate (deg/sec) with noise
    base_turn_deg_per_sec = random.uniform(*HEADING_CHANGE_DEG_PER_MIN) / 60.0

    # altitude starts near 1000–3000 ft AGL
    alt_ft = random.uniform(1500, 3000)

    # timestamps relative (start at 0) to make windowing easy
    t0 = 0
    rows: List[dict] = []

    # helper to step one second with given GS/VS targets & phase drift
    def step_second(gs_target_kt: float, vs_target_fpm: float, sign_vs: int):
        nonlocal lat, lon, heading_deg, alt_ft, t0
        # smooth fluctuations
        gs_kt = np.clip(np.random.normal(gs_target_kt, 10.0), 150, 520)
        vs_fpm = sign_vs * abs(np.random.normal(vs_target_fpm, vs_target_fpm*0.15))
        # update heading (slow wander + small noise)
        heading_deg = (heading_deg + base_turn_deg_per_sec + np.random.normal(0, HEADING_NOISE_DEG/60.0)) % 360.0
        # integrate position
        gs_mps = kt_to_mps(gs_kt)
        dx_km = (gs_mps * 1.0) / 1000.0 * math.sin(math.radians(heading_deg))  # east
        dy_km = (gs_mps * 1.0) / 1000.0 * math.cos(math.radians(heading_deg))  # north
        lat, lon = offset_latlon(lat, lon, dx_km, dy_km)
        # altitude
        alt_ft = max(0.0, alt_ft + fpm_to_mps(vs_fpm) * 1.0 / 0.3048)  # convert m/s to ft/s via /0.3048
        # record 1Hz; will sub-sample later
        rows.append({
            "timestamp": t0,
            "latitude": lat,
            "longitude": lon,
            "baro_altitude": ft_to_m(alt_ft),
            "velocity": gs_mps,
            "heading": heading_deg,
            "vertical_rate": fpm_to_mps(vs_fpm),
            "on_ground": False,
        })
        t0 += 1

    # CLIMB
    for _ in range(climb_T):
        if alt_ft < cruise_alt_ft:
            step_second(climb_gs_kt, climb_vs_fpm, sign_vs=+1)
        else:
            # reached cruise early
            step_second(cruise_gs_kt, 0.0, sign_vs=+1)

    # CRUISE
    for _ in range(cruise_T):
        # small +/- 300 ft deviation around cruise altitude
        alt_deviation = np.clip(np.random.normal(0, 50), -300, 300)
        target_alt_ft = cruise_alt_ft + alt_deviation
        sign = +1 if alt_ft < target_alt_ft else -1
        step_second(cruise_gs_kt, 100.0, sign_vs=sign)  # ~100 fpm trim up/down

    # DESCENT
    for _ in range(descent_T):
        if alt_ft > 1500.0:
            step_second(descent_gs_kt, descent_vs_fpm, sign_vs=-1)
        else:
            step_second(descent_gs_kt, 0.0, sign_vs=-1)

    # sub-sample to requested rate (e.g., every 5s)
    df1hz = pd.DataFrame(rows)
    df = df1hz.iloc[::SAMPLE_EVERY_SEC].copy().reset_index(drop=True)

    # annotate OpenSky-like meta columns
    icao_hex = "".join(np.random.choice(list("0123456789abcdef"), size=6))
    callsign = f"SYN{np.random.randint(100,999)}"
    origin = "SYN"
    df["icao24"] = icao_hex
    df["callsign"] = callsign
    df["origin_country"] = origin

    # make timestamp absolute epoch if you prefer:
    epoch0 = now_epoch()
    df["timestamp"] = df["timestamp"].astype(int) + epoch0

    # reorder columns
    cols = ["timestamp","icao24","callsign","origin_country",
            "longitude","latitude","baro_altitude","velocity",
            "heading","vertical_rate","on_ground"]
    return df[cols]

# ------------- Main ----------------------
def main():
    all_routes = []
    for i in range(NUM_ROUTES):
        df = synthesize_one_route()
        df["route_id"] = i
        all_routes.append(df)
    full = pd.concat(all_routes, ignore_index=True)
    full.to_csv(OUT_CSV, index=False)
    print(f"Saved {len(full)} rows from {NUM_ROUTES} synthetic routes to {OUT_CSV}")
    # quick stats
    print("Example head:\n", full.head())

if __name__ == "__main__":
    main()
