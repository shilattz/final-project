# -*- coding: utf-8 -*-
"""
Build global, fixed sensor array (5 sensors around all routes together),
generate per-sensor observations from synthetic_routes.csv,
and plot routes + sensor locations in 3D.

Input  (must exist): synthetic_routes.csv with columns:
  ['timestamp','icao24','callsign','origin_country',
   'longitude','latitude','baro_altitude','velocity',
   'heading','vertical_rate','on_ground','route_id']

Output:
  - sensor_observations.csv
  - a 3D plot window (routes + sensors)
"""

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3D)

# ========= Config =========
INPUT_CSV  = "synthetic_routes.csv"
OUTPUT_CSV = "sensor_observations.csv"

# Global, deterministic sensor layout
NUM_SENSORS      = 5
GPS_SENSORS      = [1, 5]      # sensor_1 and sensor_5 are GPS
ANGULAR_SENSORS  = [i for i in range(1, NUM_SENSORS+1) if i not in GPS_SENSORS]
RADIUS_FRACTION  = 0.20        # radius as fraction of global lat/lon span
SENSOR_ALT_M     = 100.0       # ground sensor altitude (meters)

# ========= Helpers =========
LAT_KM = 111.0  # approximate km per degree latitude
def lon_km(lat_deg: float) -> float:
    # km/deg longitude at given latitude
    return LAT_KM * max(math.cos(math.radians(lat_deg)), 1e-6)

def compute_relative_observation(sensor_pos, ac_pos):
    """
    From sensor (lat,lon,alt_m) to aircraft (lat,lon,alt_m),
    compute range [km], yaw [deg], pitch [deg].
    yaw: bearing in XY plane, pitch: elevation angle.
    """
    s_lat, s_lon, s_alt = sensor_pos
    a_lat, a_lon, a_alt = ac_pos

    # XY in km, Z in km
    dx = (a_lon - s_lon) * lon_km((a_lat + s_lat) / 2.0)  # east-west
    dy = (a_lat - s_lat) * LAT_KM                          # north-south
    dz = (a_alt - s_alt) / 1000.0                          # meters -> km

    dist = math.sqrt(dx*dx + dy*dy + dz*dz)
    if dist < 1e-9:
        return 0.0, 0.0, 0.0

    yaw_deg = math.degrees(math.atan2(dx, dy))            # [-180,180]
    # clip for numerical safety
    arg = max(-1.0, min(1.0, dz / dist))
    pitch_deg = math.degrees(math.asin(arg))
    return float(dist), float(yaw_deg), float(pitch_deg)

# ========= Load routes =========
df = pd.read_csv(INPUT_CSV)

required = {"timestamp","latitude","longitude","baro_altitude","route_id"}
missing = required - set(df.columns)
if missing:
    raise ValueError(f"Missing required columns: {missing}")

# keep only valid points
df = df.dropna(subset=["latitude","longitude","baro_altitude","route_id","timestamp"]).copy()

# ========= Global sensor layout (single fixed array) =========
# global bounding box across ALL routes
min_lat, max_lat = df["latitude"].min(), df["latitude"].max()
min_lon, max_lon = df["longitude"].min(), df["longitude"].max()

center_lat = (min_lat + max_lat) / 2.0
center_lon = (min_lon + max_lon) / 2.0

# circle radii in degrees
radius_lat = (max_lat - min_lat) * RADIUS_FRACTION
radius_lon = (max_lon - min_lon) * RADIUS_FRACTION

# deterministic angles:
# put GPS at opposite bearings (0°, 180°), others evenly fill the gaps
angles = []
if NUM_SENSORS >= 2 and len(GPS_SENSORS) == 2:
    angles = [0.0, math.pi]  # positions for sensor_1 and sensor_5
    remain = NUM_SENSORS - 2
    if remain > 0:
        # spread remaining sensors between (0, 2π), skipping the two fixed angles
        # Simple choice: interleave extra points roughly evenly
        # e.g., for 3 remaining -> ~72°, 144°, 216°
        extra = [2*math.pi * (k+1) / (remain+1) for k in range(remain)]
        angles += extra
else:
    angles = [2*math.pi * i / NUM_SENSORS for i in range(NUM_SENSORS)]

# Build sensor metadata (fixed once for all routes)
sensors = []
for i in range(NUM_SENSORS):
    sensor_idx = i + 1
    ang = angles[i]
    s_lat = center_lat + radius_lat * math.sin(ang)
    s_lon = center_lon + radius_lon * math.cos(ang)
    s_type = "GPS" if sensor_idx in GPS_SENSORS else "Angular"
    sensors.append({
        "sensor": f"sensor_{sensor_idx}",
        "type":   s_type,
        "lat":    float(s_lat),
        "lon":    float(s_lon),
        "alt_m":  float(SENSOR_ALT_M),
        "angle":  float(math.degrees(ang)),
    })

print("Fixed sensor array (global):")
for s in sensors:
    print(s)

# ========= Generate observations =========
records = []
for _, row in df.iterrows():
    ts   = int(row["timestamp"])
    rid  = row["route_id"]
    lat  = float(row["latitude"])
    lon  = float(row["longitude"])
    altm = float(row["baro_altitude"])  # meters

    for s in sensors:
        if s["type"] == "GPS":
            # GPS observes the (noiseless) absolute position
            rec = {
                "route_id": rid,
                "timestamp": ts,
                "sensor": s["sensor"],
                "type": "GPS",
                "lat": lat,
                "lon": lon,
                "alt": altm,
                "sensor_lat": s["lat"],
                "sensor_lon": s["lon"],
                "sensor_alt_m": s["alt_m"],
            }
        else:
            # Angular/Range observes relative geometry
            rng_km, yaw_deg, pitch_deg = compute_relative_observation(
                (s["lat"], s["lon"], s["alt_m"]),
                (lat, lon, altm)
            )
            rec = {
                "route_id": rid,
                "timestamp": ts,
                "sensor": s["sensor"],
                "type": "Angular",
                "range_km": rng_km,
                "yaw_deg": yaw_deg,
                "pitch_deg": pitch_deg,
                "sensor_lat": s["lat"],
                "sensor_lon": s["lon"],
                "sensor_alt_m": s["alt_m"],
            }
        records.append(rec)

obs_df = pd.DataFrame(records)
obs_df.sort_values(["route_id","timestamp","sensor"], inplace=True)
obs_df.to_csv(OUTPUT_CSV, index=False)
print(f"\nSaved observations -> {OUTPUT_CSV}")
print(obs_df.head(10))

# ========= 3D plot: routes + sensors =========
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# plot all routes (thin lines)
for rid, g in df.groupby("route_id", sort=False):
    g2 = g.sort_values("timestamp")
    ax.plot(
        g2["longitude"].values,
        g2["latitude"].values,
        g2["baro_altitude"].values,
        linewidth=1, alpha=0.5
    )

# plot sensors
