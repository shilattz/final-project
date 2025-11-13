# plot_routes_3d.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# === CONFIG ===
CSV_PATH = "synthetic_routes.csv"  # change if needed
ROUTE_COL = "route_id"
LAT_COL, LON_COL, ALT_COL = "latitude", "longitude", "baro_altitude"   # degrees, degrees, meters

# === LOAD ===
df = pd.read_csv(CSV_PATH)
required = {ROUTE_COL, LAT_COL, LON_COL, ALT_COL}
missing = required - set(df.columns)
if missing:
    raise ValueError(f"Missing required columns: {missing}")

# basic cleaning
df = df.dropna(subset=[ROUTE_COL, LAT_COL, LON_COL, ALT_COL]).copy()

# === 3D PLOT ===
routes = sorted(df[ROUTE_COL].unique())
cmap = get_cmap("tab20")
fig = plt.figure(figsize=(11, 8))
ax = fig.add_subplot(111, projection='3d')

for i, rid in enumerate(routes):
    g = df[df[ROUTE_COL] == rid].sort_values("timestamp")
    # Keep consistent scaling: altitude often in meters, lon/lat in degrees.
    # To visualize nicer, you may normalize alt (e.g., km) or shift to start at 0.
    x = g[LON_COL].to_numpy()
    y = g[LAT_COL].to_numpy()
    z = g[ALT_COL].to_numpy() / 1000.0  # km for readability

    color = cmap(i % cmap.N)
    ax.plot(x, y, z, label=f"Route {rid}", lw=2, color=color)
    # optional: mark start/end
    ax.scatter(x[:1], y[:1], z[:1], color=color, marker='o', s=30)
    ax.scatter(x[-1:], y[-1:], z[-1:], color=color, marker='^', s=30)

ax.set_title("3D Flight Routes (lon, lat, altitude)")
ax.set_xlabel("Longitude [deg]")
ax.set_ylabel("Latitude [deg]")
ax.set_zlabel("Altitude [km]")
ax.legend(loc="upper left", bbox_to_anchor=(1.05, 1.0))
plt.tight_layout()
plt.show()

# === (Optional) altitude vs. time per route ===
# Uncomment to also visualize altitude profiles.
# import matplotlib.dates as mdates
# fig2, ax2 = plt.subplots(figsize=(10, 5))
# for i, rid in enumerate(routes):
#     g = df[df[ROUTE_COL] == rid].sort_values("timestamp")
#     t = pd.to_datetime(g["timestamp"], unit="s", errors="coerce")
#     ax2.plot(t, g[ALT_COL]/1000.0, label=f"Route {rid}", lw=1.8)
# ax2.set_title("Altitude profile over time")
# ax2.set_xlabel("Time")
# ax2.set_ylabel("Altitude [km]")
# ax2.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0))
# fig2.autofmt_xdate()
# plt.tight_layout()
# plt.show()
