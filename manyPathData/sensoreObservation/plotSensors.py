# plot_routes_and_sensors.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ============ Load routes ============
CSV_PATH = "synthetic_routes.csv"
df = pd.read_csv(CSV_PATH)

# ============ Define global sensors ============
sensors = [
    {"sensor": "sensor_1", "type": "GPS", "lat": 31.5, "lon": 35.5, "alt_m": 100},
    {"sensor": "sensor_5", "type": "GPS", "lat": 31.7, "lon": 35.3, "alt_m": 100},
    {"sensor": "sensor_2", "type": "Angular", "lat": 31.6, "lon": 35.6, "alt_m": 100},
    {"sensor": "sensor_3", "type": "Angular", "lat": 31.8, "lon": 35.7, "alt_m": 100},
    {"sensor": "sensor_4", "type": "Angular", "lat": 31.4, "lon": 35.4, "alt_m": 100},
]

# helper to compute nice bounds
def nice_bounds(series, pad_ratio=0.05):
    lo, hi = float(series.min()), float(series.max())
    pad = (hi - lo) * pad_ratio
    return lo - pad, hi + pad

# helper to downsample
def downsample_idx(n, step=4):
    if n <= step:
        return np.arange(n)
    return np.arange(0, n, step)

# ============ 3D Plot ============
def plot_3d(df, sensors):
    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111, projection="3d")

    xlo, xhi = nice_bounds(df["longitude"])
    ylo, yhi = nice_bounds(df["latitude"])
    zlo, zhi = nice_bounds(df["baro_altitude"])

    for rid, g in df.groupby("route_id", sort=False):
        g2 = g.sort_values("timestamp")
        idx = downsample_idx(len(g2), step=3)
        ax.plot(
            g2["longitude"].values[idx],
            g2["latitude"].values[idx],
            g2["baro_altitude"].values[idx],
            linewidth=1.2, alpha=0.35,
        )

    gps = [s for s in sensors if s["type"] == "GPS"]
    ang = [s for s in sensors if s["type"] == "Angular"]

    ax.scatter([s["lon"] for s in gps], [s["lat"] for s in gps], [s["alt_m"] for s in gps],
               marker="^", s=220, label="GPS sensors")
    ax.scatter([s["lon"] for s in ang], [s["lat"] for s in ang], [s["alt_m"] for s in ang],
               marker="o", s=220, label="Angular sensors")

    for s in sensors:
        ax.text(s["lon"], s["lat"], s["alt_m"] + 80.0, s["sensor"],
                fontsize=10, bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=1.5))

    ax.set_xlim(xlo, xhi); ax.set_ylim(ylo, yhi); ax.set_zlim(zlo, zhi)
    ax.view_init(elev=22, azim=-55)
    ax.grid(True, alpha=0.2)
    ax.set_title("Synthetic Routes + Global Sensor Array (3D)")
    ax.set_xlabel("Longitude [deg]"); ax.set_ylabel("Latitude [deg]"); ax.set_zlabel("Altitude [m]")
    ax.legend(loc="upper left", framealpha=0.9)

    plt.tight_layout()
    plt.savefig("routes_and_sensors_3d.png", dpi=300)
    plt.show()
    print("Saved -> routes_and_sensors_3d.png")

# ============ 2D Plot ============
def plot_2d(df, sensors):
    plt.figure(figsize=(9, 7))
    for rid, g in df.groupby("route_id", sort=False):
        g2 = g.sort_values("timestamp")
        idx = downsample_idx(len(g2), step=2)
        plt.plot(g2["longitude"].values[idx], g2["latitude"].values[idx],
                 linewidth=0.9, alpha=0.35)

    gps = [s for s in sensors if s["type"] == "GPS"]
    ang = [s for s in sensors if s["type"] == "Angular"]

    plt.scatter([s["lon"] for s in gps], [s["lat"] for s in gps], s=160, marker="^", label="GPS sensors")
    plt.scatter([s["lon"] for s in ang], [s["lat"] for s in ang], s=160, marker="o", label="Angular sensors")

    for s in sensors:
        plt.text(s["lon"], s["lat"], s["sensor"], fontsize=9,
                 bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1))

    xlo, xhi = nice_bounds(df["longitude"]); ylo, yhi = nice_bounds(df["latitude"])
    plt.xlim(xlo, xhi); plt.ylim(ylo, yhi)
    plt.title("Routes + Sensors (Top-down 2D)")
    plt.xlabel("Longitude [deg]"); plt.ylabel("Latitude [deg]")
    plt.legend(loc="upper left", framealpha=0.95)
    plt.tight_layout()
    plt.savefig("routes_and_sensors_2d.png", dpi=300)
    plt.show()
    print("Saved -> routes_and_sensors_2d.png")

if __name__ == "__main__":
    plot_3d(df, sensors)
    plot_2d(df, sensors)
