import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

import numpy as np
import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from clearml import Task

# Initialize ClearML task
# Task.close()
task = Task.init(
    project_name="Sensores Fault Detection - one path hard noises",
    task_name="XGBoost",
    task_type=Task.TaskTypes.training
)

SEED = 42
np.random.seed(SEED)

CSV_PATH = "OnePathData/withFeatures-fixed/realisticNoises/full_faulty_dataset_realistic.csv"

SENSORS = [f"sensor_{i}" for i in range(1, 6)]
ANG_SENSORS = ["sensor_2", "sensor_4", "sensor_5"]
GPS_SENSORS = ["sensor_1", "sensor_3"]

ANG_COLS = ["range_km", "yaw_deg", "pitch_deg"]
GPS_COLS = ["lat", "lon", "alt"]
ALL_COLS = ANG_COLS + GPS_COLS

SENSOR2IDX = {s: i for i, s in enumerate(SENSORS)}

# ---------- robust stats helpers ----------
def _safe_array(arr):
    x = np.asarray(arr, dtype=float)
    if x.size == 0 or np.all(np.isnan(x)):
        return None
    return x

def nan_iqr(x):
    x = _safe_array(x)
    if x is None:
        return 0.0
    with np.errstate(all='ignore'):
        q75 = np.nanpercentile(x, 75)
        q25 = np.nanpercentile(x, 25)
    if not np.isfinite(q75) or not np.isfinite(q25):
        return 0.0
    return float(q75 - q25)

def nan_mad(x):
    x = _safe_array(x)
    if x is None:
        return 0.0
    with np.errstate(all='ignore'):
        med = np.nanmedian(x)
        mad = np.nanmedian(np.abs(x - med))
    if not np.isfinite(mad):
        return 0.0
    return float(mad)

def frac_spikes_z(x, z=3.0):
    x = _safe_array(x)
    if x is None:
        return 0.0
    with np.errstate(all='ignore'):
        mu = np.nanmean(x)
        sd = np.nanstd(x)
    if (not np.isfinite(sd)) or sd <= 1e-9:
        return 0.0
    zscores = (x - mu) / sd
    with np.errstate(all='ignore'):
        frac = np.nanmean(np.abs(zscores) > z)
    return float(0.0 if not np.isfinite(frac) else frac)

def series_stats(s: pd.Series, name_prefix: str):
    if not isinstance(s, pd.Series) or s.empty:
        return {
            f"{name_prefix}_mean": 0.0, f"{name_prefix}_std": 0.0,
            f"{name_prefix}_iqr": 0.0,  f"{name_prefix}_mad": 0.0,
            f"{name_prefix}_min": 0.0,  f"{name_prefix}_max": 0.0,
            f"{name_prefix}_miss_rate": 1.0,
            f"{name_prefix}_spike_frac": 0.0,
        }
    x = s.to_numpy(dtype=float)
    if x.size == 0 or np.all(np.isnan(x)):
        return {
            f"{name_prefix}_mean": 0.0, f"{name_prefix}_std": 0.0,
            f"{name_prefix}_iqr": 0.0,  f"{name_prefix}_mad": 0.0,
            f"{name_prefix}_min": 0.0,  f"{name_prefix}_max": 0.0,
            f"{name_prefix}_miss_rate": 1.0,
            f"{name_prefix}_spike_frac": 0.0,
        }
    with np.errstate(all='ignore'):
        mean = np.nanmean(x)
        std  = np.nanstd(x)
        iqr  = nan_iqr(x)
        mad  = nan_mad(x)
        mn   = np.nanmin(x)
        mx   = np.nanmax(x)
        miss_rate = np.mean(~np.isfinite(x) | np.isnan(x))
        spike     = frac_spikes_z(x, z=3.0)
    def _f(v): return float(v) if np.isfinite(v) else 0.0
    return {
        f"{name_prefix}_mean": _f(mean),
        f"{name_prefix}_std":  _f(std),
        f"{name_prefix}_iqr":  _f(iqr),
        f"{name_prefix}_mad":  _f(mad),
        f"{name_prefix}_min":  _f(mn),
        f"{name_prefix}_max":  _f(mx),
        f"{name_prefix}_miss_rate": _f(miss_rate),
        f"{name_prefix}_spike_frac": _f(spike),
    }

# ---------- GPS dynamics ----------
R_EARTH_M = 6_371_000.0

def haversine_m(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return R_EARTH_M * c

def derive_gps_dynamics(sensor_df: pd.DataFrame):
    if sensor_df.empty:
        idx = pd.Index([], name="timestamp", dtype="float64")
        return (pd.Series(dtype=float, index=idx, name="speed_h"),
                pd.Series(dtype=float, index=idx, name="accel_h"),
                pd.Series(dtype=float, index=idx, name="v_speed"))
    s = sensor_df.sort_values("timestamp")
    lat = s["lat"].astype(float)
    lon = s["lon"].astype(float)
    alt = s["alt"].astype(float)
    ts  = s["timestamp"].astype(float)

    lat_prev = lat.shift(1); lon_prev = lon.shift(1); alt_prev = alt.shift(1)
    dt = ts - ts.shift(1)
    dt = dt.where(dt > 0, np.nan)

    with np.errstate(all='ignore'):
        dist_m = haversine_m(lat_prev, lon_prev, lat, lon)
        speed_h = dist_m / dt
    with np.errstate(all='ignore'):
        v_speed = np.abs(alt - alt_prev) / dt
    speed_prev = speed_h.shift(1)
    with np.errstate(all='ignore'):
        accel_h = (speed_h - speed_prev) / dt

    speed_h.name = "speed_h"
    accel_h.name = "accel_h"
    v_speed.name = "v_speed"
    out_idx = s["timestamp"]
    return speed_h.set_axis(out_idx.values), accel_h.set_axis(out_idx.values), v_speed.set_axis(out_idx.values)

def build_gps_dynamic_rel(route_df: pd.DataFrame):
    per_sensor_dyn = {}
    for s in GPS_SENSORS:
        s_df = route_df[route_df["sensor"] == s]
        per_sensor_dyn[s] = derive_gps_dynamics(s_df)

    pivots = {}
    for k in ["speed_h", "accel_h", "v_speed"]:
        cols = []
        for s in GPS_SENSORS:
            ser = per_sensor_dyn[s][["speed_h", "accel_h", "v_speed"].index(k)]
            ser = ser.rename(s)
            cols.append(ser)
        piv = pd.concat(cols, axis=1)
        med = piv.median(axis=1, skipna=True)
        pivots[k] = (piv, med)
    return pivots

# ---------- relative features ----------
def build_relative_feats(route_df: pd.DataFrame, col: str, sensors_in_group):
    sub = route_df[route_df["sensor"].isin(sensors_in_group)]
    piv = sub.pivot(index="timestamp", columns="sensor", values=col)
    med = piv.median(axis=1, skipna=True)
    return piv, med

def sensor_series_or_nan(pivot: pd.DataFrame, sensor: str):
    if sensor in pivot.columns:
        return pivot[sensor]
    return pd.Series([np.nan] * len(pivot.index), index=pivot.index)

# ---------- feature extraction per route ----------
def build_features_for_route(route_df: pd.DataFrame, aircraft):
    faulty = route_df.drop_duplicates(["aircraft"])["faulty_sensor"].iloc[0]

    ang_rel = {c: build_relative_feats(route_df, c, ANG_SENSORS) for c in ANG_COLS}
    gps_rel = {c: build_relative_feats(route_df, c, GPS_SENSORS) for c in GPS_COLS}
    gps_dyn_rel = build_gps_dynamic_rel(route_df)

    rows = []
    for s in SENSORS:
        row = {"aircraft": aircraft, "sensor": s, "label": 1 if s == faulty else 0}

        s_df = route_df[route_df["sensor"] == s].sort_values("timestamp")
        s_ts = s_df.set_index("timestamp") if not s_df.empty else pd.DataFrame(index=[])
        for c in ALL_COLS:
            ser = s_ts[c] if (not s_ts.empty and c in s_ts.columns) else pd.Series(dtype=float)
            row.update(series_stats(ser, name_prefix=f"{c}"))

        if s in ANG_SENSORS:
            for c in ANG_COLS:
                piv, med = ang_rel[c]
                s_series = sensor_series_or_nan(piv, s).reindex(med.index)
                diff = s_series - med
                row.update(series_stats(diff, name_prefix=f"rel_{c}"))
                row[f"rel_{c}_abs_mean"] = float(np.nanmean(np.abs(diff))) if len(diff) > 0 else 0.0
                row[f"rel_{c}_abs_max"]  = float(np.nanmax(np.abs(diff)))  if len(diff) > 0 else 0.0
        else:
            for c in GPS_COLS:
                piv, med = gps_rel[c]
                s_series = sensor_series_or_nan(piv, s).reindex(med.index)
                diff = s_series - med
                row.update(series_stats(diff, name_prefix=f"rel_{c}"))
                row[f"rel_{c}_abs_mean"] = float(np.nanmean(np.abs(diff))) if len(diff) > 0 else 0.0
                row[f"rel_{c}_abs_max"]  = float(np.nanmax(np.abs(diff)))  if len(diff) > 0 else 0.0

            sp, ac, vs = derive_gps_dynamics(s_df)
            row.update(series_stats(sp, name_prefix="speed_h"))
            row.update(series_stats(ac, name_prefix="accel_h"))
            row.update(series_stats(vs, name_prefix="v_speed"))

            for k, (piv, med) in gps_dyn_rel.items():
                s_series = sensor_series_or_nan(piv, s).reindex(med.index)
                diff = s_series - med
                row.update(series_stats(diff, name_prefix=f"rel_{k}"))
                row[f"rel_{k}_abs_mean"] = float(np.nanmean(np.abs(diff))) if len(diff) > 0 else 0.0
                row[f"rel_{k}_abs_max"]  = float(np.nanmax(np.abs(diff)))  if len(diff) > 0 else 0.0

        rows.append(row)
    return rows

def build_dataset(df: pd.DataFrame):
    feats = []
    for rid, sub in df.groupby("aircraft"):
        sub = sub.sort_values(["timestamp", "sensor"])
        feats.extend(build_features_for_route(sub, rid))
    return pd.DataFrame(feats)

# ---------- training + evaluation ----------
def train_xgb_binary_rank(X: pd.DataFrame):
    try:
        from xgboost import XGBClassifier
    except Exception as e:
        raise SystemExit("xgboost is missing. Install with: pip install xgboost\n" + str(e))

    route_true = (X[X["label"] == 1][["aircraft", "sensor"]]
                  .drop_duplicates("aircraft")
                  .rename(columns={"sensor": "true_sensor"}))
    route_true["true_idx"] = route_true["true_sensor"].map(SENSOR2IDX)

    all_routes = route_true["aircraft"].to_numpy()
    y_routes   = route_true["true_idx"].to_numpy()

    rt_train, rt_tmp, y_tr_r, y_tmp_r = train_test_split(
        all_routes, y_routes, test_size=0.30, random_state=SEED, stratify=y_routes
    )
    rt_val, rt_test, y_va_r, y_te_r = train_test_split(
        rt_tmp, y_tmp_r, test_size=0.50, random_state=SEED, stratify=y_tmp_r
    )

    def mask_routes(rts): return X["aircraft"].isin(rts)

    train_mask = mask_routes(rt_train)
    val_mask   = mask_routes(rt_val)
    test_mask  = mask_routes(rt_test)

    ignore = {"aircraft", "sensor", "label"}
    feat_cols = [c for c in X.columns if c not in ignore]

    X_train, y_train = X.loc[train_mask, feat_cols], X.loc[train_mask, "label"].astype(int)
    X_val,   y_val   = X.loc[val_mask,   feat_cols], X.loc[val_mask,   "label"].astype(int)
    X_test,  y_test  = X.loc[test_mask,  feat_cols], X.loc[test_mask,  "label"].astype(int)

    pos = int(y_train.sum())
    neg = int(len(y_train) - pos)
    scale_pos_weight = (neg / max(pos, 1))

    print(f"Train size: {len(y_train)} (pos={pos}, neg={neg}) | scale_pos_weight≈{scale_pos_weight:.2f}")
    print(f"Val size:   {len(y_val)}")
    print(f"Test size:  {len(y_test)}")
    print(f"#features:  {len(feat_cols)}")

    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        n_estimators=1100,
        learning_rate=0.045,
        max_depth=6,
        subsample=0.85,
        colsample_bytree=0.9,
        min_child_weight=4,
        reg_lambda=1.0,
        reg_alpha=0.0,
        gamma=0.0,
        random_state=SEED,
        tree_method="hist",
        scale_pos_weight=scale_pos_weight,
        n_jobs=0
    )

    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    # (route, sensor) predictions for TEST
    p_test = model.predict_proba(X_test)[:, 1]
    test_meta = X.loc[test_mask, ["aircraft", "sensor"]].copy()
    test_meta["prob"] = p_test

    # per-route decision: argmax probability
    pred_route = (test_meta
                  .sort_values(["aircraft", "prob"], ascending=[True, False])
                  .groupby("aircraft")
                  .first()
                  .reset_index()
                  .rename(columns={"sensor": "pred_sensor"}))

    route_true_sub = route_true[route_true["aircraft"].isin(pred_route["aircraft"])]
    merged = pred_route.merge(route_true_sub, on="aircraft", how="inner")
    merged["pred_idx"] = merged["pred_sensor"].map(SENSOR2IDX)
    merged["true_idx"] = merged["true_sensor"].map(SENSOR2IDX)

    # route-level "test" accuracy
    test_acc = float((merged["pred_sensor"] == merged["true_sensor"]).mean())
    print(f"\nTest (route-level) accuracy: {test_acc*100:.1f}%  (routes={merged.shape[0]})")

    # classification report
    print("\nClassification report (route-level):")
    print(classification_report(merged["true_idx"], merged["pred_idx"],
                                target_names=SENSORS, digits=3))

    # confusion matrix (route-level)
    cm = confusion_matrix(merged["true_idx"], merged["pred_idx"], labels=list(range(5)))
    print("\nConfusion matrix (route-level):\n", cm)

    # ------ ClearML ------
    logger = task.get_logger()

    # 1) Test Accuracy (scalar)
    logger.report_scalar(
        title="Evaluation",
        series="Test Accuracy",
        value=test_acc,
        iteration=0
    )

    # 2) Confusion Matrix (native)
    logger.report_confusion_matrix(
        title="Confusion Matrix",
        series="Route-Level",
        matrix=cm.tolist(),
        xlabels=SENSORS,
        ylabels=SENSORS,
        iteration=0
    )
    # -------------------------------------

    # local visualization (not logged)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=SENSORS, yticklabels=SENSORS)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix (Route-level, XGBoost + GPS dynamics)")
    plt.tight_layout()
    plt.show()

    return model, merged, test_acc, cm

# ---------- main ----------
def main():
    usecols = ["aircraft", "timestamp", "sensor", "type"] + ALL_COLS + ["faulty_sensor"]
    df = pd.read_csv(CSV_PATH, usecols=usecols).sort_values(["aircraft", "timestamp", "sensor"]).reset_index(drop=True)
    print(f"Rows: {df.shape[0]} | Routes: {df['aircraft'].nunique()}")

    X = build_dataset(df)
    cnt = Counter(X["aircraft"])
    bad = [rid for rid, c in cnt.items() if c != 5]
    if bad:
        print(f"Warning: some routes do not contain 5 sensors. Examples: {bad[:5]}")

    _model, _merged, _acc, _cm = train_xgb_binary_rank(X)

if __name__ == "__main__":
    main()
