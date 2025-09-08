import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns
from clearml import Task

# -------------------- ClearML task --------------------
task = Task.init(
    project_name="Sensores Fault Detection - many paths",
    task_name="Transformer",
    task_type=Task.TaskTypes.training
)

# -------------------- Reproducibility --------------------
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# -------------------- Config --------------------
CSV_PATH = "manyPathData/faultySensors/full_faulty_dataset.csv"

SENSORS = [f"sensor_{i}" for i in range(1, 6)]
ANG_SENSORS = ["sensor_1", "sensor_2", "sensor_3"]
GPS_SENSORS = ["sensor_4", "sensor_5"]
ANG_COLS = ["range_km", "yaw_deg", "pitch_deg"]
GPS_COLS = ["lat", "lon", "alt"]
ALL_COLS = ANG_COLS + GPS_COLS
SENSOR2IDX = {s: i for i, s in enumerate(SENSORS)}

# ==================== Feature engineering (mirrors your XGBoost code) ====================
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

def series_stats(s: pd.Series, prefix: str):
    if not isinstance(s, pd.Series) or s.empty:
        return {
            f"{prefix}_mean": 0.0, f"{prefix}_std": 0.0, f"{prefix}_iqr": 0.0, f"{prefix}_mad": 0.0,
            f"{prefix}_min": 0.0, f"{prefix}_max": 0.0, f"{prefix}_miss_rate": 1.0, f"{prefix}_spike_frac": 0.0,
        }
    x = s.to_numpy(dtype=float)
    if x.size == 0 or np.all(np.isnan(x)):
        return {
            f"{prefix}_mean": 0.0, f"{prefix}_std": 0.0, f"{prefix}_iqr": 0.0, f"{prefix}_mad": 0.0,
            f"{prefix}_min": 0.0, f"{prefix}_max": 0.0, f"{prefix}_miss_rate": 1.0, f"{prefix}_spike_frac": 0.0,
        }
    with np.errstate(all='ignore'):
        mean = np.nanmean(x); std = np.nanstd(x)
        iqr = nan_iqr(x); mad = nan_mad(x)
        mn = np.nanmin(x); mx = np.nanmax(x)
        miss_rate = np.mean(~np.isfinite(x) | np.isnan(x))
        spike = frac_spikes_z(x, z=3.0)
    def _f(v): return float(v) if np.isfinite(v) else 0.0
    return {
        f"{prefix}_mean": _f(mean),
        f"{prefix}_std": _f(std),
        f"{prefix}_iqr": _f(iqr),
        f"{prefix}_mad": _f(mad),
        f"{prefix}_min": _f(mn),
        f"{prefix}_max": _f(mx),
        f"{prefix}_miss_rate": _f(miss_rate),
        f"{prefix}_spike_frac": _f(spike),
    }

R_EARTH_M = 6_371_000.0
def haversine_m(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1; dlon = lon2 - lon1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1)*np.cos(lat2)*(np.sin(dlon/2.0)**2)
    c = 2*np.arcsin(np.sqrt(a))
    return R_EARTH_M * c

def derive_gps_dynamics(sensor_df: pd.DataFrame):
    if sensor_df.empty:
        idx = pd.Index([], name="timestamp", dtype="float64")
        return (pd.Series(dtype=float, index=idx, name="speed_h"),
                pd.Series(dtype=float, index=idx, name="accel_h"),
                pd.Series(dtype=float, index=idx, name="v_speed"))
    s = sensor_df.sort_values("timestamp")
    lat = s["lat"].astype(float); lon = s["lon"].astype(float); alt = s["alt"].astype(float)
    ts  = s["timestamp"].astype(float)

    lat_prev = lat.shift(1); lon_prev = lon.shift(1); alt_prev = alt.shift(1)
    dt = ts - ts.shift(1)
    dt = dt.where(dt > 0, np.nan)

    with np.errstate(all='ignore'):
        dist_m  = haversine_m(lat_prev, lon_prev, lat, lon)
        speed_h = dist_m / dt
        v_speed = np.abs(alt - alt_prev) / dt
        accel_h = (speed_h - speed_h.shift(1)) / dt

    speed_h.name = "speed_h"; accel_h.name = "accel_h"; v_speed.name = "v_speed"
    out_idx = s["timestamp"]
    return speed_h.set_axis(out_idx.values), accel_h.set_axis(out_idx.values), v_speed.set_axis(out_idx.values)

def build_relative_feats(route_df: pd.DataFrame, col: str, sensors_in_group):
    sub = route_df[route_df["sensor"].isin(sensors_in_group)]
    piv = sub.pivot(index="timestamp", columns="sensor", values=col)
    med = piv.median(axis=1, skipna=True)
    return piv, med

def sensor_series_or_nan(pivot: pd.DataFrame, sensor: str):
    if sensor in pivot.columns:
        return pivot[sensor]
    return pd.Series([np.nan]*len(pivot.index), index=pivot.index)

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
            cols.append(ser.rename(s))
        piv = pd.concat(cols, axis=1)
        med = piv.median(axis=1, skipna=True)
        pivots[k] = (piv, med)
    return pivots

def build_features_for_route(route_df: pd.DataFrame, route_id):
    faulty = route_df.drop_duplicates(["route_id"])["faulty_sensor"].iloc[0]

    ang_rel = {c: build_relative_feats(route_df, c, ANG_SENSORS) for c in ANG_COLS}
    gps_rel = {c: build_relative_feats(route_df, c, GPS_SENSORS) for c in GPS_COLS}
    gps_dyn_rel = build_gps_dynamic_rel(route_df)

    rows = []
    for s in SENSORS:
        row = {"route_id": route_id, "sensor": s, "label": 1 if s == faulty else 0}

        s_df = route_df[route_df["sensor"] == s].sort_values("timestamp")
        s_ts = s_df.set_index("timestamp") if not s_df.empty else pd.DataFrame(index=[])

        # stats of raw series
        for c in ALL_COLS:
            ser = s_ts[c] if (not s_ts.empty and c in s_ts.columns) else pd.Series(dtype=float)
            row.update(series_stats(ser, prefix=f"{c}"))

        # relative-to-median stats per group
        if s in ANG_SENSORS:
            for c in ANG_COLS:
                piv, med = ang_rel[c]
                s_series = sensor_series_or_nan(piv, s).reindex(med.index)
                diff = s_series - med
                row.update(series_stats(diff, prefix=f"rel_{c}"))
                row[f"rel_{c}_abs_mean"] = float(np.nanmean(np.abs(diff))) if len(diff) else 0.0
                row[f"rel_{c}_abs_max"]  = float(np.nanmax(np.abs(diff))) if len(diff) else 0.0
        else:
            for c in GPS_COLS:
                piv, med = gps_rel[c]
                s_series = sensor_series_or_nan(piv, s).reindex(med.index)
                diff = s_series - med
                row.update(series_stats(diff, prefix=f"rel_{c}"))
                row[f"rel_{c}_abs_mean"] = float(np.nanmean(np.abs(diff))) if len(diff) else 0.0
                row[f"rel_{c}_abs_max"]  = float(np.nanmax(np.abs(diff))) if len(diff) else 0.0

            sp, ac, vs = derive_gps_dynamics(s_df)
            row.update(series_stats(sp, prefix="speed_h"))
            row.update(series_stats(ac, prefix="accel_h"))
            row.update(series_stats(vs, prefix="v_speed"))

            for k, (piv, med) in gps_dyn_rel.items():
                s_series = sensor_series_or_nan(piv, s).reindex(med.index)
                diff = s_series - med
                row.update(series_stats(diff, prefix=f"rel_{k}"))
                row[f"rel_{k}_abs_mean"] = float(np.nanmean(np.abs(diff))) if len(diff) else 0.0
                row[f"rel_{k}_abs_max"]  = float(np.nanmax(np.abs(diff))) if len(diff) else 0.0

        rows.append(row)
    return rows

def build_engineered_dataset(df: pd.DataFrame):
    feats = []
    for rid, sub in df.groupby("route_id"):
        sub = sub.sort_values(["timestamp", "sensor"])
        feats.extend(build_features_for_route(sub, rid))
    return pd.DataFrame(feats)

# -------------------- Load raw & build engineered --------------------
usecols = ["route_id", "timestamp", "sensor", "type"] + ALL_COLS + ["faulty_sensor"]
raw = pd.read_csv(CSV_PATH, usecols=usecols).sort_values(["route_id", "timestamp", "sensor"]).reset_index(drop=True)
Xdf = build_engineered_dataset(raw)

# -------------------- Route-level labels --------------------
route_true = (Xdf[Xdf["label"] == 1][["route_id", "sensor"]]
              .drop_duplicates("route_id")
              .rename(columns={"sensor": "true_sensor"}))
route_true["y"] = route_true["true_sensor"].map(SENSOR2IDX)

routes = route_true["route_id"].to_numpy()
y_routes = route_true["y"].to_numpy()
rt_train, rt_tmp, y_tr_r, y_tmp_r = train_test_split(routes, y_routes, test_size=0.20, random_state=SEED, stratify=y_routes)
rt_val,   rt_test, y_va_r, y_te_r = train_test_split(rt_tmp, y_tmp_r, test_size=0.50, random_state=SEED, stratify=y_tmp_r)

def mask_routes(rts): return Xdf["route_id"].isin(rts)

# -------------------- Safe scaling (prevents NaN/Inf, zero-std) --------------------
def debug_nan_inf(df, cols, tag):
    bad = np.isinf(df[cols]).to_numpy().sum() + np.isnan(df[cols]).to_numpy().sum()
    print(f"[{tag}] bad values (NaN/Inf) count: {bad}")

ignore_cols = {"route_id", "sensor", "label"}
feat_cols = [c for c in Xdf.columns if c not in ignore_cols]

# sanitize before stats
Xdf[feat_cols] = Xdf[feat_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
debug_nan_inf(Xdf, feat_cols, "before scaling")

train_mask = mask_routes(rt_train)
mu = Xdf.loc[train_mask, feat_cols].mean(axis=0).to_numpy(dtype=np.float64)
std = Xdf.loc[train_mask, feat_cols].std(axis=0, ddof=0).to_numpy(dtype=np.float64)

eps = 1e-8
std = np.where(std < eps, 1.0, std)

Xdf[feat_cols] = (Xdf[feat_cols].to_numpy(dtype=np.float64) - mu) / std
Xdf[feat_cols] = (pd.DataFrame(Xdf[feat_cols]).replace([np.inf, -np.inf], 0.0).fillna(0.0)).astype(np.float32)
debug_nan_inf(Xdf, feat_cols, "after scaling")

# -------------------- Pack tensors (N_routes, 5, F) --------------------
def build_tensor_pack(route_ids):
    rows, labels = [], []
    for rid in route_ids:
        sub = Xdf[Xdf["route_id"] == rid]
        mats = []
        for s in SENSORS:
            row = sub[sub["sensor"] == s]
            mats.append(row.iloc[0][feat_cols].to_numpy(dtype=np.float32) if not row.empty
                        else np.zeros(len(feat_cols), dtype=np.float32))
        rows.append(np.stack(mats))
        labels.append(int(route_true.loc[route_true["route_id"] == rid, "y"].iloc[0]))
    X = np.stack(rows).astype(np.float32)
    y = np.array(labels, dtype=np.int64)
    types = np.array([[0 if s in GPS_SENSORS else 1 for s in SENSORS] for _ in range(len(route_ids))], dtype=np.int64)
    return X, types, y

def make_finite(X):
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

X_tr, T_tr, y_tr = build_tensor_pack(rt_train)
X_va, T_va, y_va = build_tensor_pack(rt_val)
X_te, T_te, y_te = build_tensor_pack(rt_test)

X_tr, X_va, X_te = make_finite(X_tr), make_finite(X_va), make_finite(X_te)

class SensorDatasetDual(Dataset):
    def __init__(self, X, types, y):
        self.X = torch.tensor(X)
        self.types = torch.tensor(types)
        self.y = torch.tensor(y)
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.types[i], self.y[i]

train_loader = DataLoader(SensorDatasetDual(X_tr, T_tr, y_tr), batch_size=32, shuffle=True)
val_loader   = DataLoader(SensorDatasetDual(X_va, T_va, y_va), batch_size=32, shuffle=False)
test_loader  = DataLoader(SensorDatasetDual(X_te, T_te, y_te), batch_size=32, shuffle=False)

# -------------------- Model --------------------
class SensorClassifierDualEncoder(nn.Module):
    def __init__(self, input_dim, embed_dim=64, num_classes=5):
        super().__init__()
        self.gps_encoder = nn.Sequential(nn.Linear(input_dim, embed_dim), nn.ReLU())
        self.angular_encoder = nn.Sequential(nn.Linear(input_dim, embed_dim), nn.ReLU())
        enc_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=4, batch_first=True)
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=2)
        self.fc1 = nn.Linear(5 * embed_dim, 128)
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, 64)
        self.dropout2 = nn.Dropout(0.2)
        self.out = nn.Linear(64, num_classes)

    def forward(self, x, sensor_types):
        # runtime sanitization as extra guard
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

        B, S, _ = x.shape
        embs = []
        for i in range(S):
            xi = x[:, i, :]
            is_gps = (sensor_types[:, i] == 0)
            e = torch.zeros((B, self.gps_encoder[0].out_features), device=x.device)
            if is_gps.any():
                e[is_gps] = self.gps_encoder(xi[is_gps])
            if (~is_gps).any():
                e[~is_gps] = self.angular_encoder(xi[~is_gps])
            embs.append(e.unsqueeze(1))
        x = torch.cat(embs, dim=1)
        x = self.transformer(x)
        x = x.view(B, -1)
        x = self.dropout1(self.fc1(x))
        x = self.dropout2(self.fc2(x))
        return self.out(x)

# -------------------- Train --------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SensorClassifierDualEncoder(input_dim=len(feat_cols)).to(device)

class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_tr), y=y_tr)
criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float, device=device))
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

EPOCHS = 20
best_val = -1.0
best_state = None
patience = 5
no_improve = 0

for ep in range(1, EPOCHS + 1):
    model.train()
    tot_loss, correct, total = 0.0, 0, 0
    for xb, tb, yb in train_loader:
        xb, tb, yb = xb.to(device), tb.to(device), yb.to(device)
        optimizer.zero_grad()
        logits = model(xb, tb)
        loss = criterion(logits, yb)
        if torch.isnan(loss) or torch.isinf(loss):
            raise RuntimeError("NaN/Inf detected in loss; check inputs/scaling.")
        loss.backward()
        optimizer.step()
        tot_loss += loss.item()
        correct += (logits.argmax(1) == yb).sum().item()
        total += yb.size(0)
    tr_acc = correct / max(total, 1)

    model.eval()
    v_correct, v_total = 0, 0
    with torch.no_grad():
        for xb, tb, yb in val_loader:
            xb, tb, yb = xb.to(device), tb.to(device), yb.to(device)
            logits = model(xb, tb)
            v_correct += (logits.argmax(1) == yb).sum().item()
            v_total += yb.size(0)
    va_acc = v_correct / max(v_total, 1)
    print(f"Epoch {ep:02d} | TrainLoss: {tot_loss:.2f} | TrainAcc: {tr_acc:.3f} | ValAcc: {va_acc:.3f}")

    if va_acc > best_val:
        best_val = va_acc
        best_state = {k: v.cpu() for k, v in model.state_dict().items()}
        no_improve = 0
    else:
        no_improve += 1
        if no_improve >= patience:
            print(f"Early stopping at epoch {ep}")
            break

if best_state is not None:
    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

# -------------------- Test + ClearML logs (accuracy + native CM only) --------------------
model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for xb, tb, yb in test_loader:
        xb, tb = xb.to(device), tb.to(device)
        logits = model(xb, tb)
        all_preds.extend(logits.argmax(1).cpu().numpy())
        all_labels.extend(yb.numpy())

test_acc = float((np.array(all_preds) == np.array(all_labels)).mean())
print(f"\nTest Accuracy: {test_acc:.3f}")

logger = task.get_logger()
logger.report_scalar(title="Evaluation", series="Test Accuracy", value=test_acc, iteration=0)

cm = confusion_matrix(all_labels, all_preds, labels=list(range(5)))
logger.report_confusion_matrix(
    title="Confusion Matrix",
    series="Route-Level",
    matrix=cm.tolist(),
    xlabels=SENSORS,
    ylabels=SENSORS,
    iteration=0
)

print("\nClassification report:\n", classification_report(all_labels, all_preds, target_names=SENSORS, digits=3))
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=SENSORS, yticklabels=SENSORS)
plt.xlabel("Predicted"); plt.ylabel("True"); plt.title("Confusion Matrix (Transformer + engineered features, safe)")
plt.tight_layout(); plt.show()
