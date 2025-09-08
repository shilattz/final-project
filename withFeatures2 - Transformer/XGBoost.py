# -*- coding: utf-8 -*-
"""
Transformer over time-windows (tokens=time) + generic pairwise-diff features.
- No hard-coded knowledge about sensor types.
- Windows: L=48, STRIDE=4  (multiple windows per run)
- Train on windows; evaluate both window-level and run-level (soft-vote per run).
"""

import pandas as pd
import numpy as np
import random
import warnings
from collections import defaultdict, Counter

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

import seaborn as sns
import matplotlib.pyplot as plt

# ---------------- Reproducibility ----------------
SEED = 42
random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
warnings.filterwarnings("ignore", category=UserWarning)

# ---------------- Paths & Config ----------------
CSV_PATH = "full_all_features_faulty_dataset2.csv"

# data / windowing
L        = 48
STRIDE   = 4

# training
BATCH_SIZE   = 64
EPOCHS       = 60
PATIENCE     = 10
LR           = 1e-3
WEIGHT_DECAY = 3e-4
LABEL_SMOOTH = 0.02

# transformer
D_MODEL  = 64
N_HEAD   = 4
N_LAYERS = 3
DROPOUT  = 0.1

# ---------------- Columns & labels ----------------
feature_cols = [
    'range_km', 'yaw_deg', 'pitch_deg', 'lat', 'lon', 'alt',
    'range_km_diff_median', 'range_km_mean_abs_diff',
    'yaw_deg_diff_median', 'yaw_deg_mean_abs_diff',
    'pitch_deg_diff_median', 'pitch_deg_mean_abs_diff',
    'lat_diff_median', 'lat_mean_abs_diff',
    'lon_diff_median', 'lon_mean_abs_diff',
    'alt_diff_median', 'alt_mean_abs_diff',
    'lat_diff_gps_median', 'lat_std_gps',
    'lon_diff_gps_median', 'lon_std_gps',
    'alt_diff_gps_median', 'alt_std_gps',
    'range_km_diff_ang_median', 'range_km_std_ang',
    'yaw_deg_diff_ang_median', 'yaw_deg_std_ang',
    'pitch_deg_diff_ang_median', 'pitch_deg_std_ang'
]

# ---------------- Load & basic clean ----------------
df = pd.read_csv(CSV_PATH)

# sensors order (no assumptions about types)
if 'sensor' not in df.columns:
    raise ValueError("Expected a 'sensor' column.")
sensor_order = sorted(df['sensor'].unique().tolist())

# labels
sensor_to_idx = {name: i for i, name in enumerate(sensor_order)}
faulty_to_idx = {name: i for i, name in enumerate(sorted(df['faulty_sensor'].unique()))}
# Keep the original order if you want specific display names:
display_order = ['sensor_1','sensor_2','sensor_3','sensor_4','sensor_5']
if set(display_order).issubset(set(sensor_order)):
    sensor_order = display_order
    faulty_to_idx = {name: i for i, name in enumerate(sensor_order)}
label_map = faulty_to_idx
df['label'] = df['faulty_sensor'].map(label_map).astype(int)

# numeric features
for c in feature_cols:
    df[c] = pd.to_numeric(df[c], errors='coerce')
df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan)

if 'timestamp' not in df.columns:
    raise ValueError("timestamp column is required for sequence windowing.")

df = df.sort_values(['run_id','sensor','timestamp']).reset_index(drop=True)

def interp_group(g: pd.DataFrame) -> pd.DataFrame:
    g[feature_cols] = g[feature_cols].interpolate(limit_direction='both')
    med = g[feature_cols].median(numeric_only=True)
    g[feature_cols] = g[feature_cols].fillna(med)
    return g

df = df.groupby(['run_id','sensor'], group_keys=False).apply(interp_group)

global_median = df[feature_cols].median(numeric_only=True)
df[feature_cols] = df[feature_cols].fillna(global_median)

# ---------------- Split by run_id ----------------
run_labels = df.groupby('run_id')['faulty_sensor'].first().map(label_map).astype(int)
run_ids = run_labels.index.to_numpy()
run_y   = run_labels.values

train_runs, temp_runs, y_train_runs, y_temp_runs = train_test_split(
    run_ids, run_y, test_size=0.2, random_state=SEED, stratify=run_y
)
val_runs, test_runs, y_val_runs, y_test_runs = train_test_split(
    temp_runs, y_temp_runs, test_size=0.5, random_state=SEED, stratify=y_temp_runs
)

# ---------------- Scale on TRAIN only ----------------
scaler = StandardScaler()
scaler.fit(df[df['run_id'].isin(train_runs)][feature_cols])
df[feature_cols] = scaler.transform(df[feature_cols])
df[feature_cols] = np.clip(df[feature_cols], -8, 8)
assert np.isfinite(df[feature_cols].to_numpy()).all()

# ---------------- Build (T,S,F) per run ----------------
def build_run_tensor(g_run: pd.DataFrame, sensors_in_order) -> tuple:
    ts = np.sort(g_run['timestamp'].unique())
    F = len(feature_cols); S = len(sensors_in_order); T = len(ts)
    X = np.zeros((T, S, F), dtype=np.float32)
    for si, sname in enumerate(sensors_in_order):
        gi = g_run[g_run['sensor'] == sname].set_index('timestamp').sort_index()
        gi = gi.reindex(ts)
        gi[feature_cols] = gi[feature_cols].interpolate(limit_direction='both').fillna(0.0)
        X[:, si, :] = gi[feature_cols].to_numpy(dtype=np.float32)
    y_run = label_map[g_run['faulty_sensor'].iloc[0]]
    return X, y_run, ts

def windows_from_run(X_run: np.ndarray, y_run: int, L: int, stride: int):
    T = X_run.shape[0]
    if T < L: 
        return []
    return [(X_run[s:s+L], y_run, s) for s in range(0, T - L + 1, stride)]

# --------- Generic pairwise-diff augmentation (no sensor-type knowledge) ---------
def augment_with_pairwise_abs_diffs(x_window: np.ndarray) -> np.ndarray:
    """
    x_window: (L, S, F)
    return per-time-step vector:
      concat[ flatten(S*F), flatten(pairwise |diff| over sensors) ]
    shape -> (L, S*F + C(S,2)*F)
    """
    Lw, S, F = x_window.shape
    base = x_window.reshape(Lw, S*F)  # (L, S*F)
    diffs = []
    for i in range(S):
        for j in range(i+1, S):
            diffs.append(np.abs(x_window[:, i, :] - x_window[:, j, :]))  # (L, F)
    if len(diffs) > 0:
        pdiff = np.concatenate(diffs, axis=1)  # (L, C(S,2)*F)
        out = np.concatenate([base, pdiff], axis=1)  # (L, S*F + C(S,2)*F)
    else:
        out = base
    return out.astype(np.float32)

def make_windows_for_runs(df, run_ids, sensors_in_order, L, stride):
    X_list, y_list, run_of_window = [], [], []
    for rid in run_ids:
        g = df[df['run_id'] == rid]
        X_run, y_run, ts = build_run_tensor(g, sensors_in_order)
        wins = windows_from_run(X_run, y_run, L, stride)
        for xw, yw, sidx in wins:
            x_aug = augment_with_pairwise_abs_diffs(xw)  # (L, D_per_timestep)
            X_list.append(x_aug); y_list.append(yw); run_of_window.append(rid)
    if len(X_list) == 0:
        D = len(sensors_in_order)*len(feature_cols)
        return np.zeros((0, L, D), dtype=np.float32), np.zeros((0,), dtype=np.int64), []
    X = np.stack(X_list).astype(np.float32)
    y = np.array(y_list, dtype=np.int64)
    return X, y, run_of_window

X_tr, y_tr, runs_tr = make_windows_for_runs(df, train_runs, sensor_order, L, STRIDE)
X_va, y_va, runs_va = make_windows_for_runs(df, val_runs,   sensor_order, L, STRIDE)
X_te, y_te, runs_te = make_windows_for_runs(df, test_runs,  sensor_order, L, STRIDE)

print(f"Train windows: {X_tr.shape} | Val: {X_va.shape} | Test: {X_te.shape}")
print("Class counts (train):", np.bincount(y_tr) if len(y_tr) else "[]")

# ---------------- Dataset / Loader ----------------
class WindowDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X)
        self.y = torch.tensor(y)
    def __len__(self): return len(self.X)
    def __getitem__(self, idx):
        x = self.X[idx]  # (L, D_per_timestep)
        y = self.y[idx]
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        return x, y

train_loader = DataLoader(WindowDataset(X_tr, y_tr), batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(WindowDataset(X_va, y_va), batch_size=BATCH_SIZE, shuffle=False)
test_loader  = DataLoader(WindowDataset(X_te, y_te), batch_size=BATCH_SIZE, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------- Model (token = time) ----------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))
    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class TimeTransformer(nn.Module):
    """
    Input per example: (B, L, Din) where Din = S*F + C(S,2)*F
    """
    def __init__(self, din, num_classes, d_model=64, nhead=4, nlayers=3, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(din, d_model), nn.ReLU(), nn.Dropout(dropout)
        )
        self.pos = PositionalEncoding(d_model, max_len=5000)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, batch_first=True, dropout=dropout, activation='gelu'
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=nlayers)
        self.pre_ln  = nn.LayerNorm(d_model)
        self.post_ln = nn.LayerNorm(d_model)
        self.cls_token = nn.Parameter(torch.zeros(1,1,d_model))
        self.head = nn.Sequential(
            nn.Linear(d_model, 128), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(128, 64),     nn.GELU(), nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )
    def forward(self, x):  # (B,L,Din)
        B, L, _ = x.shape
        x = torch.nan_to_num(x, 0.0, 0.0, 0.0)
        x = self.input_proj(x)
        x = self.pre_ln(x)
        x = self.pos(x)
        cls = self.cls_token.expand(B, 1, -1)
        x = torch.cat([cls, x], dim=1)        # (B, L+1, d_model)
        x = self.encoder(x)
        x_cls = self.post_ln(x[:,0,:])        # CLS pooling
        return self.head(x_cls)

# ---------------- Train / Eval helpers ----------------
def train_model(model, train_loader, val_loader, y_train, epochs=EPOCHS, lr=LR,
                weight_decay=WEIGHT_DECAY, label_smoothing=LABEL_SMOOTH, patience=PATIENCE):
    model = model.to(device)
    # class weights
    classes = np.unique(y_train)
    counts  = np.bincount(y_train, minlength=classes.max()+1)
    inv_freq = counts.sum() / np.maximum(counts, 1)
    weights = inv_freq / inv_freq.mean()
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(weights, dtype=torch.float).to(device),
        label_smoothing=label_smoothing
    )
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='max', factor=0.5, patience=3)

    best_val, best_state, no_imp = -1.0, None, 0
    for ep in range(1, epochs+1):
        model.train()
        tot_loss, correct, total = 0.0, 0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tot_loss += loss.item()
            correct += (logits.argmax(1) == yb).sum().item()
            total   += yb.size(0)
        train_acc = correct / max(total,1)
        val_acc = evaluate(model, val_loader)
        old_lr = opt.param_groups[0]['lr']
        sched.step(val_acc)
        new_lr = opt.param_groups[0]['lr']
        if new_lr != old_lr:
            print(f"[Transformer] LR {old_lr:.6f} -> {new_lr:.6f}")
        print(f"[Transformer] Epoch {ep:02d} | TrainLoss {tot_loss/len(train_loader):.3f} "
              f"| TrainAcc {train_acc:.3f} | ValAcc {val_acc:.3f}")
        if val_acc > best_val:
            best_val = val_acc
            best_state = {k: v.detach().cpu() for k,v in model.state_dict().items()}
            no_imp = 0
        else:
            no_imp += 1
            if no_imp >= patience:
                print(f"[Transformer] Early stopping at epoch {ep} (patience={patience})")
                break
    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k,v in best_state.items()})
    return model, best_val

@torch.no_grad()
def evaluate(model, data_loader):
    model.eval()
    correct, total = 0, 0
    for xb, yb in data_loader:
        xb, yb = xb.to(device), yb.to(device)
        pred = model(xb).argmax(1)
        correct += (pred == yb).sum().item()
        total   += yb.size(0)
    return correct / max(total,1)

@torch.no_grad()
def predict_proba(model, X, batch_size=256):
    model.eval()
    probs = []
    for i in range(0, len(X), batch_size):
        xb = torch.tensor(X[i:i+batch_size]).to(device)
        p  = torch.softmax(model(xb), dim=1).cpu().numpy()
        probs.append(p)
    return np.vstack(probs)

def plot_cm(cm, labels, title):
    plt.figure(figsize=(7,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted"); plt.ylabel("True"); plt.title(title); plt.show()

# ---------------- Train ----------------
Din = X_tr.shape[2]  # S*F + C(S,2)*F
num_classes = len(label_map)
model = TimeTransformer(din=Din, num_classes=num_classes,
                        d_model=D_MODEL, nhead=N_HEAD, nlayers=N_LAYERS, dropout=DROPOUT)

model, best_val = train_model(model, train_loader, val_loader, y_tr,
                              epochs=EPOCHS, lr=LR, weight_decay=WEIGHT_DECAY,
                              label_smoothing=LABEL_SMOOTH, patience=PATIENCE)

print(f"\nBest ValAcc: {best_val:.3f}")

# ---------------- Evaluate: window-level ----------------
win_probs = predict_proba(model, X_te)
win_pred  = win_probs.argmax(1)
win_acc   = accuracy_score(y_te, win_pred)
print(f"\n[Window] Accuracy: {win_acc:.3f}")
print("Classification Report (window-level):")
print(classification_report(y_te, win_pred, target_names=sensor_order))
cm_win = confusion_matrix(y_te, win_pred)
plot_cm(cm_win, sensor_order, "Confusion Matrix [Transformer-Time] (Window)")

# ---------------- Evaluate: run-level (soft-vote over windows of the same run) ----------------
# קיבוץ הסתברויות לפי run_id וביצוע ממוצע (soft vote)
by_run_probs = defaultdict(list)
by_run_true  = {}
for rid, p, y in zip(runs_te, win_probs, y_te):
    by_run_probs[rid].append(p)
    by_run_true[rid] = y

run_ids_sorted = sorted(by_run_probs.keys())
run_true = []
run_pred = []
for rid in run_ids_sorted:
    P = np.mean(by_run_probs[rid], axis=0)   # ממוצע הסתברויות
    run_pred.append(np.argmax(P))
    run_true.append(by_run_true[rid])
run_true = np.array(run_true); run_pred = np.array(run_pred)

run_acc = accuracy_score(run_true, run_pred)
print(f"\n[Run] Accuracy (soft-vote over windows): {run_acc:.3f}")
print("Classification Report (run-level):")
print(classification_report(run_true, run_pred, target_names=sensor_order))
cm_run = confusion_matrix(run_true, run_pred)
plot_cm(cm_run, sensor_order, "Confusion Matrix [Transformer-Time] (Run soft-vote)")
