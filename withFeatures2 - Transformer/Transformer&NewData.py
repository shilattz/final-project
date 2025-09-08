# =============================================
# Colab-friendly FAST pipeline:
# 1) Generate improved synthetic dataset
# 2) Train a lightweight Transformer quickly
# =============================================

import os, random, warnings, math
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix

# ---------------- Colab/CPU tweaks ----------------
torch.set_num_threads(min(8, os.cpu_count() or 8))
NUM_WORKERS = 2  # יציב ברוב סביבות Colab; אם איטי/זוכרון קצר – הורידי ל-0/1

# =======================
# ----- Config: Data ----
# =======================
INPUT_FILE  = "sensor_observations.csv"            # קובץ הבסיס התקין
OUTPUT_FILE = "full_all_features_faulty_dataset_gen.csv"
NUM_RUNS    = 800           # אפשר להגדיל/להקטין לפי זמן עיבוד
RANDOM_SEED = 42

# טווח אורך ריצה יחסי לבסיס (נגביל תמיד לפחות L+10 בהמשך)
MIN_RUN_FRAC = 0.7
MAX_RUN_FRAC = 1.0

# Missing (מוחל על הערכים המנויזים)
POINT_DROP_PROB      = 0.02
BURST_PROB_PER_SENSOR= 0.20
BURST_LEN_MIN, BURST_LEN_MAX = 3, 20

# GPS noise (meters)
GPS_BIAS_STD_M = 1.0
GPS_RW_STEP_STD_M = 0.5
GPS_WHITE_STD_M = 1.2
GPS_MULTIPATH_PROB = 0.005
GPS_MULTIPATH_JUMP_MU = 8.0
GPS_MULTIPATH_JUMP_SIGMA = 5.0

# GPS – faulty amplification
GPS_FAULTY_MULT_STD = 4.0
GPS_FAULTY_MULTIPATH_PROB = 0.03

# Angular / Range noise – healthy (degrees, km)
RANGE_A_KM = 0.02
RANGE_B = 0.005
ANG_BIAS_STD_DEG = 0.15
ANG_RW_STEP_STD_DEG = 0.02
ANG_WHITE_STD_DEG = 0.10
ANG_SPIKE_PROB = 0.003
ANG_SPIKE_JUMP_MU = 1.2
ANG_SPIKE_JUMP_SIGMA = 0.7

# Angular – faulty amplification
ANG_FAULTY_STD_MULT = 3.0
ANG_FAULTY_SPIKE_PROB = 0.02

# healthy angular pull-to-median (0..1)
ANG_PULL_HEALTHY = 0.15

# =========================
# ----- Config: Train -----
# =========================
# FAST MODE (לקיצור זמן):
L = 32                 # אורך חלון קצר
STRIDE = 8             # צעד גדול
USE_PAIR_TOKENS = False

EMBED_DIM    = 32
TIME_LAYERS  = 1
TIME_HEADS   = 2
CROSS_LAYERS = 0        # אין שלב cross-token (מהיר יותר)
CROSS_HEADS  = 2
DROPOUT      = 0.10

BATCH_SIZE = 64
EPOCHS     = 12
PATIENCE   = 3
LR         = 3e-4
WEIGHT_DECAY = 1e-4

# caps: סאב-סמפול לחלונות כדי לקצר זמן
MAX_TRAIN_WINDOWS = 4000
MAX_VAL_WINDOWS   = 600
MAX_TEST_WINDOWS  = 600

SEED = 42

# =========================================================
# ================== Data Generation ======================
# =========================================================
def ensure_time_index(df):
    if 'timestamp' in df.columns:
        return df['timestamp'].rank(method='first').astype(int).values
    if 't' in df.columns:
        return df['t'].rank(method='first').astype(int).values
    return np.arange(len(df))

def meters_to_deg(lat_deg, dx_m, dy_m):
    lat_rad = np.deg2rad(lat_deg)
    m_per_deg_lat = 111_320.0
    m_per_deg_lon = 111_320.0 * np.maximum(np.cos(lat_rad), 1e-6)
    dlat = dy_m / m_per_deg_lat
    dlon = dx_m / m_per_deg_lon
    return dlat, dlon

def random_walk(n, step_std, rng):
    if n <= 0:
        return np.array([])
    steps = rng.normal(0.0, step_std, size=n)
    return np.cumsum(steps)

def add_gps_noise_per_sensor(df_sens, is_faulty, rng):
    n = len(df_sens)
    if n == 0:
        return df_sens['lat'].values, df_sens['lon'].values, df_sens['alt'].values

    lat = df_sens['lat'].values.astype(float)
    lon = df_sens['lon'].values.astype(float)
    alt = df_sens['alt'].values.astype(float)

    rw_step  = GPS_RW_STEP_STD_M
    white_std= GPS_WHITE_STD_M
    multi_p  = GPS_MULTIPATH_PROB
    if is_faulty:
        rw_step  *= GPS_FAULTY_MULT_STD
        white_std*= GPS_FAULTY_MULT_STD
        multi_p   = GPS_FAULTY_MULTIPATH_PROB

    bias_x = rng.normal(0.0, GPS_BIAS_STD_M)
    bias_y = rng.normal(0.0, GPS_BIAS_STD_M)
    bias_alt = rng.normal(0.0, 1.0)

    rw_x = random_walk(n, rw_step, rng)
    rw_y = random_walk(n, rw_step, rng)
    rw_alt = random_walk(n, rw_step, rng)

    w_x = rng.normal(0.0, white_std, size=n)
    w_y = rng.normal(0.0, white_std, size=n)
    w_alt = rng.normal(0.0, white_std, size=n)

    jumps_mask = rng.random(size=n) < multi_p
    jump_mag = np.abs(rng.normal(GPS_MULTIPATH_JUMP_MU, GPS_MULTIPATH_JUMP_SIGMA, size=n))
    jump_sign_x = np.where(rng.random(size=n) < 0.5, -1.0, 1.0)
    jump_sign_y = np.where(rng.random(size=n) < 0.5, -1.0, 1.0)
    j_x = np.where(jumps_mask, jump_sign_x * jump_mag, 0.0)
    j_y = np.where(jumps_mask, jump_sign_y * jump_mag, 0.0)
    j_alt = np.where(jumps_mask, rng.normal(0.0, 2.0, size=n), 0.0)

    lat_ref = pd.Series(lat).fillna(np.nanmedian(lat) if np.isfinite(np.nanmedian(lat)) else 0.0).values
    dlat_bias, dlon_bias = meters_to_deg(lat_ref, bias_x, bias_y)
    dlat_rw, dlon_rw     = meters_to_deg(lat_ref, rw_x, rw_y)
    dlat_w, dlon_w       = meters_to_deg(lat_ref, w_x, w_y)
    dlat_j, dlon_j       = meters_to_deg(lat_ref, j_x, j_y)

    noisy_lat = lat + dlat_bias + dlat_rw + dlat_w + dlat_j
    noisy_lon = lon + dlon_bias + dlon_rw + dlon_w + dlon_j
    noisy_alt = alt + bias_alt + rw_alt + w_alt + j_alt
    return noisy_lat, noisy_lon, noisy_alt

def add_angular_noise_per_sensor(df_sens, is_faulty, rng, sensor_id=None, fault_mode=None):
    n = len(df_sens)
    if n == 0:
        return df_sens['range_km'].values, df_sens['yaw_deg'].values, df_sens['pitch_deg'].values

    rng_vals = df_sens['range_km'].values.astype(float)
    yaw   = df_sens['yaw_deg'].values.astype(float)
    pitch = df_sens['pitch_deg'].values.astype(float)

    a_km, b = RANGE_A_KM, RANGE_B
    ang_bias_std = ANG_BIAS_STD_DEG
    ang_rw_step  = ANG_RW_STEP_STD_DEG
    ang_white_std= ANG_WHITE_STD_DEG
    ang_spike_p  = ANG_SPIKE_PROB

    if is_faulty:
        b           *= ANG_FAULTY_STD_MULT
        ang_rw_step *= ANG_FAULTY_STD_MULT
        ang_white_std *= ANG_FAULTY_STD_MULT
        ang_spike_p  = ANG_FAULTY_SPIKE_PROB

    # חתימת תקלה אופיינית לכל סנסור זוויתי
    if fault_mode is None and is_faulty:
        sid = str(sensor_id)
        if sid in ['sensor_2','2']:
            fault_mode = rng.choice(['yaw_drift','mild_spikes','range_bias'], p=[0.6,0.2,0.2])
        elif sid in ['sensor_3','3']:
            fault_mode = rng.choice(['pitch_spikes','yaw_spikes','range_bias'], p=[0.6,0.3,0.1])
        elif sid in ['sensor_4','4']:
            fault_mode = rng.choice(['range_scale','range_bias','mixed'], p=[0.5,0.3,0.2])
        else:
            fault_mode = rng.choice(['yaw_drift','pitch_spikes','range_bias','range_scale','mixed'])

    range_scale, range_bias = 1.0, 0.0
    yaw_bias_extra, pitch_bias_extra = 0.0, 0.0
    yaw_spike_mu, pitch_spike_mu = ANG_SPIKE_JUMP_MU, ANG_SPIKE_JUMP_MU

    if is_faulty and fault_mode is not None:
        if fault_mode == 'yaw_drift':
            ang_rw_step *= 1.8
            yaw_bias_extra = rng.normal(0.4, 0.15)
        elif fault_mode == 'pitch_spikes':
            pitch_spike_mu *= 1.8; ang_spike_p *= 1.5
        elif fault_mode == 'yaw_spikes':
            yaw_spike_mu *= 1.8;   ang_spike_p *= 1.5
        elif fault_mode == 'range_bias':
            range_bias = rng.normal(0.08, 0.03)  # ~80m (ב-km)
        elif fault_mode == 'range_scale':
            range_scale = rng.normal(1.06, 0.02)
        elif fault_mode == 'mixed':
            range_scale = rng.normal(1.03, 0.015)
            yaw_bias_extra = rng.normal(0.2, 0.1)
            pitch_spike_mu *= 1.4

    # Range
    rng_std  = a_km + b * np.nan_to_num(rng_vals, nan=0.0)
    w_range  = rng.normal(0.0, 1.0, size=n) * rng_std
    spike_r  = rng.random(size=n) < (ang_spike_p * 0.5)
    spikes_range = np.where(spike_r, rng.normal(0.0, 0.2, size=n), 0.0)
    noisy_range  = (rng_vals * range_scale) + range_bias + w_range + spikes_range

    # Angles
    yaw_bias   = rng.normal(0.0, ang_bias_std) + yaw_bias_extra
    pitch_bias = rng.normal(0.0, ang_bias_std) + pitch_bias_extra
    yaw_rw   = random_walk(n, ang_rw_step, rng)
    pitch_rw = random_walk(n, ang_rw_step, rng)
    yaw_w    = rng.normal(0.0, ang_white_std, size=n)
    pitch_w  = rng.normal(0.0, ang_white_std, size=n)
    spike_m  = rng.random(size=n) < ang_spike_p
    yaw_spikes   = np.where(spike_m, rng.normal(0.0, yaw_spike_mu, size=n), 0.0)
    pitch_spikes = np.where(spike_m, rng.normal(0.0, pitch_spike_mu, size=n), 0.0)

    noisy_yaw   = yaw   + yaw_bias   + yaw_rw   + yaw_w   + yaw_spikes
    noisy_pitch = pitch + pitch_bias + pitch_rw + pitch_w + pitch_spikes
    return noisy_range, noisy_yaw, noisy_pitch

def build_missing_mask(n, is_faulty, rng):
    point_mask = rng.random(size=n) < POINT_DROP_PROB
    burst_prob = BURST_PROB_PER_SENSOR * (2.0 if is_faulty else 1.0)
    burst_mask = np.zeros(n, dtype=bool)
    if rng.random() < burst_prob:
        start  = rng.integers(0, max(1, n))
        length = int(rng.integers(BURST_LEN_MIN, int(BURST_LEN_MAX*(1.5 if is_faulty else 1.0))+1))
        end    = min(n, start + length)
        burst_mask[start:end] = True
    return point_mask | burst_mask

def apply_missing_to_noisy(df_run_noisy, sens_idx, cols, is_faulty, rng):
    n = len(sens_idx)
    if n == 0 or not cols:
        return
    miss_mask = build_missing_mask(n, is_faulty, rng)
    for c in cols:
        if c in df_run_noisy.columns:
            vals = df_run_noisy.loc[sens_idx, c].values.astype(float)
            vals[miss_mask] = np.nan
            df_run_noisy.loc[sens_idx, c] = vals

def choose_run_length(n_base, rng, min_frac=MIN_RUN_FRAC, max_frac=MAX_RUN_FRAC):
    # מבטיחים לפחות L+10 נקודות כדי שתמיד יהיו חלונות
    min_needed = L + 10
    Lr = int(max(min_needed, n_base * rng.uniform(min_frac, max_frac)))
    Lr = min(Lr, n_base)  # לא לחרוג מהבסיס
    return Lr

def generate_dataset():
    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(
            f"'{INPUT_FILE}' לא נמצא. העלי את הקובץ או שימי אותו בתיקייה הנוכחית."
        )
    if RANDOM_SEED is not None:
        np.random.seed(RANDOM_SEED); random.seed(RANDOM_SEED); torch.manual_seed(RANDOM_SEED)

    base = pd.read_csv(INPUT_FILE)
    for rc in ['sensor', 'type']:
        if rc not in base.columns:
            raise ValueError(f"Missing required column: {rc}")

    base = base.copy()
    base['_time_idx'] = ensure_time_index(base)

    ANG_COLS = [c for c in ['range_km','yaw_deg','pitch_deg'] if c in base.columns]
    GPS_COLS = [c for c in ['lat','lon','alt'] if c in base.columns]
    FEATURE_COLS = ANG_COLS + GPS_COLS

    all_runs = []
    unique_sensors = base['sensor'].unique()

    # הערכת אורך בסיס פר סנסור (מניחים דומה)
    any_sensor = base['sensor'].iloc[0]
    n_base = len(base[base['sensor']==any_sensor])

    for run_id in range(NUM_RUNS):
        rng = np.random.default_rng(np.random.randint(0, 2**31-1))
        df_run = base.copy()

        # חיתוך חלון זמן אקראי לכל סנסור באותה ריצה
        Lr = choose_run_length(n_base, rng)
        start = rng.integers(0, max(1, n_base - Lr + 1))
        keep_idx = df_run.groupby('sensor').apply(lambda g: g.iloc[start:start+Lr].index).explode().values
        df_run = df_run.loc[keep_idx].sort_values(['sensor','_time_idx']).reset_index(drop=True)

        faulty_sensor = rng.choice(unique_sensors)
        df_run_noisy = df_run.copy()

        for sensor_id, df_sens in df_run.groupby('sensor'):
            sens_idx  = df_sens.index.values
            sens_type = df_sens['type'].iloc[0]
            is_faulty = (sensor_id == faulty_sensor)

            if sens_type == 'GPS' and set(GPS_COLS) >= {'lat','lon','alt'}:
                nl, no, na = add_gps_noise_per_sensor(df_sens[GPS_COLS+['_time_idx']], is_faulty, rng)
                df_run_noisy.loc[sens_idx, 'lat'] = nl
                df_run_noisy.loc[sens_idx, 'lon'] = no
                df_run_noisy.loc[sens_idx, 'alt'] = na
            elif sens_type == 'Angular' and set(ANG_COLS) >= {'range_km','yaw_deg','pitch_deg'}:
                nr, ny, npit = add_angular_noise_per_sensor(df_sens[ANG_COLS+['_time_idx']], is_faulty, rng,
                                                            sensor_id=sensor_id, fault_mode=None)
                df_run_noisy.loc[sens_idx, 'range_km']  = nr
                df_run_noisy.loc[sens_idx, 'yaw_deg']   = ny
                df_run_noisy.loc[sens_idx, 'pitch_deg'] = npit

            # חסרים על noisy
            cols_for_missing = [c for c in FEATURE_COLS if c in df_sens.columns]
            if cols_for_missing:
                apply_missing_to_noisy(df_run_noisy, sens_idx, cols_for_missing, is_faulty, rng)

        # משיכת בריאים למדיֵן הזוויתיים (התקול נשאר רחוק)
        if set(ANG_COLS).issubset(df_run_noisy.columns):
            ang_mask = (df_run_noisy['type']=='Angular')
            g_ang = df_run_noisy.loc[ang_mask]
            if not g_ang.empty:
                med_by_t = g_ang.groupby('_time_idx')[ANG_COLS].transform('median')
                for idx in g_ang.index:
                    sid = df_run_noisy.loc[idx, 'sensor']
                    if sid != faulty_sensor:
                        df_run_noisy.loc[idx, ANG_COLS] = (
                            (1 - ANG_PULL_HEALTHY) * df_run_noisy.loc[idx, ANG_COLS].values +
                            ANG_PULL_HEALTHY * med_by_t.loc[idx, ANG_COLS].values
                        )

        df_run_noisy['faulty_sensor'] = faulty_sensor
        df_run_noisy['run_id'] = run_id
        all_runs.append(df_run_noisy)

    full_dataset = pd.concat(all_runs, ignore_index=True)
    full_dataset.drop(columns=['_time_idx'], errors='ignore', inplace=True)

    # one-hot type
    type_dummies = pd.get_dummies(full_dataset['type'], prefix='type')
    full_dataset = pd.concat([full_dataset, type_dummies], axis=1)

    # ---- Feature engineering (כמו אצלך, עם הרחבות) ----
    feature_cols = [c for c in ['range_km','yaw_deg','pitch_deg','lat','lon','alt'] if c in full_dataset.columns]
    new_feature_dfs = []
    for rid, group in full_dataset.groupby("run_id", sort=False):
        if 'type' in group.columns:
            if set(['lat','lon','alt']).issubset(group.columns):
                gps_mask = (group['type']=='GPS')
                gps_group = group.loc[gps_mask]
                for col in ['lat','lon','alt']:
                    if gps_group[col].notna().sum()>1:
                        gps_median = np.nanmedian(gps_group[col])
                        gps_std    = np.nanstd(gps_group[col])
                        group.loc[gps_group.index, f"{col}_diff_gps_median"] = np.abs(gps_group[col] - gps_median)
                        group.loc[gps_group.index, f"{col}_std_gps"] = gps_std
            if set(['range_km','yaw_deg','pitch_deg']).issubset(group.columns):
                ang_mask = (group['type']=='Angular')
                ang_group = group.loc[ang_mask]
                for col in ['range_km','yaw_deg','pitch_deg']:
                    if ang_group[col].notna().sum()>1:
                        ang_median = np.nanmedian(ang_group[col])
                        ang_std    = np.nanstd(ang_group[col])
                        group.loc[ang_group.index, f"{col}_diff_ang_median"] = np.abs(ang_group[col] - ang_median)
                        group.loc[ang_group.index, f"{col}_std_ang"] = ang_std

        for col in feature_cols:
            vals = group[col].values
            if np.all(np.isnan(vals)): continue
            median = np.nanmedian(vals)
            group[f"{col}_diff_median"] = np.abs(group[col] - median)

            v = pd.Series(vals)
            mad_list = []
            for i, xi in enumerate(v):
                if pd.isna(xi):
                    mad_list.append(np.nan); continue
                others = v.drop(v.index[i]).values
                mad_list.append(np.nanmean(np.abs(xi - others)))
            group[f"{col}_mean_abs_diff"] = mad_list

        new_feature_dfs.append(group)

    full_dataset = pd.concat(new_feature_dfs, ignore_index=True)
    full_dataset.to_csv(OUTPUT_FILE, index=False)
    print(f"\n{NUM_RUNS} runs created. Saved: {OUTPUT_FILE}")
    return OUTPUT_FILE

# =========================================================
# ================== Transformer (FAST) ===================
# =========================================================
def train_transformer(csv_path):
    # Repro
    random.seed(SEED); np.random.seed(SEED)
    torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # --- Load ---
    df = pd.read_csv(csv_path)
    sensor_order = ['sensor_1','sensor_2','sensor_3','sensor_4','sensor_5']
    label_map = {s:i for i,s in enumerate(sensor_order)}
    df['label'] = df['faulty_sensor'].map(label_map)

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
    angular_base_feats = ['yaw_deg','pitch_deg','range_km']
    angular_feat_idx = [feature_cols.index(f) for f in angular_base_feats]

    # סדר + אינטרפולציה
    sort_cols = [c for c in ['run_id','sensor','timestamp'] if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols).reset_index(drop=True)
    df[feature_cols] = df[feature_cols].interpolate(limit_direction='both')

    # חלוקה לפי run_id
    run_labels_series = df.groupby('run_id')['faulty_sensor'].first().map(label_map)
    run_ids = run_labels_series.index.to_numpy()
    run_y   = run_labels_series.values

    train_ids, temp_ids, y_train, y_temp = train_test_split(
        run_ids, run_y, test_size=0.2, random_state=SEED, stratify=run_y
    )
    val_ids, test_ids, y_val, y_test = train_test_split(
        temp_ids, y_temp, test_size=0.5, random_state=SEED, stratify=y_temp
    )

    # סקיילר (fit רק על train)
    scaler = StandardScaler()
    scaler.fit(df[df['run_id'].isin(train_ids)][feature_cols])
    df[feature_cols] = scaler.transform(df[feature_cols])

    # ---------- Dataset ----------
    class SensorSequenceDataset(Dataset):
        def __init__(self, df, run_ids, sensor_order, feature_cols, angular_feat_idx, L=32, stride=8, use_pair_tokens=False):
            self.sensor_order = sensor_order
            self.feature_cols = feature_cols
            self.angular_feat_idx = angular_feat_idx
            self.L = L
            self.use_pair_tokens = use_pair_tokens
            self.samples = []

            for run_id, g_run in df[df['run_id'].isin(run_ids)].groupby('run_id'):
                label = label_map[g_run['faulty_sensor'].iloc[0]]
                sensors_seq, types = [], []
                min_len = np.inf
                for s in sensor_order:
                    g = g_run[g_run['sensor']==s]
                    if 'timestamp' in g.columns:
                        g = g.sort_values('timestamp')
                    seq = g[self.feature_cols].to_numpy(dtype=np.float32)
                    if len(seq)==0:
                        seq = np.zeros((1, len(self.feature_cols)), dtype=np.float32)
                        # ניחוש סוג לפי השם
                        tval = 0 if s in ['sensor_1','sensor_5'] else 1
                    else:
                        tval = 0 if g['type'].iloc[0]=='GPS' else 1
                    sensors_seq.append(seq)
                    types.append(tval)
                    min_len = min(min_len, len(seq))

                min_len = int(min_len) if np.isfinite(min_len) else 0
                for start in range(0, max(min_len - L + 1, 0), stride):
                    window = [seq[start:start+L] for seq in sensors_seq]
                    if not all(len(w)==L for w in window): 
                        continue

                    X_list = window.copy()
                    t_list = types.copy()

                    # pair tokens כבוי במצב מהיר
                    if self.use_pair_tokens:
                        w2, w3, w4 = window[1], window[2], window[3]
                        def time_diff(x):
                            v = np.zeros_like(x)
                            v[1:, self.angular_feat_idx] = x[1:, self.angular_feat_idx] - x[:-1, self.angular_feat_idx]
                            return v
                        def make_pair(a,b):
                            d_abs = np.zeros_like(a); d_abs[:, self.angular_feat_idx] = np.abs(a[:, self.angular_feat_idx]-b[:, self.angular_feat_idx])
                            d_signed = np.zeros_like(a); d_signed[:, self.angular_feat_idx] = (a[:, self.angular_feat_idx]-b[:, self.angular_feat_idx])
                            v_abs = time_diff(d_abs); v_signed = time_diff(d_signed)
                            return (d_abs + d_signed + v_abs + v_signed)/4.0
                        X_list += [make_pair(w2,w3), make_pair(w2,w4), make_pair(w3,w4)]
                        t_list += [2,2,2]

                    X = np.stack(X_list, axis=0)  # (S,L,F)
                    t = np.array(t_list, dtype=np.int64)
                    self.samples.append((X, t, int(label), int(run_id)))

        def __len__(self): return len(self.samples)
        def __getitem__(self, idx):
            X, types, y, rid = self.samples[idx]
            return torch.tensor(X), torch.tensor(types), torch.tensor(y), int(rid)

    train_ds = SensorSequenceDataset(df, train_ids, sensor_order, feature_cols, angular_feat_idx, L=L, stride=STRIDE, use_pair_tokens=USE_PAIR_TOKENS)
    val_ds   = SensorSequenceDataset(df, val_ids,   sensor_order, feature_cols, angular_feat_idx, L=L, stride=STRIDE, use_pair_tokens=USE_PAIR_TOKENS)
    test_ds  = SensorSequenceDataset(df, test_ids,  sensor_order, feature_cols, angular_feat_idx, L=L, stride=STRIDE, use_pair_tokens=USE_PAIR_TOKENS)

    # --- Subsample (caps) ---
    def subsample_samples(sample_list, max_n):
        if max_n is None or len(sample_list) <= max_n:
            return sample_list
        idx = list(range(len(sample_list)))
        random.shuffle(idx)
        idx = idx[:max_n]
        return [sample_list[i] for i in idx]

    train_ds.samples = subsample_samples(train_ds.samples, MAX_TRAIN_WINDOWS)
    val_ds.samples   = subsample_samples(val_ds.samples,   MAX_VAL_WINDOWS)
    test_ds.samples  = subsample_samples(test_ds.samples,  MAX_TEST_WINDOWS)

    print(f"Train windows: {len(train_ds)}  Val: {len(val_ds)}  Test: {len(test_ds)}")

    # --- DataLoaders ---
    pin = (device.type == 'cuda')
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=NUM_WORKERS, pin_memory=pin, persistent_workers=NUM_WORKERS>0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=pin, persistent_workers=NUM_WORKERS>0)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=pin, persistent_workers=NUM_WORKERS>0)

    # --- Loss weights (balanced CE) ---
    train_labels_windows = np.array([lbl for _,_,lbl,_ in train_ds.samples])
    classes = np.unique(train_labels_windows)
    w = compute_class_weight(class_weight='balanced', classes=classes, y=train_labels_windows)
    cw = torch.tensor(w, dtype=torch.float).to(device)

    # ---------- Model ----------
    class LearnedPositionalEncoding(nn.Module):
        def __init__(self, max_len, d_model):
            super().__init__()
            self.pe = nn.Embedding(max_len, d_model)
        def forward(self, x):
            B,L_,E = x.shape
            pos = torch.arange(L_, device=x.device)
            return x + self.pe(pos).unsqueeze(0).expand(B,L_,E)

    class TinyTransformer(nn.Module):
        def __init__(self, F, embed_dim=32, num_classes=5, num_tokens=5, time_layers=1, time_heads=2, dropout=0.1, L=32):
            super().__init__()
            E = embed_dim
            self.E = E
            self.num_tokens = num_tokens

            # אנקודרים פר-טוקן (חמישה סנסורים אמיתיים)
            self.encoders = nn.ModuleList([nn.Sequential(nn.Linear(F,E), nn.ReLU(), nn.LayerNorm(E)) for _ in range(5)])
            self.posenc   = LearnedPositionalEncoding(L, E)
            t_layer = nn.TransformerEncoderLayer(d_model=E, nhead=time_heads, batch_first=True, dropout=dropout, activation='gelu')
            self.temporal = nn.TransformerEncoder(t_layer, num_layers=time_layers)

            # הטמעת זהות טוקן (מבחינה בין טוקנים אחרי סיכום בזמן)
            self.token_id_emb = nn.Embedding(num_tokens, E)

            self.head = nn.Sequential(
                nn.Linear(num_tokens*E, 128), nn.ReLU(), nn.Dropout(0.25),
                nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.15),
                nn.Linear(64, num_classes)
            )

        def forward(self, X, types):
            # X: (B,S,L,F), types: (B,S)  | S=5
            B,S,L_,Fdim = X.shape
            outs = []
            for i in range(S):
                Xi = X[:,i]                # (B,L,F)
                Zi = self.encoders[i](Xi)  # (B,L,E)
                Zi = self.posenc(Zi)
                Zi = self.temporal(Zi).mean(dim=1)  # (B,E)
                Zi = Zi + self.token_id_emb(torch.full((B,), i, device=X.device))
                outs.append(Zi)
            Z = torch.stack(outs, dim=1)       # (B,S,E)
            H = Z.reshape(B, -1)               # (B,S*E)
            logits = self.head(H)              # (B,5)
            return logits

    model = TinyTransformer(
        F=len(feature_cols), embed_dim=EMBED_DIM, num_classes=5, num_tokens=5,
        time_layers=TIME_LAYERS, time_heads=TIME_HEADS, dropout=DROPOUT, L=L
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    def run_epoch(dl, train=False):
        model.train() if train else model.eval()
        loss_sum, correct, total = 0.0, 0, 0
        for X, types, y, _ in dl:
            X, types, y = X.to(device), types.to(device), y.to(device)
            if train: optimizer.zero_grad()
            logits = model(X, types)
            loss = F.cross_entropy(logits, y, weight=cw)
            if train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            loss_sum += loss.item()
            _, pred = logits.max(1)
            correct += (pred==y).sum().item()
            total += y.size(0)
        acc = correct/total if total else 0.0
        return loss_sum, acc

    best_val = -1.0
    best_state = None
    no_improve = 0
    for epoch in range(1, EPOCHS+1):
        tr_loss, tr_acc = run_epoch(train_loader, train=True)
        val_loss, val_acc = run_epoch(val_loader,   train=False)
        print(f"Epoch {epoch:02d} | TrainLoss: {tr_loss:.2f} | TrainAcc: {tr_acc:.3f} | ValAcc: {val_acc:.3f}")
        if val_acc > best_val:
            best_val = val_acc
            best_state = {k: v.detach().cpu() for k,v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print("Early stopping.")
                break
    if best_state:
        model.load_state_dict({k: v.to(device) for k,v in best_state.items()})

    # -------- Window-level eval --------
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for X, types, y, _ in test_loader:
            X, types = X.to(device), types.to(device)
            logits = model(X, types)
            _, pred = logits.max(1)
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(y.numpy())

    print("\nWindow-level Accuracy:", np.mean(np.array(all_preds)==np.array(all_labels)))
    print(classification_report(all_labels, all_preds, target_names=sensor_order, zero_division=0))
    print(confusion_matrix(all_labels, all_preds))

    # -------- Run-level eval (ממוצע לוגיטים לריצה) --------
    from collections import defaultdict
    logit_sum_by_run = defaultdict(lambda: None)
    cnt_by_run = defaultdict(int)
    true_by_run = {}

    with torch.no_grad():
        for X, types, y, rids in test_loader:
            X, types = X.to(device), types.to(device)
            logits = model(X, types).cpu().numpy()
            rids = [int(r) for r in rids]
            for rid, yt in zip(rids, y):
                true_by_run[rid] = int(yt)
            for rid, logit in zip(rids, logits):
                if logit_sum_by_run[rid] is None:
                    logit_sum_by_run[rid] = logit.astype(np.float64, copy=True)
                else:
                    logit_sum_by_run[rid] += logit
                cnt_by_run[rid] += 1

    y_true_run, y_pred_run = [], []
    for rid, logit_sum in logit_sum_by_run.items():
        avg_logits = logit_sum / max(cnt_by_run[rid], 1)
        y_pred_run.append(int(np.argmax(avg_logits)))
        y_true_run.append(true_by_run[rid])

    y_true_run = np.array(y_true_run); y_pred_run = np.array(y_pred_run)
    print("\nRun-level Accuracy:", (y_true_run==y_pred_run).mean())
    print(classification_report(y_true_run, y_pred_run, target_names=sensor_order, zero_division=0))
    print(confusion_matrix(y_true_run, y_pred_run))

# =======================
# ---------- Main -------
# =======================
if __name__ == "__main__":
    out_csv = generate_dataset()      # יוצר את הדאטה הסינתטי ושומר ל-CSV
    train_transformer(out_csv)        # מאמן טרנספורמר "מהיר" ומדפיס תוצאות
