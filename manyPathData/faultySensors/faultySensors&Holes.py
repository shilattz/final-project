# generate_faulty_dataset_with_features.py
# -*- coding: utf-8 -*-
"""
Create a realistic, labeled dataset for sensor-fault detection
from your deterministic sensor_observations.csv.

Key points:
- One faulty sensor per run (chosen uniformly from existing sensor IDs).
- Add realistic noise to GPS and Angular sensors (random walk + white + occasional jumps/spikes).
- Apply missingness on the *noisy* values (not on the clean base) with a per-segment cap
  to avoid "all-NaN slice" issues.
- Add robust, missing-aware per-(run_id,sensor) features and within-run relative deviations.
- Outputs a single CSV ready for modeling.

Input  : sensor_observations.csv  (columns like: run_id, t or timestamp, sensor, type, lat/lon/alt or range_km/yaw_deg/pitch_deg)
Output : full_all_features_faulty_dataset.csv
"""

import numpy as np
import pandas as pd
from typing import Tuple

# =========================
# Config
# =========================
INPUT_FILE  = "manyPathData/sensoreObservation/sensor_observations.csv"
OUTPUT_FILE = "manyPathData/faultySensors/full_faulty_dataset.csv"

# How many synthetic runs to generate.
# If None -> use the distinct run_id count from the input as-is (1× each route)
# If int  -> replicate the base routes until reaching approximately NUM_RUNS runs
NUM_RUNS: int | None = None   # e.g. 400 or None

RANDOM_SEED = 42  # set for reproducibility (or None)

# Missing params (moderate) + CAP per (run,sensor,column)
POINT_DROP_PROB        = 0.01   # per-sample missing probability
BURST_PROB_PER_SENSOR  = 0.07   # probability to start a short burst (faulty doubles)
BURST_LEN_MIN, BURST_LEN_MAX = 3, 8
MAX_MISS_FRAC          = 0.40   # cap: at most 40% missing per (run,sensor,column)
KEEP_EDGES             = True   # keep first/last sample present

# GPS noise (meters) – healthy (moderate)
GPS_BIAS_STD_M          = 1.0
GPS_RW_STEP_STD_M       = 0.5
GPS_WHITE_STD_M         = 1.2
GPS_MULTIPATH_PROB      = 0.005
GPS_MULTI_JUMP_MU       = 4.0    # meters
GPS_MULTI_JUMP_SIGMA    = 2.0

# GPS – faulty multipliers (separate for RW and white)
GPS_FAULTY_RW_MULT      = 3.4
GPS_FAULTY_WHITE_MULT   = 1.8
GPS_FAULTY_MULTIPATH_PROB = 0.010

# Angular / Range noise – healthy
RANGE_A_KM              = 0.02
RANGE_B                 = 0.005
ANG_BIAS_STD_DEG        = 0.15
ANG_RW_STEP_STD_DEG     = 0.02
ANG_WHITE_STD_DEG       = 0.10
ANG_SPIKE_PROB          = 0.003
ANG_SPIKE_JUMP_MU       = 0.25   # deg
ANG_SPIKE_JUMP_SIGMA    = 0.15   # deg

# Angular – faulty (separate multipliers)
ANG_FAULTY_RW_MULT      = 2.6
ANG_FAULTY_WHITE_MULT   = 1.7
ANG_FAULTY_RANGE_B_MULT = 1.6
ANG_FAULTY_SPIKE_PROB   = 0.008

# Cap for range noise so distant targets don't whiteout (km)
RANGE_NOISE_CAP_KM      = 0.12   # ~120 m

# Common-mode drifts shared by all sensors in a run (soft)
COMMON_DRIFT_STD_M_GPS  = 0.20   # meters (x,y,z)
COMMON_DRIFT_STD_DEG    = 0.008  # yaw/pitch (deg)
COMMON_DRIFT_STD_KM     = 0.003  # range (km)

# Healthy per-sensor variability (lognormal ~ 1.0 ±10%)
HEALTHY_LN_MEAN         = 0.0
HEALTHY_LN_SIGMA        = 0.10

# Minimal valid samples to compute time-dispersion stats safely
MIN_VALID               = 3


# =========================
# Helpers
# =========================
def ensure_time_index(df: pd.DataFrame) -> np.ndarray:
    if 'timestamp' in df.columns:
        return df['timestamp'].rank(method='first').astype(int).values
    if 't' in df.columns:
        return df['t'].rank(method='first').astype(int).values
    # fallback: per-run ordering later
    return np.arange(len(df))


def meters_to_deg(lat_deg, dx_m, dy_m):
    lat_rad = np.deg2rad(lat_deg)
    m_per_deg_lat = 111_320.0
    m_per_deg_lon = 111_320.0 * np.maximum(np.cos(lat_rad), 1e-6)
    dlat = dy_m / m_per_deg_lat
    dlon = dx_m / m_per_deg_lon
    return dlat, dlon


def random_walk(n: int, step_std: float, rng: np.random.Generator) -> np.ndarray:
    if n <= 0:
        return np.array([])
    steps = rng.normal(0.0, step_std, size=n)
    return np.cumsum(steps)


# =========================
# Missing (capped) builders
# =========================
def build_missing_mask_capped(
    n: int,
    base_point_p: float,
    burst_prob: float,
    len_min: int,
    len_max: int,
    rng: np.random.Generator,
    is_faulty: bool = False,
    max_frac: float = MAX_MISS_FRAC,
    keep_edges: bool = KEEP_EDGES,
) -> np.ndarray:
    m = rng.random(size=n) < base_point_p
    # burst (faulty doubles probability)
    bprob = burst_prob * (2.0 if is_faulty else 1.0)
    if rng.random() < bprob:
        start = rng.integers(0, max(1, n))
        length = int(rng.integers(len_min, int(len_max * (1.5 if is_faulty else 1.0)) + 1))
        end = min(n, start + length)
        m[start:end] = True

    # cap missing fraction
    k = int(max_frac * n)
    miss_idx = np.flatnonzero(m)
    if len(miss_idx) > k:
        keep_idx = rng.choice(miss_idx, size=k, replace=False)
        m[:] = False
        m[keep_idx] = True

    # keep edges
    if keep_edges and n > 0:
        m[0] = False
        m[-1] = False
    return m


def apply_missing_to_noisy(
    df_run_noisy: pd.DataFrame,
    sens_idx: np.ndarray,
    cols: list[str],
    is_faulty: bool,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    n = len(sens_idx)
    if n == 0 or not cols:
        return np.zeros(n, dtype=bool), np.zeros(n, dtype=bool)

    miss_any = np.zeros(n, dtype=bool)
    burst_any = np.zeros(n, dtype=bool)  # placeholder (we keep simple indicator)

    for c in cols:
        if c not in df_run_noisy.columns:
            continue
        m = build_missing_mask_capped(
            n=n,
            base_point_p=POINT_DROP_PROB,
            burst_prob=BURST_PROB_PER_SENSOR,
            len_min=BURST_LEN_MIN,
            len_max=BURST_LEN_MAX,
            rng=rng,
            is_faulty=is_faulty,
            max_frac=MAX_MISS_FRAC,
            keep_edges=KEEP_EDGES,
        )
        vals = df_run_noisy.loc[sens_idx, c].to_numpy(dtype=float, copy=True)
        vals[m] = np.nan
        df_run_noisy.loc[sens_idx, c] = vals
        miss_any |= m
        # (optional) compute real "burst_any" by scanning m for runs>1 if you need it later
    return miss_any, burst_any


# =========================
# Noise models
# =========================
def add_gps_noise_per_sensor(df_sens: pd.DataFrame,
                             is_faulty: bool,
                             rng: np.random.Generator,
                             common_rw_xyz=None,
                             healthy_mult: float = 1.0):
    """
    df_sens columns: ['lat','lon','alt','_time_idx']
    """
    n = len(df_sens)
    if n == 0:
        return df_sens['lat'].values, df_sens['lon'].values, df_sens['alt'].values

    lat = df_sens['lat'].values.astype(float)
    lon = df_sens['lon'].values.astype(float)
    alt = df_sens['alt'].values.astype(float)

    # base (with light healthy variability)
    rw_step   = GPS_RW_STEP_STD_M * healthy_mult
    white_std = GPS_WHITE_STD_M   * healthy_mult
    multi_p   = GPS_MULTIPATH_PROB

    if is_faulty:
        rw_step   *= GPS_FAULTY_RW_MULT
        white_std *= GPS_FAULTY_WHITE_MULT
        multi_p    = GPS_FAULTY_MULTIPATH_PROB

    # biases (m)
    bias_x   = rng.normal(0.0, GPS_BIAS_STD_M)
    bias_y   = rng.normal(0.0, GPS_BIAS_STD_M)
    bias_alt = rng.normal(0.0, 1.0)

    # RW & white (m)
    rw_x = random_walk(n, rw_step, rng)
    rw_y = random_walk(n, rw_step, rng)
    rw_alt = random_walk(n, rw_step, rng)

    w_x = rng.normal(0.0, white_std, size=n)
    w_y = rng.normal(0.0, white_std, size=n)
    w_alt = rng.normal(0.0, white_std, size=n)

    # multipath (m)
    jumps_mask = rng.random(size=n) < multi_p
    jump_mag   = np.abs(rng.normal(GPS_MULTI_JUMP_MU, GPS_MULTI_JUMP_SIGMA, size=n))
    j_sign_x   = np.where(rng.random(size=n) < 0.5, -1.0, 1.0)
    j_sign_y   = np.where(rng.random(size=n) < 0.5, -1.0, 1.0)
    j_x = np.where(jumps_mask, j_sign_x * jump_mag, 0.0)
    j_y = np.where(jumps_mask, j_sign_y * jump_mag, 0.0)
    j_alt = np.where(jumps_mask, rng.normal(0.0, 1.0, size=n), 0.0)

    # common-mode drift (meters)
    if common_rw_xyz is not None:
        cm_x, cm_y, cm_alt = common_rw_xyz
    else:
        cm_x = cm_y = cm_alt = np.zeros(n)

    # convert meters->degrees
    lat_ref = pd.Series(lat).fillna(np.nanmedian(lat) if np.isfinite(np.nanmedian(lat)) else 0.0).values
    dlat_bias, dlon_bias = meters_to_deg(lat_ref, bias_x, bias_y)
    dlat_rw,   dlon_rw   = meters_to_deg(lat_ref, rw_x, rw_y)
    dlat_w,    dlon_w    = meters_to_deg(lat_ref, w_x, w_y)
    dlat_j,    dlon_j    = meters_to_deg(lat_ref, j_x, j_y)
    dlat_cm,   dlon_cm   = meters_to_deg(lat_ref, cm_x, cm_y)

    noisy_lat = lat + dlat_bias + dlat_rw + dlat_w + dlat_j + dlat_cm
    noisy_lon = lon + dlon_bias + dlon_rw + dlon_w + dlon_j + dlon_cm
    noisy_alt = alt + bias_alt + rw_alt + w_alt + j_alt + cm_alt

    return noisy_lat, noisy_lon, noisy_alt


def add_angular_noise_per_sensor(df_sens: pd.DataFrame,
                                 is_faulty: bool,
                                 rng: np.random.Generator,
                                 common_drifts=None,
                                 healthy_mult: float = 1.0):
    """
    df_sens columns: ['range_km','yaw_deg','pitch_deg','_time_idx']
    """
    n = len(df_sens)
    if n == 0:
        return df_sens['range_km'].values, df_sens['yaw_deg'].values, df_sens['pitch_deg'].values

    rng_vals = df_sens['range_km'].values.astype(float)
    yaw      = df_sens['yaw_deg'].values.astype(float)
    pitch    = df_sens['pitch_deg'].values.astype(float)

    b             = RANGE_B * healthy_mult
    ang_rw_step   = ANG_RW_STEP_STD_DEG   * healthy_mult
    ang_white_std = ANG_WHITE_STD_DEG     * healthy_mult
    ang_spike_p   = ANG_SPIKE_PROB

    if is_faulty:
        b             *= ANG_FAULTY_RANGE_B_MULT
        ang_rw_step   *= ANG_FAULTY_RW_MULT
        ang_white_std *= ANG_FAULTY_WHITE_MULT
        ang_spike_p    = ANG_FAULTY_SPIKE_PROB

    # range noise (capped)
    rng_std = RANGE_A_KM + b * np.nan_to_num(rng_vals, nan=0.0)
    rng_std = np.minimum(rng_std, RANGE_NOISE_CAP_KM)
    w_range = rng.normal(0.0, 1.0, size=n) * rng_std

    # small spikes in range (~20 m -> 0.02 km)
    spike_mask_range = rng.random(size=n) < (ang_spike_p * 0.5)
    spikes_range = np.where(spike_mask_range, rng.normal(0.0, 0.02, size=n), 0.0)
    noisy_range = rng_vals + w_range + spikes_range

    # angles
    yaw_bias   = rng.normal(0.0, ANG_BIAS_STD_DEG)
    pitch_bias = rng.normal(0.0, ANG_BIAS_STD_DEG)
    yaw_rw     = random_walk(n, ang_rw_step, rng)
    pitch_rw   = random_walk(n, ang_rw_step, rng)
    yaw_w      = rng.normal(0.0, ang_white_std, size=n)
    pitch_w    = rng.normal(0.0, ang_white_std, size=n)
    spike_mask = rng.random(size=n) < ang_spike_p
    yaw_spikes   = np.where(spike_mask,   rng.normal(0.0, ANG_SPIKE_JUMP_MU, size=n), 0.0)
    pitch_spikes = np.where(spike_mask,   rng.normal(0.0, ANG_SPIKE_JUMP_MU, size=n), 0.0)

    if common_drifts is not None:
        cm_range, cm_yaw, cm_pitch = common_drifts
    else:
        cm_range = cm_yaw = cm_pitch = np.zeros(n)

    noisy_yaw   = yaw   + yaw_bias   + yaw_rw   + yaw_w   + yaw_spikes   + cm_yaw
    noisy_pitch = pitch + pitch_bias + pitch_rw + pitch_w + pitch_spikes + cm_pitch
    noisy_range = noisy_range + cm_range

    return noisy_range, noisy_yaw, noisy_pitch


# =========================
# Robust feature helpers
# =========================
def longest_nan_run(x: pd.Series) -> int:
    m = pd.isna(x).astype(int).values
    best = cur = 0
    for v in m:
        cur = cur + 1 if v else 0
        best = max(best, cur)
    return int(best)


def count_nan_bursts(x: pd.Series) -> int:
    m = pd.isna(x).astype(int).values
    if m.size == 0:
        return 0
    return int(np.sum((m[1:] == 1) & (m[:-1] == 0)) + (m[0] == 1))


def spike_count_series(x: pd.Series) -> int:
    v = x.values
    if np.sum(~np.isnan(v)) < 2:
        return 0
    diffs = (x - x.shift(1)).abs()
    if diffs.notna().sum() == 0:
        return 0
    med = np.nanmedian(diffs)
    mad = np.nanmedian(np.abs(diffs - med))
    thr = 3.0 * (mad if mad > 0 else (np.nanstd(diffs) if np.isfinite(np.nanstd(diffs)) else 0.0))
    if not np.isfinite(thr) or thr == 0:
        return 0
    return int(np.nansum((diffs > thr).astype(float)))


def safe_stats_over_time(s: pd.Series) -> tuple[float, float, float, float, int]:
    """Return (std, mean, median, iqr, valid_count) with NaN/0-safe fallbacks."""
    valid = s.dropna().values
    cnt = valid.size
    if cnt >= MIN_VALID:
        std_   = float(np.nanstd(valid)) if np.isfinite(np.nanstd(valid)) else 0.0
        mean_  = float(np.nanmean(valid))
        med_   = float(np.nanmedian(valid))
        iqr_   = float(np.nanpercentile(valid, 75) - np.nanpercentile(valid, 25))
        return std_, mean_, med_, iqr_, cnt
    else:
        return 0.0, float(np.nan), float(np.nan), 0.0, cnt


# =========================
# Main pipeline
# =========================
def main():
    if RANDOM_SEED is not None:
        np.random.seed(RANDOM_SEED)

    base = pd.read_csv(INPUT_FILE)
    required = {'route_id', 'sensor', 'type'}
    if not required.issubset(base.columns):
        raise ValueError(f"Missing columns in {INPUT_FILE}: {required - set(base.columns)}")

    # normalize time column
    if 'timestamp' not in base.columns and 't' not in base.columns:
        # create per-(run_id) time index
        base = base.sort_values(['route_id']).reset_index(drop=True)
        base['t'] = base.groupby('route_id').cumcount()
    base['_time_idx'] = ensure_time_index(base)

    ANG_COLS = [c for c in ['range_km', 'yaw_deg', 'pitch_deg'] if c in base.columns]
    GPS_COLS = [c for c in ['lat', 'lon', 'alt'] if c in base.columns]
    FEATURE_COLS = ANG_COLS + GPS_COLS

    # Determine how many runs to synthesize
    base_run_ids = sorted(base['route_id'].unique())
    base_n_runs = len(base_run_ids)

    if NUM_RUNS is None or NUM_RUNS <= base_n_runs:
        synth_plan = [(rid, 0) for rid in base_run_ids]  # single pass of each
    else:
        # replicate base runs in round-robin until ~NUM_RUNS
        repeats = int(np.ceil(NUM_RUNS / base_n_runs))
        synth_plan = []
        k = 0
        for rep in range(repeats):
            for rid in base_run_ids:
                synth_plan.append((rid, rep))
                k += 1
                if k >= NUM_RUNS:
                    break
            if k >= NUM_RUNS:
                break

    # Prepare seeds per synthesized run (deterministic if RANDOM_SEED set)
    if RANDOM_SEED is not None:
        ss = np.random.SeedSequence(RANDOM_SEED)
        child_seeds = ss.spawn(len(synth_plan))
    else:
        child_seeds = [None] * len(synth_plan)

    all_runs = []
    unique_sensors = base['sensor'].unique()

    # Build synthetic runs
    for new_run_id, (src_run_id, rep_idx) in enumerate(synth_plan):
        rng = (np.random.default_rng(child_seeds[new_run_id])
               if child_seeds[new_run_id] is not None else np.random.default_rng())

        df_run = base[base['route_id'] == src_run_id].copy().reset_index(drop=True)
        # choose faulty sensor uniformly from available sensors
        faulty_sensor = rng.choice(unique_sensors)

        df_run_noisy = df_run.copy()

        n_all = len(df_run_noisy)

        # Common-mode drifts across the *whole* run, then sliced per sensor indices
        cm_rng = (np.random.default_rng(child_seeds[new_run_id].entropy)
                  if child_seeds[new_run_id] is not None else np.random.default_rng())
        cm_rw_x_full    = random_walk(n_all, COMMON_DRIFT_STD_M_GPS, cm_rng)
        cm_rw_y_full    = random_walk(n_all, COMMON_DRIFT_STD_M_GPS, cm_rng)
        cm_rw_alt_full  = random_walk(n_all, COMMON_DRIFT_STD_M_GPS, cm_rng)
        cm_rw_range_full= random_walk(n_all, COMMON_DRIFT_STD_KM,    cm_rng)
        cm_rw_yaw_full  = random_walk(n_all, COMMON_DRIFT_STD_DEG,   cm_rng)
        cm_rw_pitch_full= random_walk(n_all, COMMON_DRIFT_STD_DEG,   cm_rng)

        # Apply noise + missingness per sensor block (keep df order!)
        for sensor_id, df_sens in df_run_noisy.groupby('sensor', sort=False):
            sens_idx  = df_sens.index.values
            sens_type = df_sens['type'].iloc[0]
            is_faulty = (sensor_id == faulty_sensor)

            healthy_mult = rng.lognormal(mean=HEALTHY_LN_MEAN, sigma=HEALTHY_LN_SIGMA)

            cm_xyz = (cm_rw_x_full[sens_idx], cm_rw_y_full[sens_idx], cm_rw_alt_full[sens_idx])
            cm_ang = (cm_rw_range_full[sens_idx], cm_rw_yaw_full[sens_idx], cm_rw_pitch_full[sens_idx])

            if sens_type == 'GPS' and set(GPS_COLS) >= {'lat','lon','alt'}:
                noisy_lat, noisy_lon, noisy_alt = add_gps_noise_per_sensor(
                    df_sens[GPS_COLS + ['_time_idx']], is_faulty, rng, common_rw_xyz=cm_xyz, healthy_mult=healthy_mult
                )
                df_run_noisy.loc[sens_idx, 'lat'] = noisy_lat
                df_run_noisy.loc[sens_idx, 'lon'] = noisy_lon
                df_run_noisy.loc[sens_idx, 'alt'] = noisy_alt

            elif sens_type == 'Angular' and set(ANG_COLS) >= {'range_km','yaw_deg','pitch_deg'}:
                noisy_range, noisy_yaw, noisy_pitch = add_angular_noise_per_sensor(
                    df_sens[ANG_COLS + ['_time_idx']], is_faulty, rng, common_drifts=cm_ang, healthy_mult=healthy_mult
                )
                df_run_noisy.loc[sens_idx, 'range_km'] = noisy_range
                df_run_noisy.loc[sens_idx, 'yaw_deg']  = noisy_yaw
                df_run_noisy.loc[sens_idx, 'pitch_deg']= noisy_pitch

            # apply missing AFTER noise
            cols_for_missing = [c for c in FEATURE_COLS if c in df_sens.columns]
            if cols_for_missing:
                _miss_mask, _burst_mask = apply_missing_to_noisy(
                    df_run_noisy, sens_idx, cols_for_missing, is_faulty, rng
                )

        df_run_noisy['faulty_sensor'] = faulty_sensor
        df_run_noisy['route_id'] = new_run_id  # reindex synthesized runs
        all_runs.append(df_run_noisy)

    full_dataset = pd.concat(all_runs, ignore_index=True)
    full_dataset.drop(columns=['_time_idx'], errors='ignore', inplace=True)

    # One-hot type
    type_dummies = pd.get_dummies(full_dataset['type'], prefix='type')
    full_dataset = pd.concat([full_dataset, type_dummies], axis=1)

    # =========================
    # Per-(run_id, sensor) missing/spike/time-dispersion features (robust)
    # =========================
    feature_cols = [c for c in ['range_km','yaw_deg','pitch_deg','lat','lon','alt'] if c in full_dataset.columns]

    rs_dfs = []
    for (rid, sid), g in full_dataset.groupby(['route_id','sensor'], sort=False):
        gg = g.copy()
        for col in feature_cols:
            s = gg[col]
            miss_rate   = float(s.isna().mean())
            longest_gap = longest_nan_run(s)
            burst_count = count_nan_bursts(s)
            spk_count   = spike_count_series(s)
            std_t, mean_t, med_t, iqr_t, _valid_cnt = safe_stats_over_time(s)

            gg[f"{col}_miss_rate_rs"]   = miss_rate
            gg[f"{col}_longest_gap_rs"] = longest_gap
            gg[f"{col}_burst_count_rs"] = burst_count
            gg[f"{col}_spike_count_rs"] = spk_count
            gg[f"{col}_stdtime_rs"]     = std_t
            gg[f"{col}_mean_rs"]        = mean_t
            gg[f"{col}_median_rs"]      = med_t
            gg[f"{col}_iqr_rs"]         = iqr_t
        rs_dfs.append(gg)

    full_dataset = pd.concat(rs_dfs, ignore_index=True)

    # =========================
    # Within-run relative deviations (overall + by type)
    # =========================
    new_feature_dfs = []
    for rid, group in full_dataset.groupby("route_id", sort=False):
        group = group.copy()

        # Per-type blocks
        if 'type' in group.columns:
            # GPS block
            if set(['lat','lon','alt']).issubset(group.columns):
                gps_mask  = (group['type'] == 'GPS')
                gps_group = group.loc[gps_mask]
                for col in ['lat','lon','alt']:
                    if gps_group[col].notna().sum() > 1:
                        gps_median = np.nanmedian(gps_group[col])
                        gps_std    = np.nanstd(gps_group[col])
                        group.loc[gps_group.index, f"{col}_diff_gps_median"] = np.abs(gps_group[col] - gps_median)
                        group.loc[gps_group.index, f"{col}_std_gps"]         = gps_std

            # Angular block
            if set(['range_km','yaw_deg','pitch_deg']).issubset(group.columns):
                ang_mask  = (group['type'] == 'Angular')
                ang_group = group.loc[ang_mask]
                for col in ['range_km','yaw_deg','pitch_deg']:
                    if ang_group[col].notna().sum() > 1:
                        ang_median = np.nanmedian(ang_group[col])
                        ang_std    = np.nanstd(ang_group[col])
                        group.loc[ang_group.index, f"{col}_diff_ang_median"] = np.abs(ang_group[col] - ang_median)
                        group.loc[ang_group.index, f"{col}_std_ang"]         = ang_std

        # Overall robust deviations (within run)
        for col in feature_cols:
            vals = group[col].values
            if np.all(np.isnan(vals)):
                continue
            median = np.nanmedian(vals)
            group[f"{col}_diff_median"] = np.abs(group[col] - median)

            # mean abs diff to others (robust fallback)
            v = pd.Series(vals)
            mad_list = []
            for i, xi in enumerate(v):
                if pd.isna(xi):
                    mad_list.append(np.nan)
                    continue
                others = v.drop(v.index[i]).values
                mad_list.append(np.nanmean(np.abs(xi - others)))
            group[f"{col}_mean_abs_diff"] = mad_list

        new_feature_dfs.append(group)

    full_dataset = pd.concat(new_feature_dfs, ignore_index=True)

    # =========================
    # Save
    # =========================
    full_dataset.to_csv(OUTPUT_FILE, index=False)
    n_runs_out = full_dataset['route_id'].nunique()
    print(f"\nDone. Wrote {len(full_dataset):,} rows across {n_runs_out} runs to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
