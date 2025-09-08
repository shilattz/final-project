import numpy as np
import pandas as pd

'''
New feature: Deviation relative to other sensors of the same type
Runs the noise addition process several times, each time selecting a different faulty sensor.
The result is a large dataset in which each example contains observations on
a trajectory, and includes a label - which sensor is faulty.
'''

# observation file (without noise)
base_observations_df = pd.read_csv("sensor_observations.csv")

# How many iterations to perform (each with a different faulty sensor)
num_runs = 200

# noises:

# Angular
strong_noise_std = {
    'range_km': 300,
    'yaw_deg': 10,
    'pitch_deg': 5
}
weak_noise_std = {
    'range_km': 30,
    'yaw_deg': 2,
    'pitch_deg': 1
}
# GPS
gps_noise_std = {
    'lat': 0.000045, # 5 meters in deg
    'lon': 0.000045,
    'alt': 5
}


all_runs = []

# Each run makes data with a different faulty sensor.
for run_id in range(num_runs):
    observations_df = base_observations_df.copy()

    # Randomly selecting a faulty sensor
    faulty_sensor = np.random.choice(observations_df['sensor'].unique())

     # Adding noise according to the type of sensor and whether it is faulty
    noisy_df = observations_df.copy()
    for i, row in noisy_df.iterrows():
        sensor = row['sensor']
        sensor_type = row['type']
        is_faulty = (sensor == faulty_sensor)

        # GPS
        if sensor_type == 'GPS':
            # Multiplies the noise by 5 times for the faulty
            stds = gps_noise_std if not is_faulty else {k: v * 5 for k, v in gps_noise_std.items()}
            for field in ['lat', 'lon', 'alt']:
                if not pd.isna(row[field]):
                    noisy_df.at[i, field] = row[field] + np.random.normal(0, stds[field])

        # Angular
        elif sensor_type == 'Angular':
            stds = strong_noise_std if is_faulty else weak_noise_std
            for field in ['range_km', 'yaw_deg', 'pitch_deg']:
                if not pd.isna(row[field]):
                    noisy_df.at[i, field] = row[field] + np.random.normal(0, stds[field])

    # Adding holes - missing data (2% of rows)
    drop_probability = 0.02
    for col in ['range_km', 'yaw_deg', 'pitch_deg', 'lat', 'lon', 'alt']:
        mask = np.random.rand(len(noisy_df)) < drop_probability
        noisy_df.loc[mask, col] = np.nan

    # Adding a label -  which sensor failed in this run
    noisy_df['faulty_sensor'] = faulty_sensor
    noisy_df['run_id'] = run_id

    all_runs.append(noisy_df)

# all runs into one file
full_dataset = pd.concat(all_runs, ignore_index=True)

# One-hot encoding for type column
# type_GPS
# type_Angular
type_dummies = pd.get_dummies(full_dataset['type'], prefix='type')
full_dataset = pd.concat([full_dataset, type_dummies], axis=1)

# Adding new features to each sensor within each run_id
feature_cols = ['range_km', 'yaw_deg', 'pitch_deg', 'lat', 'lon', 'alt']

# Calculate deviations for each run_id
new_feature_dfs = []

for run_id, group in full_dataset.groupby("run_id"):
# Calculate separate statistics by sensor type within the same run 
    gps_group = group[group['type'] == 'GPS']
    angular_group = group[group['type'] == 'Angular']

    for col in ['lat', 'lon', 'alt']:
        if gps_group[col].notna().sum() > 1:
            gps_median = np.nanmedian(gps_group[col])
            gps_std = np.nanstd(gps_group[col])
            group.loc[gps_group.index, f"{col}_diff_gps_median"] = np.abs(gps_group[col] - gps_median)
            group.loc[gps_group.index, f"{col}_std_gps"] = gps_std

    for col in ['range_km', 'yaw_deg', 'pitch_deg']:
        if angular_group[col].notna().sum() > 1:
            ang_median = np.nanmedian(angular_group[col])
            ang_std = np.nanstd(angular_group[col])
            group.loc[angular_group.index, f"{col}_diff_ang_median"] = np.abs(angular_group[col] - ang_median)
            group.loc[angular_group.index, f"{col}_std_ang"] = ang_std

    group = group.copy()

    for col in feature_cols:
        if group[col].isna().all():
            continue  # There is no information in this column in running sensors

        values = group[col].values
        median = np.nanmedian(values)

        # Deviation from the median
        group[f"{col}_diff_median"] = np.abs(group[col] - median)

        # Average difference from the rest (excluding itself)
        mean_diffs = []
        for i in range(len(values)):
            others = np.delete(values, i)
            mean_diff = np.nanmean(np.abs(values[i] - others))
            mean_diffs.append(mean_diff)
        group[f"{col}_mean_abs_diff"] = mean_diffs

    new_feature_dfs.append(group)

# all runs into one file
full_dataset = pd.concat(new_feature_dfs, ignore_index=True)


# saving to the file
full_dataset.to_csv("full_all_features_faulty_dataset.csv", index=False)

print(f"\n {num_runs} runs were created with a different faulty sensor each time.")
print("The file is saved as: full_all_features_faulty_dataset.csv")

