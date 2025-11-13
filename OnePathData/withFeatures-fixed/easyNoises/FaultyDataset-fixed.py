# import numpy as np
# import pandas as pd

# '''
# New feature: Deviation relative to other sensors of the same type
# Runs the noise addition process several times, each time selecting a different faulty sensor.
# The result is a large dataset in which each example contains observations on
# a trajectory, and includes a label - which sensor is faulty.
# '''

# # observation file (without noise)
# base_observations_df = pd.read_csv("OnePathData/sensorObservation/sensor_observations.csv")

# # How many iterations to perform (each with a different faulty sensor)
# num_runs = 200

# # noises:

# # Angular
# strong_noise_std = {
#     'range_km': 300,
#     'yaw_deg': 10,
#     'pitch_deg': 5
# }
# weak_noise_std = {
#     'range_km': 30,
#     'yaw_deg': 2,
#     'pitch_deg': 1
# }
# # GPS
# gps_noise_std = {
#     'lat': 0.000045, # 5 meters in deg
#     'lon': 0.000045,
#     'alt': 5
# }


# all_runs = []

# # Each run makes data with a different faulty sensor.
# for run_id in range(num_runs):
#     observations_df = base_observations_df.copy()

#     # Randomly selecting a faulty sensor
#     faulty_sensor = np.random.choice(observations_df['sensor'].unique())

#      # Adding noise according to the type of sensor and whether it is faulty
#     noisy_df = observations_df.copy()
#     for i, row in noisy_df.iterrows():
#         sensor = row['sensor']
#         sensor_type = row['type']
#         is_faulty = (sensor == faulty_sensor)

#         # GPS
#         if sensor_type == 'GPS':
#             # Multiplies the noise by 5 times for the faulty
#             stds = gps_noise_std if not is_faulty else {k: v * 5 for k, v in gps_noise_std.items()}
#             for field in ['lat', 'lon', 'alt']:
#                 if not pd.isna(row[field]):
#                     noisy_df.at[i, field] = row[field] + np.random.normal(0, stds[field])

#         # Angular
#         elif sensor_type == 'Angular':
#             stds = strong_noise_std if is_faulty else weak_noise_std
#             for field in ['range_km', 'yaw_deg', 'pitch_deg']:
#                 if not pd.isna(row[field]):
#                     noisy_df.at[i, field] = row[field] + np.random.normal(0, stds[field])

#     # Adding holes - missing data (2% of rows)
#     drop_probability = 0.02
#     for col in ['range_km', 'yaw_deg', 'pitch_deg', 'lat', 'lon', 'alt']:
#         mask = np.random.rand(len(noisy_df)) < drop_probability
#         noisy_df.loc[mask, col] = np.nan

#     # Adding a label -  which sensor failed in this run
#     noisy_df['faulty_sensor'] = faulty_sensor
#     noisy_df['run_id'] = run_id

#     all_runs.append(noisy_df)

# # all runs into one file
# full_dataset = pd.concat(all_runs, ignore_index=True)

# # saving to the file
# full_dataset.to_csv("OnePathData/withFeatures-fixed/full_faulty_dataset_fixed.csv", index=False)

# print(f"\n {num_runs} runs were created with a different faulty sensor each time.")
# print("The file is saved as: full_faulty_dataset_fixed.csv")

import numpy as np
import pandas as pd

'''
Generate noisy raw sensor observations for multiple runs (fast version).
Each run randomly selects a faulty sensor, adds noise and missing values,
and labels the run with `faulty_sensor` and `run_id`.
No engineered features are written to the output.
'''

np.random.seed(42)

# === Base observations ===
base_observations_df = pd.read_csv("OnePathData/sensorObservation/sensor_observations2.csv")
num_runs = 200

# Noise levels
strong_noise_std = {'range_km': 300, 'yaw_deg': 10, 'pitch_deg': 5}
weak_noise_std   = {'range_km': 30,  'yaw_deg': 2,  'pitch_deg': 1}
gps_noise_std    = {'lat': 0.000045, 'lon': 0.000045, 'alt': 5}

all_runs = []
n_rows = len(base_observations_df)

for run_id in range(num_runs):
    noisy_df = base_observations_df.copy()
    faulty_sensor = np.random.choice(noisy_df['sensor'].unique())

    # Masks
    gps_mask      = noisy_df['type'] == 'GPS'
    angular_mask  = noisy_df['type'] == 'Angular'
    faulty_mask   = noisy_df['sensor'] == faulty_sensor
    gps_faulty    = gps_mask & faulty_mask
    gps_normal    = gps_mask & ~faulty_mask
    ang_faulty    = angular_mask & faulty_mask
    ang_normal    = angular_mask & ~faulty_mask

    # ---- GPS noise ----
    for field in ['lat', 'lon', 'alt']:
        std = gps_noise_std[field]
        # regular
        noisy_df.loc[gps_normal, field] += np.random.normal(0, std, size=gps_normal.sum())
        # noisy
        noisy_df.loc[gps_faulty, field] += np.random.normal(0, std * 5, size=gps_faulty.sum())

    # ---- Angular noise ----
    for field in ['range_km', 'yaw_deg', 'pitch_deg']:
        std_weak = weak_noise_std[field]
        std_strong = strong_noise_std[field]
        # regular
        noisy_df.loc[ang_normal, field] += np.random.normal(0, std_weak, size=ang_normal.sum())
        # noisy
        noisy_df.loc[ang_faulty, field] += np.random.normal(0, std_strong, size=ang_faulty.sum())

    # ---- Missing data ----
    drop_probability = 0.02
    drop_mask = np.random.rand(n_rows, 6) < drop_probability
    cols = ['range_km', 'yaw_deg', 'pitch_deg', 'lat', 'lon', 'alt']
    noisy_df.loc[:, cols] = noisy_df[cols].mask(drop_mask)

    # ---- Labels ----
    noisy_df['faulty_sensor'] = faulty_sensor
    noisy_df['route_id'] = run_id

    all_runs.append(noisy_df)


# === Combine all runs ===
full_dataset = pd.concat(all_runs, ignore_index=True)
full_dataset.to_csv("OnePathData/withFeatures-fixed/full_faulty_dataset_fixed.csv", index=False)

print(f"\n {num_runs} runs created. Saved as full_faulty_dataset_raw.csv")
