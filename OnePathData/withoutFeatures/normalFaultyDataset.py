import numpy as np
import pandas as pd

'''
This code creating data without additional features but only raw data.
Runs the noise addition process several times, each time choosing a different faulty sensor.
The result is a large dataset in which each example contains observations on a track, 
and includes a label - which sensor is faulty.
Sensor types: GPS & Angular
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
# Gps
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

# saving to the file
full_dataset.to_csv("full_faulty_dataset.csv", index=False)

print(f"\n {num_runs} runs were created with a different faulty sensor each time.")
print("The file is saved as: full_faulty_dataset_raw.csv")

