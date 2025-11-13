import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

'''
This code simulates observations of an aircraft by multiple sensors (GPS and Angular),
placed in a circle around the flight area. 
It generates sensor readings over time and saves them to a CSV - sensor_observations.csv.
'''

# one aircraft data
df = pd.read_csv("OnePathData/pathCreation/one_aircraft_data.csv")
df = df.dropna(subset=['latitude', 'longitude', 'baro_altitude', 'timestamp'])

# the boundaries of the area
min_lat, max_lat = df['latitude'].min(), df['latitude'].max()
min_lon, max_lon = df['longitude'].min(), df['longitude'].max()

# Calculate the center of the area
center_lat = (min_lat + max_lat) / 2
center_lon = (min_lon + max_lon) / 2

# Ask the user how many sensors to place
num_sensors = int(input("How many sensors?"))

# Determine radius (20% of distances)
radius_lat = (max_lat - min_lat) * 0.2
radius_lon = (max_lon - min_lon) * 0.2

# Calculate sensor locations in a circle around the center
sensors = {}
for i in range(num_sensors):
    angle = 2 * np.pi * i / num_sensors
    lat = center_lat + radius_lat * np.sin(angle)
    lon = center_lon + radius_lon * np.cos(angle)
    alt = np.random.uniform(50, 300)  # ground level in meters
    sensor_type = np.random.choice(['GPS', 'Angular'])  # sensor type
    sensors[f"sensor_{i+1}"] = {'pos': (lat, lon, alt), 'type': sensor_type}


LAT_KM = 111 # one km
def lon_km(lat): return LAT_KM * np.cos(np.radians(lat))

def compute_relative_observation(sensor_pos, aircraft_pos):
    lat_s, lon_s, alt_s = sensor_pos
    lat_a, lon_a, alt_a = aircraft_pos

    dx = (lon_a - lon_s) * lon_km((lat_a + lat_s) / 2)
    dy = (lat_a - lat_s) * LAT_KM
    dz = (alt_a - alt_s) / 1000  # Elevation in meters per kilometer

    distance = np.sqrt(dx**2 + dy**2 + dz**2)
    yaw = np.degrees(np.arctan2(dx, dy))
    pitch = np.degrees(np.arcsin(dz / distance))
    return distance, yaw, pitch

# Processing each time point in the route
results = []
icao = df['icao24'].iloc[0]  # the aircraft identifier

for i, row in df.iterrows():
    aircraft_pos = (row['latitude'], row['longitude'], row['baro_altitude'])
    timestamp = row['timestamp']

    for name, sensor in sensors.items():
        s_pos = sensor['pos']
        s_type = sensor['type']

        if s_type == 'GPS':
            obs = {
                'aircraft': icao,
                'sensor': name,
                'type': 'GPS',
                'timestamp': timestamp,
                'lat': aircraft_pos[0],
                'lon': aircraft_pos[1],
                'alt': aircraft_pos[2]
            }
        else:
            rng, yaw, pitch = compute_relative_observation(s_pos, aircraft_pos)
            obs = {
                'aircraft': icao,
                'sensor': name,
                'type': 'Angular',
                'timestamp': timestamp,
                'range_km': round(rng, 3),
                'yaw_deg': round(yaw, 1),
                'pitch_deg': round(pitch, 1)
            }
        results.append(obs)

# Create a DataFrame
obs_df = pd.DataFrame(results)

# print("\nדוגמה לתצפיות מכל החיישנים על המסלול:")
# print(obs_df.head())

# save to file
obs_df.to_csv("sensor_observations2.csv", index=False)



