import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load the CSV file
df = pd.read_csv("one_aircraft_data.csv")

# Drop rows with missing values in critical columns
df = df.dropna(subset=['latitude', 'longitude', 'baro_altitude'])

# Extract the 3D coordinates
lat = df['latitude']
lon = df['longitude']
alt = df['baro_altitude']

# Create a 3D plot
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot(lon, lat, alt, color='blue', label='Aircraft Path')

# Set axis labels
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
ax.set_zlabel('Altitude [m]')
ax.set_title('3D Flight Path')

# Optional: Add a legend
ax.legend()

# Show the plot
plt.tight_layout()
plt.show()
