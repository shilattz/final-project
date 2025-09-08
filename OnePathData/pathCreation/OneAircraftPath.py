import requests
import time
import pandas as pd

username = ""
password = ""
interval = 5  # seconds
duration = 10 * 60  # 10 minutes
output_csv = "1_aircraft_data.csv"


# Obtain initial aircraft data (select one aircraft with location)
def get_states():
    url = "https://opensky-network.org/api/states/all"
    response = requests.get(url, auth=(username, password) if username else None)
    if response.status_code == 200:
        return response.json()
    else:
        print("Failed to fetch states:", response.status_code)
        return None


def extract_aircraft(states):
    aircrafts = []
    for s in states["states"]:
        if s[5] is not None and s[6] is not None:  # lat, lon
            aircrafts.append(s)
    return aircrafts


print("Looking for 1 aircraft with position data...")
while True:
    result = get_states()
    if result:
        candidates = extract_aircraft(result)
        if len(candidates) >= 1:
            selected = candidates[:1]
            icao24_list = [s[0] for s in selected]
            print("Selected aircraft:", icao24_list)
            break
    time.sleep(5)

# Collect data every 5 seconds
print("Starting tracking...")
start_time = time.time()
data = []

while time.time() - start_time < duration:
    result = get_states()
    if result:
        now = int(time.time())
        for s in result["states"]:
            if s[0] in icao24_list:
                data.append({
                    "timestamp": now,
                    "icao24": s[0],
                    "callsign": s[1].strip() if s[1] else None,
                    "origin_country": s[2],
                    "longitude": s[5],
                    "latitude": s[6],
                    "baro_altitude": s[7],
                    "velocity": s[9],
                    "heading": s[10],
                    "vertical_rate": s[11],
                    "on_ground": s[8]
                })
    time.sleep(interval)

# saving the data
df = pd.DataFrame(data)
df.to_csv(output_csv, index=False)
print(f"Saved {len(df)} records to {output_csv}")
