import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
from clearml import Task
import matplotlib.pyplot as plt
import seaborn as sns
import io
from PIL import Image

'''
Takes the vectors of all sensors with the same run_id
Calculates the distance of each sensor from the median of all
The sensor with the largest deviation from the median is considered "failed"
'''

# Initialize ClearML Task
task = Task.init(
    project_name="Sensores Fault Detection - no features",
    task_name="Statistical Baseline Model",
    task_type=Task.TaskTypes.inference
)

# Load dataset
df = pd.read_csv("withoutFeatures/full_faulty_dataset.csv")

# Fill missing values
feature_cols = ['range_km', 'yaw_deg', 'pitch_deg', 'lat', 'lon', 'alt']
df[feature_cols] = df[feature_cols].interpolate(limit_direction='both')

# Map faulty sensor labels to integers
label_map = {'sensor_1': 0, 'sensor_2': 1, 'sensor_3': 2, 'sensor_4': 3, 'sensor_5': 4}
df['label'] = df['faulty_sensor'].map(label_map)

# Aggregate data by run_id and sensor
aggregated = df.groupby(['run_id', 'sensor']).mean(numeric_only=True).reset_index()
types = df[['run_id', 'sensor', 'type']].drop_duplicates()
aggregated = aggregated.merge(types, on=['run_id', 'sensor'], how='left')

# Run statistical "prediction"
y_true = []
y_pred = []

for run_id, group in aggregated.groupby("run_id"):
    sensor_vectors = []
    sensor_names = []
    
    for sensor_name in ['sensor_1', 'sensor_2', 'sensor_3', 'sensor_4', 'sensor_5']:
        row = group[group['sensor'] == sensor_name]
        if row.empty:
            vector = np.zeros(6)
        else:
            sensor_type = row['type'].values[0]
            if sensor_type == 'GPS':
                vector = row[['lat', 'lon', 'alt']].values[0]
                vector = np.pad(vector, (0, 3), constant_values=0)
            else:
                vector = row[['range_km', 'yaw_deg', 'pitch_deg']].values[0]
                vector = np.pad(vector, (3, 0), constant_values=0)
        sensor_vectors.append(vector)
        sensor_names.append(sensor_name)

    sensor_vectors = np.stack(sensor_vectors)  # shape: (5, 6)
    median = np.median(sensor_vectors, axis=0)
    diffs = np.linalg.norm(sensor_vectors - median, axis=1)
    predicted_faulty = np.argmax(diffs)
    
    label = group['label'].iloc[0]
    y_true.append(label)
    y_pred.append(predicted_faulty)

# Evaluate
acc = accuracy_score(y_true, y_pred)
print(f"Statistical Baseline Accuracy: {acc:.2f}")

task.get_logger().report_scalar(
    title="Evaluation",
    series="Accuracy",
    value=acc,
    iteration=0
)

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)
label_names = ['sensor_1', 'sensor_2', 'sensor_3', 'sensor_4', 'sensor_5']
cm_df = pd.DataFrame(cm, index=label_names, columns=label_names)


plt.figure(figsize=(8,6))
sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix - Statistical Baseline")
plt.ylabel("True Label")
plt.xlabel("Predicted Label")
plt.tight_layout()
plt.show()

# Send to ClearML
buf = io.BytesIO()
plt.savefig(buf, format='png')
buf.seek(0)
img = Image.open(buf).convert("RGB")

task.get_logger().report_image(
    title="Confusion Matrix",
    series="Matrix",
    iteration=0,
    image=img
)
