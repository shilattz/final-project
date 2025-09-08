import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from xgboost import XGBClassifier
from clearml import Task
import matplotlib.pyplot as plt
import seaborn as sns
import io
from PIL import Image

# Initialize ClearML Task
task = Task.init(
    project_name="Sensores Fault Detection - no features",
    task_name="XGBoost - no features",
    task_type=Task.TaskTypes.training
)

# Load dataset
df = pd.read_csv("withoutFeatures/full_faulty_dataset.csv")

# Fill missing values and normalize features
feature_cols = ['range_km', 'yaw_deg', 'pitch_deg', 'lat', 'lon', 'alt']
df[feature_cols] = df[feature_cols].interpolate(limit_direction='both')
scaler = StandardScaler()
df[feature_cols] = scaler.fit_transform(df[feature_cols])

# Map faulty sensor labels to integers
label_map = {'sensor_1': 0, 'sensor_2': 1, 'sensor_3': 2, 'sensor_4': 3, 'sensor_5': 4}
df['label'] = df['faulty_sensor'].map(label_map)

# Aggregate data by run_id and sensor
aggregated = df.groupby(['run_id', 'sensor']).mean(numeric_only=True).reset_index()
types = df[['run_id', 'sensor', 'type']].drop_duplicates()
aggregated = aggregated.merge(types, on=['run_id', 'sensor'], how='left')

# Construct feature vectors for each run
runs = []
labels = []

for run_id, group in aggregated.groupby("run_id"):
    sensors = []
    for sensor_name in ['sensor_1', 'sensor_2', 'sensor_3', 'sensor_4', 'sensor_5']:
        row = group[group['sensor'] == sensor_name]
        if row.empty:
            sensors.append(np.zeros(7))
        else:
            sensor_type = row['type'].values[0]
            type_code = 1 if sensor_type == 'Angular' else 0

            if sensor_type == 'GPS':
                sensor_data = row[['lat', 'lon', 'alt']].values[0]
                padded = np.pad(sensor_data, (0, 3), constant_values=0)
            else:
                sensor_data = row[['range_km', 'yaw_deg', 'pitch_deg']].values[0]
                padded = np.pad(sensor_data, (3, 0), constant_values=0)

            full_vector = np.concatenate([padded, [type_code]])  # Total 7 features
            sensors.append(full_vector)

    runs.append(np.concatenate(sensors))  # Flattened to 35 features (5 sensors × 7)
    labels.append(group['label'].iloc[0])

X = np.stack(runs)
y = np.array(labels)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train the XGBoost model
model = XGBClassifier(
    objective='multi:softmax',
    num_class=5,
    eval_metric='mlogloss',
    random_state=42
)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {acc:.2f}")

# Report accuracy to ClearML
task.get_logger().report_scalar(
    title="Evaluation",
    series="Accuracy",
    value=acc,
    iteration=0
)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
label_names = ['sensor_1', 'sensor_2', 'sensor_3', 'sensor_4', 'sensor_5']
cm_df = pd.DataFrame(cm, index=label_names, columns=label_names)



plt.figure(figsize=(8,6))
sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.ylabel("True Label")
plt.xlabel("Predicted Label")
plt.tight_layout()

plt.show()

# Report confusion matrix image to ClearML
buf = io.BytesIO()
plt.savefig(buf, format='png')
buf.seek(0)
img = Image.open(buf).convert("RGB")  # Convert RGBA to RGB

task.get_logger().report_image(
    title="Confusion Matrix",
    series="Matrix",
    iteration=0,
    image=img
)
