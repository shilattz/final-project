import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from clearml import Task
import io
from PIL import Image

# Starting a ClearML task
# task.close()
task = Task.init(
    project_name="Sensores Fault Detection - many paths",
    task_name="Statistical Baseline",
    task_type=Task.TaskTypes.training
)

# data
df = pd.read_csv("manyPathData/faultySensors/full_faulty_dataset.csv")

sensor_order = ['sensor_1', 'sensor_2', 'sensor_3', 'sensor_4', 'sensor_5']
label_map = {name: i for i, name in enumerate(sensor_order)}
df['label'] = df['faulty_sensor'].map(label_map)

# Features to choose from (raw only)
feature_cols = [
    'range_km', 'yaw_deg', 'pitch_deg', 'lat', 'lon', 'alt'
]

# Building predictions by run_id
y_true = []
y_pred = []

for route_id, group in df.groupby("route_id"):
    group = group.set_index("sensor")
    vectors = []
    for sensor in sensor_order:
        if sensor in group.index:
            vectors.append(group.loc[sensor, feature_cols].values)
        else:
            vectors.append(np.zeros(len(feature_cols)))
    sensor_matrix = np.stack(vectors)  # shape (5, F)

    # Deviation index: sum of distance from median for each sensor
    median_vector = np.median(sensor_matrix, axis=0)
    deviations = np.sum(np.abs(sensor_matrix - median_vector), axis=1)
    predicted_sensor_index = np.argmax(deviations)

    y_pred.append(predicted_sensor_index)
    y_true.append(label_map[group['faulty_sensor'].iloc[0]])

acc = accuracy_score(y_true, y_pred)
print(f"\n Statistical Model Accuracy: {acc:.2%}")

# Report to clearml
task.get_logger().report_scalar(
    title="Evaluation",
    series="Test Accuracy",
    value=acc,
    iteration=0
)

print("\n Classification Report:")
print(classification_report(y_true, y_pred, target_names=sensor_order))

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=sensor_order, yticklabels=sensor_order)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Statistical Model Confusion Matrix")
plt.show()

# Reporting the confusion matrix to clearml
buf = io.BytesIO()
plt.savefig(buf, format='png')
buf.seek(0)
img = Image.open(buf)

task.get_logger().report_image(
    title="Confusion Matrix",
    series="Matrix",
    iteration=0,
    image=img
)
