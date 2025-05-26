import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from clearml import Task
import io
from PIL import Image


# Starting a ClearML task
# task.close()
task = Task.init(
    project_name="Sensores Fault Detection - with additional features",
    task_name="XGBoost",
    task_type=Task.TaskTypes.training
)

#  data
df = pd.read_csv("withFeatures/full_all_features_faulty_dataset.csv")

sensor_order = ['sensor_1', 'sensor_2', 'sensor_3', 'sensor_4', 'sensor_5']
label_map = {name: i for i, name in enumerate(sensor_order)}
df['label'] = df['faulty_sensor'].map(label_map)

# Features we will use
feature_cols = [
    'range_km', 'yaw_deg', 'pitch_deg', 'lat', 'lon', 'alt',
    'range_km_diff_median', 'range_km_mean_abs_diff',
    'yaw_deg_diff_median', 'yaw_deg_mean_abs_diff',
    'pitch_deg_diff_median', 'pitch_deg_mean_abs_diff',
    'lat_diff_median', 'lat_mean_abs_diff',
    'lon_diff_median', 'lon_mean_abs_diff',
    'alt_diff_median', 'alt_mean_abs_diff',
    'lat_diff_gps_median', 'lat_std_gps',
    'lon_diff_gps_median', 'lon_std_gps',
    'alt_diff_gps_median', 'alt_std_gps',
    'range_km_diff_ang_median', 'range_km_std_ang',
    'yaw_deg_diff_ang_median', 'yaw_deg_std_ang',
    'pitch_deg_diff_ang_median', 'pitch_deg_std_ang'
]

# normalization
df[feature_cols] = df[feature_cols].interpolate(limit_direction='both')
scaler = StandardScaler()
df[feature_cols] = scaler.fit_transform(df[feature_cols])

# Create a DataFrame by run_id and sensor
rows = []
for run_id, group in df.groupby("run_id"):
    run_features = []
    for sensor in sensor_order:
        row = group[group['sensor'] == sensor]
        if not row.empty:
            run_features.extend(row[feature_cols].values[0])
        else:
            run_features.extend([0.0] * len(feature_cols))
    rows.append(run_features)

X = np.array(rows)
y = df.drop_duplicates("run_id")['label'].values

# Train & Test division
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# training XGBoost 
model = XGBClassifier(
    objective='multi:softmax',
    num_class=5,
    eval_metric='mlogloss',
    use_label_encoder=False,
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

model.fit(X_train, y_train)

# Model evaluation
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\n XGBoost Accuracy: {accuracy:.2%}")

# Report to clearml
task.get_logger().report_scalar(
    title="Evaluation",
    series="Accuracy",
    value=accuracy,
    iteration=0
)


print("\n Classification Report:")
print(classification_report(y_test, y_pred, target_names=sensor_order))

#  Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=sensor_order, yticklabels=sensor_order)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("XGBoost Confusion Matrix")
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
