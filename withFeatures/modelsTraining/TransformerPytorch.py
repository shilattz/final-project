import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from clearml import Task
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import io
from PIL import Image

# Starting a ClearML task
# task.close()
task = Task.init(
    project_name="Sensores Fault Detection - with additional features",
    task_name="Transformer with PyTorch",
    task_type=Task.TaskTypes.training
)

# data
df = pd.read_csv("withFeatures/full_all_features_faulty_dataset.csv")

# Label Mapping 
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

# Construct X, sensor_types, y by run_id
runs, types_list, labels = [], [], []
for run_id, group in df.groupby("run_id"):
    sensors = []
    sensor_types = []
    for sensor in sensor_order:
        row = group[group['sensor'] == sensor]
        if row.empty:
            sensors.append(np.zeros(len(feature_cols)))
            sensor_types.append(0)
        else:
            sensors.append(row[feature_cols].values[0])
            sensor_types.append(0 if row['type'].values[0] == 'GPS' else 1)
    runs.append(np.stack(sensors))
    types_list.append(np.array(sensor_types))
    labels.append(label_map[group['faulty_sensor'].iloc[0]])

X = np.stack(runs).astype(np.float32)               # (N, 5, F)
sensor_types = np.stack(types_list).astype(int)     # (N, 5)
y = np.array(labels).astype(np.int64)               # (N,)

# Dataset adapted to two types of sensors
class SensorDatasetDual(Dataset):
    def __init__(self, X, sensor_types, y):
        self.X = torch.tensor(X)
        self.sensor_types = torch.tensor(sensor_types)
        self.y = torch.tensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.sensor_types[idx], self.y[idx]

dataset = SensorDatasetDual(X, sensor_types, y)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# The model with two encoders by sensor type
class SensorClassifierDualEncoder(nn.Module):
    def __init__(self, input_dim, embed_dim=64, num_classes=5):
        super().__init__()
        self.gps_encoder = nn.Sequential(nn.Linear(input_dim, embed_dim), nn.ReLU())
        self.angular_encoder = nn.Sequential(nn.Linear(input_dim, embed_dim), nn.ReLU())
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=4, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.fc1 = nn.Linear(5 * embed_dim, 128)
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, 64)
        self.dropout2 = nn.Dropout(0.2)
        self.output = nn.Linear(64, num_classes)

    def forward(self, x, sensor_types):
        B, S, _ = x.shape
        embeddings = []

        for i in range(S):
            sensor_feat = x[:, i, :]
            is_gps = (sensor_types[:, i] == 0)
            embed = torch.zeros((B, self.gps_encoder[0].out_features), device=x.device)
            if is_gps.any():
                embed[is_gps] = self.gps_encoder(sensor_feat[is_gps])
            if (~is_gps).any():
                embed[~is_gps] = self.angular_encoder(sensor_feat[~is_gps])
            embeddings.append(embed.unsqueeze(1))

        x_embed = torch.cat(embeddings, dim=1)
        x = self.transformer(x_embed)
        x = x.view(B, -1)
        x = self.fc1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        return self.output(x)

# training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SensorClassifierDualEncoder(input_dim=len(feature_cols)).to(device)

class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y), y=y)
criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float).to(device))
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(10):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    for inputs, types, labels in dataloader:
        inputs, types, labels = inputs.to(device), types.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs, types)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    acc = correct / total
    print(f"Epoch {epoch+1} | Loss: {total_loss:.2f} | Accuracy: {acc:.2f}")


# Model evaluation
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for inputs, types, labels in dataloader:
        inputs, types = inputs.to(device), types.to(device)
        outputs = model(inputs, types)
        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.numpy())


accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
print(f"\n Final Model Accuracy: {accuracy:.2%}")

# Report to clearml
task.get_logger().report_scalar(
    title="Evaluation",
    series="Accuracy",
    value=accuracy,
    iteration=0
)

# Classification report
print("\n Classification Report:")
print(classification_report(all_labels, all_preds, target_names=sensor_order))

# Confusion matrix
cm = confusion_matrix(all_labels, all_preds)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=sensor_order, yticklabels=sensor_order)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
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


