# pytorch + transformer
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import io
from PIL import Image
from clearml import Task


# Starting a ClearML task
# task.close()
task = Task.init(
    project_name="Sensores Fault Detection - no features",
    task_name="Transformer with PyTorch",
    task_type=Task.TaskTypes.training
)

# data
df = pd.read_csv("withoutFeatures/full_faulty_dataset.csv")

# Normalization and imputation of missing values
feature_cols = ['range_km', 'yaw_deg', 'pitch_deg', 'lat', 'lon', 'alt']
df[feature_cols] = df[feature_cols].interpolate(limit_direction='both')
scaler = StandardScaler()
df[feature_cols] = scaler.fit_transform(df[feature_cols])

# Label Mapping 
label_map = {'sensor_1': 0, 'sensor_2': 1, 'sensor_3': 2, 'sensor_4': 3, 'sensor_5': 4}
df['label'] = df['faulty_sensor'].map(label_map)

# Average by sensor while running
aggregated = df.groupby(['run_id', 'sensor']).mean(numeric_only=True).reset_index()
types = df[['run_id', 'sensor', 'type']].drop_duplicates()
aggregated = aggregated.merge(types, on=['run_id', 'sensor'], how='left')

# Building Tensors
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

            full_vector = np.concatenate([padded, [type_code]])  # 6 + 1 = 7 features
            sensors.append(full_vector)

    runs.append(np.stack(sensors))
    label = group['label'].iloc[0]
    labels.append(label)

X = np.stack(runs).astype(np.float32)  # (200, 5, 7)
y = np.array(labels).astype(np.int64)  # (200,)

# Dataset and DataLoader
class SensorDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X)
        self.y = torch.tensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

dataset = SensorDataset(X, y)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# the modal
class SensorClassifier(nn.Module):
    def __init__(self, input_dim=7, model_dim=64, num_classes=5):
        super().__init__()
        self.embedding = nn.Linear(input_dim, model_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=4, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(5 * model_dim, 128)
        self.dropout = nn.Dropout(0.3)
        self.output = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.embedding(x)        # (batch, 5, model_dim)
        x = self.transformer(x)      # (batch, 5, model_dim)
        x = self.flatten(x)          # (batch, 5*model_dim)
        x = self.fc1(x)              # (batch, 128)
        x = self.dropout(x)
        out = self.output(x)         # (batch, num_classes)
        return out

# Training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SensorClassifier().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(10):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    acc = correct / total
    print(f"Epoch {epoch+1} | Loss: {total_loss:.2f} | Accuracy: {acc:.2f}")

from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# assessment
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Calculate accuracy
acc = accuracy_score(all_labels, all_preds)
print(f"\nFinal Accuracy: {acc:.2f}")


# Report to clearml
task.get_logger().report_scalar(
    title="Evaluation",
    series="Accuracy",
    value=acc,
    iteration=0
)

# Calculate confusion matrix
cm = confusion_matrix(all_labels, all_preds)
print("\nConfusion Matrix:")
print(cm)

# Graphic matrix
labels_names = ['sensor_1', 'sensor_2', 'sensor_3', 'sensor_4', 'sensor_5']
cm_df = pd.DataFrame(cm, index=labels_names, columns=labels_names)

plt.figure(figsize=(8,6))
sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.title('Confusion Matrix')
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
