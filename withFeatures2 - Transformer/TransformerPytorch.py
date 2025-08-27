import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from clearml import Task
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import io
from PIL import Image

from sklearn.model_selection import train_test_split
import random
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

'''
Transformer with Train, Validation Test
'''

# Fix seed for all libraries
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# Verify determinism in cudnn
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


#Starting a ClearML task
# task.close()
task = Task.init(
    project_name="Sensores Fault Detection - with initial data division in training",
    task_name="Transformer with PyTorch",
    task_type=Task.TaskTypes.training
)

# data
df = pd.read_csv("withFeatures1/full_all_features_faulty_dataset.csv")

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

# Split by run_id (to avoid leakage)
run_labels_series = df.groupby('run_id')['faulty_sensor'].first().map(label_map)
run_ids = run_labels_series.index.to_numpy()
run_y   = run_labels_series.values

train_ids, temp_ids, y_train_runs, y_temp_runs = train_test_split(
    run_ids, run_y, test_size=0.2, random_state=42, stratify=run_y
)
val_ids, test_ids, y_val_runs, y_test_runs = train_test_split(
    temp_ids, y_temp_runs, test_size=0.5, random_state=42, stratify=y_temp_runs
)

# Interpolate + Scale (fit on TRAIN only) 
df = df.sort_values(['run_id', 'sensor']).reset_index(drop=True)
df[feature_cols] = df[feature_cols].interpolate(limit_direction='both')

scaler = StandardScaler()
scaler.fit(df[df['run_id'].isin(train_ids)][feature_cols])
df[feature_cols] = scaler.transform(df[feature_cols])

def build_xy(ids, df_src):
    runs, types_list, labels = [], [], []
    for run_id, group in df_src[df_src['run_id'].isin(ids)].groupby("run_id"):
        sensors = []
        sensor_types = []
        for sensor in sensor_order:
            row = group[group['sensor'] == sensor]
            if row.empty:
                sensors.append(np.zeros(len(feature_cols), dtype=np.float32))
                sensor_types.append(0)
            else:
                sensors.append(row[feature_cols].values[0].astype(np.float32))
                sensor_types.append(0 if row['type'].values[0] == 'GPS' else 1)
        runs.append(np.stack(sensors))
        types_list.append(np.array(sensor_types))
        labels.append(label_map[group['faulty_sensor'].iloc[0]])
    X = np.stack(runs).astype(np.float32)         # (N, 5, F)
    T = np.stack(types_list).astype(np.int64)     # (N, 5)
    y = np.array(labels).astype(np.int64)         # (N,)
    return X, T, y

X_tr, T_tr, y_tr = build_xy(train_ids, df)
X_va, T_va, y_va = build_xy(val_ids,   df)
X_te, T_te, y_te = build_xy(test_ids,  df)


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

dataset_train = SensorDatasetDual(X_tr, T_tr, y_tr)
dataset_val   = SensorDatasetDual(X_va, T_va, y_va)
dataset_test  = SensorDatasetDual(X_te, T_te, y_te)

train_loader = DataLoader(dataset_train, batch_size=32, shuffle=True)
val_loader   = DataLoader(dataset_val,   batch_size=32, shuffle=False)
test_loader  = DataLoader(dataset_test,  batch_size=32, shuffle=False)


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

class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_tr), y=y_tr)

criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float).to(device))
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


EPOCHS = 15
best_val_acc = -1.0
best_state = None

for epoch in range(EPOCHS):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for inputs, types, labels in train_loader:
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
    train_acc = correct / total

    # Validation 
    model.eval()
    v_correct, v_total = 0, 0
    with torch.no_grad():
        for inputs, types, labels in val_loader:
            inputs, types, labels = inputs.to(device), types.to(device), labels.to(device)
            outputs = model(inputs, types)
            _, predicted = torch.max(outputs, 1)
            v_correct += (predicted == labels).sum().item()
            v_total += labels.size(0)
    val_acc = v_correct / v_total if v_total > 0 else 0.0

    print(f"Epoch {epoch+1} | TrainLoss: {total_loss:.2f} | TrainAcc: {train_acc:.3f} | ValAcc: {val_acc:.3f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_state = {k: v.cpu() for k, v in model.state_dict().items()}

# Claiming the best model we found based on validation
if best_state is not None:
    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})


# Model evaluation on TEST ONLY
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for inputs, types, labels in test_loader:
        inputs, types = inputs.to(device), types.to(device)
        outputs = model(inputs, types)
        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.numpy())

accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
print(f"\n Test Accuracy: {accuracy:.2%}")

# Report to clearml
task.get_logger().report_scalar(
    title="Evaluation",
    series="Test Accuracy",
    value=float(accuracy),
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


