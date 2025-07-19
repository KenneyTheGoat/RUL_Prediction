# Convolutional neural network to classify different battery types.
# AUthor: Kenneth Kamogelo Baloyi
# Date: 06 July 2025
# NNetwork.py

# Import necessary libraries
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import joblib
from collections import Counter

# ---------------- LOAD DATA -------------------
df = pd.read_csv("balanced_discharge_data.csv")
features = ["voltage_measured", "current_measured", "temperature_measured", "capacity", "cycle"]

grouped = df.groupby("battery")
window_size = 128
step_size = 1 # Step size determining which data points go in the training batch

X, y = [], []
le = LabelEncoder() # Encoder of features to labels

for battery_id, group in grouped:
    series = group[features].values
    for start in range(0, len(series) - window_size + 1, step_size):
        window = series[start:start + window_size].T  # features and window_size
        X.append(window)
        y.append(battery_id)

X = np.stack(X)
y_enc = le.fit_transform(y)
num_samples, num_features, seq_len = X.shape

# ---------------- SCALE FEATURES -------------------
X_flat = X.transpose(0, 2, 1).reshape(-1, num_features)
scaler = StandardScaler()
X_scaled_flat = scaler.fit_transform(X_flat)
X_scaled = X_scaled_flat.reshape(num_samples, seq_len, num_features).transpose(0, 2, 1)

num_classes = len(le.classes_)

# ---------------- SPLIT & CLASS WEIGHTS -------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_enc, test_size=0.2, random_state=42
)

present_classes = np.unique(y_train)
present_weights = compute_class_weight('balanced', classes=present_classes, y=y_train)
full_weights = np.ones(num_classes, dtype=np.float32)
for cls, w in zip(present_classes, present_weights):
    full_weights[cls] = w
class_weights_tensor = torch.tensor(full_weights, dtype=torch.float32)

# ---------------- TORCH DATASETS -------------------
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=32, shuffle=True)
test_loader = DataLoader(TensorDataset(X_test_tensor, y_test_tensor), batch_size=32)

# MODEL DEFINITION
class Battery1DCNN(nn.Module):
    def __init__(self, num_features, seq_len, num_classes):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(num_features, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )
    def forward(self, x):
        return self.cnn(x)

model = Battery1DCNN(num_features, seq_len, num_classes)

# TRAINING
criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
optimizer = optim.Adam(model.parameters(), lr=0.001)
epochs = 30 

for epoch in range(epochs):
    model.train()
    total_loss = 0
    for xb, yb in train_loader:
        optimizer.zero_grad()
        out = model(xb)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}")

# SAVE 
torch.save(model.state_dict(), "battery_cnn_model.pth")
joblib.dump(scaler, "cnn_scaler.save")
joblib.dump(le, "cnn_label_encoder.save")
print("Model and preprocessors saved.")

# EVALUATION 
model.eval()
with torch.no_grad():
    y_pred_logits = model(X_test_tensor)
    y_pred = torch.argmax(y_pred_logits, dim=1)

accuracy = (y_pred == y_test_tensor).sum().item() / len(y_test_tensor) * 100
print(f"\nAccuracy: {accuracy:.2f}%")

present_labels = np.unique(y_test_tensor.numpy())
present_class_names = [le.classes_[i] for i in present_labels]
print("\nClassification Report:")
print(classification_report(
    y_test_tensor, y_pred,
    labels=present_labels,
    target_names=present_class_names
))
print("Confusion Matrix:")
print(confusion_matrix(y_test_tensor, y_pred))

#  CLASS DISTRIBUTIONS 
def log_class_distribution(label_array, label_encoder, set_name=""):
    counts = Counter(label_array)
    print(f"\n {set_name} Set Class Distribution:")
    for idx in range(len(label_encoder.classes_)):
        print(f"  - Class {idx} ({label_encoder.classes_[idx]}): {counts.get(idx, 0)} samples")

log_class_distribution(y_enc, le, "Full")
log_class_distribution(y_train, le, "Train")
log_class_distribution(y_test, le, "Test")


