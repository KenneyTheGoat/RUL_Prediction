# Predict.py
# Program to predict battery type and RUL
# AUthor: Kenneth Kamogelo Baloyi
# Date: 11 July 2025

#Import necessary libraries
import argparse
import os
from collections import Counter

import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn



# CONSTANTS

CNN_FEATURES   = ["voltage_measured", "current_measured", "temperature_measured", "capacity", "cycle"]
RF_FEATURES    = ["voltage_measured", "current_measured", "temperature_measured", "capacity", "cycle"]
SEQ_LEN        = 128          # window length used during CNN training
STEP           = 1            # step size



# CNN ARCHITECTURE
class Battery1DCNN(nn.Module):
    def __init__(self, n_features: int, n_classes: int):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(n_features, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(64, n_classes),
        )

    def forward(self, x):
        return self.cnn(x)



# RULE‑BASED CLASSIFICATION‑TREE
def _ct_single_row(row):
    #Decision rules
    if row["current_measured"] < -0.9981662:
        return "B0007"
    if row["current_measured"] < -0.001243135:
        if row["voltage_measured"] >= 3.466922:
            return "B0006"
        if row["temperature_measured"] >= 37.75091:
            return "B0006"
        return "B0018"
    if row["temperature_measured"] >= 38.67214:
        return "B0005"
    if row["voltage_measured"] >= 3.46407:
        if row["temperature_measured"] >= 23.66534:
            if row["voltage_measured"] < 3.599884:
                return "B0005"
            if row["voltage_measured"] >= 4.18646:
                return "B0005"
            return "B0006"
        return "B0018"
    return "B0018"


# MAIN CLASS 

class BatteryPredictor:
    """
    model_type : {"cnn", "rf", "ct"}
    csv_path   : path to battery discharge CSV
    device     : torch device (only relevant for cnn)
    """
    def __init__(self, model_type: str, device: str = "cpu"):
        self.model_type = model_type.lower()
        self.device     = torch.device(device)
        if self.model_type not in {"cnn", "rf", "ct"}:
            raise ValueError("model_type must be one of {'cnn','rf','ct'}")

        
        # LOAD TRAINED MODEL
        if self.model_type == "cnn":
            self._load_cnn()
        elif self.model_type == "rf":
            self._load_rf()
        #  ct has no external architecture

    # RF
    def _load_rf(self):
        self.rf_clf        = joblib.load("best_random_forest_model.save")
        self.rf_scaler     = joblib.load("rf_scaler.save")
        self.rf_encoder    = joblib.load("rf_label_encoder.save")
        try:
            self.rf_ytrain = joblib.load("rf_y_train.save")
        except FileNotFoundError:
            self.rf_ytrain = None  # sample‑cap feature will be disabled


    # CNN

    def _load_cnn(self):
        # load scaler & label‑encoder
        self.cnn_scaler   = joblib.load("cnn_scaler.save")
        self.cnn_encoder  = joblib.load("cnn_label_encoder.save")
        n_classes         = len(self.cnn_encoder.classes_)
        n_features        = len(CNN_FEATURES)
        # build network & load weights
        self.cnn_net      = Battery1DCNN(n_features, n_classes).to(self.device)
        self.cnn_net.load_state_dict(torch.load("battery_cnn_model.pth", map_location=self.device))
        self.cnn_net.eval()


    # PREDICTION ENTRY POINT
    def predict(self, csv_path: str, limit: int | None = None):
        if not os.path.exists(csv_path):
            raise FileNotFoundError(csv_path)
        if self.model_type == "ct":
            return self._predict_ct(csv_path, limit)
        if self.model_type == "rf":
            return self._predict_rf(csv_path, limit)
        return self._predict_cnn(csv_path, limit)


    # CT
    def _predict_ct(self, csv_path, limit):
        df = pd.read_csv(csv_path)
        df = df.head(limit) if limit else df
        for col in ["current_measured", "voltage_measured", "temperature_measured"]:
            if col not in df.columns:
                raise ValueError(f"Missing column {col} for CT model.")
        preds = [_ct_single_row(r) for _, r in df.iterrows()]
        return self._majority_vote(preds)


    # RF

    def _predict_rf(self, csv_path, limit):
        df = pd.read_csv(csv_path)
        for col in RF_FEATURES:
            if col not in df.columns:
                raise ValueError(f"Missing required feature {col} for RF model.")
        # dynamic sample cap – match Predict.py behaviour
        if limit is None and self.rf_ytrain is not None:
            min_train_class = min(Counter(self.rf_ytrain).values())
            limit = min_train_class
        X_raw   = df[RF_FEATURES].values[:limit] if limit else df[RF_FEATURES].values
        X_scaled = self.rf_scaler.transform(X_raw)
        idx_preds = self.rf_clf.predict(X_scaled)
        label_preds = self.rf_encoder.inverse_transform(idx_preds)
        return self._majority_vote(label_preds)

    
    # CNN

    def _predict_cnn(self, csv_path, limit):
        df = pd.read_csv(csv_path)
        for col in CNN_FEATURES:
            if col not in df.columns:
                raise ValueError(f"Missing required feature {col} for CNN model.")
        series = df[CNN_FEATURES].values
        windows = []
        for start in range(0, len(series) - SEQ_LEN + 1, STEP):
            if limit and len(windows) >= limit:
                break
            window = series[start : start + SEQ_LEN].T  # (features, seq_len)
            windows.append(window)
        if not windows:
            raise ValueError(f"Not enough rows for a {SEQ_LEN}-sample window.")
        X = np.stack(windows)  # shape [n_windows, features, seq_len]
        # scale features the same way they were scaled in training
        X_flat = X.transpose(0, 2, 1).reshape(-1, X.shape[1])
        X_scaled_flat = self.cnn_scaler.transform(X_flat)
        X_scaled = X_scaled_flat.reshape(len(X), SEQ_LEN, X.shape[1]).transpose(0, 2, 1)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            logits = self.cnn_net(X_tensor)
            idx_preds = torch.argmax(logits, dim=1).cpu().numpy()
        label_preds = self.cnn_encoder.inverse_transform(idx_preds)
        return self._majority_vote(label_preds)

    
    # helper function to get majority predictions
    @staticmethod
    def _majority_vote(labels):
        counts = Counter(labels)
        top, freq = counts.most_common(1)[0]
        return {
            "predicted_label": top,
            "majority_count":  freq,
            "distribution":   dict(counts)
        }


# CLI
def main():
    parser = argparse.ArgumentParser(description="Battery model selector")
    parser.add_argument("--csv",   required=True, help="Path to battery CSV")
    parser.add_argument("--model", required=True, choices=["cnn", "rf", "ct", "lr"],
                        help="Which model to use")
    parser.add_argument("--device", default="cpu",
                        help="cpu | cuda  (only matters for cnn)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Optional: max samples/windows to use for prediction")
    args = parser.parse_args()

    predictor = BatteryPredictor(args.model, device=args.device)
    result    = predictor.predict(args.csv, limit=args.limit)
    

    # pretty print
    battery_file = os.path.splitext(os.path.basename(args.csv))[0]
    print("\n---------------------------------------")
    print(f"CSV Supplied : {battery_file}")
    print(f"Model Used   : {args.model.upper()}")
    print(f"Prediction   : {result['predicted_label']}  "
          f"(majority {result['majority_count']} votes)")
    print(f"Distribution : {result['distribution']}")
    print("---------------------------------------\n")


if __name__ == "__main__":
    main()
