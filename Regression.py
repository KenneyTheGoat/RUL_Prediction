# Regression.py
# Linear Regression for Capacity Prediction and RUL Estimation
# Author: Kenneth Kamogelo Baloyi
# Date: 08 July 2025

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os


# -------------------------------
# Save model and RMSE
# -------------------------------
def save_model(model, rmse, filename='rul_model.pkl'):
    joblib.dump({'model': model, 'rmse': rmse}, filename)
    print(f"Model saved to {filename}")

# -------------------------------
# Load model and RMSE
# -------------------------------
def load_model(filename='rul_model.pkl'):
    if not os.path.exists(filename):
        raise FileNotFoundError(f"No saved model found at {filename}")
    data = joblib.load(filename)
    print(f"Model loaded from {filename}")
    return data['model'], data['rmse']

# -------------------------------
# Train the model from data
# -------------------------------
def train_model(file_path, feature_cols, target_col):
    df = pd.read_csv(file_path)

    # Clean and prepare data
    df = df.dropna()
    if not set(feature_cols + [target_col]).issubset(df.columns):
        raise ValueError(f"Missing required columns: {feature_cols + [target_col]}")

    X = df[feature_cols]
    y = df[target_col]

    model = LinearRegression() # Apply the least squares algorithm
    model.fit(X, y)

    y_pred = model.predict(X)
    rmse, r2 = evaluate_model(y, y_pred)


    print("Model trained.")
   
    coef_dict = dict(zip(feature_cols, model.coef_))
    coef_dict = {k: round(float(v), 6) for k, v in coef_dict.items()}
    print("Model coefficients:", coef_dict)
    print("Intercept:", model.intercept_)
    print("Performance:", rmse, r2)

    return model, rmse

# -------------------------------
# Predict using the model
# -------------------------------
def predict_with_model(model, input_dict):
    X_input = pd.DataFrame([input_dict])
    return model.predict(X_input)[0]

# -------------------------------
# Evaluate model performance
# -------------------------------
def evaluate_model(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    print("RMSE: " + str(round(rmse, 4)) +", R²: " + str(round(r2, 4)))
    return rmse, r2


# -------------------------------
# Estimate RUL
# -------------------------------
def estimate_rul(model, current_cycle, current_current, capacity_threshold=0.2):
    # Get coefficients
    intercept = model.intercept_
    coef = model.coef_

    # Assume cycle and current are in the order of features used in training
    beta_cycle = coef[0]
    beta_current = coef[1]

    if beta_cycle >= 0:
        raise ValueError("Cycle coefficient is non-negative. Cannot estimate EOL.")

    # Solve for EOL cycle
    eol_cycle = (capacity_threshold - intercept - beta_current * current_current) / beta_cycle
    rul = max(eol_cycle - current_cycle, 0)

    return "EOL_Cycle: "+ str(round(eol_cycle, 2)) + ", RUL: " + str(round(rul, 2))


# -------------------------------
    #USAGE
# -------------------------------
file_path = 'Test_balanced_discharge_data.csv'
features = ['cycle', 'current_measured']
target = 'capacity'

# Only train if there is no saved model in the directory
model_path = 'rul_model.pkl'
if os.path.exists(model_path):
    model, rmse = load_model(model_path)
else:
    model, rmse = train_model(file_path, features, target)
    save_model(model, rmse, model_path)

input_data = {'cycle': 97, 'current_measured': 0.002966027}
predicted_capacity = predict_with_model(model, input_data)
print("Predicted Capacity: "+ str(round(predicted_capacity, 4)) + " ± "+ str(round(rmse, 2)))

rul_info = estimate_rul(model, current_cycle=input_data.get('cycle'), current_current=input_data.get('current_measured'))
print("Estimated RUL:", rul_info)
