# XXGBoost.py
# XGBoost Regression for Capacity Prediction and RUL Estimation
# Author: Kenneth Kamogelo Baloyi
# Date: 12 July 2025

import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os

# -------------------------------
# Save model and RMSE
# -------------------------------
def save_model(model, rmse, filename='rulxg_model.pkl'):
    joblib.dump({'model': model, 'rmse': rmse}, filename)
    print(f"Model saved to {filename}")

# -------------------------------
# Load model and RMSE
# -------------------------------
def load_model(filename='rulxg_model.pkl'):
    if not os.path.exists(filename):
        raise FileNotFoundError(f"No saved model found at {filename}")
    data = joblib.load(filename)
    print(f"Model loaded from {filename}")
    return data['model'], data['rmse']

# -------------------------------
# Train the model from data
# -------------------------------
def train_model(file_path, feature_cols, target_col):
    """
    Trains an XGBoost model on the provided dataset with a monotonicity constraint.

    Returns:
        Trained XGBoost model  and RMSE of the model on the training data.
    """
    try:
        df = pd.read_csv(file_path)
        df = df.dropna()

        if not set(feature_cols + [target_col]).issubset(df.columns):
            raise ValueError(f"Missing required columns: {set(feature_cols + [target_col]) - set(df.columns)}")

        X = df[feature_cols]
        y = df[target_col]

        # Enforce decreasing capacity with increasing cycles
        monotone_constraints = (-1 if 'cycle' in feature_cols else 0, 0 if 'current' in feature_cols else 0)
        model = XGBRegressor(
            objective='reg:squarederror',
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42,
            monotone_constraints=monotone_constraints
        )
        model.fit(X, y)

        y_pred = model.predict(X)
        rmse, r2 = evaluate_model(y, y_pred)

        importance_dict = dict(zip(feature_cols, model.feature_importances_))
        importance_dict = {k: round(float(v), 6) for k, v in importance_dict.items()}
        print("Model trained.")
        print("Feature importances:", importance_dict)
        print("Performance:", rmse, r2)

        return model, rmse

    except Exception as e:
        print(f"Error in training: {str(e)}")
        raise

# -------------------------------
# Predict using the model
# -------------------------------
def predict_with_model(model, input_dict, feature_cols):
    """
    Predict capacity for a single input using the trained model.
    
    Returns:
         Predicted capacity.
    """
    try:
        if not all(key in input_dict for key in feature_cols):
            raise ValueError(f"Input dictionary missing required keys: {set(feature_cols) - set(input_dict.keys())}")
        X_input = pd.DataFrame([input_dict])[feature_cols]
        return model.predict(X_input)[0]
    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        raise

# -------------------------------
# Evaluate model performance
# -------------------------------
def evaluate_model(y_true, y_pred):
    """
    Evaluates model performance using RMSE and R² metrics.

    Returns:
        RMSE and R² score.
    """
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    print("RMSE: " + str(round(rmse, 4)) + ", R²: " + str(round(r2, 4)))
    return rmse, r2

# -------------------------------
# Estimate RUL
# -------------------------------
def estimate_rul(model, current_cycle, current_current, feature_cols, capacity_threshold=0.5, file_path=None):
    """
    Estimate RUL using a local linear approximation at the user-provided sample point.

    Returns:
        Formatted string with EOL cycle and RUL.
    """
    try:
        # Compute initial capacity at cycle=0 for the given current
        initial_capacity = predict_with_model(model, {'cycle': 0, 'current': current_current}, feature_cols)

        eol_capacity = capacity_threshold * initial_capacity

        # Validate inputs against dataset ranges
        if file_path:
            df = pd.read_csv(file_path)
            df = df.dropna()
            if not (df['cycle'].min() <= current_cycle <= df['cycle'].max()):
                print(f"Warning: current_cycle={current_cycle} outside dataset range [{df['cycle'].min()}, {df['cycle'].max()}]")
            if 'current' in feature_cols and not (df['current'].min() <= current_current <= df['current'].max()):
                print(f"Warning: current_current={current_current} outside dataset range [{df['current'].min()}, {df['current'].max()}]")

        # Predict capacity at current_cycle
        input_dict = {'cycle': current_cycle, 'current': current_current}
        current_capacity = predict_with_model(model, input_dict, feature_cols)
        
        # Check if already at EOL
        if current_capacity <= eol_capacity:
            return "EOL_Cycle: " + str(round(current_cycle, 2)) + ", RUL: 0.0"

        # Estimate local slope by averaging over dataset-informed cycle steps
        if file_path:
            df = pd.read_csv(file_path)
            max_cycle = df['cycle'].max()
            cycle_steps = [min(5, max_cycle - current_cycle), min(10, max_cycle - current_cycle), min(20, max_cycle - current_cycle)]
        else:
            cycle_steps = [5, 10, 20]

        slopes = []
        for step in cycle_steps:
            if step > 0:  # Ensure valid step
                next_cycle = current_cycle + step
                next_input_dict = {'cycle': next_cycle, 'current': current_current}
                next_capacity = predict_with_model(model, next_input_dict, feature_cols)
                slope = (next_capacity - current_capacity) / step
                if slope < 0:  # Only include negative slopes
                    slopes.append(slope)
        
        if not slopes:
            print(f"Warning: No negative slopes found for cycle steps {cycle_steps}. Cannot estimate RUL.")
            return "EOL_Cycle: None, RUL: None"

        slope = np.mean(slopes)  # Average negative slopes

        # Estimate EOL cycle using local linear approximation
        eol_cycle = current_cycle + (eol_capacity - current_capacity) / slope
        rul = max(eol_cycle - current_cycle, 0)

        # Validate RUL
        if eol_cycle < current_cycle or rul <= 0:
            print(f"Warning: Invalid EOL cycle ({eol_cycle:.2f}) or RUL ({rul:.2f}). Capacity: {current_capacity:.4f}, EOL threshold: {eol_capacity:.4f}, Slope: {slope:.6f}")
            return "EOL_Cycle: None, RUL: None"

        return "EOL_Cycle: " + str(round(eol_cycle, 2)) + ", RUL: " + str(round(rul, 2))

    except Exception as e:
        print(f"Error in RUL estimation: {str(e)}")
        raise

# -------------------------------
# USAGE
# -------------------------------
if __name__ == "__main__":
    file_path = 'Test_balanced_discharge_data.csv'
    features = ['cycle', 'current']
    target = 'capacity'

    try:
        # Only train if there is no saved model
        model_path = 'rulxg_model.pkl'
        if os.path.exists(model_path):
            model, rmse = load_model(model_path)
        else:
            model, rmse = train_model(file_path, features, target)
            save_model(model, rmse, model_path)

        # input for prediction and RUL estimation
        input_cycle = 97
        input_current = 0.002966027  # discharge meausured current
        input_data = {'cycle': input_cycle, 'current': input_current}
        
        # Predict capacity
        predicted_capacity = predict_with_model(model, input_data, features)
        print("Predicted Capacity: " + str(round(predicted_capacity, 4)) + " ± " + str(round(rmse, 2)))

        # Estimate RUL 
        rul_info = estimate_rul(
            model,
            current_cycle=input_cycle,
            current_current=input_current,
            feature_cols=features,
            capacity_threshold=0.2,  
            file_path=file_path
        )
        print("Estimated RUL:", rul_info)

    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
    except Exception as e:
        print(f"Error: {str(e)}")