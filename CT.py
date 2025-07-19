# Program that implements decision rules from a classification tree
# Author: Kenneth Kamogelo Baloyi
# Date: 11 July 2025

#CT.py
#Import necessary libraries
import pandas as pd
from collections import Counter
import os

def classify_sample(row):
    # Manually coded decision rules from the R tree
    if row['current_measured'] < -0.9981662:
        return 'B0007'
    else:
        if row['current_measured'] < -0.001243135:
            if row['voltage_measured'] >= 3.466922:
                return 'B0006'
            else:
                if row['temperature_measured'] >= 37.75091:
                    return 'B0006'
                else:
                    return 'B0018'
        else:
            if row['temperature_measured'] >= 38.67214:
                return 'B0005'
            else:
                if row['voltage_measured'] >= 3.46407:
                    if row['temperature_measured'] >= 23.66534:
                        if row['voltage_measured'] < 3.599884:
                            return 'B0005'
                        else:
                            if row['voltage_measured'] >= 4.18646:
                                return 'B0005'
                            else:
                                return 'B0006'
                    else:
                        return 'B0018'
                else:
                    return 'B0018'

def classify_battery_csv(csv_path, limit=2916):
    df = pd.read_csv(csv_path)
    
    # Limit rows if needed
    df = df.head(limit)

    required_cols = ['current_measured', 'voltage_measured', 'temperature_measured']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")
    
    predictions = []
    for _, row in df.iterrows():
        predicted = classify_sample(row)
        predictions.append(predicted)
    
    counts = Counter(predictions)
    battery_name = os.path.splitext(os.path.basename(csv_path))[0]

    print(f"\nActual Battery File: {battery_name}")
    print("Predicted Class Distribution:")
    for k, v in counts.items():
        print(f"  - {k}: {v}")
    
    most_common = counts.most_common(1)[0]
    print(f"\nMajority Predicted Class: {most_common[0]} (Count: {most_common[1]})")

# USAGE
classify_battery_csv("B0028.csv")
