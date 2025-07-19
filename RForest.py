# Random forest algorithm to classify different battery types.
# AUthor: Kenneth Kamogelo Baloyi
# Date: 06 July 2025


#RForest.py
#Import necessary libraries
import os
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

#LOAD DATA 
df = pd.read_csv("balanced_discharge_data.csv")
features = ["voltage_measured", "current_measured", "temperature_measured", "capacity", "cycle"]

X = df[features].values
y = df["battery"].values

# ENCODING FEATURES TO LABELS
le = LabelEncoder()
y_enc = le.fit_transform(y)

# SCALING 
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#STRATIFIED SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_enc, test_size=0.2, random_state=42, stratify=y_enc
)

# CHECK CLASS DISTRIBUTION
def log_class_distribution(label_array, encoder, name):
    counts = Counter(label_array)
    print(f"\n{name} Set Distribution:")
    for i in range(len(encoder.classes_)):
        print(f"  - Class {i} ({encoder.classes_[i]}): {counts.get(i, 0)} samples")

log_class_distribution(y_enc, le, "Full")
log_class_distribution(y_train, le, "Train")
log_class_distribution(y_test, le, "Test")

# HYPERPARAMETER GRID SEARCH
param_grid = {
    'n_estimators': [100, 300],
    'max_depth': [10, 15, None],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'max_features': ['sqrt', 'log2'],
    'class_weight': ['balanced']
}

grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42, n_jobs=-1),
    param_grid=param_grid,
    cv=3,
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)
best_clf = grid_search.best_estimator_

# SAVE COMPONENTS
joblib.dump(best_clf, "best_random_forest_model.save")
joblib.dump(scaler, "rf_scaler.save")
joblib.dump(le, "rf_label_encoder.save")
joblib.dump(y_train, "rf_y_train.save")
print("\nBest model, scaler, and label encoder saved.")

# CROSS-VALIDATION SCORE
cv_scores = cross_val_score(best_clf, X_train, y_train, cv=5)
print(f"\nCross-validation accuracy: {cv_scores.mean():.2f} ± {cv_scores.std():.2f}")

#  FINAL EVALUATION 
y_pred = best_clf.predict(X_test)
accuracy = (y_pred == y_test).sum() / len(y_test) * 100

print(f"\nRandom Forest Accuracy on Test Set: {accuracy:.2f}%\n")
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# FEATURE IMPORTANCE PLOT 
importances = best_clf.feature_importances_
sorted_idx = np.argsort(importances)[::-1]
sorted_features = [features[i] for i in sorted_idx]
"""
plt.figure(figsize=(8, 4))
plt.title("Feature Importances")
plt.bar(range(len(importances)), importances[sorted_idx], align="center")
plt.xticks(range(len(importances)), sorted_features, rotation=45)
plt.tight_layout()
plt.show()

"""
