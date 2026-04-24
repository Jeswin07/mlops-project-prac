# src/train.py

import os
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# -----------------------------
# Step 1: Create dataset (simulate real data pipeline)
# -----------------------------
os.makedirs("data", exist_ok=True)

data = load_iris()

df = pd.DataFrame(data.data, columns=data.feature_names)#type: ignore
df["target"] = data.target#type: ignore

# Save dataset
df.to_csv("data/dataset.csv", index=False)

print("Dataset saved to data/dataset.csv")

# -----------------------------
# Step 2: Load dataset (IMPORTANT for MLOps flow)
# -----------------------------
df = pd.read_csv("data/dataset.csv")

X = df.drop("target", axis=1)
y = df["target"]

# -----------------------------
# Step 3: Train model
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier()
model.fit(X_train, y_train)

# -----------------------------
# Step 4: Save model
# -----------------------------
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/model.pkl")

print("Model trained and saved at models/model.pkl")