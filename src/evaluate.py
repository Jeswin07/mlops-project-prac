# src/evaluate.py

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import json

# -----------------------------
# Step 1: Load dataset (IMPORTANT)
# -----------------------------
df = pd.read_csv("data/dataset.csv")

X = df.drop("target", axis=1)
y = df["target"]

# -----------------------------
# Step 2: Same split as training
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# Step 3: Load model
# -----------------------------
model = joblib.load("models/model.pkl")

# -----------------------------
# Step 4: Predict
# -----------------------------
preds = model.predict(X_test)

# -----------------------------
# Step 5: Evaluate
# -----------------------------
acc = accuracy_score(y_test, preds)
print("Accuracy:", acc)

# -----------------------------
# Step 6: Save metrics (for DVC)
# -----------------------------
with open("metrics.json", "w") as f:
    json.dump({"accuracy": acc}, f)