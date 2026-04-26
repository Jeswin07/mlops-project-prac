# src/evaluate.py

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import json
import os
import mlflow

# -----------------------------
# MLflow setup (IMPORTANT)
# -----------------------------
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("mlops-exp")

with mlflow.start_run():

    # -----------------------------
    # Step 1: Load dataset
    # -----------------------------
    df = pd.read_csv("data/dataset.csv")

    X = df.drop("target", axis=1)
    y = df["target"]

    # -----------------------------
    # Step 2: Same split
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

    # Save metrics for DVC
    with open("metrics.json", "w") as f:
        json.dump({"accuracy": acc}, f)

    # Log to MLflow
    mlflow.log_metric("accuracy", acc) #type: ignore

    # -----------------------------
    # Monitoring logic
    # -----------------------------
    THRESHOLD = 0.8

    if acc < THRESHOLD:
        print("⚠️ WARNING: Model performance dropped!")

        with open("retrain.flag", "w") as f:
            f.write("retrain needed")
    else:
        if os.path.exists("retrain.flag"):
            os.remove("retrain.flag")