# src/train.py

import os
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
import mlflow
import mlflow.sklearn

# -----------------------------
# MLflow setup (IMPORTANT)
# -----------------------------
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("mlops-exp")

# -----------------------------
# Force training for experiments
# -----------------------------
FORCE_TRAIN = True

if not FORCE_TRAIN:
    if os.path.exists("models/model.pkl") and not os.path.exists("retrain.flag"):
        print("Model is up-to-date. Skipping training.")
        exit()

print("Training model...")

with mlflow.start_run():

    # -----------------------------
    # Step 1: Create dataset
    # -----------------------------
    os.makedirs("data", exist_ok=True)

    data = load_iris()

    df = pd.DataFrame(data.data, columns=data.feature_names) #type: ignore
    df["target"] = data.target#type: ignore

    df.to_csv("data/dataset.csv", index=False)
    print("Dataset saved")

    # -----------------------------
    # Step 2: Prepare data
    # -----------------------------
    X = df.drop("target", axis=1)
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # -----------------------------
    # Step 3: Train model
    # -----------------------------
    n_estimators = 100

    model = RandomForestClassifier(n_estimators=n_estimators)
    model.fit(X_train, y_train)

    # -----------------------------
    # Step 4: Save model
    # -----------------------------
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/model.pkl")

    print("Model trained and saved")

    # -----------------------------
    # Step 5: MLflow logging
    # -----------------------------
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.sklearn.log_model(model, name="model")#type: ignore