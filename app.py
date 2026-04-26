# app.py

from fastapi import FastAPI
import joblib
import numpy as np

app = FastAPI()

# Load model once
model = joblib.load("models/model.pkl")

@app.get("/")
def home():
    return {"message": "ML Model API is running"}

@app.post("/predict")
def predict(features: list[float]):
    data = np.array(features).reshape(1, -1)
    prediction = model.predict(data)
    return {"prediction": int(prediction[0])}