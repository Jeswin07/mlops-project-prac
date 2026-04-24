# src/evaluate.py
import joblib
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Load data
data = load_iris()
X, y = data.data, data.target # type: ignore

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Load model
model = joblib.load("models/model.pkl")

# Predict
preds = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, preds))