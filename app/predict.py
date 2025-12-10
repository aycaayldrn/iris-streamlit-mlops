import joblib
import numpy as np
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "model.joblib"   # model in app/model.joblib

# 1. load the trained model from the file
try:
    print(f"DEBUG: Attempting to load model from: {MODEL_PATH}")
    model = joblib.load(str(MODEL_PATH))
    print("Model loaded successfully.")
except FileNotFoundError:
    print(f"Error: Model file not found at {MODEL_PATH}")
    model = None
except Exception as e:
    print(f"Error loading model: {e}")
    model = None


def predict(features):
    """
    Takes a list of 4 features and returns the predicted Iris species index.
    """
    if model is None:
        return "Error: Model is not loaded."

    try:
        # model expects a 2D array, so reshape the [1, 2, 3, 4] list
        features_array = np.array(features).reshape(1, -1)

        prediction = model.predict(features_array)

        return prediction
    except Exception as e:
        return f"Error during prediction: {e}"


