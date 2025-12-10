import json
from pathlib import Path

import numpy as np
import streamlit as st

from predict import predict

# ---------- load metadata ----------

APP_DIR = Path(__file__).resolve().parent
META_PATH = APP_DIR / "model_meta.json"


def load_metadata():
    if not META_PATH.exists():
        # Fallback if metadata is missing
        return {
            "version": "unknown",
            "best_model": "unknown",
            "mlflow_run_id": "unknown",
            "metrics": {}
        }
    with META_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


meta = load_metadata()
version = meta.get("version", "unknown")
best_model = meta.get("best_model", "unknown")
run_id = meta.get("mlflow_run_id", "unknown")
metrics = meta.get("metrics", {})

# accuracy comes directly from your JSON
accuracy = metrics.get("accuracy")

# ---------- page layout ----------

st.set_page_config(page_title="Iris Species Predictor", layout="centered")

st.title("Iris Species Predictor 💋🌸💋")
st.write(
    "Enter the measurements of an Iris flower to predict its species. "
    "This app uses a machine learning model to make a prediction."
)

# species names for predictions
species_names = ["Setosa", "Versicolor", "Virginica"]

# input fields
st.header("Enter flower measurements (in cm)")

col1, col2 = st.columns(2)

with col1:
    sepal_length = st.number_input("Sepal length", min_value=0.0, format="%.1f")
    petal_length = st.number_input("Petal length", min_value=0.0, format="%.1f")

with col2:
    sepal_width = st.number_input("Sepal width", min_value=0.0, format="%.1f")
    petal_width = st.number_input("Petal width", min_value=0.0, format="%.1f")

# prediction button
if st.button("Predict species"):
    features = [sepal_length, sepal_width, petal_length, petal_width]

    prediction_result = predict(features)

    if isinstance(prediction_result, (int, list, np.ndarray)):
        # normalise to an index
        if isinstance(prediction_result, int):
            species_index = prediction_result
        else:
            species_index = prediction_result[0]

        try:
            species_index = int(species_index)
            predicted_species = species_names[species_index]
        except (IndexError, ValueError, TypeError):
            st.error(f"Unexpected prediction output: {prediction_result}")
        else:
            st.success(f"The predicted species is: **{predicted_species}**")

            if predicted_species == "Setosa":
                st.image(
                    "https://upload.wikimedia.org/wikipedia/commons/5/56/"
                    "Kosaciec_szczecinkowaty_Iris_setosa.jpg",
                    caption="Iris setosa",
                )
            elif predicted_species == "Versicolor":
                st.image(
                    "https://upload.wikimedia.org/wikipedia/commons/4/41/"
                    "Iris_versicolor_3.jpg",
                    caption="Iris versicolor",
                )
            elif predicted_species == "Virginica":
                st.image(
                    "https://upload.wikimedia.org/wikipedia/commons/9/9f/"
                    "Iris_virginica.jpg",
                    caption="Iris virginica",
                )
    else:
        # predict() returned an error string or similar
        st.error(prediction_result)

# ---------- footer with metadata ----------

st.markdown("---")
st.markdown("### Model information")

# clickable link to MLflow UI (adjust if you use a different host/port)
mlflow_ui_url = "http://localhost:5000"

# format accuracy nicely if available
if isinstance(accuracy, (int, float)):
    accuracy_str = f"{accuracy:.3f}"
else:
    accuracy_str = "unknown"

footer_text = (
    f"Version: **{version}** • "
    f"Best model: **{best_model}** • "
    f"MLflow run: `{run_id}` • "
    f"Accuracy: **{accuracy_str}** • "
    f"[Open MLflow UI]({mlflow_ui_url})"
)

st.markdown(footer_text)
