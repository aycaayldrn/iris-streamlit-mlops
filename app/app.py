import streamlit as st
from predict import predict
import numpy as np

# 1. set up the page title and description
st.title("Iris Species Predictor 💋🌸💋")
st.write(
    "Enter the measurements of an Iris flower to predict its species. "
    "This app uses a machine learning model to make a prediction."
)

# 2. define the species names for predictions.
# the model predicts an index (0, 1, or 2), which we map to these names.
species_names = ['Setosa', 'Versicolor', 'Virginica']

# 3. create input fields for flower measurements
st.header("Enter Flower Measurements (in cm)")

col1, col2 = st.columns(2)

with col1:
    sepal_length = st.number_input("Sepal Length", min_value=0.0, format="%.1f")
    petal_length = st.number_input("Petal Length", min_value=0.0, format="%.1f")

with col2:
    sepal_width = st.number_input("Sepal Width", min_value=0.0, format="%.1f")
    petal_width = st.number_input("Petal Width", min_value=0.0, format="%.1f")

# 4. create the prediction button
if st.button("Predict Species"):
    # 5. collect user inputs into a list
    features = [sepal_length, sepal_width, petal_length, petal_width]

    # 6. prediction
    prediction_result = predict(features)

    # 7. display the prediction and image
    if isinstance(prediction_result, (int, list, np.ndarray)):
        species_index = prediction_result[0]
        predicted_species = species_names[species_index]

        st.success(f"The predicted species is: **{predicted_species}**")

        if predicted_species == 'Setosa':
            st.image("https://upload.wikimedia.org/wikipedia/commons/5/56/Kosaciec_szczecinkowaty_Iris_setosa.jpg",
                     caption="Iris Setosa")
        elif predicted_species == 'Versicolor':
            st.image("https://upload.wikimedia.org/wikipedia/commons/4/41/Iris_versicolor_3.jpg",
                     caption="Iris Versicolor")
        elif predicted_species == 'Virginica':
            st.image("https://upload.wikimedia.org/wikipedia/commons/9/9f/Iris_virginica.jpg", caption="Iris Virginica")

    else:
        st.error(prediction_result)