import streamlit as st
from model import predict_species

st.set_page_config(
    page_title="Iris Flower Classifier",
    page_icon="🌸",
    layout="centered"
)

st.title("🌸 Iris Flower Classification")

st.write(
    "Predict the species of an Iris flower using a trained K-Nearest Neighbors (KNN) model."
)

st.divider()

sepal_length = st.number_input(
    "Sepal Length (cm)",
    min_value=0.0,
    format="%.1f"
)

sepal_width = st.number_input(
    "Sepal Width (cm)",
    min_value=0.0,
    format="%.1f"
)

petal_length = st.number_input(
    "Petal Length (cm)",
    min_value=0.0,
    format="%.1f"
)

petal_width = st.number_input(
    "Petal Width (cm)",
    min_value=0.0,
    format="%.1f"
)

if st.button("Predict Species"):
    prediction = predict_species(
        sepal_length,
        sepal_width,
        petal_length,
        petal_width
    )

    st.success(f"🌸 Predicted Species: {prediction.title()}")