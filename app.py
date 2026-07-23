import streamlit as st
from model import predict_species

st.set_page_config(
    page_title="Iris Flower Classification",
    page_icon="🌸",
    layout="centered"
)


st.sidebar.title("🌸 Iris Classifier")

st.sidebar.markdown("""
### About

This application predicts the species of an Iris flower using a **K-Nearest Neighbors (KNN)** Machine Learning model.

### Model

- Algorithm: KNN
- Dataset: Iris Dataset
- Features: 4
- Classes: 3

### Species

- Setosa
- Versicolor
- Virginica
""")


 
st.title("🌸 Iris Flower Classification")

st.write(
    "Enter the flower measurements below to predict its species."
)

st.divider()



col1, col2 = st.columns(2)

with col1:

    sepal_length = st.number_input(
        "Sepal Length (cm)",
        min_value=0.0,
        value=5.1
    )

    petal_length = st.number_input(
        "Petal Length (cm)",
        min_value=0.0,
        value=1.4
    )

with col2:

    sepal_width = st.number_input(
        "Sepal Width (cm)",
        min_value=0.0,
        value=3.5
    )

    petal_width = st.number_input(
        "Petal Width (cm)",
        min_value=0.0,
        value=0.2
    )

st.write("")


if st.button("Predict Species", use_container_width=True):

    prediction = predict_species(
        sepal_length,
        sepal_width,
        petal_length,
        petal_width
    )

    st.success(f"🌸 Predicted Species: **{prediction.title()}**")

    if prediction == "setosa":
        st.image("images/setosa.jpg", width=350)

    elif prediction == "versicolor":
        st.image("images/versicolor.jpg", width=350)

    else:
        st.image("images/virginica.jpg", width=350)

st.divider()

# =========================
# Confusion Matrix
# =========================

st.subheader("📊 Model Evaluation")

st.image(
    "images/confusion_matrix.png",
    caption="Confusion Matrix"
)

st.divider()

# =========================
# Dataset Information
# =========================

st.subheader("📁 Dataset Information")

st.markdown("""
- **Dataset:** Iris Dataset
- **Samples:** 150
- **Features:** 4
- **Classes:** 3

### Input Features

- Sepal Length
- Sepal Width
- Petal Length
- Petal Width

### Target Classes

- Setosa
- Versicolor
- Virginica
""")

st.divider()

st.caption("Built with ❤️ using Python, Scikit-learn and Streamlit")