import streamlit as st
from model import predict_species, plot_confusion_matrix

# ===========================
# Page Configuration
# ===========================

st.set_page_config(
    page_title="Iris Flower Classification",
    page_icon="🌸",
    layout="centered"
)

# ===========================
# Sidebar
# ===========================

st.sidebar.title("🌸 Iris Flower Classifier")

st.sidebar.markdown("""
### About

This application predicts the species of an Iris flower using a **K-Nearest Neighbors (KNN)** Machine Learning model.

### Model Information

- Algorithm: KNN
- Dataset: Iris Dataset
- Features: 4
- Classes: 3

### Iris Species

- 🌸 Setosa
- 🌼 Versicolor
- 🌺 Virginica
""")

# ===========================
# Main Title
# ===========================

st.title("🌸 Iris Flower Classification")

st.write(
    "Enter the flower measurements below to predict the species of an Iris flower."
)

st.divider()

# ===========================
# Input Section
# ===========================

col1, col2 = st.columns(2)

with col1:
    sepal_length = st.number_input(
        "Sepal Length (cm)",
        min_value=0.0,
        value=5.1,
        step=0.1
    )

    petal_length = st.number_input(
        "Petal Length (cm)",
        min_value=0.0,
        value=1.4,
        step=0.1
    )

with col2:
    sepal_width = st.number_input(
        "Sepal Width (cm)",
        min_value=0.0,
        value=3.5,
        step=0.1
    )

    petal_width = st.number_input(
        "Petal Width (cm)",
        min_value=0.0,
        value=0.2,
        step=0.1
    )

st.write("")

# ===========================
# Prediction
# ===========================

if st.button("🔍 Predict Species", use_container_width=True):

    prediction = predict_species(
        sepal_length,
        sepal_width,
        petal_length,
        petal_width
    )

    st.success(f"### 🌸 Predicted Species: **{prediction.title()}**")

    if prediction == "setosa":
        st.info("""
### 🌸 Setosa

- Small petals
- Wide sepals
- Easiest species to classify
        """)

    elif prediction == "versicolor":
        st.info("""
### 🌼 Versicolor

- Medium-sized petals
- Medium-sized sepals
- Lies between Setosa and Virginica
        """)

    else:
        st.info("""
### 🌺 Virginica

- Largest petals
- Larger flower measurements
- Often confused with Versicolor
        """)

st.divider()

# ===========================
# Confusion Matrix
# ===========================

st.subheader("📊 Model Evaluation")

fig = plot_confusion_matrix()

st.pyplot(fig)

st.divider()

# ===========================
# Dataset Information
# ===========================

st.subheader("📁 Dataset Information")

st.markdown("""
**Dataset:** Iris Dataset

- Total Samples: **150**
- Features: **4**
- Classes: **3**

### Input Features

- Sepal Length (cm)
- Sepal Width (cm)
- Petal Length (cm)
- Petal Width (cm)

### Target Classes

- Setosa
- Versicolor
- Virginica
""")

st.divider()

# ===========================
# Footer
# ===========================

st.caption("Built with ❤️ using Python, Scikit-learn and Streamlit")