🌸 Iris Flower Classification

A simple machine learning web app that predicts the species of an Iris flower from its sepal and petal measurements, using a K-Nearest Neighbors (KNN) classifier and an interactive Streamlit interface.

Overview

This project trains a KNN classifier on the classic Iris dataset and wraps it in a Streamlit app where a user can enter flower measurements and get a real-time species prediction, along with a confusion matrix showing how the model performs on held-out test data.

Features
Interactive input for sepal length/width and petal length/width
Real-time species prediction (Setosa, Versicolor, Virginica)
Confusion matrix visualization of model performance on the test set
Species information panel with distinguishing characteristics
Tech Stack
Python
Scikit-learn — KNN classifier, train/test split, evaluation
Pandas — data handling
Matplotlib / Seaborn — confusion matrix visualization
Streamlit — web interface
Model Details
Algorithm: K-Nearest Neighbors (k=3)
Dataset: Iris dataset (150 samples, 4 features, 3 classes)
Split: 80% train / 20% test (random_state=42)
Features: Sepal length, sepal width, petal length, petal width
Classes: Setosa, Versicolor, Virginica
Project Structure
├── app.py              # Streamlit app — UI, inputs, prediction display
├── model.py             # Data loading, KNN training, prediction & confusion matrix functions
├── test.py               # Tests
├── images/               # Project images/screenshots
└── requirements.txt
Setup & Usage
Clone the repository
bash
   git clone https://github.com/Abdullah1dev/Data-Classification-Using-AI.git
   cd Data-Classification-Using-AI
Install dependencies
bash
   pip install streamlit scikit-learn pandas matplotlib seaborn
Run the app
bash
   streamlit run app.py
Open the local URL Streamlit prints (usually http://localhost:8501) and enter measurements to get a prediction.
Possible Improvements
Compare KNN against other classifiers (Logistic Regression, Random Forest, SVM) and report accuracy/precision/recall for each
Add cross-validation to justify the choice of k
Swap in a real-world dataset instead of the built-in Iris data
Add unit tests for the prediction function
License

This project is open source and available for learning purposes.
