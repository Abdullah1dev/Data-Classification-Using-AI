import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from app.py import plot_confusion_matrix
from app.py import predict_species

# Load the Iris dataset
iris = load_iris()

# Create DataFrame
df = pd.DataFrame(
    data=iris.data,
    columns=iris.feature_names
)

# Add target column
df["target"] = iris.target

# Features and target
X = df.drop(columns=["target"])
y = df["target"]

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train the KNN model
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)


def predict_species(sepal_length, sepal_width, petal_length, petal_width):
    """
    Predict the species of an Iris flower based on its measurements.
    """

    prediction = model.predict([
        [
            sepal_length,
            sepal_width,
            petal_length,
            petal_width
        ]
    ])

    return iris.target_names[prediction[0]]