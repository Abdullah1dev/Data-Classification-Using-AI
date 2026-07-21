import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier



iris = load_iris()

df = pd.DataFrame(data=iris.data , columns= iris.feature_names)



df["species"] = iris.target
df["target"] = iris.target





df["species"] = df["target"].map({
    0 : "setosa",
    1 : "versicolor",
    2 : "virginica"
    
}
    
)





print(df.head())

#EDA 

print("DF Shape \n")
print(df.shape)

print("Dataset Information \n")
df.info()


print("Statistical Summary \n")
print(df.describe())

print("Missing Values")
print(df.isnull().sum())

print("Specie Distribution \n")
print(df["species"].value_counts())


X = df.drop(columns=["target" , "species"])
y = df["target"]

print("X-Shape")
print(X.shape)

print("Y-shape")
print(y.shape)

X_train , X_test , y_train , y_test = train_test_split (
    X,
    y,
    test_size=0.2,
    random_state=42
    
    
)

print("Training Features:", X_train.shape)
print("Testing Features :", X_test.shape)

print("Training Labels  :", y_train.shape)
print("Testing Labels   :", y_test.shape)


model = KNeighborsClassifier(n_neighbors  = 3)

model.fit(X_train , y_train)
print("Model trained Successfully")




