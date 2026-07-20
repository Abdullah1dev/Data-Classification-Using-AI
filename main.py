import pandas as pd
from sklearn.datasets import load_iris


iris = load_iris()

df = pd.DataFrame(data=iris.data , columns= iris.feature_names)

print("Feature Names \n")
print(iris.feature_names)

print("Target Names")
print(iris.target_names)


df = pd.DataFrame(
    data = iris.data,
    columns = iris.feature_names
)
df["species"] = iris.target




df["species"] = df["species"].replace({
    0 : "setosa",
    1 : "versicolor",
    2 : "virgincia"
}
    
)

print(df.head())

#EDA 

print("DF Shape \n")
print(df.shape)

print("Dataset Information \n")
print(df.info())

print("Statistical Summary \n")
print(df.describe())

print("Missing Values")
print(df.isnull().sum())

print("Specie Distribution \n")
print(df["species"].value_counts())



