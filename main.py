import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns





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



#now do the prdeiction part

y_pred = model.predict(X_test)
print(y_pred)



#comparison

comparison = pd.DataFrame({
    "Actual" : [iris.target_names[i] for i in y_test],
    "Predicted"  : [iris.target_names[i] for i in y_pred]
    
})

print(comparison)


#accuracy

accuracy = accuracy_score(y_test , y_pred)
print("Model Accuracy is :")


print(f"Accuracy: {accuracy:.2%}")


#confusion Metrix

cm = confusion_matrix(y_test , y_pred)
print("Confusion Metrix")

print(cm)


#classification Report

report = classification_report(
    y_test,
    y_pred,
    target_names = iris.target_names

)

print("Classification Report")
print(report)


#create the heatmap

plt.figure(figsize=(6 , 5))

sns.heatmap(
    cm,
    annot = True,
    cmap = "Blues",
    xticklabels=iris.target_names,
    yticklabels=iris.target_names
    
)

plt.title("Confusion Metrix")
plt.xlabel("Predicted Labels")
plt.ylabel("Acutal Lable")

plt.show()

