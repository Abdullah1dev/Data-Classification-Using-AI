import pandas as pd
from sklearn.datasets import load_iris


iris = load_iris()

df = pd.DataFrame(data=iris.data , columns= iris.feature_names)

print("Feature Names \n")
print(iris.feature_names)

print("Target Names")
print(iris.target_names)
