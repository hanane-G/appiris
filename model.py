import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle

# Notre data iris.csv
df=pd.read_csv("iris.csv")

print(df.head())

# Les features
X = df[["Sepal_Length", "Sepal_Width", "Petal_Length", "Petal_Width"]]
# Le target
y = df["Class"]


# train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=50)


# Instancier le modèle avec 100 estimateurs (arbre)
classifier = RandomForestClassifier(n_estimators=100)

# Entrainer le modèle
classifier.fit(X_train, y_train)

# Sauvegarder le modèle entrainé dans un fichier pickle
pickle.dump(classifier, open("model.pkl", "wb"))