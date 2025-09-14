import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load the dataset
data = pd.read_csv("train.csv")

# Quick look at the data
print(data.head())

# Select features and target
features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare"]
data["Sex"] = data["Sex"].map({"male": 0, "female": 1})
data["Age"].fillna(data["Age"].median(), inplace=True)

X = data[features]
y = data["Survived"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Predict and evaluate
y_pred = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
