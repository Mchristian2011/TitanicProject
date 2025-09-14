"""
Author: Your Name
Date: 09/07/2025
Course: Machine Learning
Description: Supervised classification on Titanic dataset using Decision Tree.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Load Titanic dataset
data = pd.read_csv("titanic.csv")

# Select features and target
features = data[['Pclass', 'Age', 'Sex', 'Fare']].dropna()
features['Sex'] = features['Sex'].map({'male': 0, 'female': 1})  # Encode gender
target = data.loc[features.index, 'Survived']

# Split data
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=42)

# Train Decision Tree
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

# Predict and evaluate
preds = clf.predict(X_test)
acc = accuracy_score(y_test, preds)

print(f"Decision Tree Accuracy: {acc:.2f}")
print("Sample Predictions:", preds[:10])
