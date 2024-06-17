# Data Science project to identify the species of iris flower
import seaborn as sns
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
#load the dataset
iris=pd.read_csv("/content/Iris.csv")
print(iris)

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix



#Data Preprocessing
X = iris.drop('Species', axis=1)
y = iris['Species']
le = LabelEncoder()
y = le.fit_transform(y)

#Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Choose a classification model (Random Forest Classifier in this case)
model = RandomForestClassifier()

#Next train the model on the training data
model.fit(X_train, y_train)

#Predict the target variable on the test set
y_pred = model.predict(X_test)

#Evaluate the model performance
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print("Accuracy:", accuracy)
print("Classification Report:\n", report)
print("Confusion Matrix:\n", conf_matrix)

#Define a new sample to test the model
new_sample = [[5.8, 2.7, 5.1, 1.9, 0]]  # Add the missing feature with a placeholder value

#Make a prediction on the new sample
prediction = model.predict(new_sample)

#Get the predicted species
predicted_species = le.inverse_transform(prediction)[0]

print("Predicted species:", predicted_species)
