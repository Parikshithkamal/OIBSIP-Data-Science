#This is a data science project on unemployment analysis

import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt
#load the dataset
unemployment=pd.read_csv("/content/Unemployment_Rate_upto_11_2020.csv")
print(unemployment)

#exploring the dataset
print(unemployment.head())  # Display the first 5 rows
print(unemployment.info())  # Display information about the dataset
print(unemployment.describe())  # Display statistical summary of numeric columns

#Checking for missing values
print(unemployment.isnull().sum())

#Now let us visualise the data
# Create a bar plot to visualize the unemployment rate by region
plt.figure(figsize=(10, 6))
sns.barplot(x='Region', y=unemployment.columns[unemployment.columns.str.contains('Estimated Unemployment Rate')][0], data=unemployment)
plt.xticks(rotation=90)
plt.title('Unemployment Rate by Region')
plt.show()

#creating a scatter plot
plt.figure(figsize=(8, 6))
sns.scatterplot(x=unemployment.columns[unemployment.columns.str.contains('Estimated Employed')][0], y=unemployment.columns[unemployment.columns.str.contains('Estimated Unemployment Rate')][0], data=unemployment)
plt.title('Relationship between Estimated Employed and Unemployment Rate')
plt.xlabel('Estimated Employed')
plt.ylabel('Unemployment Rate')
plt.show()

# Importing necessary libraries
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

target_column = unemployment.columns[unemployment.columns.str.contains('Estimated Unemployment Rate')][0]
X = unemployment[[unemployment.columns[unemployment.columns.str.contains('Estimated Employed')][0]]]  # Feature(s)
y = unemployment[target_column]

#splitting the data into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
#Training the model using the training data
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluating the model
mse = mean_squared_error(y_test, y_pred)
print('Mean Squared Error:', mse)

#We can also visualize the predicted vs. actual values
plt.figure(figsize=(8, 6))
plt.scatter(X_test, y_test, color='blue', label='Actual')
plt.plot(X_test, y_pred, color='red', linewidth=2, label='Predicted')
plt.title('Actual vs. Predicted Unemployment Rate')
plt.xlabel('Estimated Employed')
plt.ylabel('Unemployment Rate')
plt.legend()
plt.show()

from sklearn.metrics import mean_absolute_error, r2_score
#Computing the Mean Absolute Error (MAE)
mae = mean_absolute_error(y_test, y_pred)
print('Mean Absolute Error:', mae)
