# Data science project on car price prediction

import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt
#load the dataset
car=pd.read_csv("/content/car data.csv")
print(car)

print(car.head())
print(car.isnull().sum())

# Exploring the data
print(car.info())
print(car.describe())

car = car.dropna()

#Visualizing the data
#Creating a bar plot for the distribution of fuel types
plt.figure(figsize=(10, 6))
sns.countplot(x='Fuel_Type', data=car)
plt.title('Distribution of Fuel Types')
plt.xlabel('Fuel Type')
plt.ylabel('Count')
plt.show()

#Now let us create a scatter plot for the relationship between driven kilometers and selling price
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Driven_kms', y='Selling_Price', data=car)
plt.title('Relationship between Driven Kilometers and Selling Price')
plt.xlabel('Driven Kilometers')
plt.ylabel('Selling Price')
plt.show()

# Create a histogram for the distribution of selling prices
plt.figure(figsize=(10, 6))
sns.histplot(car['Selling_Price'], bins=50)
plt.title('Distribution of Selling Prices')
plt.xlabel('Selling Price')
plt.ylabel('Frequency')
plt.show()

#Spliting the data into training and testing sets
from sklearn.model_selection import train_test_split
X = car.drop(['Car_Name', 'Year', 'Selling_Price'], axis=1)
y = car['Selling_Price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Encoding the categorical variables
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
X_train['Fuel_Type'] = le.fit_transform(X_train['Fuel_Type'])
X_test['Fuel_Type'] = le.transform(X_test['Fuel_Type'])
X_train['Selling_type'] = le.fit_transform(X_train['Selling_type'])
X_test['Selling_type'] = le.transform(X_test['Selling_type'])
X_train['Transmission'] = le.fit_transform(X_train['Transmission'])
X_test['Transmission'] = le.transform(X_test['Transmission'])
X_train['Owner'] = le.fit_transform(X_train['Owner'])
X_test['Owner'] = le.transform(X_test['Owner'])

#Now training a machine learning model
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

#Making predictions on the test set
y_pred = model.predict(X_test)

#Now let us evaluate the model
from sklearn.metrics import mean_squared_error, r2_score
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print('Mean Squared Error:', mse)
print('R-squared:', r2)

from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print('Mean Squared Error:', mse)
print('R-squared:', r2)
print('Mean Absolute Error:', mae)

