#importing the dependencies
import numpy as np
import pandas as pd
#for Numerical operations and handling datasets
import matplotlib.pyplot as plt
import seaborn as sns
#for plotting
from sklearn.preprocessing import LabelEncoder
#for encoding categorical variables
from sklearn.model_selection import train_test_split
#for spkitting data into training and testing sets
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
#for building reegressionn models
from sklearn import metrics
#for evaluating models

# Loading the data from the Walmart CSV file to a Pandas DataFrame
walmart_data = pd.read_csv('/content/Walmart.csv')

# Displaying the first 5 rows of the DataFrame
walmart_data.head()

# Checking the number of data points & number of features
print(walmart_data.shape)

# Getting some information about the dataset
walmart_data.info()

# Checking for missing values
print(walmart_data.isnull().sum())

# Descriptive statistics of the dataset
print(walmart_data.describe())

# Setting seaborn style
sns.set()

#Weekly_Sales
plt.figure(figsize=(6, 6))
sns.distplot(walmart_data['Weekly_Sales'])
plt.show()

#Temperature
plt.figure(figsize=(6, 6))
sns.distplot(walmart_data['Temperature'])
plt.show()

#Fuel_Price
plt.figure(figsize=(6, 6))
sns.distplot(walmart_data['Fuel_Price'])
plt.show()

#CPI
plt.figure(figsize=(6, 6))
sns.distplot(walmart_data['CPI'])
plt.show()

#Unemployment
plt.figure(figsize=(6, 6))
sns.distplot(walmart_data['Unemployment'])
plt.show()

# Convert 'Date' column to datetime
walmart_data['Date'] = pd.to_datetime(walmart_data['Date'])

# Extract relevant date features
walmart_data['Year'] = walmart_data['Date'].dt.year
walmart_data['Month'] = walmart_data['Date'].dt.month
walmart_data['Day'] = walmart_data['Date'].dt.day

# Drop the original 'Date' column
walmart_data = walmart_data.drop(columns=['Date'])

# Assuming 'Store' and 'Holiday_Flag' are categorical columns, use label encoding
encoder = LabelEncoder()
walmart_data['Store'] = encoder.fit_transform(walmart_data['Store'])
walmart_data['Holiday_Flag'] = encoder.fit_transform(walmart_data['Holiday_Flag'])

# Split the data into features (X) and target variable (Y)
X_walmart = walmart_data.drop(columns=['Weekly_Sales'], axis=1)
Y_walmart = walmart_data['Weekly_Sales']

# Splitting the data into training and testing sets with a random seed
X_train_walmart, X_test_walmart, Y_train_walmart, Y_test_walmart = train_test_split(
    X_walmart, Y_walmart, test_size=0.2, random_state=2)

# Using XGBoost as one of the regressors
xgboost_regressor_walmart = XGBRegressor()
xgboost_regressor_walmart.fit(X_train_walmart, Y_train_walmart)

# Prediction on training data using XGBoost
xgboost_training_data_prediction_walmart = xgboost_regressor_walmart.predict(X_train_walmart)
r2_train_xgboost_walmart = metrics.r2_score(Y_train_walmart, xgboost_training_data_prediction_walmart)
print('R Squared value for training data (XGBoost) =', r2_train_xgboost_walmart)

# Prediction on test data using XGBoost
xgboost_test_data_prediction_walmart = xgboost_regressor_walmart.predict(X_test_walmart)
r2_test_xgboost_walmart = metrics.r2_score(Y_test_walmart, xgboost_test_data_prediction_walmart)
print('R Squared value for test data (XGBoost) =', r2_test_xgboost_walmart)

# Using RandomForestRegressor as another regressor (ensemble)
random_forest_regressor_walmart = RandomForestRegressor()
random_forest_regressor_walmart.fit(X_train_walmart, Y_train_walmart)

# Prediction on training data using RandomForestRegressor
rf_training_data_prediction_walmart = random_forest_regressor_walmart.predict(X_train_walmart)
r2_train_rf_walmart = metrics.r2_score(Y_train_walmart, rf_training_data_prediction_walmart)
print('R Squared value for training data (Random Forest) =', r2_train_rf_walmart)

# Prediction on test data using RandomForestRegressor
rf_test_data_prediction_walmart = random_forest_regressor_walmart.predict(X_test_walmart)
r2_test_rf_walmart = metrics.r2_score(Y_test_walmart, rf_test_data_prediction_walmart)
print('R Squared value for test data (Random Forest) =', r2_test_rf_walmart)

# Ensemble Learning: Averaging predictions from XGBoost and RandomForestRegressor
ensemble_prediction_walmart = (xgboost_test_data_prediction_walmart + rf_test_data_prediction_walmart) / 2
r2_ensemble_walmart = metrics.r2_score(Y_test_walmart, ensemble_prediction_walmart)
print('R Squared value for test data (Ensemble) =', r2_ensemble_walmart)

