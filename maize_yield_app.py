import pandas as pd
import streamlit as st

# Example data embedded directly into the script
data = [
    [1.5759, 0.8919, -0.7268, -0.0478, -0.6570, -1.1133, -0.8776, 0.9946, -0.2997, -0.7618, -62.956],
    [0.5355, 1.2667, -1.7799, 1.2090, -0.1132, -0.9711, 1.0642, -0.5553, 0.7413, -0.9875, 108.722],
    [-0.6863, 0.1571, 0.1458, -1.4327, -0.6668, 0.1762, 0.5853, 0.1979, -0.4044, -0.6075, -115.541],
    # Add more rows as needed
]

# Convert the data to a DataFrame
df = pd.DataFrame(data, columns=['SoilPH', 'P2O5', 'K2O', 'Zn', 'ClayContent', 'ECa', 'DraughtForce', 'ConeIndex', 'Precipitation', 'Temperature', 'MaizeYield'])

# Use the DataFrame as before
st.write(df.head())

except FileNotFoundError as e:
    st.error(f'Failed to load the dataset: {e}')

df = pd.read_csv('maize_yield_prediction_dataset.csv')


# Display the first few rows of the dataframe
print(df.head())

# Fill missing values with the forward fill method
df_filled = df_filled = df.bfill()

from sklearn.model_selection import train_test_split

# Assuming 'MaizeYield' is your target column
X = df_filled.drop('MaizeYield', axis=1)
y = df_filled['MaizeYield']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

import xgboost as xgb

# Initialize and train the XGBoost model
model = xgb.XGBRegressor(objective ='reg:squarederror', colsample_bytree = 0.3, learning_rate = 0.1,
                max_depth = 5, alpha = 10, n_estimators = 10)
model.fit(X_train, y_train)

from sklearn.metrics import mean_squared_error
import numpy as np

# Make predictions
predictions = model.predict(X_test)

# Calculate RMSE
rmse = np.sqrt(mean_squared_error(y_test, predictions))
print(f"RMSE: {rmse}")
import pandas as pd

# Define the path to your dataset
file_path = r'C:\Users\wasagu\OneDrive\Desktop\maize_yield_app\maize_yield_prediction_dataset.csv'

# Load the dataset
df = pd.read_csv(file_path)

# Display the first few rows of the dataframe
print(df.head())

# Fill missing values with the forward fill method
df_filled = df_filled = df.bfill()

from sklearn.model_selection import train_test_split

# Assuming 'MaizeYield' is your target column
X = df_filled.drop('MaizeYield', axis=1)
y = df_filled['MaizeYield']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

import xgboost as xgb

# Initialize and train the XGBoost model
model = xgb.XGBRegressor(objective ='reg:squarederror', colsample_bytree = 0.3, learning_rate = 0.1,
                max_depth = 5, alpha = 10, n_estimators = 10)
model.fit(X_train, y_train)

from sklearn.metrics import mean_squared_error
import numpy as np

# Make predictions
predictions = model.predict(X_test)

# Calculate RMSE
rmse = np.sqrt(mean_squared_error(y_test, predictions))
print(f"RMSE: {rmse}")
import joblib

# After training the model and calculating RMSE
# Save the model to a file
joblib.dump(model, 'maize_yield_model.pkl')

print("Model saved successfully!")
