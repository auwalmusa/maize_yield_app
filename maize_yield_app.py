import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import mean_squared_error
import numpy as np
import joblib

# Attempt to load and display the dataset from CSV file
try:
    # Load the dataset
    df = pd.read_csv('maize_yield_prediction_dataset.csv')
    # Display the first few rows of the dataframe
    st.write(df.head())
except FileNotFoundError as e:
    st.error(f'Failed to load the dataset: {e}')
    # If the CSV file can't be found, create a DataFrame manually for demonstration
    data = [
        [1.5759, 0.8919, -0.7268, -0.0478, -0.6570, -1.1133, -0.8776, 0.9946, -0.2997, -0.7618, -62.956],
        [0.5355, 1.2667, -1.7799, 1.2090, -0.1132, -0.9711, 1.0642, -0.5553, 0.7413, -0.9875, 108.722],
        [-0.6863, 0.1571, 0.1458, -1.4327, -0.6668, 0.1762, 0.5853, 0.1979, -0.4044, -0.6075, -115.541],
    ]
    df = pd.DataFrame(data, columns=['SoilPH', 'P2O5', 'K2O', 'Zn', 'ClayContent', 'ECa', 'DraughtForce', 'ConeIndex', 'Precipitation', 'Temperature', 'MaizeYield'])
    st.write("Using backup data due to CSV load failure:", df.head())

# Fill missing values with the backward fill method
df_filled = df.bfill()

# Splitting dataset into features and target
X = df_filled.drop('MaizeYield', axis=1)
y = df_filled['MaizeYield']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the XGBoost model
model = xgb.XGBRegressor(objective ='reg:squarederror', colsample_bytree = 0.3, learning_rate = 0.1,
                max_depth = 5, alpha = 10, n_estimators = 10)
model.fit(X_train, y_train)

# Make predictions and calculate RMSE
predictions = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
st.write(f"RMSE: {rmse}")

# Save the model to a file
joblib.dump(model, 'maize_yield_model.pkl')
st.write("Model saved successfully!")
