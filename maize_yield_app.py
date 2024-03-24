import pandas as pd

# Define the path to your dataset

# Load the dataset
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
