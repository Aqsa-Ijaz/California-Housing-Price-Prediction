import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
from sklearn.model_selection import train_test_split

data = fetch_california_housing(as_frame=True)
df = data.frame
print(df.head())
print("\nDataset shape:")
print(df.shape)

print("\nColumn names:")
print(df.columns)

print("\nData types:")
print(df.dtypes)

print("\nMissing values:")
print(df.isnull().sum())

print("\nDataset statistics:")
print(df.describe())

X=df.drop("MedHouseVal", axis=1)
y=df["MedHouseVal"]

print("\n X shape")
print(X.shape)
print("\n y shape")
print(y.shape)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("X_train:", X_train.shape)
print("X_test:", X_test.shape)
print("y_train:", y_train.shape)
print("y_test:", y_test.shape)

from sklearn.linear_model import LinearRegression

model = LinearRegression()

model.fit(X_train, y_train)

print("Model trained successfully!")

y_pred = model.predict(X_test)

print("First 5 predictions:")
print(y_pred[:5])
print("First 5 actual values:")
print(y_test.iloc[:5].values)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)
print("R² Score:", r2)


import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))

plt.scatter(y_test[:300], y_pred[:300], alpha=0.5)

# Perfect prediction line
plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()]
)

plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Actual vs Predicted Values (First 300 Samples)")

plt.show()

# random forest
print("\nRandom Forest Regressor")
rf_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train, y_train)

rf_pred = rf_model.predict(X_test)

rf_mae = mean_absolute_error(y_test, rf_pred)
rf_mse = mean_squared_error(y_test, rf_pred)
rf_rmse = np.sqrt(rf_mse)
rf_r2 = r2_score(y_test, rf_pred)

print("Random Forest Results")
print("MAE:", rf_mae)
print("MSE:", rf_mse)
print("RMSE:", rf_rmse)
print("R² Score:", rf_r2)

# Feature Importance

importances = rf_model.feature_importances_

for feature, importance in zip(X.columns, importances):
    print(feature, ":", importance)


plt.bar(X.columns, importances)

plt.xlabel("Features")
plt.ylabel("Importance")
plt.title("Random Forest Feature Importance")

plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


plt.scatter(y_test, rf_pred, alpha=0.5)

plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()])

plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Actual vs Predicted House Prices")

plt.tight_layout()
plt.show()

residuals = y_test - rf_pred

print("First 10 residuals:")
print(residuals[:10])

plt.scatter(rf_pred, residuals)

plt.axhline(y=0)

plt.xlabel("Predicted Values")
plt.ylabel("Residuals")
plt.title("Residual Plot")

plt.tight_layout()
plt.show()

import joblib

joblib.dump(rf_model, "random_forest_model.pkl")

print("Model saved successfully!")

#where it is saved
import os

print(os.getcwd())
# how to load it 
loaded_model = joblib.load("models/random_forest_model.pkl")

print("Model loaded successfully!")
# test the loaded model
predictions_loaded = loaded_model.predict(X_test)

print("First 5 predictions:")
print(predictions_loaded[:5])

original_predictions = rf_model.predict(X_test)

print("Do both models give the same predictions?")
print(np.allclose(original_predictions, predictions_loaded))