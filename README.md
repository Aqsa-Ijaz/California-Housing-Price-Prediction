# California Housing Price Prediction

## Project Overview

This project uses Machine Learning to predict California house prices based on different housing-related features.

The project compares two regression models:

* Linear Regression
* Random Forest Regressor

The Random Forest model was trained and evaluated to determine its performance in predicting median house values.

## Dataset

The project uses the California Housing dataset provided by Scikit-learn.

The dataset contains several features related to California housing, including:

* Median Income
* House Age
* Average Number of Rooms
* Average Number of Bedrooms
* Population
* Average Occupancy
* Latitude
* Longitude

The target variable is:

* Median House Value (`MedHouseVal`)

## Project Workflow

1. Load the California Housing dataset.
2. Explore and inspect the dataset.
3. Check for missing values.
4. Separate features and the target variable.
5. Split the data into training and testing sets.
6. Train a Linear Regression model.
7. Evaluate the Linear Regression model.
8. Train a Random Forest Regressor.
9. Evaluate and compare model performance.
10. Analyze feature importance.
11. Visualize actual vs predicted values.
12. Analyze residuals.
13. Save the trained Random Forest model.
14. Load the saved model and make predictions.

## Models Used

### Linear Regression

Linear Regression was used as a baseline model for predicting house prices.

### Random Forest Regressor

A Random Forest Regressor was trained using 100 decision trees.

The model was configured with:

* `n_estimators = 100`
* `random_state = 42`
* `n_jobs = -1`

## Evaluation Metrics

The models were evaluated using:

* Mean Absolute Error (MAE)
* Mean Squared Error (MSE)
* Root Mean Squared Error (RMSE)
* R² Score

## Feature Importance

The Random Forest model was also used to identify which features have the greatest influence on house price predictions.

## Visualizations

The project includes:

* Actual vs Predicted Values plot
* Random Forest Feature Importance plot
* Residual Plot

## Saved Model

The trained Random Forest model is saved using Joblib.

```text
models/random_forest_model.pkl
```

The saved model can later be loaded and used to make predictions without training the model again.

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Joblib

## Project Structure

```text
California-Housing-Price-Prediction/
│
├── notebooks/
├── models/
│   └── random_forest_model.pkl
│
├── train_model.py
└── README.md
```

## Future Improvements

Possible future improvements include:

* Hyperparameter tuning
* Comparing additional machine learning models
* Adding cross-validation
* Creating a web application for house price prediction
* Deploying the trained model
