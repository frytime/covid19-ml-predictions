import kagglehub
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt


# Download latest version of the dataset
path = kagglehub.dataset_download("dhruvildave/covid19-deaths-dataset")

print("Path to dataset files:", path)

# Define dataset path
dataset_path = os.path.join(path, "all_weekly_excess_deaths.csv") 

# Load dataset
df = pd.read_csv(dataset_path)

# Convert date columns to datetime format
df['start_date'] = pd.to_datetime(df['start_date'])
df['end_date'] = pd.to_datetime(df['end_date'])

# Drop unnecessary columns
df = df.drop(columns=['country', 'region', 'region_code'])

# Fill missing values if any
df.fillna(0, inplace=True)

# Display basic information
print(df.info())
print(df.head())

# Define Features and Target Variable
features = [
    'days', 'year', 'week', 'population', 'total_deaths', 'expected_deaths', 
    'excess_deaths', 'non_covid_deaths', 'covid_deaths_per_100k'
]

target = 'covid_deaths'

# Extract features (X) and target (y)
X = df[features]
y = df[target]

# Split Data for Training and Testing (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Machine Learning Model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions on test set
y_pred = model.predict(X_test)

# Evaluate Model Performance
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"Mean Absolute Error (MAE): {mae}")
print(f"Root Mean Squared Error (RMSE): {rmse}")

# Hyperparameter Tuning for Better Performance
# Define the parameter grid
param_grid = {
    # Number of trees in the forest
    'n_estimators': [50, 100, 200], 
    # Maximum depth of the tree
    'max_depth': [None, 10, 20],  
    # Minimum samples required to split a node
    'min_samples_split': [2, 5, 10]
}

# Perform Grid Search with Cross-Validation
grid_search = GridSearchCV(
    RandomForestRegressor(random_state=42),  # Base model
    param_grid,  # Hyperparameter grid
    cv=3,  # 3-fold cross-validation
    scoring='neg_mean_absolute_error',  # Optimization metric
    n_jobs=-1  # Use all CPU cores for faster computation
)

# Fit the Grid Search model
grid_search.fit(X_train, y_train)

# Get the best model from Grid Search
best_model = grid_search.best_estimator_
print("Best Parameters:", grid_search.best_params_)

# Example for Future Prediction
# Ensure the new input matches training features
future_data = pd.DataFrame([{
    'days': 300,
    'year': 2021,
    'week': 10,
    'population': 50000000,
    'total_deaths': 100000,
    'expected_deaths': 95000,
    'excess_deaths': 5000,
    'non_covid_deaths': 94000,
    'covid_deaths_per_100k': 5.0
}])

# Ensure future_data has the same column order as X_train
future_data = future_data[X_train.columns]

# Predict using the optimized model
future_pred = best_model.predict(future_data)

print(f"Year: {future_data['year'].values[0]}")
print(f"Predicted COVID Deaths: {future_pred[0]}")


# Sort dataset by date before training
df = df.sort_values(by='start_date')

# Reduce overcrowding in plot by sampling data
sample_size = 200  # Adjust the number of points displayed
y_test_sample = y_test[:sample_size]
y_pred_sample = y_pred[:sample_size]

# Create a line plot to show trends in actual vs predicted deaths
plt.figure(figsize=(10, 6))
plt.plot(y_test_sample.values, label='Actual', marker='o', linestyle='-')
plt.plot(y_pred_sample, label='Predicted', marker='x', linestyle='-')

plt.xlabel('Data Points')
plt.ylabel('COVID-19 Deaths')
plt.title("Actual vs. Predicted COVID-19 Deaths (Reduced Sample)")
plt.legend()
plt.show()