import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Load dataset (Replace with actual dataset path)
df = pd.read_csv("housing.csv")

# Handle missing values
df["total_bedrooms"] = df["total_bedrooms"].fillna(df["total_bedrooms"].median())

# Identify categorical and numerical features
categorical_features = ["ocean_proximity"]
numerical_features = df.drop(columns=["median_house_value"] + categorical_features).columns.tolist()

# Define preprocessing pipeline
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numerical_features),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
])

# Define features and target
X = df.drop(columns=["median_house_value"])
y = df["median_house_value"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a pipeline with preprocessing and RandomForestRegressor
pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("rf", RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1))
])

# Define hyperparameter grid
param_grid = {
    "rf__n_estimators": [100, 200, 300],  # Number of trees
    "rf__max_depth": [10, 20, None],      # Tree depth
    "rf__min_samples_split": [2, 5, 10],  # Minimum samples per split
    "rf__min_samples_leaf": [1, 2, 4],    # Minimum samples per leaf
    "rf__max_features": ["auto", "sqrt"]  # Number of features considered per split
}

# Perform Grid Search
grid_search = GridSearchCV(pipeline, param_grid, cv=3, scoring='r2', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Best model evaluation
y_pred = grid_search.best_estimator_.predict(X_test)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

# Print results
print(f"Best Parameters: {grid_search.best_params_}")
print(f"R² Score: {r2:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
