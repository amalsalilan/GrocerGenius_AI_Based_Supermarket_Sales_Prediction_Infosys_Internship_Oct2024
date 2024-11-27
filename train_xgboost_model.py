import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error, r2_score
from joblib import dump
import numpy as np

# 1. Generate a sample dataset for regression (replace with actual data if available)
X, y = make_regression(n_samples=100, n_features=5, noise=0.1, random_state=42)

# 2. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Initialize and train the XGBoost model
xg_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=1000, learning_rate=0.05, max_depth=5, random_state=42)
xg_model.fit(X_train, y_train)

# 4. Evaluate the model
y_pred = xg_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
print(f"R^2 Score: {r2}")

# 5. Save the trained model to a .joblib file
dump(xg_model, 'xgboost_model.joblib')
print("Model saved successfully.")
