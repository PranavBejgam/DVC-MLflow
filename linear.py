import mlflow
import mlflow.sklearn
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np

# Sample data
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([2, 4, 6, 8, 10])

# Start an MLflow run
mlflow.set_tracking_uri(uri="http://127.0.0.1:5000")  # Custom tracking server
mlflow.set_experiment("Linear Regression Experiment")

with mlflow.start_run():
    # Train a simple model
    model = LinearRegression()
    model.fit(X, y)
    
    # Predict and calculate metric
    y_pred = model.predict(X)
    mse = mean_squared_error(y, y_pred)
    
    # Log parameters and metrics
    mlflow.log_param("model_type", "LinearRegression")
    mlflow.log_metric("mse", mse)
    
    # Log the model
    mlflow.sklearn.log_model(model, "model")
    
    # Log an artifact (e.g., a text file)
    with open("output.txt", "w") as f:
        f.write("Model training complete!")
    mlflow.log_artifact("output.txt")

print("Run logged successfully!")