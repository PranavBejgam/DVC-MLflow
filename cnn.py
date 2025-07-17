import mlflow
import mlflow.tensorflow
from mlflow.models import infer_signature

mlflow.set_tracking_uri(uri="http://127.0.0.1:5000")  # Custom tracking server
mlflow.set_experiment("MNIST-CNN")

import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import numpy as np

tf.random.set_seed(42)

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 🔴 Start MLflow run
with mlflow.start_run(run_name="mnist-cnn-run"):

    # Log parameters manually (optional)
    mlflow.log_param("optimizer", "adam")
    mlflow.log_param("batch_size", 32)
    mlflow.log_param("epochs", 10)

    # Train model
    history = model.fit(
        x_train, y_train,
        epochs=10,
        batch_size=32,
        validation_split=0.2,
        verbose=1
    )

    # Evaluate
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)

    # Log metrics
    mlflow.log_metric("test_accuracy", float(test_accuracy))
    mlflow.log_metric("test_loss", float(test_loss))

    # Log model
    mlflow.tensorflow.log_model(model, artifact_path="model")

    # (Optional) Log signature for model input/output
    signature = infer_signature(x_test, model.predict(x_test))
    mlflow.tensorflow.log_model(model, "model", signature=signature)

# print(f"Test Accuracy: {test_accuracy:.4f}")
