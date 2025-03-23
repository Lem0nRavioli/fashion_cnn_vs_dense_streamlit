import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.datasets import fashion_mnist
import mlflow.keras
import time

# # Chargement des données depuis CSV
# train_df = pd.read_csv("data/fashion-mnist_train.csv")
# test_df = pd.read_csv("data/fashion-mnist_test.csv")

# X_train = train_df.drop("label", axis=1).values.reshape(-1, 28, 28) / 255.0
# y_train = train_df["label"].values
# X_test = test_df.drop("label", axis=1).values.reshape(-1, 28, 28) / 255.0
# y_test = test_df["label"].values

# load fashion mnist dataset
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

# reshape data lightly
X_train = X_train.reshape(-1, 28, 28, 1) / 255.0
X_test = X_test.reshape(-1, 28, 28, 1) / 255.0

# create a simple dense model
model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

# compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# create a tensorboard callback
log_dir = f"logs/dense_{int(time.time())}"
tensorboard_callback = TensorBoard(log_dir=log_dir)

# train the model with mlflow autologging and save the model
mlflow.set_experiment("fashion_mnist")
with mlflow.start_run(run_name="Dense_Model"):
    mlflow.keras.autolog()
    model.fit(X_train, y_train, epochs=10, validation_split=0.2, callbacks=[tensorboard_callback])
    model.save("model_dense.h5")
