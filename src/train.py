import os
import csv
import yaml
import numpy as np
import tensorflow as tf
from tensorflow import keras

def load_params():
    with open("params.yaml", "r") as f:
        return yaml.safe_load(f)

def build_model(dense_units, dropout_rate, learning_rate):
    model = keras.Sequential([
        keras.layers.Input(shape=(28, 28)),
        keras.layers.Flatten(),
        keras.layers.Dense(dense_units, activation="relu"),
        keras.layers.Dropout(dropout_rate),
        keras.layers.Dense(10, activation="softmax")
    ])
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

def train():
    params = load_params()["train"]
    dense_units = params["dense_units"]
    dropout_rate = params["dropout_rate"]
    learning_rate = params["learning_rate"]
    epochs = params["epochs"]
    batch_size = params["batch_size"]

    print("Loading processed data for training...")
    x_train = np.load("data/processed/x_train.npy")
    y_train = np.load("data/processed/y_train.npy")
    x_val = np.load("data/processed/x_val.npy")
    y_val = np.load("data/processed/y_val.npy")

    print(f"Building ANN (dense_units={dense_units}, dropout={dropout_rate}, lr={learning_rate})...")
    model = build_model(dense_units, dropout_rate, learning_rate)
    model.summary()

    print(f"Training model for {epochs} epochs (batch_size={batch_size})...")
    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1
    )

    os.makedirs("models", exist_ok=True)
    model_path = "models/model.h5"
    model.save(model_path)
    print(f"Model saved to {model_path}")

    history_path = "models/history.csv"
    with open(history_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "loss", "accuracy", "val_loss", "val_accuracy"])
        for epoch in range(len(history.history["loss"])):
            writer.writerow([
                epoch + 1,
                history.history["loss"][epoch],
                history.history["accuracy"][epoch],
                history.history["val_loss"][epoch],
                history.history["val_accuracy"][epoch]
            ])
    print(f"Training history saved to {history_path}")

if __name__ == "__main__":
    train()
