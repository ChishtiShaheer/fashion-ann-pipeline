import os
import yaml
import numpy as np
from sklearn.model_selection import train_test_split

def load_params():
    with open("params.yaml", "r") as f:
        return yaml.safe_load(f)

def normalize(x):
    return (x.astype("float32") / 255.0 - 0.2860) / 0.3530     # standardization

def preprocess():
    params = load_params()
    val_size = params["preprocess"]["val_size"]
    seed = params["preprocess"]["seed"]

    print("Loading raw arrays...")
    x_train_raw = np.load("data/raw/x_train.npy")
    y_train_raw = np.load("data/raw/y_train.npy")
    x_test_raw = np.load("data/raw/x_test.npy")
    y_test_raw = np.load("data/raw/y_test.npy")

    print("Normalizing pixel values to [0, 1]...")
    x_train_norm = normalize(x_train_raw)
    x_test_norm = normalize(x_test_raw)

    print(f"Splitting validation set (val_size={val_size}, seed={seed})...")
    x_train, x_val, y_train, y_val = train_test_split(
        x_train_norm,
        y_train_raw,
        test_size=val_size,
        random_state=seed,
        stratify=y_train_raw
    )

    os.makedirs("data/processed", exist_ok=True)
    np.save("data/processed/x_train.npy", x_train)
    np.save("data/processed/y_train.npy", y_train)
    np.save("data/processed/x_val.npy", x_val)
    np.save("data/processed/y_val.npy", y_val)
    np.save("data/processed/x_test.npy", x_test_norm)
    np.save("data/processed/y_test.npy", y_test_raw)
    print("Preprocessed data saved to data/processed/")

if __name__ == "__main__":
    preprocess()
# work in progress
