"""
Original file is located at
    https://colab.research.google.com/github/freeCodeCamp/boilerplate-linear-regression-health-costs-calculator/blob/master/fcc_predict_health_costs_with_regression.ipynb
"""

import os
import urllib.request
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ---------------------------------------------------
# Reproducibility
# ---------------------------------------------------
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ---------------------------------------------------
# Data download (safe for script execution)
# ---------------------------------------------------
DATA_URL = "https://cdn.freecodecamp.org/project-data/health-costs/insurance.csv"
DATA_FILE = "insurance.csv"

def download_dataset(url, output_path):
    # Attempt 1: urllib with headers
    try:
        print("Attempting download via urllib...")
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0"}
        )
        with urllib.request.urlopen(req) as response, open(output_path, "wb") as out:
            out.write(response.read())
        return True
    except Exception as e:
        print(f"urllib failed: {e}")

    # Attempt 2: system wget
    try:
        print("Attempting download via wget...")
        exit_code = os.system(f"wget -q {url} -O {output_path}")
        if exit_code == 0:
            return True
    except Exception as e:
        print(f"wget failed: {e}")

    return False


if not os.path.exists(DATA_FILE):
    print("Dataset not found. Downloading...")
    success = download_dataset(DATA_URL, DATA_FILE)
    if not success:
        raise RuntimeError(
            "Failed to download dataset. "
            "Check your internet connection or download insurance.csv manually."
        )
else:
    print("Dataset already exists. Skipping download.")

# ---------------------------------------------------
# Load data
# ---------------------------------------------------
dataset = pd.read_csv(DATA_FILE)

# ---------------------------------------------------
# Encode categorical features
# ---------------------------------------------------
dataset = pd.get_dummies(
    dataset,
    columns=["sex", "smoker", "region"],
    drop_first=True
)

# ---------------------------------------------------
# Train / test split (80 / 20)
# ---------------------------------------------------
train_dataset = dataset.sample(frac=0.8, random_state=SEED)
test_dataset = dataset.drop(train_dataset.index)

# ---------------------------------------------------
# Separate labels
# ---------------------------------------------------
train_labels = train_dataset.pop("expenses")
test_labels = test_dataset.pop("expenses")

# ---------------------------------------------------
# Feature normalization
# ---------------------------------------------------
normalizer = layers.Normalization(axis=-1)
normalizer.adapt(np.array(train_dataset))
print(f'\nTrain Shape: {train_dataset.shape}\n')

# ---------------------------------------------------
# Build regression model
# ---------------------------------------------------
model = keras.Sequential([
    layers.Input(shape=(train_dataset.shape[-1],)),
    normalizer,
    layers.Dense(32, activation="gelu"),
    layers.Dropout(0.2),
    layers.Dense(64, activation="gelu"),
    layers.Dropout(0.3),
    layers.Dense(128, activation="gelu"),
    layers.Dropout(0.4),
    layers.Dense(256, activation="gelu"),
    layers.Dropout(0.5),
    layers.Dense(1)
])

model.compile(
    optimizer=keras.optimizers.AdamW(learning_rate=1e-3),
    loss="mse",
    metrics=["mae", "mse"]
)

model.summary()

# ---------------------------------------------------
# Train model
# ---------------------------------------------------
callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True, verbose=1),
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.9, patience=10, min_lr=1e-6),
]

history = model.fit(
    train_dataset,
    train_labels,
    validation_split=0.2,
    epochs=1000,
    callbacks=callbacks,
    verbose=1
)

# ---------------------------------------------------
# Evaluate model (FCC-style check)
# ---------------------------------------------------
loss, mae, mse = model.evaluate(test_dataset, test_labels, verbose=2)

print(f"\nTesting set Mean Abs Error: {mae:5.2f} expenses")

if mae < 3500:
    print("You passed the challenge. Great job! 🎉")
else:
    print("The Mean Abs Error must be less than 3500. Keep trying.")

# ---------------------------------------------------
# Plot predictions
# ---------------------------------------------------
test_predictions = model.predict(test_dataset).flatten()

plt.figure(figsize=(6, 6))
plt.scatter(test_labels, test_predictions, alpha=0.6)
plt.xlabel("True values (expenses)")
plt.ylabel("Predictions (expenses)")
lims = [0, 50000]
plt.xlim(lims)
plt.ylim(lims)
plt.plot(lims, lims)
plt.tight_layout()
plt.show()
