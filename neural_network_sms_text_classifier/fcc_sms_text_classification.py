import os
import urllib.request
import numpy as np
import pandas as pd
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
# Robust dataset download
# ---------------------------------------------------
TRAIN_URL = "https://cdn.freecodecamp.org/project-data/sms/train-data.tsv"
TEST_URL = "https://cdn.freecodecamp.org/project-data/sms/valid-data.tsv"

TRAIN_FILE = "train-data.tsv"
TEST_FILE = "valid-data.tsv"

def download_file(url, filename):
    if os.path.exists(filename):
        print(f"{filename} already exists. Skipping download.")
        return

    print(f"Downloading {filename}...")
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0"}
    )
    with urllib.request.urlopen(req) as response, open(filename, "wb") as out:
        out.write(response.read())

download_file(TRAIN_URL, TRAIN_FILE)
download_file(TEST_URL, TEST_FILE)

# ---------------------------------------------------
# Load data
# ---------------------------------------------------
train_df = pd.read_csv(TRAIN_FILE, sep="\t", header=None, names=["label", "text"])
test_df = pd.read_csv(TEST_FILE, sep="\t", header=None, names=["label", "text"])

# Convert labels to binary
train_df["label"] = train_df["label"].map({"ham": 0, "spam": 1})
test_df["label"] = test_df["label"].map({"ham": 0, "spam": 1})

train_texts = train_df["text"].values
train_labels = train_df["label"].values
test_texts = test_df["text"].values
test_labels = test_df["label"].values

# ---------------------------------------------------
# Text vectorization
# ---------------------------------------------------
max_tokens = 10000
sequence_length = 128

vectorizer = layers.TextVectorization(
    max_tokens=max_tokens,
    output_mode="int",
    output_sequence_length=sequence_length
)

vectorizer.adapt(train_texts)

# ---------------------------------------------------
# Build model
# ---------------------------------------------------
print(f'\ntrain_texts.shape: {train_texts.shape}\n')
model = keras.Sequential([
    keras.Input(shape=(1,), dtype=tf.string, name="text_input"),
    vectorizer,
    layers.Embedding(max_tokens, 64, mask_zero=True),
    layers.Bidirectional(layers.LSTM(64)),
    layers.Dropout(0.5),
    layers.Dense(64, activation="gelu"),
    layers.Dense(1, activation="sigmoid")
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ---------------------------------------------------
# Train model
# ---------------------------------------------------
model.fit(
    train_texts,
    train_labels,
    epochs=5,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

# ---------------------------------------------------
# Prediction function (FCC required)
# ---------------------------------------------------
def predict_message(pred_text):
    pred_input = tf.constant([pred_text])
    pred = model(pred_input, training=False).numpy()[0][0]
    label = "spam" if pred >= 0.5 else "ham"
    return [float(pred), label]

# ---------------------------------------------------
# Manual test
# ---------------------------------------------------
if __name__ == "__main__":
    sample = "how are you doing today?"
    print(predict_message(sample))

    # FCC test harness
    def test_predictions():
        test_messages = [
            "how are you doing today",
            "sale today! to stop texts call 98912460324",
            "i dont want to go. can we try it a different day? available sat",
            "our new mobile video service is live. just install on your phone to start watching.",
            "you have won £1000 cash! call to claim your prize.",
            "i'll bring it tomorrow. don't forget the milk.",
            "wow, is your arm alright. that happened to me one time too"
        ]

        test_answers = ["ham", "spam", "ham", "spam", "spam", "ham", "ham"]
        passed = True

        for msg, ans in zip(test_messages, test_answers):
            prediction = predict_message(msg)
            if prediction[1] != ans:
                passed = False

        if passed:
            print("You passed the challenge. Great job! 🎉")
        else:
            print("You haven't passed yet. Keep trying.")

    test_predictions()
