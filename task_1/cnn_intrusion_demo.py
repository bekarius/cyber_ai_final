#!/usr/bin/env python3
import numpy as np
import pandas as pd
from pathlib import Path
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

def load_csv_as_images(csv_path, side=8):
    df = pd.read_csv(csv_path)
    y = df["label"].values.astype(int)
    X = df.drop(columns=["label"]).values.astype(np.float32)
    X = X.reshape(-1, side, side, 1)  # (N, 8, 8, 1)
    return X, y

def build_cnn(input_shape=(8,8,1)):
    return models.Sequential([
        layers.Conv2D(16, (3,3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(32, (3,3), activation='relu'),
        layers.Flatten(),
        layers.Dense(32, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])

def main(csv_path="data/flows.csv", epochs=8, batch_size=32):
    X, y = load_csv_as_images(csv_path)
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=42
    )

    model = build_cnn(input_shape=Xtr.shape[1:])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    history = model.fit(Xtr, ytr, epochs=epochs, batch_size=batch_size,
                        validation_split=0.2, verbose=0)

    y_prob = model.predict(Xte)
    y_pred = (y_prob > 0.5).astype(int).ravel()

    print("\n=== Evaluation ===")
    print("Accuracy:", np.mean(y_pred == yte))
    print("Confusion Matrix:\n", confusion_matrix(yte, y_pred))
    print(classification_report(yte, y_pred, target_names=["normal","attack"]))

if __name__ == "__main__":
    main()
