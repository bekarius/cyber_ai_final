#!/usr/bin/env python3
import argparse, os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models

def build_model(T=60, F=8):
    inp = layers.Input(shape=(T, F))
    x = layers.Conv1D(32, 5, padding="same")(inp)
    x = layers.BatchNormalization()(x); x = layers.ReLU()(x)
    x = layers.MaxPool1D(2)(x)
    x = layers.Conv1D(64, 3, padding="same")(x)
    x = layers.BatchNormalization()(x); x = layers.ReLU()(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.25)(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    model = models.Model(inp, out)
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy", tf.keras.metrics.AUC(name="auc")])
    return model

def plot_history(hist, outpath):
    plt.figure()
    plt.plot(hist.history["accuracy"], label="train_acc")
    plt.plot(hist.history["val_accuracy"], label="val_acc")
    plt.plot(hist.history["loss"], label="train_loss")
    plt.plot(hist.history["val_loss"], label="val_loss")
    plt.legend(); plt.title("Training history")
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    plt.savefig(outpath, dpi=160); plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/ddos_windows.npz")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--model", default="models/cnn_ddos.keras")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    d = np.load(args.data)
    X, y = d["X"], d["y"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=args.seed)

    model = build_model(T=X.shape[1], F=X.shape[2])
    cb = [tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)]
    hist = model.fit(Xtr, ytr, validation_data=(Xte, yte), epochs=args.epochs, batch_size=args.batch, callbacks=cb, verbose=1)

    os.makedirs(os.path.dirname(args.model), exist_ok=True)
    model.save(args.model)
    print(f"Saved model to {args.model}")

    plot_history(hist, "figs/history_acc_loss.png")

if __name__ == "__main__":
    main()
