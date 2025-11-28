#!/usr/bin/env python3
import argparse, os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
import tensorflow as tf

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/ddos_windows.npz")
    ap.add_argument("--model", default="models/cnn_ddos.keras")
    args = ap.parse_args()

    d = np.load(args.data)
    X, y = d["X"], d["y"]

    model = tf.keras.models.load_model(args.model)
    p = model.predict(X, verbose=0).ravel()
    yhat = (p >= 0.5).astype("int64")

    cm = confusion_matrix(y, yhat)
    print("\n=== Evaluation ===")
    print(f"Accuracy: {(yhat==y).mean():.4f}, ROC-AUC: {roc_auc_score(y, p):.4f}")
    print("Confusion Matrix:\n", cm)
    print(classification_report(y, yhat, target_names=["normal","ddos"]))

    os.makedirs("figs", exist_ok=True)
    plt.figure()
    im = plt.imshow(cm, cmap="Blues")
    plt.title("Confusion Matrix"); plt.colorbar(im)
    plt.xticks([0,1], ["normal","ddos"]); plt.yticks([0,1], ["normal","ddos"])
    for i in range(2):
        for j in range(2):
            plt.text(j, i, cm[i,j], ha="center", va="center")
    plt.tight_layout(); plt.savefig("figs/confusion_matrix.png", dpi=160); plt.close()
    print("Saved: figs/confusion_matrix.png")

if __name__ == "__main__":
    main()
