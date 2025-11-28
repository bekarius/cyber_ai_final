#!/usr/bin/env python3
import argparse, numpy as np, tensorflow as tf

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/cnn_ddos.keras")
    ap.add_argument("--data", default="data/ddos_windows.npz")
    ap.add_argument("--index", type=int, default=-1, help="sample index; -1=random")
    args = ap.parse_args()

    d = np.load(args.data)
    X, y = d["X"], d["y"]
    if args.index < 0:
        idx = np.random.default_rng().integers(0, len(X))
    else:
        idx = max(0, min(args.index, len(X)-1))

    model = tf.keras.models.load_model(args.model)
    prob = model.predict(X[idx:idx+1], verbose=0).ravel()[0]
    pred = "ddos" if prob >= 0.5 else "normal"
    print(f"Sample #{idx}: true={ 'ddos' if y[idx]==1 else 'normal' }  prob_ddos={prob:.4f}  -> {pred}")

if __name__ == "__main__":
    main()
