#!/usr/bin/env python3
import numpy as np
import pandas as pd
from pathlib import Path

def main(n_normal=500, n_attack=500, seed=42, out_csv="data/flows.csv"):
    rng = np.random.default_rng(seed)
    # 8x8 feature maps flattened -> 64 features per sample
    X_normal = rng.normal(loc=0.0, scale=1.0, size=(n_normal, 64))
    X_attack = rng.normal(loc=2.0, scale=1.0, size=(n_attack, 64))
    X = np.vstack([X_normal, X_attack])
    y = np.array([0]*n_normal + [1]*n_attack)

    cols = [f"f{i}" for i in range(64)]
    df = pd.DataFrame(X, columns=cols)
    df["label"] = y

    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    # Print a compact preview to embed in PDF
    print("\n=== Data preview (first 8 rows) ===")
    print(df.head(8).to_string(index=False))
    print(f"\nSaved dataset to: {out_path.resolve()} (shape={df.shape})")

if __name__ == "__main__":
    main()
