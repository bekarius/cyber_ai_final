#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

def sinusoidal_positional_encoding(max_pos=200, d_model=32):
    P = np.zeros((max_pos, d_model), dtype=np.float32)
    positions = np.arange(max_pos)[:, None]
    div_terms = np.exp(np.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
    P[:, 0::2] = np.sin(positions * div_terms)
    P[:, 1::2] = np.cos(positions * div_terms)
    return P

P = sinusoidal_positional_encoding(max_pos=200, d_model=16)

plt.figure(figsize=(7,4))
for d in [0,1,2,3,4,5]:
    plt.plot(P[:, d], label=f"dim {d}")
plt.title("Sinusoidal Positional Encoding (selected dimensions)")
plt.xlabel("position")
plt.ylabel("value")
plt.legend(ncol=3, fontsize=8)
plt.tight_layout()
plt.savefig("figs/positional_encoding.png", dpi=160)
print("Saved figs/positional_encoding.png")
