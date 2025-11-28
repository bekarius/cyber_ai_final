#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

tokens = ["GET", "/login", "?", "user", "=", "admin", "&", "q", "=", "union", "select", "1"]
np.random.seed(42)
seq_len, dk = len(tokens), 8
X = np.random.randn(seq_len, dk).astype(np.float32)
Wq = np.random.randn(dk, dk).astype(np.float32)
Wk = np.random.randn(dk, dk).astype(np.float32)
Wv = np.random.randn(dk, dk).astype(np.float32)

Q, K, V = X @ Wq, X @ Wk, X @ Wv
scores = (Q @ K.T) / np.sqrt(dk)
# softmax row-wise
scores -= scores.max(axis=1, keepdims=True)
A = np.exp(scores); A /= A.sum(axis=1, keepdims=True)

plt.figure(figsize=(6,5))
plt.imshow(A, aspect='auto')
plt.colorbar(label="attention weight")
plt.xticks(np.arange(seq_len), tokens, rotation=45, ha='right')
plt.yticks(np.arange(seq_len), tokens)
plt.title("Scaled Dot-Product Attention (toy example)")
plt.tight_layout()
plt.savefig("figs/attention_demo.png", dpi=160)
print("Saved figs/attention_demo.png")
