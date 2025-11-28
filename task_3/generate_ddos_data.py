#!/usr/bin/env python3
import argparse, os
import numpy as np

def synth_window_normal(T=60, rng=None):
    r = rng or np.random.default_rng()
    base = r.normal(0.0, 1.0, (T, 8))
    # shape it to plausible scales
    feat = np.zeros_like(base)
    feat[:,0] = np.abs(50 + 10*base[:,0])          # pkts_in
    feat[:,1] = np.abs(45 + 10*base[:,1])          # pkts_out
    feat[:,2] = np.abs(5e4 + 1.5e4*base[:,2])      # bytes_in
    feat[:,3] = np.abs(4e4 + 1.5e4*base[:,3])      # bytes_out
    feat[:,4] = np.clip(5 + 2*base[:,4], 1, 20)    # uniq_src_ips
    feat[:,5] = np.clip(8 + 3*base[:,5], 1, 40)    # uniq_dst_ports
    feat[:,6] = np.clip(5 + 2*base[:,6], 0, 25)    # syn_rate
    feat[:,7] = np.clip(2 + 1*base[:,7], 0, 10)    # rst_rate
    return feat

def synth_window_ddos(T=60, rng=None, strength=3.0):
    r = rng or np.random.default_rng()
    w = synth_window_normal(T, r)
    # elevate inbound metrics + uniq_src_ips + syn_rate
    w[:,0] *= (1.5 + 0.3*r.normal(0,1,T))          # pkts_in
    w[:,2] *= (1.8 + 0.3*r.normal(0,1,T))          # bytes_in
    w[:,4] += np.abs(60 + 10*r.normal(0,1,T))      # uniq_src_ips
    w[:,6] += np.abs(80 + 15*r.normal(0,1,T))      # syn_rate
    # optional short bursts to mimic attack dynamics
    for _ in range(3):
        s = r.integers(0, T-5)
        w[s:s+5,0] *= 2.0
        w[s:s+5,2] *= 2.0
        w[s:s+5,6] *= 2.5
    # outbound often doesn't rise similarly
    return w

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/ddos_windows.npz")
    ap.add_argument("--n-normal", type=int, default=2000)
    ap.add_argument("--n-ddos", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--time-steps", type=int, default=60)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    Xn = np.stack([synth_window_normal(args.time_steps, rng) for _ in range(args.n_normal)], axis=0)
    Xd = np.stack([synth_window_ddos(args.time_steps, rng) for _ in range(args.n_ddos)], axis=0)
    X = np.concatenate([Xn, Xd], axis=0).astype("float32")
    y = np.concatenate([np.zeros(len(Xn)), np.ones(len(Xd))]).astype("int64")

    # standardize per-feature over time (optional: do in pipeline)
    mu = X.mean(axis=(0,1), keepdims=True)
    sd = X.std(axis=(0,1), keepdims=True) + 1e-6
    Xz = (X - mu)/sd

    np.savez_compressed(args.out, X=Xz, y=y)
    print(f"Saved dataset: {args.out} (X={Xz.shape}, y={y.shape})")

if __name__ == "__main__":
    main()
