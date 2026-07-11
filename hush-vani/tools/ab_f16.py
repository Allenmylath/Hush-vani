"""Paired f32-vs-f16 kernel benchmark.

Same discipline as ab_bench.py: the two builds run in INTERLEAVED blocks and the ratio is
taken within each turn, so thermal drift (which on this machine dwarfs the effect) cancels.
"""
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
EXE = os.path.join(ROOT, "target", "release", "hush-devtool.exe")

SECS = 5
BLOCK = 8      # samples per turn
TURNS = 10     # interleaved turns per arm
THREADS = int(os.environ.get("TH", "1"))
NN_ONLY = os.environ.get("NN", "0") == "1"


def run(basename, n):
    env = dict(os.environ, HUSH_W=basename)
    cmd = [EXE, "raw", str(SECS), str(n), str(THREADS)] + (["nn"] if NN_ONLY else [])
    out = subprocess.run(cmd, capture_output=True, text=True, check=True, env=env)
    return [float(x) for x in out.stdout.split()]


def main():
    run("weights", 2)
    run("weights.f16", 2)  # warm

    a32, a16, ratios = [], [], []
    for i in range(TURNS):
        if i % 2 == 0:
            b32 = run("weights", BLOCK)
            b16 = run("weights.f16", BLOCK)
        else:
            b16 = run("weights.f16", BLOCK)
            b32 = run("weights", BLOCK)
        a32 += b32
        a16 += b16
        ratios.append(np.median(b32) / np.median(b16))  # >1 => f16 faster

    a32, a16, ratios = np.array(a32), np.array(a16), np.array(ratios)
    logr = np.log(ratios)
    rng = np.random.default_rng(0)
    boot = [np.exp(np.mean(rng.choice(logr, logr.size))) for _ in range(20000)]
    lo, hi = np.percentile(boot, [2.5, 97.5])
    geo = float(np.exp(logr.mean()))

    scope = "NN only" if NN_ONLY else "full pipeline"
    print(f"\n=== f32 vs f16 kernels — {scope}, {THREADS} thread(s), {SECS}s audio ===")
    print(f"    {TURNS} interleaved turns x {BLOCK} samples per arm\n")
    print(f"  f32 weights: median {np.median(a32):7.2f} ms   (min {a32.min():.2f})")
    print(f"  f16 weights: median {np.median(a16):7.2f} ms   (min {a16.min():.2f})")
    print(f"\n  paired speedup (drift-immune): {geo:.3f}x   95% CI [{lo:.3f}, {hi:.3f}]")
    print(f"  f16 faster in {int((ratios>1).sum())}/{len(ratios)} turns")
    sig = lo > 1.0 or hi < 1.0
    verdict = ("f16 FASTER" if geo > 1 else "f16 SLOWER") if sig else "no significant difference"
    print(f"  verdict: {verdict}")


if __name__ == "__main__":
    main()
