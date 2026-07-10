"""Interleaved A/B benchmark: Rust vs onnxruntime, 50 samples each.

Runs the two in alternating blocks so thermal drift / turbo state / background load hit
both roughly equally. Reports mean, median, stddev and a bootstrap CI on the difference,
plus a Mann-Whitney U test (no normality assumption -- timing tails are not Gaussian).
"""

import os
import subprocess
import sys
import time

import numpy as np
import soundfile as sf

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
ORT_DIR = os.path.join(ROOT, "ort")
EXE = os.path.join(os.path.dirname(HERE), "target", "release", "hush-devtool.exe")
sys.path.insert(0, ORT_DIR)
from hush_ort import HushORT  # noqa: E402

SECS = 5
TOTAL = int(os.environ.get("AB_TOTAL", "50"))
BLOCK = 10  # samples per turn before handing over


def rust_block(n, threads, nn=False):
    cmd = [EXE, "raw", str(SECS), str(n), str(threads)] + (["nn"] if nn else [])
    out = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return [float(x) for x in out.stdout.split()]


def ort_block(model, audio, n):
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        model.enhance(audio)
        ts.append((time.perf_counter() - t0) * 1e3)
    return ts


def ort_nn_block(model, feats, n):
    """time only the three graph Run() calls, features precomputed"""
    fe, fs = feats
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        e0, e1, e2, e3, emb, c0, lsnr = model.enc.run(
            None, {"feat_erb": fe, "feat_spec": fs})
        model.erb_dec.run(None, {"emb": emb, "e3": e3, "e2": e2, "e1": e1, "e0": e0})
        model.df_dec.run(None, {"emb": emb, "c0": c0})
        ts.append((time.perf_counter() - t0) * 1e3)
    return ts


def boot_ci(a, b, n=20000, seed=0):
    """bootstrap CI on median(b) - median(a)"""
    rng = np.random.default_rng(seed)
    d = [np.median(rng.choice(b, b.size)) - np.median(rng.choice(a, a.size)) for _ in range(n)]
    return np.percentile(d, [2.5, 97.5])


def mannwhitney(a, b):
    """two-sided U test via normal approximation with tie correction"""
    x = np.concatenate([a, b])
    r = np.argsort(np.argsort(x)) + 1.0
    # average ranks for ties
    for v in np.unique(x):
        m = x == v
        if m.sum() > 1:
            r[m] = r[m].mean()
    n1, n2 = len(a), len(b)
    u1 = r[:n1].sum() - n1 * (n1 + 1) / 2
    u = min(u1, n1 * n2 - u1)
    mu = n1 * n2 / 2
    _, cnt = np.unique(x, return_counts=True)
    tie = (cnt**3 - cnt).sum()
    n = n1 + n2
    sd = np.sqrt(n1 * n2 / 12 * ((n + 1) - tie / (n * (n - 1))))
    z = (u - mu) / sd
    from math import erfc, sqrt

    return 2 * 0.5 * erfc(abs(z) / sqrt(2))


def describe(name, t):
    t = np.asarray(t)
    print(f"  {name:22s} n={t.size:3d}  mean={t.mean():6.2f}  median={np.median(t):6.2f}  "
          f"sd={t.std(ddof=1):5.2f}  min={t.min():6.2f}  p95={np.percentile(t,95):6.2f}")
    return t


def one_replicate(threads, nn, model, noisy, feats):
    """Returns all samples plus per-turn (rust_median, ort_median) pairs.

    The two sides are measured adjacent in time within each turn, so a turn-level ratio is
    immune to thermal drift across the run. Pooling raw samples is not: this machine warms
    by ~20% over three replicates, and that between-replicate variance swamps a 5% effect.
    """
    a_blk = (lambda n: rust_block(n, threads, nn))
    b_blk = (lambda n: ort_nn_block(model, feats, n)) if nn else (lambda n: ort_block(model, noisy, n))
    rust, ort, turns = [], [], []
    for i in range(TOTAL // BLOCK):
        if i % 2 == 0:  # alternate who goes first, too
            rb = a_blk(BLOCK)
            ob = b_blk(BLOCK)
        else:
            ob = b_blk(BLOCK)
            rb = a_blk(BLOCK)
        rust += rb
        ort += ob
        turns.append((float(np.median(rb)), float(np.median(ob))))
    return np.array(rust), np.array(ort), turns


def run(threads, nn=False, reps=3):
    noisy, sr = sf.read(os.path.join(ORT_DIR, "sample_raw.wav"), dtype="float32")
    dur = len(noisy) / sr
    model = HushORT(intra_threads=threads, inter_threads=1)
    _, fe, fs = model.features(noisy)
    for _ in range(3):
        model.enhance(noisy)
    rust_block(3, threads, nn)  # warm the exe/page cache

    scope = "NN only (no DSP)" if nn else "full pipeline"
    print(f"\n=== {scope}, {threads} thread(s), {SECS}s audio ===")
    print(f"    {reps} replicates x {TOTAL} samples/side, interleaved in blocks of {BLOCK}")

    all_r, all_o, all_turns, per_rep = [], [], [], []
    for k in range(reps):
        r, o, turns = one_replicate(threads, nn, model, noisy, (fe, fs))
        s = np.median(o) / np.median(r)
        per_rep.append(s)
        all_r.append(r)
        all_o.append(o)
        all_turns += turns
        print(f"    replicate {k+1}: rust {np.median(r):7.2f}  ort {np.median(o):7.2f}  -> {s:.3f}x")

    r = np.concatenate(all_r)
    o = np.concatenate(all_o)
    print()
    describe("onnxruntime", o)
    describe("rust", r)

    # --- paired, drift-immune: one ratio per interleaved turn ---
    ratios = np.array([ob / rb for rb, ob in all_turns])
    logr = np.log(ratios)
    rng = np.random.default_rng(0)
    boot = [np.exp(np.mean(rng.choice(logr, logr.size))) for _ in range(20000)]
    rlo, rhi = np.percentile(boot, [2.5, 97.5])
    geo = float(np.exp(logr.mean()))
    wins = int((ratios > 1).sum())
    print(f"\n  paired by turn (n={len(ratios)} turns, drift-immune):")
    print(f"    speedup = {geo:.3f}x   95% CI [{rlo:.3f}, {rhi:.3f}]   "
          f"rust faster in {wins}/{len(ratios)} turns")
    paired_sig = rlo > 1.0 or rhi < 1.0
    print(f"    verdict: {'significant' if paired_sig else 'NOT significant (CI includes 1.0)'}")

    # --- unpooled reference (inflated by drift across replicates) ---
    lo, hi = boot_ci(r, o)
    print(f"  unpaired pooled (for reference, drift-contaminated):")
    print(f"    median diff {np.median(o)-np.median(r):+.2f} ms, CI [{lo:+.2f}, {hi:+.2f}], "
          f"p={mannwhitney(r, o):.1e}")
    print(f"  rust RTF={np.median(r)/1e3/dur:.5f}   ort RTF={np.median(o)/1e3/dur:.5f}")
    return geo


if __name__ == "__main__":
    run(1, nn=True)   # apples-to-apples: the neural network alone
    run(1, nn=False)  # what a user actually pays
    run(2, nn=False)
