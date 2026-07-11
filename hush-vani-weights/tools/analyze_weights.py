"""Distribution analysis of the Hush weights, to judge compressibility.

Question: are the weights concentrated (near a central value, low spread)? If so, the f32
arena wastes bits and quantization / entropy coding will shrink it. Uses Polars for the
tabular stats and numpy for the quantization error math.
"""

import gzip
import os
import sys

import numpy as np
import polars as pl

sys.stdout.reconfigure(encoding="utf-8")  # Polars/box chars vs Windows cp1252

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BIN = os.path.join(HERE, "data", "weights.bin")
MAN = os.path.join(HERE, "data", "weights.txt")


def load():
    raw = np.fromfile(BIN, dtype="<f4")
    rows = []
    for line in open(MAN):
        line = line.strip()
        if not line:
            continue
        head, off, n = line.rsplit(" ", 2)
        name, shape = head.rsplit(" ", 1)
        off, n = int(off), int(n)
        rows.append((name, shape, off, n))
    return raw, rows


def main():
    raw, rows = load()
    total = raw.size

    # tensor id per element, for per-tensor grouping in Polars
    tid = np.empty(total, dtype=np.int32)
    kind = []  # coarse category by name
    for i, (name, _shape, off, n) in enumerate(rows):
        tid[off:off + n] = i
        if "GRU" in name:
            k = "gru"
        elif "linear" in name or "MatMul" in name or "df_out" in name or "fc" in name:
            k = "linear"
        elif "Conv" in name or "conv" in name or "convt" in name:
            k = "conv"
        else:
            k = "other"
        kind.append(k)

    df = pl.DataFrame({"w": raw, "tid": tid})
    tnames = pl.DataFrame({
        "tid": np.arange(len(rows), dtype=np.int32),
        "name": [r[0] for r in rows],
        "kind": kind,
        "numel": [r[3] for r in rows],
    })

    print("=" * 78)
    print(f"Hush weights: {total:,} f32 across {len(rows)} tensors  ({total*4/1e6:.2f} MB raw)")
    print("=" * 78)

    # ---------------- global distribution ----------------
    g = df.select(
        mean=pl.col("w").mean(),
        median=pl.col("w").median(),
        std=pl.col("w").std(),
        mn=pl.col("w").min(),
        mx=pl.col("w").max(),
        p01=pl.col("w").quantile(0.01),
        p50=pl.col("w").quantile(0.50),
        p99=pl.col("w").quantile(0.99),
        absmean=pl.col("w").abs().mean(),
        absmed=pl.col("w").abs().median(),
    ).row(0, named=True)
    w = raw.astype(np.float64)
    sd = w.std()
    kurt = ((w - w.mean()) ** 4).mean() / (sd ** 4)  # 3 = Gaussian, >3 = peaky/heavy-tailed
    print("\n--- global distribution ---")
    print(f"  mean   {g['mean']:+.5f}      median {g['median']:+.5f}   (both ~0 => centered)")
    print(f"  std    {g['std']:.5f}       abs-mean {g['absmean']:.5f}   abs-median {g['absmed']:.5f}")
    print(f"  range  [{g['mn']:+.4f}, {g['mx']:+.4f}]   p1..p99 [{g['p01']:+.4f}, {g['p99']:+.4f}]")
    print(f"  excess kurtosis {kurt-3:+.1f}   (0=Gaussian; large + => sharply peaked at centre)")
    for k in (0.01, 0.02, 0.05, 0.1):
        frac = float((np.abs(w) < k).mean())
        print(f"  |w| < {k:<5}: {frac*100:5.1f}% of all weights")

    # ---------------- per-tensor centering ----------------
    per = (
        df.group_by("tid")
        .agg(
            t_mean=pl.col("w").mean(),
            t_median=pl.col("w").median(),
            t_absmax=pl.col("w").abs().max(),
            t_std=pl.col("w").std(),
        )
        .join(tnames, on="tid")
    )
    print("\n--- per-tensor centering (are all tensors ~zero-mean?) ---")
    c = per.select(
        max_abs_mean=pl.col("t_mean").abs().max(),
        frac_meanlt_1pct_std=(pl.col("t_mean").abs() < 0.1 * pl.col("t_std")).mean(),
    ).row(0, named=True)
    print(f"  largest |per-tensor mean|: {c['max_abs_mean']:.4f}")
    print(f"  tensors whose |mean| < 0.1*std: {c['frac_meanlt_1pct_std']*100:.0f}%  "
          f"(high => centred at 0, not some offset)")
    print("\n  by layer kind:")
    by = per.group_by("kind").agg(
        tensors=pl.len(),
        weights=pl.col("numel").sum(),
        mean_of_means=pl.col("t_mean").mean(),
        med_absmax=pl.col("t_absmax").median(),
        med_std=pl.col("t_std").median(),
    ).sort("weights", descending=True)
    print(f"    {'kind':7} {'tensors':>7} {'weights':>10} {'mean(mean)':>11} {'med(absmax)':>11} {'med(std)':>9}")
    for r in by.iter_rows(named=True):
        print(f"    {r['kind']:7} {r['tensors']:7d} {r['weights']:10,d} "
              f"{r['mean_of_means']:+11.4f} {r['med_absmax']:11.4f} {r['med_std']:9.4f}")

    # ---------------- compression scenarios ----------------
    print("\n--- what compression buys (error vs size) ---")
    raw_gz = len(gzip.compress(raw.tobytes(), 9))
    print(f"  f32 raw           {total*4/1e6:6.2f} MB   gzip -> {raw_gz/1e6:5.2f} MB  (1.00x baseline)")

    def report(name, q_bytes, deq, extra_bytes=0):
        err = raw - deq
        sig = np.mean(raw ** 2)
        snr = 10 * np.log10(sig / (np.mean(err ** 2) + 1e-30))
        gz = len(gzip.compress(q_bytes, 9)) + extra_bytes
        print(f"  {name:17} {len(q_bytes)/1e6:6.2f} MB   gzip -> {gz/1e6:5.2f} MB  "
              f"({raw_gz/gz:4.2f}x vs f32-gz)   SNR {snr:5.1f} dB   max|err| {np.abs(err).max():.2e}")

    # f16
    h = raw.astype(np.float16)
    report("f16", h.tobytes(), h.astype(np.float32))

    # int8 symmetric, per-tensor scale
    q = np.empty(total, dtype=np.int8)
    deq = np.empty(total, dtype=np.float32)
    scales = np.empty(len(rows), dtype=np.float32)
    for i, (_n, _s, off, n) in enumerate(rows):
        seg = raw[off:off + n]
        amax = np.abs(seg).max()
        s = amax / 127.0 if amax > 0 else 1.0
        scales[i] = s
        qi = np.clip(np.round(seg / s), -127, 127).astype(np.int8)
        q[off:off + n] = qi
        deq[off:off + n] = qi.astype(np.float32) * s
    report("int8 per-tensor", q.tobytes(), deq, extra_bytes=scales.nbytes)

    # int8 global scale (simpler, one number) for contrast
    amax = np.abs(raw).max()
    s = amax / 127.0
    qg = np.clip(np.round(raw / s), -127, 127).astype(np.int8)
    report("int8 global", qg.tobytes(), qg.astype(np.float32) * s, extra_bytes=4)

    print("\n  (SNR ~= 6*bits+2; end-to-end audio tolerates far more error than tensor SNR")
    print("   suggests, since the model output is already 129.7 dB above onnxruntime.)")


if __name__ == "__main__":
    main()
