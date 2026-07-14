"""Does per-channel int8 rescue the accuracy that per-tensor int8 threw away?

The shipped int8 experiment used ONE scale per tensor. The GRU weights carry +/-31
outliers while 99% of their values sit inside +/-0.23, so a single scale spends almost
the whole int8 range on a handful of entries and quantises the rest to a few levels.
That cost 8.4 dB of noise reduction on 01_munching -- the model got worse at its job.

Per-channel gives each output ROW of a matmul its own scale, so an outlier only wrecks
its own row. It is free at inference: the kernel already multiplies by a scale at the
end, and a per-row scale is a vector load instead of a broadcast.

The subtlety is which axis. The GRU tensors are [1, 3*hidden, input] -- leading dim is
num_directions, so grouping by shape[0] gives ONE group and silently reproduces
per-tensor. The contraction axis is the LAST one, so the row stride is shape[-1].

Everything here is simulated by quantise -> dequantise back to f32, then run through the
real f32 kernels, exactly as bench_samples.py does. That measures the ACCURACY of a
scheme without needing the int8 kernel to exist yet. Speed is a separate question,
answered only after a scheme earns the kernel work.
"""
import json
import os
import shutil
import subprocess
import sys
import time

import numpy as np
import soundfile as sf

sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__))
CRATE = os.path.dirname(HERE)
ROOT = os.path.dirname(CRATE)
ASSETS = os.path.join(CRATE, "assets")
BS = os.path.join(ROOT, "bench_samples")
OUT = os.path.join(ROOT, "quant_schemes")
CLI = os.path.join(ROOT, "target", "release", "hush-vani.exe")
SR = 16000
DELAY = 160

BIN = os.path.join(ASSETS, "weights.bin")
MAN = os.path.join(ASSETS, "weights.txt")


def tensors():
    """(name, shape, offset, numel) -- keeping the shape this time; it is the whole point."""
    for line in open(MAN):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        head, off, n = line.rsplit(" ", 2)
        name, shape_s = head.rsplit(" ", 1)
        shape = [int(d) for d in shape_s.split(",")] if shape_s else []
        yield name, shape, int(off), int(n)


def q_dq(seg, group):
    """Symmetric int8 quantise->dequantise with one scale per `group` contiguous elements."""
    if group <= 0 or seg.size % group != 0:
        group = seg.size  # fall back to per-tensor
    rows = seg.reshape(-1, group)
    amax = np.abs(rows).max(axis=1, keepdims=True)
    s = np.where(amax > 0, amax / 127.0, 1.0)
    return (np.clip(np.round(rows / s), -127, 127) * s).reshape(-1), s.size


# scheme -> (group size for a tensor, whether to quantise 1D tensors at all)
def group_for(scheme, shape, numel):
    if scheme == "per_tensor":
        return numel
    if scheme in ("per_channel", "per_channel_f32bias"):
        # contraction axis is the last one; one scale per output row
        return shape[-1] if len(shape) >= 2 else numel
    if scheme == "per_channel_g64":
        # finer still: split each row into 64-wide blocks (block-wise quantisation)
        row = shape[-1] if len(shape) >= 2 else numel
        return 64 if row % 64 == 0 else row
    raise ValueError(scheme)


SCHEMES = ["per_tensor", "per_channel", "per_channel_f32bias", "per_channel_g64"]


def build(scheme, raw):
    q = raw.copy()
    nscales = 0
    for _name, shape, off, n in tensors():
        seg = raw[off:off + n]
        if scheme == "per_channel_f32bias" and len(shape) < 2:
            continue  # leave biases / 1-D tensors in f32: 16 tensors, a rounding error of the total
        g = group_for(scheme, shape, n)
        q[off:off + n], k = q_dq(seg, g)
        nscales += k
    return q, nscales


def weight_snr(raw, q):
    err = q - raw
    return 10 * np.log10((raw @ raw + 1e-12) / (err @ err + 1e-12))


def db(x):
    return 20 * np.log10(max(float(x), 1e-12))


def si_sdr(est, ref):
    est, ref = est - est.mean(), ref - ref.mean()
    a = np.dot(est, ref) / (np.dot(ref, ref) + 1e-12)
    p = a * ref
    n = est - p
    return 10 * np.log10((np.dot(p, p) + 1e-12) / (np.dot(n, n) + 1e-12))


def floor_and_speech(x):
    fl = 320
    nf = len(x) // fl
    e = 10 * np.log10(np.mean(x[: nf * fl].reshape(nf, fl) ** 2, 1) + 1e-12)
    return float(np.percentile(e, 10)), float(np.percentile(e, 90))


def main():
    raw = np.fromfile(BIN, dtype="<f4")
    os.makedirs(OUT, exist_ok=True)

    # ---------- stage 1: weight-level, instant ----------
    print("=" * 78)
    print("WEIGHT RECONSTRUCTION (int8 quantise -> dequantise vs true f32)")
    print("=" * 78)
    print(f"  {'scheme':<22} {'scales':>9} {'scale bytes':>12} {'weight SNR':>12}")
    blobs = {}
    for s in SCHEMES:
        q, nsc = build(s, raw)
        blobs[s] = q
        print(f"  {s:<22} {nsc:>9,} {nsc * 4:>11,}B {weight_snr(raw, q):>9.1f} dB")

    # where do the outliers actually live?
    print("\n  worst tensors under per-tensor (SNR, and the outlier ratio that causes it):")
    rows = []
    for name, shape, off, n in tensors():
        seg = raw[off:off + n]
        if seg.size == 0 or np.abs(seg).max() == 0:
            continue
        pt, _ = q_dq(seg, n)
        pc, _ = q_dq(seg, group_for("per_channel", shape, n))
        e1, e2 = pt - seg, pc - seg
        s1 = 10 * np.log10((seg @ seg + 1e-12) / (e1 @ e1 + 1e-12))
        s2 = 10 * np.log10((seg @ seg + 1e-12) / (e2 @ e2 + 1e-12))
        ratio = np.abs(seg).max() / (np.percentile(np.abs(seg), 99) + 1e-12)
        rows.append((s1, s2, ratio, name, shape))
    rows.sort()
    print(f"    {'tensor':<28} {'shape':<14} {'max/p99':>8} {'per-tensor':>11} {'per-chan':>10}")
    for s1, s2, ratio, name, shape in rows[:6]:
        print(f"    {name:<28} {','.join(map(str,shape)):<14} {ratio:>8.0f}x "
              f"{s1:>8.1f} dB {s2:>7.1f} dB")

    # ---------- stage 2: end-to-end audio ----------
    ins = sorted(os.listdir(os.path.join(BS, "in16k")))
    if not ins:
        print("\nno inputs in bench_samples/in16k -- run bench_samples.py first")
        return

    wdirs = {}
    for s in SCHEMES:
        d = os.path.join(OUT, "_w", s)
        os.makedirs(d, exist_ok=True)
        blobs[s].astype("<f4").tofile(os.path.join(d, "weights.bin"))
        shutil.copy(MAN, os.path.join(d, "weights.txt"))
        wdirs[s] = d
    d32 = os.path.join(OUT, "_w", "f32")
    os.makedirs(d32, exist_ok=True)
    shutil.copy(BIN, os.path.join(d32, "weights.bin"))
    shutil.copy(MAN, os.path.join(d32, "weights.txt"))

    print("\n" + "=" * 78)
    print("END-TO-END on the 6 DeepFilterNet2 demo samples (vs f32 reference)")
    print("=" * 78)

    results = []
    for fn in ins:
        stem = fn[:2]
        inp = os.path.join(BS, "in16k", fn)
        x16 = sf.read(inp, dtype="float32")[0]

        outs, ms = {}, {}
        for s in ["f32"] + SCHEMES:
            dst = os.path.join(OUT, f"{stem}_{s}.wav")
            wd = d32 if s == "f32" else wdirs[s]
            t0 = time.perf_counter()
            r = subprocess.run([CLI, inp, dst, "--weights", wd], capture_output=True, text=True)
            ms[s] = (time.perf_counter() - t0) * 1e3
            if r.returncode != 0:
                print("  FAIL", s, r.stderr[:120])
                return
            outs[s] = sf.read(dst, dtype="float32")[0]

        ref = outs["f32"]
        nz = x16[: len(ref) - DELAY]
        f_in, _ = floor_and_speech(nz)
        f32_red = f_in - floor_and_speech(ref[DELAY:][: len(nz)])[0]

        row = {"id": stem, "f32_reduction": round(f32_red, 1)}
        print(f"\n  {stem}   (f32 removes {f32_red:+.1f} dB of noise)")
        print(f"    {'scheme':<22} {'noise removed':>14} {'vs f32':>9} {'SI-SDR':>9} {'error':>11}")
        for s in SCHEMES:
            en = outs[s][DELAY:][: len(nz)]
            red = f_in - floor_and_speech(en)[0]
            n = min(len(outs[s]), len(ref))
            resid = db(np.sqrt(np.mean((outs[s][:n] - ref[:n]) ** 2)))
            sd = si_sdr(outs[s][:n], ref[:n])
            row[s] = {"reduction": round(red, 1), "lost": round(red - f32_red, 1),
                      "sisdr": round(sd, 1), "resid": round(resid, 1), "ms": round(ms[s], 1)}
            print(f"    {s:<22} {red:+11.1f} dB {red - f32_red:+6.1f} dB "
                  f"{sd:>6.1f} dB {resid:>7.1f} dBFS")
        results.append(row)

    # ---------- aggregate ----------
    print("\n" + "=" * 78)
    print(f"AGGREGATE over {len(results)} samples")
    print("=" * 78)
    print(f"  {'scheme':<22} {'noise lost vs f32':>18} {'worst sample':>13} {'SI-SDR':>9} {'error':>11}")
    agg = {}
    for s in SCHEMES:
        lost = np.mean([r[s]["lost"] for r in results])
        worst = min(r[s]["lost"] for r in results)
        sd = np.mean([r[s]["sisdr"] for r in results])
        rs = np.mean([r[s]["resid"] for r in results])
        agg[s] = {"lost": round(float(lost), 2), "worst": round(float(worst), 1),
                  "sisdr": round(float(sd), 1), "resid": round(float(rs), 1)}
        print(f"  {s:<22} {lost:+15.2f} dB {worst:+10.1f} dB {sd:>6.1f} dB {rs:>7.1f} dBFS")

    print("\n  Bar to clear: a 16-bit WAV holds ~96 dBFS of range. f16 lands at -94 dBFS")
    print("  error and loses 0.0 dB of noise reduction. per-tensor int8 lost 2.1 dB.")

    shutil.rmtree(os.path.join(OUT, "_w"))
    json.dump({"results": results, "agg": agg}, open(os.path.join(OUT, "results.json"), "w"), indent=1)
    print(f"\nwrote {os.path.join(OUT, 'results.json')}")


if __name__ == "__main__":
    main()
