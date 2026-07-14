"""Confirm the exact shipping format end-to-end before it is frozen into a published crate.

Candidate: int8 payload, one f16 scale per 64 contiguous weights along the contraction
axis (rows shorter than 64, or not a multiple of it, get a single per-row scale).

This simulates ONLY the weight quantisation -- the dequantised f32 weights run through the
existing f32 kernels, which is exactly what the loader will do. It is therefore an honest
measurement of the crate as it will actually ship. Quantising the ACTIVATIONS too (what the
VNNI kernel needs to be fast) is a separate, later question.
"""
import os
import shutil
import subprocess
import sys
import time

import numpy as np
import soundfile as sf

sys.stdout.reconfigure(encoding="utf-8")
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(ROOT)
ASSETS = os.path.join(ROOT, "hush-vani", "assets")
BS = os.path.join(ROOT, "bench_samples")
OUT = os.path.join(ROOT, "quant_schemes")
CLI = os.path.join(ROOT, "target", "release", "hush-vani.exe")
DELAY = 160
BIN, MAN = os.path.join(ASSETS, "weights.bin"), os.path.join(ASSETS, "weights.txt")

CANDIDATES = {"int8_b64_f16s": 64, "int8_b32_f16s": 32}


def tensors():
    for line in open(MAN):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        head, off, n = line.rsplit(" ", 2)
        name, shape_s = head.rsplit(" ", 1)
        shape = [int(d) for d in shape_s.split(",")] if shape_s else []
        yield name, shape, int(off), int(n)


def quantize(raw, target):
    out = raw.copy()
    for _n, shape, off, n in tensors():
        seg = raw[off:off + n]
        row = shape[-1] if len(shape) >= 2 else n
        b = target if row % target == 0 else row
        r = seg.reshape(-1, b)
        amax = np.abs(r).max(axis=1, keepdims=True)
        s = np.where(amax > 0, amax / 127.0, 1.0).astype(np.float16).astype(np.float32)
        s[s == 0] = 1.0
        q = np.clip(np.round(r / s), -127, 127).astype(np.int8)
        out[off:off + n] = (q.astype(np.float32) * s).reshape(-1)
    return out


def db(x):
    return 20 * np.log10(max(float(x), 1e-12))


def si_sdr(est, ref):
    est, ref = est - est.mean(), ref - ref.mean()
    a = np.dot(est, ref) / (np.dot(ref, ref) + 1e-12)
    p = a * ref
    n = est - p
    return 10 * np.log10((np.dot(p, p) + 1e-12) / (np.dot(n, n) + 1e-12))


def floor10(x):
    fl = 320
    nf = len(x) // fl
    e = 10 * np.log10(np.mean(x[: nf * fl].reshape(nf, fl) ** 2, 1) + 1e-12)
    return float(np.percentile(e, 10))


def main():
    raw = np.fromfile(BIN, dtype="<f4")
    os.makedirs(OUT, exist_ok=True)
    dirs = {}
    for name, b in CANDIDATES.items():
        d = os.path.join(OUT, "_w", name)
        os.makedirs(d, exist_ok=True)
        quantize(raw, b).astype("<f4").tofile(os.path.join(d, "weights.bin"))
        shutil.copy(MAN, os.path.join(d, "weights.txt"))
        dirs[name] = d
    d32 = os.path.join(OUT, "_w", "f32")
    os.makedirs(d32, exist_ok=True)
    shutil.copy(BIN, os.path.join(d32, "weights.bin"))
    shutil.copy(MAN, os.path.join(d32, "weights.txt"))
    # f16, the thing we are trying to beat
    d16 = os.path.join(OUT, "_w", "f16")
    os.makedirs(d16, exist_ok=True)
    shutil.copy(os.path.join(ASSETS, "weights.f16.bin"), os.path.join(d16, "weights.bin"))
    shutil.copy(os.path.join(ASSETS, "weights.f16.txt"), os.path.join(d16, "weights.txt"))
    dirs = {"f16": d16, **dirs}

    names = list(dirs)
    print("=" * 84)
    print("SHIPPING-FORMAT CONFIRMATION  (noise removed vs the f32 reference, per sample)")
    print("=" * 84)
    print(f"  {'sample':<8} {'f32 ref':>9} " + " ".join(f"{n:>16}" for n in names))

    lost = {n: [] for n in names}
    sisdr = {n: [] for n in names}
    resid = {n: [] for n in names}
    for fn in sorted(os.listdir(os.path.join(BS, "in16k"))):
        stem, inp = fn[:2], os.path.join(BS, "in16k", fn)
        x16 = sf.read(inp, dtype="float32")[0]
        outs = {}
        for n, wd in [("f32", d32)] + list(dirs.items()):
            dst = os.path.join(OUT, f"{stem}_{n}.wav")
            r = subprocess.run([CLI, inp, dst, "--weights", wd], capture_output=True, text=True)
            if r.returncode != 0:
                print("FAIL", n, r.stderr[:150])
                return
            outs[n] = sf.read(dst, dtype="float32")[0]
        ref = outs["f32"]
        nz = x16[: len(ref) - DELAY]
        f_in = floor10(nz)
        ref_red = f_in - floor10(ref[DELAY:][: len(nz)])
        cells = []
        for n in names:
            red = f_in - floor10(outs[n][DELAY:][: len(nz)])
            k = min(len(outs[n]), len(ref))
            lost[n].append(red - ref_red)
            sisdr[n].append(si_sdr(outs[n][:k], ref[:k]))
            resid[n].append(db(np.sqrt(np.mean((outs[n][:k] - ref[:k]) ** 2))))
            cells.append(f"{red:+7.1f} ({red - ref_red:+4.1f})")
        print(f"  {stem:<8} {ref_red:+8.1f} dB " + " ".join(f"{c:>16}" for c in cells))

    print("\n" + "=" * 84)
    print(f"  {'format':<16} {'avg noise lost':>15} {'worst':>8} {'SI-SDR':>9} {'error':>11} {'size':>9}")
    sizes = {"f16": 4.56, "int8_b64_f16s": 2.36, "int8_b32_f16s": 2.43}
    for n in names:
        print(f"  {n:<16} {np.mean(lost[n]):+12.2f} dB {min(lost[n]):+6.1f} dB "
              f"{np.mean(sisdr[n]):>6.1f} dB {np.mean(resid[n]):>7.1f} dBFS {sizes[n]:>7.2f} MB")
    print("\n  per-tensor int8 (the rejected one) for reference:  -2.07 dB   -8.3 dB   "
          "28.9 dB   -57.1 dBFS   2.28 MB")
    shutil.rmtree(os.path.join(OUT, "_w"))


if __name__ == "__main__":
    main()
