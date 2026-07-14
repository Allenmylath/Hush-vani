"""Run the DeepFilterNet2 demo samples through Hush at f32 / f16 / int8 and compare.

The samples are 48 kHz; Hush is a 16 kHz model, so they are resampled to 16 kHz first.
(That also means these outputs are NOT comparable to DeepFilterNet2's own 48 kHz full-band
results -- different model, different bandwidth. The point here is the three precisions
against each other on real, varied noise.)

Each precision is read from its real blob by the Rust loader. Build them first:

    python tools/export_weights.py                       # assets/weights.{bin,txt}
    python tools/export_weights.py --f16  --out weights.f16
    python tools/export_weights.py --int8 --out weights.int8
"""
import json
import os
import shutil
import subprocess
import sys
import time

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__))
CRATE = os.path.dirname(HERE)
ROOT = os.path.dirname(CRATE)
ASSETS = os.path.join(CRATE, "assets")
BS = os.path.join(ROOT, "bench_samples")
CLI = os.path.join(ROOT, "target", "release", "hush-vani.exe")
SR = 16000
DELAY = 160
PRECS = ("f32", "f16", "int8")


def build_weight_dirs():
    """One dir per precision, each holding the REAL blob that ships for it.

    int8 used to be simulated here by quantising and dequantising back to f32 in numpy, then
    running the f32 kernels. That measured a scheme, not a crate. Every precision now goes
    through its actual on-disk blob and the Rust loader, so what is measured is what ships.
    """
    dirs = {p: os.path.join(BS, "_w", p) for p in PRECS}
    src = {"f32": "weights", "f16": "weights.f16", "int8": "weights.int8"}
    for p, d in dirs.items():
        os.makedirs(d, exist_ok=True)
        for ext in ("bin", "txt"):
            s = os.path.join(ASSETS, f"{src[p]}.{ext}")
            if not os.path.exists(s):
                sys.exit(f"missing {s} -- run: python tools/export_weights.py "
                         f"{'--' + p if p != 'f32' else ''} --out {src[p]}")
            shutil.copy(s, os.path.join(d, f"weights.{ext}"))
    return dirs


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
    dirs = build_weight_dirs()
    os.makedirs(os.path.join(BS, "in16k"), exist_ok=True)
    for p in PRECS:
        os.makedirs(os.path.join(BS, p), exist_ok=True)

    noisy_files = sorted(os.listdir(os.path.join(BS, "noisy")))
    results = []

    print("=" * 92)
    print("DeepFilterNet2 demo samples through Hush — f32 vs f16 vs int8")
    print("=" * 92)

    for fn in noisy_files:
        x48, sr = sf.read(os.path.join(BS, "noisy", fn), dtype="float32")
        if x48.ndim > 1:
            x48 = x48.mean(1)
        x16 = resample_poly(x48, SR, sr).astype(np.float32)
        stem = fn[:2]
        label = fn[3:].replace("_noisy.wav", "")
        noise = label.split("_")[2] if len(label.split("_")) > 2 else label
        inp = os.path.join(BS, "in16k", f"{stem}_noisy16k.wav")
        sf.write(inp, x16, SR)

        outs, ms = {}, {}
        for p in PRECS:
            dst = os.path.join(BS, p, f"{stem}_{p}.wav")
            t0 = time.perf_counter()
            r = subprocess.run([CLI, inp, dst, "--weights", dirs[p]], capture_output=True, text=True)
            ms[p] = (time.perf_counter() - t0) * 1e3
            if r.returncode != 0:
                print("  FAIL", r.stderr[:120])
            outs[p] = sf.read(dst, dtype="float32")[0]

        nz = x16[: len(outs["f32"]) - DELAY]
        f_in, s_in = floor_and_speech(nz)
        ref = outs["f32"]
        row = {"id": stem, "noise": noise, "dur": len(x16) / SR,
               "in_floor": round(f_in, 1), "in_snr": round(s_in - f_in, 1)}
        for p in PRECS:
            en = outs[p][DELAY:][: len(nz)]
            f_o, s_o = floor_and_speech(en)
            n = min(len(outs[p]), len(ref))
            resid = db(np.sqrt(np.mean((outs[p][:n] - ref[:n]) ** 2))) if p != "f32" else None
            row[p] = {
                "reduction": round(f_in - f_o, 1),
                "floor": round(f_o, 1),
                "dr": round(s_o - f_o, 1),
                "sisdr": None if p == "f32" else round(si_sdr(outs[p][:n], ref[:n]), 1),
                "resid": None if resid is None else round(resid, 1),
                "ms": round(ms[p], 1),
            }
        results.append(row)
        print(f"\n{stem}  {noise:14} ({row['dur']:.1f}s, input SNR ~{row['in_snr']:.0f} dB)")
        print(f"  {'':6} {'noise removed':>14} {'out floor':>10} {'dyn range':>10} "
              f"{'SI-SDR vs f32':>14} {'error':>10}")
        for p in PRECS:
            r = row[p]
            ss = "  (ref)" if r["sisdr"] is None else f"{r['sisdr']:6.1f} dB"
            rr = "     —" if r["resid"] is None else f"{r['resid']:6.1f} dBFS"
            print(f"  {p:6} {r['reduction']:+11.1f} dB {r['floor']:9.1f} dB {r['dr']:8.1f} dB "
                  f"{ss:>14} {rr:>10}")

    # ---- aggregate ----
    print("\n" + "=" * 92)
    print("AGGREGATE over", len(results), "real samples")
    print("=" * 92)
    print(f"  {'':6} {'noise removed':>16} {'SI-SDR vs f32':>15} {'error level':>13} {'speed':>10}")
    agg = {}
    for p in PRECS:
        red = np.mean([r[p]["reduction"] for r in results])
        sis = [r[p]["sisdr"] for r in results if r[p]["sisdr"] is not None]
        res = [r[p]["resid"] for r in results if r[p]["resid"] is not None]
        spd = np.mean([r[p]["ms"] for r in results])
        agg[p] = {"reduction": round(float(red), 1),
                  "sisdr": None if not sis else round(float(np.mean(sis)), 1),
                  "resid": None if not res else round(float(np.mean(res)), 1),
                  "ms": round(float(spd), 1)}
        ss = "   (reference)" if not sis else f"{np.mean(sis):8.1f} dB"
        rr = "        —" if not res else f"{np.mean(res):7.1f} dBFS"
        print(f"  {p:6} {red:+13.1f} dB {ss:>15} {rr:>13} {spd:8.0f} ms")

    shutil.rmtree(os.path.join(BS, "_w"))
    json.dump({"results": results, "agg": agg}, open(os.path.join(BS, "results.json"), "w"), indent=1)
    print(f"\nwrote {os.path.join(BS, 'results.json')}")


if __name__ == "__main__":
    main()
