"""Quantify the three precisions against each other and dump data for the listening page."""
import json
import os
import sys

import numpy as np
import soundfile as sf

sys.stdout.reconfigure(encoding="utf-8")
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CMP = os.path.join(ROOT, "comparison")
DELAY = 160  # one hop of algorithmic latency

FILES = [
    ("noisy", "0_noisy_input.wav"),
    ("f32", "1_enhanced_f32.wav"),
    ("f16", "2_enhanced_f16.wav"),
    ("int8", "3_enhanced_int8.wav"),
]


def db(x):
    return 20 * np.log10(max(float(x), 1e-12))


def si_sdr(est, ref):
    est, ref = est - est.mean(), ref - ref.mean()
    a = np.dot(est, ref) / (np.dot(ref, ref) + 1e-12)
    p = a * ref
    n = est - p
    return 10 * np.log10((np.dot(p, p) + 1e-12) / (np.dot(n, n) + 1e-12))


def main():
    au = {k: sf.read(os.path.join(CMP, f), dtype="float32")[0] for k, f in FILES}
    sr = 16000
    ref = au["f32"]

    print("=" * 74)
    print("Same 5 s clip, three weight precisions")
    print("=" * 74)
    print(f"{'':6} {'rms dBFS':>9} {'SI-SDR vs f32':>14} {'max|diff|':>11}  {'residual':>9}")
    rows = {}
    for k in ("f32", "f16", "int8"):
        x = au[k]
        n = min(len(x), len(ref))
        s = si_sdr(x[:n], ref[:n]) if k != "f32" else float("inf")
        d = np.abs(x[:n] - ref[:n]).max()
        # residual = the error signal's level, i.e. what quantisation ADDED
        res = db(np.sqrt(np.mean((x[:n] - ref[:n]) ** 2))) if k != "f32" else -np.inf
        sstr = "  (reference)" if k == "f32" else f"{s:8.1f} dB"
        rstr = "     —" if k == "f32" else f"{res:7.1f} dBFS"
        print(f"{k:6} {db(np.sqrt(np.mean(x**2))):8.2f}  {sstr:>14} {d:11.2e}  {rstr:>9}")
        rows[k] = {"sisdr": None if k == "f32" else round(float(s), 1),
                   "rms": round(db(np.sqrt(np.mean(x**2))), 2),
                   "residual": None if k == "f32" else round(float(res), 1)}

    # noise-floor reduction vs the noisy input (delay-compensated)
    print("\nnoise-floor reduction (p10 frame energy vs the noisy input):")
    nz = au["noisy"][: len(ref) - DELAY]
    fl, nf = 320, len(nz) // 320
    e_in = 10 * np.log10(np.mean(nz[: nf * fl].reshape(nf, fl) ** 2, 1) + 1e-12)
    p10_in = np.percentile(e_in, 10)
    for k in ("f32", "f16", "int8"):
        en = au[k][DELAY:][: len(nz)]
        e = 10 * np.log10(np.mean(en[: nf * fl].reshape(nf, fl) ** 2, 1) + 1e-12)
        p10, p90 = np.percentile(e, 10), np.percentile(e, 90)
        print(f"  {k:5}  floor {p10:7.2f} dB   reduction {p10_in - p10:+6.2f} dB   "
              f"dynamic range {p90 - p10:5.2f} dB")
        rows[k]["floor"] = round(float(p10), 1)
        rows[k]["reduction"] = round(float(p10_in - p10), 1)

    # where does int8's error live? (spectrum of the residual)
    print("\nwhere the int8 error sits (residual energy by band):")
    n = min(len(au["int8"]), len(ref))
    resid = au["int8"][:n] - ref[:n]
    R = np.abs(np.fft.rfft(resid * np.hanning(n)))
    S = np.abs(np.fft.rfft(ref[:n] * np.hanning(n)))
    freqs = np.fft.rfftfreq(n, 1 / sr)
    for lo, hi in [(0, 500), (500, 1500), (1500, 4000), (4000, 8000)]:
        m = (freqs >= lo) & (freqs < hi)
        r = 10 * np.log10(np.sum(R[m] ** 2) / (np.sum(S[m] ** 2) + 1e-20) + 1e-20)
        print(f"  {lo:5}-{hi:<5} Hz   residual is {r:+6.1f} dB relative to signal")

    # envelopes for the viz (peak per ~10 ms)
    env = {}
    W = 160
    for k, _ in FILES:
        x = au[k]
        m = len(x) // W
        env[k] = np.abs(x[: m * W].reshape(m, W)).max(1).round(4).tolist()

    json.dump({"rows": rows, "env": env, "sr": sr},
              open(os.path.join(CMP, "metrics.json"), "w"))
    print(f"\nwrote {os.path.join(CMP, 'metrics.json')}")


if __name__ == "__main__":
    main()
