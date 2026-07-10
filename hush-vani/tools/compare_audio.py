"""Compare the Rust output against the ORT output and the model author's reference wav."""
import os

import numpy as np
import soundfile as sf

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ORT = os.path.join(os.path.dirname(HERE), "ort")
SR = 16000
DELAY = 160  # one hop of algorithmic latency


def db(x):
    return 20 * np.log10(max(float(x), 1e-12))


def rms(x):
    return float(np.sqrt(np.mean(np.square(x)) + 1e-20))


def si_sdr(est, ref):
    est, ref = est - est.mean(), ref - ref.mean()
    a = np.dot(est, ref) / (np.dot(ref, ref) + 1e-12)
    p = a * ref
    n = est - p
    return 10 * np.log10((np.dot(p, p) + 1e-12) / (np.dot(n, n) + 1e-12))


def best_delay(y, x, max_d=512):
    best = (0, -2.0)
    for d in range(max_d + 1):
        n = min(len(x), len(y) - d)
        if n < 1000:
            break
        a, b = y[d : d + n], x[:n]
        c = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
        if c > best[1]:
            best = (d, c)
    return best


noisy, _ = sf.read(os.path.join(HERE, "sample_raw.wav"), dtype="float32")
rust, _ = sf.read(os.path.join(HERE, "sample_enhanced_rust.wav"), dtype="float32")
ref, _ = sf.read(os.path.join(HERE, "sample_denoised_ref.wav"), dtype="float32")
ort, _ = sf.read(os.path.join(ORT, "sample_enhanced_ort.wav"), dtype="float32")

print("=" * 72)
print("Rust (custom) output vs ORT output and the published reference")
print("=" * 72)

n = min(len(rust), len(ort))
print(f"\nrust vs onnxruntime (both 16-bit wav, same delay):")
print(f"  SI-SDR   = {si_sdr(rust[:n], ort[:n]):7.2f} dB")
print(f"  max|diff| = {np.abs(rust[:n]-ort[:n]).max():.2e}  (int16 LSB = 3.05e-5)")

d, c = best_delay(rust, ref)
a, b = rust[d:], ref[: len(rust) - d]
m = min(len(a), len(b))
print(f"\nrust vs published reference (sample_00006_denoised.wav):")
print(f"  aligned at delay {d} samples, corr={c:.6f}")
print(f"  SI-SDR   = {si_sdr(a[:m], b[:m]):7.2f} dB")
print(f"  max|diff| = {np.abs(a[:m]-b[:m]).max():.2e}")

# denoising effect, delay-compensated against the input
nz = noisy[: len(rust) - DELAY]
en = rust[DELAY:][: len(nz)]
fl = 320
nf = len(nz) // fl
e_in = 10 * np.log10(np.mean(nz[: nf * fl].reshape(nf, fl) ** 2, 1) + 1e-12)
e_out = 10 * np.log10(np.mean(en[: nf * fl].reshape(nf, fl) ** 2, 1) + 1e-12)
print(f"\nwhat the model did (rust output vs noisy input):")
print(f"  input  rms {db(rms(nz)):7.2f} dBFS   output rms {db(rms(en)):7.2f} dBFS")
print(f"  noise floor (p10 frame energy): {np.percentile(e_in,10):7.2f} -> "
      f"{np.percentile(e_out,10):7.2f} dB   ({np.percentile(e_in,10)-np.percentile(e_out,10):+.2f} dB reduction)")
print(f"  speech level (p90 frame energy): {np.percentile(e_in,90):7.2f} -> "
      f"{np.percentile(e_out,90):7.2f} dB   ({np.percentile(e_out,90)-np.percentile(e_in,90):+.2f} dB)")
print(f"  dynamic range (p90-p10): {np.percentile(e_in,90)-np.percentile(e_in,10):.2f} -> "
      f"{np.percentile(e_out,90)-np.percentile(e_out,10):.2f} dB")
