"""Pin down libdf's synthesis (ISTFT overlap-add) exactly."""
import numpy as np
from libdf import DF

SR, FFT, HOP = 16000, 320, 160
df = DF(sr=SR, fft_size=FFT, hop_size=HOP, nb_bands=32, min_nb_erb_freqs=2)
i = np.arange(FFT)
win = np.sin(np.pi / 2 * np.sin(np.pi * (i + 0.5) / FFT) ** 2).astype(np.float32)

rng = np.random.default_rng(3)
au = (rng.standard_normal(HOP * 6) * 0.1).astype(np.float32)
df.reset()
spec = df.analysis(au[None].copy())
df.reset()
ref = df.synthesis(spec.copy())[0]

T = spec.shape[1]
print("frames", T, "out len", ref.shape)

for name, extra in [("irfft*N, win, OLA", FFT), ("irfft, win, OLA", 1.0)]:
    buf = np.zeros(FFT, np.float64)
    out = np.zeros(T * HOP, np.float64)
    for t in range(T):
        y = np.fft.irfft(spec[0, t].astype(np.complex128), n=FFT) * extra
        y = y * win
        buf += y
        out[t * HOP : (t + 1) * HOP] = buf[:HOP]
        buf = np.concatenate([buf[HOP:], np.zeros(HOP)])
    print(f"  {name:20s} maxdiff vs libdf = {np.abs(out - ref).max():.4e}")

# Princen-Bradley check for the vorbis window at 50% overlap
s = win[:HOP] ** 2 + win[HOP:] ** 2
print("  sum of squared window halves: min", s.min(), "max", s.max())
