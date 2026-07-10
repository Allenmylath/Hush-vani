import numpy as np
from libdf import DF

SR, FFT, HOP, NB_ERB, MIN_ERB = 16000, 320, 160, 32, 2
df = DF(sr=SR, fft_size=FFT, hop_size=HOP, nb_bands=NB_ERB, min_nb_erb_freqs=MIN_ERB)

rng = np.random.default_rng(0)
audio = (rng.standard_normal((1, 16000)) * 0.05).astype(np.float32)
spec = df.analysis(audio.copy())
rec = df.synthesis(spec.copy())

x, y = audio[0], rec[0]
# cross-correlate to find integer delay of y relative to x
best = None
for d in range(0, 641):
    n = len(x) - d
    num = float(np.dot(y[d : d + n], x[:n]))
    den = float(np.linalg.norm(y[d : d + n]) * np.linalg.norm(x[:n]) + 1e-12)
    c = num / den
    if best is None or c > best[1]:
        best = (d, c)
print("best delay (samples):", best[0], "corr:", round(best[1], 6))

d = best[0]
n = len(x) - d
err = np.mean((y[d : d + n] - x[:n]) ** 2) / np.mean(x[:n] ** 2)
print(f"relerr at delay {d}: {err:.3e}")
print("energy ratio out/in:", float(np.mean(y**2) / np.mean(x**2)))

# check reset behaviour
df.reset()
spec2 = df.analysis(audio.copy())
print("analysis deterministic after reset:", np.allclose(spec, spec2))
