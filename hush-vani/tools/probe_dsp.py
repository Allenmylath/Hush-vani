"""Reverse-engineer libdf's exact DSP so it can be reimplemented in Rust."""
import numpy as np
from libdf import DF, erb, erb_norm, unit_norm

SR, FFT, HOP, NB_ERB, NB_DF, MIN_ERB = 16000, 320, 160, 32, 64, 2
ALPHA = float(np.exp(-(HOP / SR) / 1.0))
df = DF(sr=SR, fft_size=FFT, hop_size=HOP, nb_bands=NB_ERB, min_nb_erb_freqs=MIN_ERB)
W = df.erb_widths()

# ---- 1. window ----
w = np.asarray(df.fft_window())
i = np.arange(FFT)
vorbis = np.sin(np.pi / 2 * np.sin(np.pi * (i + 0.5) / FFT) ** 2)
print("window len", w.shape, "max|w - vorbis| =", np.abs(w - vorbis).max())

# ---- 2. analysis scaling ----
rng = np.default_rng if False else np.random.default_rng(1)
au = (rng.standard_normal(HOP * 4) * 0.1).astype(np.float32)
df.reset()
spec = df.analysis(au[None])
# manual: frame buffer of FFT samples, hop 160, window, rfft
buf = np.zeros(FFT, np.float32)
man = []
for t in range(4):
    buf = np.concatenate([buf[HOP:], au[t * HOP : (t + 1) * HOP]])
    man.append(np.fft.rfft(buf * vorbis))
man = np.stack(man)
for scale, nm in [(1.0, "1"), (1 / FFT, "1/N"), (1 / np.sqrt(FFT), "1/sqrt(N)")]:
    print(f"  analysis scale {nm:10s} maxdiff={np.abs(man * scale - spec[0]).max():.4e}")

# ---- 3. erb() ----
df.reset()
spec2 = df.analysis(au[None])
e = erb(spec2, W)
pw = (spec2.real**2 + spec2.imag**2)[0]  # [T,161]
edges = np.concatenate([[0], np.cumsum(W)]).astype(np.int64)
band_sum = np.stack([pw[:, edges[b] : edges[b + 1]].sum(1) for b in range(NB_ERB)], 1)
band_mean = band_sum / W[None, :]
for cand, nm in [
    (10 * np.log10(band_mean + 1e-10), "10log10(mean+1e-10)"),
    (10 * np.log10(band_sum + 1e-10), "10log10(sum+1e-10)"),
    (20 * np.log10(band_mean + 1e-10), "20log10(mean+1e-10)"),
]:
    print(f"  erb {nm:24s} maxdiff={np.abs(cand - e[0]).max():.4e}")

# ---- 4. erb_norm: state recursion + init ----
e_in = e.copy()
en = erb_norm(e_in.copy(), ALPHA)
# assume: state[i] = x*(1-a) + state*a ; out = (x - state)/40 ; find init
# solve init from first frame: out0 = (x0 - (x0*(1-a) + s0*a))/C
x0, o0 = e[0, 0], en[0, 0]
for C in [40.0, 1.0]:
    s0 = (x0 - o0 * C - x0 * (1 - ALPHA)) / ALPHA
    print(f"  erb_norm C={C:4.1f} implied init state = {np.round(s0, 4)[:6]} ...")
# verify full recursion with the implied init (C=40)
s = (e[0, 0] - en[0, 0] * 40.0 - e[0, 0] * (1 - ALPHA)) / ALPHA
out = np.empty_like(e[0])
st = s.copy()
for t in range(e.shape[1]):
    st = e[0, t] * (1 - ALPHA) + st * ALPHA
    out[t] = (e[0, t] - st) / 40.0
print("  erb_norm recursion reproduces libdf:", np.abs(out - en[0]).max())
print("  erb_norm init state:", np.round(s, 4))
lin = np.linspace(-60, -90, NB_ERB)
print("  vs linspace(-60,-90,32) maxdiff:", np.abs(s - lin).max())

# ---- 5. unit_norm ----
sd = spec2[..., :NB_DF]
un = unit_norm(sd.copy(), ALPHA)
# assume state[i] = |x|*(1-a) + state*a ; out = x / sqrt(state)
x0 = np.abs(sd[0, 0])
o0 = np.abs(un[0, 0])
# |out| = |x|/sqrt(state)  => state = (|x|/|out|)^2
st0 = (np.abs(sd[0, 0]) / np.abs(un[0, 0])) ** 2
s0 = (st0 - np.abs(sd[0, 0]) * (1 - ALPHA)) / ALPHA
print("\n  unit_norm implied init state:", np.round(s0, 6)[:6], "...")
lin2 = np.linspace(0.001, 0.0001, NB_DF)
print("  vs linspace(0.001,0.0001,64) maxdiff:", np.abs(s0 - lin2).max())
st = lin2.copy().astype(np.float32)
out = np.empty_like(sd[0])
for t in range(sd.shape[1]):
    st = np.abs(sd[0, t]) * (1 - ALPHA) + st * ALPHA
    out[t] = sd[0, t] / np.sqrt(st)
print("  unit_norm recursion reproduces libdf:", np.abs(out - un[0]).max())

# ---- 6. synthesis ----
df.reset()
sp = df.analysis(au[None])
df.reset()
rec = df.synthesis(sp.copy())
print("\n  synthesis out len", rec.shape)
