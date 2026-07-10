"""Dump the ORT reference intermediates so the Rust port can be validated stage by stage."""
import os
import sys

import numpy as np
import soundfile as sf

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "ort"))
from hush_ort import HushORT  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets")
os.makedirs(OUT, exist_ok=True)

noisy, sr = sf.read(os.path.join(ROOT, "ort", "sample_raw.wav"), dtype="float32")
o = HushORT(intra_threads=1)
out, aux = o.enhance(noisy, collect=True)

spec = aux["spec"]
_, fe, fs = o.features(noisy)
e0, e1, e2, e3, emb, c0, lsnr = o.enc.run(None, {"feat_erb": fe, "feat_spec": fs})
(m,) = o.erb_dec.run(None, {"emb": emb, "e3": e3, "e2": e2, "e1": e1, "e0": e0})
(coefs,) = o.df_dec.run(None, {"emb": emb, "c0": c0})


def save(name, a):
    a = np.ascontiguousarray(a, dtype="<f4")
    a.tofile(os.path.join(OUT, f"{name}.bin"))
    print(f"  {name:12s} {a.shape}")


save("spec_re", spec.real)
save("spec_im", spec.imag)
save("feat_erb", fe)
save("feat_spec", fs)
save("e0", e0)
save("e1", e1)
save("e2", e2)
save("e3", e3)
save("c0", c0)
save("emb", emb)
save("lsnr", lsnr)
save("mask", m)
save("coefs", coefs)
save("enhanced", out)
print(f"-> {OUT}")
