"""Run the same clip through f32 / f16 / int8 weights and store the outputs side by side.

int8 is simulated by quantising each tensor to int8 (symmetric, per-tensor) and dequantising
back to f32 -- identical numerics to an int8 deployment's weights, without needing the int8
kernels wired through the whole model.
"""
import os
import shutil
import subprocess
import sys

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__))
CRATE = os.path.dirname(HERE)
ROOT = os.path.dirname(CRATE)
ASSETS = os.path.join(CRATE, "assets")
OUT = os.path.join(ROOT, "comparison")
CLI = os.path.join(ROOT, "target", "release", "hush-vani.exe")
SAMPLE = os.path.join(CRATE, "sample_raw.wav")


def tensors(man):
    for line in open(man):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        head, off, n = line.rsplit(" ", 2)
        name, _shape = head.rsplit(" ", 1)
        yield name, int(off), int(n)


def main():
    os.makedirs(OUT, exist_ok=True)
    f32_bin = os.path.join(ASSETS, "weights.bin")
    f32_txt = os.path.join(ASSETS, "weights.txt")
    raw = np.fromfile(f32_bin, dtype="<f4")

    # --- build three weight dirs ---
    dirs = {}
    for name in ("f32", "f16", "int8"):
        d = os.path.join(OUT, "_w", name)
        os.makedirs(d, exist_ok=True)
        dirs[name] = d

    shutil.copy(f32_bin, os.path.join(dirs["f32"], "weights.bin"))
    shutil.copy(f32_txt, os.path.join(dirs["f32"], "weights.txt"))

    shutil.copy(os.path.join(ASSETS, "weights.f16.bin"), os.path.join(dirs["f16"], "weights.bin"))
    shutil.copy(os.path.join(ASSETS, "weights.f16.txt"), os.path.join(dirs["f16"], "weights.txt"))

    # int8: symmetric per-tensor round-trip, stored back as f32
    q = raw.copy()
    for _n, off, n in tensors(f32_txt):
        seg = raw[off:off + n]
        amax = np.abs(seg).max()
        s = amax / 127.0 if amax > 0 else 1.0
        q[off:off + n] = np.clip(np.round(seg / s), -127, 127) * s
    q.astype("<f4").tofile(os.path.join(dirs["int8"], "weights.bin"))
    shutil.copy(f32_txt, os.path.join(dirs["int8"], "weights.txt"))

    # --- run the model three times ---
    shutil.copy(SAMPLE, os.path.join(OUT, "0_noisy_input.wav"))
    outs = {}
    for name in ("f32", "f16", "int8"):
        dst = os.path.join(OUT, f"{ {'f32':1,'f16':2,'int8':3}[name] }_enhanced_{name}.wav")
        dst = os.path.join(OUT, f"{ {'f32':1,'f16':2,'int8':3}[name] }_enhanced_{name}.wav")
        r = subprocess.run([CLI, SAMPLE, dst, "--weights", dirs[name]],
                           capture_output=True, text=True)
        print(f"  {name:5} -> {os.path.basename(dst)}   {r.stderr.strip()}")
        outs[name] = dst

    shutil.rmtree(os.path.join(OUT, "_w"))
    print(f"\nwrote {OUT}")
    return outs


if __name__ == "__main__":
    main()
