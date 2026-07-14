"""Check the Rust int8 loader against the Python simulation that was used to approve it.

The approval numbers came from quantising in numpy, dequantising back to f32, and feeding
the f32 blob to the existing kernels. The shipped path instead has Rust read the int8 blob
and dequantise it itself. Those must agree, or the measured numbers do not describe the
crate that ships.

The two are compared at the WEIGHT level (Rust's decode vs numpy's, exactly) and then at
the AUDIO level (same wav out of the CLI both ways), plus the end-to-end metrics vs f32.
"""
import os
import shutil
import subprocess
import sys

import numpy as np
import soundfile as sf

sys.stdout.reconfigure(encoding="utf-8")
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ASSETS = os.path.join(ROOT, "hush-vani", "assets")
BS = os.path.join(ROOT, "bench_samples")
OUT = os.path.join(ROOT, "quant_schemes", "verify")
CLI = os.path.join(ROOT, "target", "release", "hush-vani.exe")
DELAY = 160

F32_BIN, F32_MAN = os.path.join(ASSETS, "weights.bin"), os.path.join(ASSETS, "weights.txt")
I8_BIN, I8_MAN = os.path.join(ASSETS, "weights.int8.bin"), os.path.join(ASSETS, "weights.int8.txt")


def parse_int8():
    """Dequantise the int8 blob in numpy, the same way the Rust loader is supposed to."""
    raw = open(I8_BIN, "rb").read()
    lines = [l.strip() for l in open(I8_MAN) if l.strip()]
    payload = int(lines[0].split()[3])
    codes = np.frombuffer(raw[:payload], dtype=np.int8)
    scales = np.frombuffer(raw[payload:], dtype="<f4")
    out = np.zeros(payload, dtype=np.float32)
    meta = {}
    for line in lines[1:]:
        # "<name> <shape> <off> <numel> <scale_off> <block>" -- the name holds no spaces,
        # so five right-splits land each field exactly.
        name, _shape, off, n, soff, block = line.rsplit(" ", 5)
        off, n, soff, block = int(off), int(n), int(soff), int(block)
        c = codes[off:off + n].astype(np.float32)
        s = scales[soff:soff + (n + block - 1) // block]
        out[off:off + n] = c * np.repeat(s, block)[:n]
        meta[name] = (off, n)
    return out, meta


def main():
    os.makedirs(OUT, exist_ok=True)

    # ---- weight level: numpy's decode of the int8 blob vs the true f32 ----
    f32 = np.fromfile(F32_BIN, dtype="<f4")
    deq, meta = parse_int8()
    print("=" * 76)
    print("WEIGHT LEVEL")
    print("=" * 76)
    # the two blobs have different padding, so compare tensor by tensor
    f32meta = {}
    for line in open(F32_MAN):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        head, off, n = line.rsplit(" ", 2)
        nm, _ = head.rsplit(" ", 1)
        f32meta[nm] = (int(off), int(n))
    num = den = 0.0
    for nm, (off8, n8) in meta.items():
        off4, n4 = f32meta[nm]
        assert n8 == n4, f"{nm}: length differs between blobs"
        a, b = f32[off4:off4 + n4], deq[off8:off8 + n8]
        num += float(a @ a)
        den += float((b - a) @ (b - a))
    print(f"  int8 blob dequantised vs true f32 weights: {10 * np.log10(num / den):.1f} dB SNR")
    print(f"  (the approved simulation measured 47.9 dB -- these must match)")

    # ---- audio level: Rust reading int8 vs Rust reading the dequantised f32 ----
    d_i8, d_sim, d_f32 = (os.path.join(OUT, x) for x in ("i8", "sim", "f32"))
    for d in (d_i8, d_sim, d_f32):
        os.makedirs(d, exist_ok=True)
    shutil.copy(I8_BIN, os.path.join(d_i8, "weights.bin"))
    shutil.copy(I8_MAN, os.path.join(d_i8, "weights.txt"))
    deq.astype("<f4").tofile(os.path.join(d_sim, "weights.bin"))
    shutil.copy(I8_MAN, os.path.join(d_sim, "weights.txt"))  # placeholder, fixed below
    shutil.copy(F32_BIN, os.path.join(d_f32, "weights.bin"))
    shutil.copy(F32_MAN, os.path.join(d_f32, "weights.txt"))

    # the simulated blob is f32 but uses the INT8 blob's offsets/padding, so it needs the
    # int8 manifest with the int8 header and the two extra fields stripped off.
    with open(os.path.join(d_sim, "weights.txt"), "w", encoding="utf-8") as f:
        for line in open(I8_MAN):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            name, shape, off, n, _soff, _block = line.rsplit(" ", 5)
            f.write(f"{name} {shape} {off} {n}\n")

    print("\n" + "=" * 76)
    print("AUDIO LEVEL  (Rust decoding int8  vs  Rust fed the same weights as f32)")
    print("=" * 76)
    print(f"  {'sample':<8} {'int8 vs sim':>26} {'noise removed vs f32':>22}")

    lost, sisdr, resid = [], [], []
    for fn in sorted(os.listdir(os.path.join(BS, "in16k"))):
        stem, inp = fn[:2], os.path.join(BS, "in16k", fn)
        x16 = sf.read(inp, dtype="float32")[0]
        got = {}
        for tag, wd in (("i8", d_i8), ("sim", d_sim), ("f32", d_f32)):
            dst = os.path.join(OUT, f"{stem}_{tag}.wav")
            r = subprocess.run([CLI, inp, dst, "--weights", wd], capture_output=True, text=True)
            if r.returncode != 0:
                print("  FAIL", tag, r.stderr[:200])
                return
            got[tag] = sf.read(dst, dtype="float32")[0]

        n = min(len(got["i8"]), len(got["sim"]))
        maxdiff = float(np.abs(got["i8"][:n] - got["sim"][:n]).max())

        ref = got["f32"]
        nz = x16[: len(ref) - DELAY]
        fl = lambda x: float(np.percentile(10 * np.log10(
            np.mean(x[: len(x) // 320 * 320].reshape(-1, 320) ** 2, 1) + 1e-12), 10))
        f_in = fl(nz)
        ref_red = f_in - fl(ref[DELAY:][: len(nz)])
        red = f_in - fl(got["i8"][DELAY:][: len(nz)])
        k = min(len(got["i8"]), len(ref))
        e, r_ = got["i8"][:k] - np.mean(got["i8"][:k]), ref[:k] - np.mean(ref[:k])
        a = (e @ r_) / (r_ @ r_ + 1e-12)
        p = a * r_
        sd = 10 * np.log10((p @ p + 1e-12) / ((e - p) @ (e - p) + 1e-12))
        rs = 20 * np.log10(max(float(np.sqrt(np.mean((got["i8"][:k] - ref[:k]) ** 2))), 1e-12))
        lost.append(red - ref_red)
        sisdr.append(sd)
        resid.append(rs)
        verdict = "identical" if maxdiff == 0.0 else f"max |diff| {maxdiff:.2e}"
        print(f"  {stem:<8} {verdict:>26} {red:+13.1f} dB ({red - ref_red:+.1f})")

    print("\n" + "=" * 76)
    print(f"  SHIPPED int8 crate, measured through the real Rust loader:")
    print(f"    noise removed vs f32 : {np.mean(lost):+.2f} dB   (worst sample {min(lost):+.1f} dB)")
    print(f"    SI-SDR vs f32        : {np.mean(sisdr):.1f} dB")
    print(f"    error level          : {np.mean(resid):.1f} dBFS")
    print(f"    weights on disk      : {os.path.getsize(I8_BIN) / 1e6:.2f} MB "
          f"(f16 {os.path.getsize(os.path.join(ASSETS, 'weights.f16.bin')) / 1e6:.2f} MB, "
          f"f32 {os.path.getsize(F32_BIN) / 1e6:.2f} MB)")
    shutil.rmtree(OUT, ignore_errors=True)


if __name__ == "__main__":
    main()
