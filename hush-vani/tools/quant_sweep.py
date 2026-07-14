"""Pick the int8 block size and scale dtype. This decides the ON-DISK FORMAT, which is
permanent once the crate is published, so it is worth sweeping rather than guessing.

Block-wise means: one scale per B contiguous weights along the contraction (last) axis.
B = row length is plain per-channel. Smaller B costs more scale bytes and buys accuracy.
Rows that are not a multiple of B fall back to one scale for the whole row.

Scales can be stored f32 or f16. f16 halves the scale overhead; a scale is a positive
magnitude with ~3 decimal digits of headroom, so f16 should be lossless in practice --
this checks that it is.
"""
import sys

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
BIN = "hush-vani/assets/weights.bin"
MAN = "hush-vani/assets/weights.txt"


def tensors():
    for line in open(MAN):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        head, off, n = line.rsplit(" ", 2)
        name, shape_s = head.rsplit(" ", 1)
        shape = [int(d) for d in shape_s.split(",")] if shape_s else []
        yield name, shape, int(off), int(n)


def blocks_for(shape, numel, target):
    """Block length actually used for this tensor."""
    row = shape[-1] if len(shape) >= 2 else numel
    if target is None or row % target != 0:
        return row
    return target


def q_dq(seg, block, f16_scale):
    rows = seg.reshape(-1, block)
    amax = np.abs(rows).max(axis=1, keepdims=True)
    s = np.where(amax > 0, amax / 127.0, 1.0).astype(np.float32)
    if f16_scale:
        s = s.astype(np.float16).astype(np.float32)
        s[s == 0] = 1.0  # an f16 underflow would divide by zero
    q = np.clip(np.round(rows / s), -127, 127).astype(np.int8)
    return (q.astype(np.float32) * s).reshape(-1), s.size


def run(target, f16_scale, raw):
    out = raw.copy()
    nscales = 0
    for _name, shape, off, n in tensors():
        seg = raw[off:off + n]
        b = blocks_for(shape, n, target)
        out[off:off + n], k = q_dq(seg, b, f16_scale)
        nscales += k
    err = out - raw
    snr = 10 * np.log10((raw @ raw) / (err @ err + 1e-30))
    return out, nscales, snr


def main():
    raw = np.fromfile(BIN, dtype="<f4")
    n = raw.size
    print(f"{n:,} weights  ->  int8 payload {n / 1e6:.2f} MB  (f32 today: {n * 4 / 1e6:.2f} MB, "
          f"f16: {n * 2 / 1e6:.2f} MB)\n")

    print(f"  {'block':>8} {'scale':>6} {'scales':>9} {'scale MB':>9} {'total MB':>9} "
          f"{'vs f16':>8} {'weight SNR':>11}")
    print("  " + "-" * 68)
    f16_mb = n * 2 / 1e6
    for target in (None, 256, 128, 64, 32, 16):
        for f16s in (False, True):
            _, nsc, snr = run(target, f16s, raw)
            sbytes = nsc * (2 if f16s else 4)
            total = (n + sbytes) / 1e6
            lbl = "per-row" if target is None else str(target)
            print(f"  {lbl:>8} {'f16' if f16s else 'f32':>6} {nsc:>9,} {sbytes / 1e6:>8.3f}M "
                  f"{total:>8.2f}M {total / f16_mb:>7.0%} {snr:>8.1f} dB")


if __name__ == "__main__":
    main()
