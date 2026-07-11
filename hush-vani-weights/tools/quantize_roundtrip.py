"""Write quantized-then-dequantized copies of the weight arena, so the Rust verifier can
measure the *audio* impact (end-to-end SI-SDR) rather than just per-tensor SNR."""

import os
import sys

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BIN = os.path.join(HERE, "data", "weights.bin")
MAN = os.path.join(HERE, "data", "weights.txt")


def tensors():
    for line in open(MAN):
        line = line.strip()
        if not line:
            continue
        head, off, n = line.rsplit(" ", 2)
        name, _shape = head.rsplit(" ", 1)
        yield name, int(off), int(n)


def main():
    raw = np.fromfile(BIN, dtype="<f4")

    # f16 round-trip
    f16 = raw.astype(np.float16).astype(np.float32)
    f16.tofile(os.path.join(HERE, "data", "weights_f16.bin"))

    # int8 per-tensor symmetric round-trip
    q = raw.copy()
    for _name, off, n in tensors():
        seg = raw[off:off + n]
        amax = np.abs(seg).max()
        s = amax / 127.0 if amax > 0 else 1.0
        q[off:off + n] = np.clip(np.round(seg / s), -127, 127) * s
    q.astype("<f4").tofile(os.path.join(HERE, "data", "weights_int8.bin"))

    print("wrote weights_f16.bin and weights_int8.bin (both f32-valued, dequantized)")
    print("verify each with: hush-devtool verify  (after swapping into assets/weights.bin)")


if __name__ == "__main__":
    main()
