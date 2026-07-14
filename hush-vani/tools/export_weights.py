"""One-time build step: flatten the ONNX initializers into a flat blob + manifest
so the Rust crate never has to parse protobuf.

  weights.bin  - little-endian tensors, concatenated in manifest order
  weights.txt  - "<graph>/<name> <shape,comma,sep> <offset_in_elements> <numel>"

  python export_weights.py           # f32,  9.12 MB  (reference)
  python export_weights.py --f16     # f16,  4.56 MB  (error -94 dBFS, below the 16-bit floor)
  python export_weights.py --int8    # int8, 2.36 MB  (error -72 dBFS, same noise removal)

Each tensor starts on a 64-byte boundary so the AVX2 kernels can use aligned loads. The
padding is counted in *elements*, so offsets differ between formats (f32 pads to 16
elements, f16 to 32, int8 to 64) -- every manifest is only valid for its own blob.

int8 layout
-----------
Header is "#dtype int8 payload <n_elements>". The blob is then

    [ int8 payload : n_elements bytes ] [ f32 scales : 4 bytes each ]

and each manifest line gains two fields:

    <name> <shape> <off> <numel> <scale_off> <block>

`block` weights share one scale: dequantise with w[i] = q[i] * scale[scale_off + i/block].

Blocks run along the CONTRACTION (last) axis, so a scale covers part of one matmul row and
never straddles two. Block 64 where the row length allows it, else one scale for the row.

Why not one scale per tensor, which is what the first int8 experiment did: the GRU weights
carry +/-31 outliers while 99% of their values sit inside +/-0.23, so a single scale spends
the int8 range on a handful of entries. That cost 8.4 dB of noise reduction on the munching
sample -- the model got measurably worse at its job. Block-wise scales give that back in
full (see tools/quant_schemes.py).

Why f32 scales and not f16, which would save 80 KB: a scale is amax/127, and for the blocks
whose weights are all around 1e-6 that lands under f16's smallest subnormal. 425 of the
39,688 scales flush to zero in f16 and another 350 go subnormal, which quietly zeroes or
mangles the 64 weights each one covers. It does not move the aggregate SNR -- those weights
are tiny -- but the on-disk format is permanent once published, and 80 KB on a 2.4 MB crate
is not worth carrying a whole class of underflow bug.
"""
import os
import sys

import numpy as np
import onnx
from onnx import numpy_helper

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
BUNDLE = os.path.join(ROOT, "ort", "bundle")
OUT = os.path.join(os.path.dirname(HERE), "assets")
os.makedirs(OUT, exist_ok=True)

F16 = "--f16" in sys.argv
INT8 = "--int8" in sys.argv
if F16 and INT8:
    sys.exit("--f16 and --int8 are mutually exclusive")

DTYPE = "int8" if INT8 else ("f16" if F16 else "f32")
ESZ = 1 if INT8 else (2 if F16 else 4)
ALIGN = 64 // ESZ  # elements, so every tensor starts 64-byte aligned

# The int8 block length. 64 keeps the scale table small (79 KB) while letting the AVX-VNNI
# kernel fold two vpdpbusd into one accumulator before it has to apply a scale.
BLOCK = 64

NAME = "weights"
if "--out" in sys.argv:
    NAME = sys.argv[sys.argv.index("--out") + 1]


def block_len(shape, numel):
    """Weights sharing one scale. The contraction axis is the last one."""
    row = shape[-1] if len(shape) >= 2 else numel
    return BLOCK if row % BLOCK == 0 else row


def quantize(a, shape):
    """Symmetric int8 with one f32 scale per block. Returns (int8 codes, f32 scales, block)."""
    b = block_len(shape, a.size)
    rows = a.reshape(-1, b)
    amax = np.abs(rows).max(axis=1, keepdims=True)
    # An all-zero block (the model has a few numerically dead tensors, ~1e-9) would divide by
    # zero; give it scale 1 and it quantises to all-zero codes, which is exactly right.
    s = np.where(amax > 0, amax / 127.0, 1.0).astype(np.float32)
    q = np.clip(np.round(rows / s), -127, 127).astype(np.int8)
    return q.reshape(-1), s.reshape(-1), b


blob = bytearray()
scales = []
lines = []
offset = 0     # in elements
scale_off = 0  # in f16 scales

for g in ["enc", "erb_dec", "df_dec"]:
    m = onnx.load(os.path.join(BUNDLE, f"{g}.onnx"))
    for init in m.graph.initializer:
        pad = (-offset) % ALIGN
        if pad:
            blob += b"\0" * (pad * ESZ)
            offset += pad
        arr = numpy_helper.to_array(init)
        shape = list(arr.shape)
        shape_s = ",".join(str(d) for d in shape)
        flat = arr.astype(np.float32).ravel()

        if INT8:
            q, s, b = quantize(flat, shape)
            blob += q.tobytes()
            scales.append(s)
            lines.append(f"{g}/{init.name} {shape_s} {offset} {q.size} {scale_off} {b}")
            scale_off += s.size
            offset += q.size
        else:
            a = flat.astype(np.dtype("<f2") if F16 else np.dtype("<f4"))
            blob += a.tobytes()
            lines.append(f"{g}/{init.name} {shape_s} {offset} {a.size}")
            offset += a.size

if INT8:
    header = [f"#dtype int8 payload {offset}"]
    blob += np.concatenate(scales).astype("<f4").tobytes()
elif F16:
    header = ["#dtype f16"]
else:
    header = []

with open(os.path.join(OUT, f"{NAME}.bin"), "wb") as f:
    f.write(blob)
with open(os.path.join(OUT, f"{NAME}.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(header + lines) + "\n")

extra = f", {scale_off:,} scales ({scale_off * 4 / 1e3:.0f} KB)" if INT8 else ""
print(f"{len(lines)} tensors, {offset:,} elements{extra}, {len(blob) / 1e6:.2f} MB "
      f"({DTYPE}) -> {NAME}.bin")
