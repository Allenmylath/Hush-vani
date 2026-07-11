"""One-time build step: flatten the ONNX initializers into a flat blob + manifest
so the Rust crate never has to parse protobuf.

  weights.bin  - little-endian f32 (or f16 with --f16), tensors concatenated in manifest order
  weights.txt  - "<graph>/<name> <shape,comma,sep> <offset_in_elements> <numel>"
                 preceded by "#dtype f16" when the blob is f16

  python export_weights.py           # f32, 9.12 MB
  python export_weights.py --f16     # f16, 4.56 MB (end-to-end SI-SDR 75.7 dB, inaudible)

Offsets/lengths are in *elements*, so they are identical between the two formats.
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
DT = np.dtype("<f2") if F16 else np.dtype("<f4")
ESZ = DT.itemsize

# --out NAME -> assets/NAME.{bin,txt}   (default: "weights")
NAME = "weights"
if "--out" in sys.argv:
    NAME = sys.argv[sys.argv.index("--out") + 1]

# Every tensor starts on a 64-byte boundary so the AVX2 kernels can use aligned loads.
ALIGN = 64 // ESZ  # elements

blob = bytearray()
lines = []
offset = 0
for g in ["enc", "erb_dec", "df_dec"]:
    m = onnx.load(os.path.join(BUNDLE, f"{g}.onnx"))
    for init in m.graph.initializer:
        pad = (-offset) % ALIGN
        if pad:
            blob += b"\0" * (pad * ESZ)
            offset += pad
        arr = numpy_helper.to_array(init)
        a = arr.astype(DT).ravel()
        shape = ",".join(str(d) for d in arr.shape)
        lines.append(f"{g}/{init.name} {shape} {offset} {a.size}")
        blob += a.tobytes()
        offset += a.size

header = ["#dtype f16"] if F16 else []
with open(os.path.join(OUT, f"{NAME}.bin"), "wb") as f:
    f.write(blob)
with open(os.path.join(OUT, f"{NAME}.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(header + lines) + "\n")

print(f"{len(lines)} tensors, {offset:,} elements, {len(blob)/1e6:.2f} MB "
      f"({'f16' if F16 else 'f32'}) -> {NAME}.bin")
