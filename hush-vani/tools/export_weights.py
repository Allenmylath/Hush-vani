"""One-time build step: flatten the ONNX initializers into a flat f32 blob + manifest
so the Rust crate never has to parse protobuf.

  weights.bin  - little-endian f32, tensors concatenated in manifest order
  weights.txt  - "<graph>/<name> <shape,comma,sep> <offset_in_floats> <numel>"
"""
import os
import struct

import onnx
from onnx import numpy_helper

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
BUNDLE = os.path.join(ROOT, "ort", "bundle")
OUT = os.path.join(os.path.dirname(HERE), "assets")
os.makedirs(OUT, exist_ok=True)

ALIGN = 16  # floats -> every tensor starts on a 64-byte boundary, so AVX2 aligned loads
            # are legal on each weight row (all row strides here are multiples of 16 floats)

blob = bytearray()
lines = []
offset = 0
for g in ["enc", "erb_dec", "df_dec"]:
    m = onnx.load(os.path.join(BUNDLE, f"{g}.onnx"))
    for init in m.graph.initializer:
        pad = (-offset) % ALIGN
        if pad:
            blob += b"\0" * (pad * 4)
            offset += pad
        a = numpy_helper.to_array(init).astype("<f4").ravel()
        shape = ",".join(str(d) for d in numpy_helper.to_array(init).shape)
        lines.append(f"{g}/{init.name} {shape} {offset} {a.size}")
        blob += a.tobytes()
        offset += a.size

with open(os.path.join(OUT, "weights.bin"), "wb") as f:
    f.write(blob)
with open(os.path.join(OUT, "weights.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")

print(f"{len(lines)} tensors, {offset} floats, {len(blob)/1e6:.2f} MB")
print(f"-> {OUT}")
