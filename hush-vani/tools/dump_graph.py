"""Dump full ONNX graph structure so the forward pass can be hand-written."""
import os
import sys

import onnx
from onnx import numpy_helper

BUNDLE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ort", "bundle")


def attrs(n):
    out = {}
    for a in n.attribute:
        if a.type == onnx.AttributeProto.INT:
            out[a.name] = a.i
        elif a.type == onnx.AttributeProto.INTS:
            out[a.name] = list(a.ints)
        elif a.type == onnx.AttributeProto.FLOAT:
            out[a.name] = a.f
        elif a.type == onnx.AttributeProto.STRING:
            out[a.name] = a.s.decode()
        elif a.type == onnx.AttributeProto.TENSOR:
            t = numpy_helper.to_array(a.t)
            out[a.name] = f"<tensor {t.shape} {t.dtype}>" if t.size > 8 else t.tolist()
        else:
            out[a.name] = f"<type {a.type}>"
    return out


def main(name):
    m = onnx.load(os.path.join(BUNDLE, name))
    g = m.graph
    inits = {i.name: numpy_helper.to_array(i) for i in g.initializer}
    print(f"##### {name}  (opset {m.opset_import[0].version})")
    print("# initializers:")
    for k, v in inits.items():
        if v.size <= 8:
            print(f"#   {k:38s} {str(v.shape):16s} {v.dtype}  = {v.reshape(-1).tolist()}")
        else:
            print(f"#   {k:38s} {str(v.shape):16s} {v.dtype}")
    print("# graph inputs :", [i.name for i in g.input])
    print("# graph outputs:", [o.name for o in g.output])
    print("# nodes:")
    for i, n in enumerate(g.node):
        a = attrs(n)
        astr = " " + str(a) if a else ""
        print(f"{i:3d} {n.op_type:14s} {list(n.input)} -> {list(n.output)}{astr}")


if __name__ == "__main__":
    name = sys.argv[1]
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), name.replace(".onnx", ".txt"))
    with open(out, "w", encoding="utf-8") as f:
        sys.stdout = f
        main(name)
    sys.stdout = sys.__stdout__
    print("wrote", out)
