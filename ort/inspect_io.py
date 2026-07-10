import onnxruntime as ort
import os

base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bundle")

for name in ["enc.onnx", "erb_dec.onnx", "df_dec.onnx"]:
    p = os.path.join(base, name)
    s = ort.InferenceSession(p, providers=["CPUExecutionProvider"])
    print("=" * 70)
    print(name)
    print("=" * 70)
    print("  INPUTS:")
    for i in s.get_inputs():
        print(f"    {i.name:24s} {str(i.shape):32s} {i.type}")
    print("  OUTPUTS:")
    for o in s.get_outputs():
        print(f"    {o.name:24s} {str(o.shape):32s} {o.type}")
    print()
