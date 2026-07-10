import onnx, os, collections

base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bundle")
for name in ["enc.onnx", "erb_dec.onnx", "df_dec.onnx"]:
    m = onnx.load(os.path.join(base, name))
    ops = collections.Counter(n.op_type for n in m.graph.node)
    params = sum(
        int.__mul__(*[1, 1]) or 0 for _ in []
    )
    nparam = 0
    for init in m.graph.initializer:
        n = 1
        for d in init.dims:
            n *= d
        nparam += n
    print(f"{name:14s} opset={m.opset_import[0].version} nodes={len(m.graph.node):3d} "
          f"params={nparam/1e6:.3f}M")
    print("   ops:", dict(ops.most_common()))
    print()
