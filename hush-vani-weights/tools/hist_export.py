"""Compute histogram / CDF data for the weight distribution and dump it as JSON for the viz."""
import json
import os
import sys

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BIN = os.path.join(HERE, "data", "weights.bin")
MAN = os.path.join(HERE, "data", "weights.txt")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hist.json")


def kind_of(name):
    if "GRU" in name:
        return "gru"
    if "linear" in name or "MatMul" in name or "df_out" in name or "fc" in name:
        return "linear"
    return "conv"


def main():
    raw = np.fromfile(BIN, dtype="<f4").astype(np.float64)
    tid_kind = {}
    kinds = np.empty(raw.size, dtype=object)
    for line in open(MAN):
        line = line.strip()
        if not line:
            continue
        head, off, n = line.rsplit(" ", 2)
        name, _ = head.rsplit(" ", 1)
        off, n = int(off), int(n)
        kinds[off:off + n] = kind_of(name)

    data = {}
    data["stats"] = {
        "n": int(raw.size),
        "mean": float(raw.mean()),
        "median": float(np.median(raw)),
        "std": float(raw.std()),
        "min": float(raw.min()),
        "max": float(raw.max()),
        "p1": float(np.percentile(raw, 1)),
        "p99": float(np.percentile(raw, 99)),
        "kurt_excess": float(((raw - raw.mean()) ** 4).mean() / raw.std() ** 4 - 3),
    }

    # 1) full range, log-y counts
    c, e = np.histogram(raw, bins=181, range=(raw.min(), raw.max()))
    data["full"] = {"edges": e.round(4).tolist(), "counts": c.tolist()}

    # 2) central zoom [-0.3, 0.3], density
    c, e = np.histogram(raw, bins=160, range=(-0.3, 0.3), density=True)
    data["central"] = {"edges": e.round(5).tolist(), "density": c.round(3).tolist()}

    # 3) per-kind central density (comparable shapes)
    data["per_kind"] = {}
    for k in ("gru", "linear", "conv"):
        seg = raw[kinds == k]
        c, e = np.histogram(seg, bins=120, range=(-0.3, 0.3), density=True)
        data["per_kind"][k] = {
            "edges": e.round(5).tolist(),
            "density": c.round(3).tolist(),
            "n": int(seg.size),
            "std": float(seg.std()),
        }

    # 4) CDF of |w|: fraction with |w| <= x
    a = np.abs(raw)
    xs = np.concatenate([np.linspace(0, 0.5, 120), np.linspace(0.5, float(a.max()), 40)[1:]])
    cdf = np.searchsorted(np.sort(a), xs, side="right") / a.size
    data["cdf_abs"] = {"x": xs.round(4).tolist(), "y": cdf.round(5).tolist()}

    # 5) end-to-end results measured earlier, for the compression panel
    data["compress"] = [
        {"scheme": "f32 (current)", "mb": 9.12, "gz": 8.55, "sisdr": 129.7},
        {"scheme": "f16", "mb": 4.56, "gz": 4.21, "sisdr": 75.7},
        {"scheme": "int8 / tensor", "mb": 2.28, "gz": 1.46, "sisdr": 29.8},
        {"scheme": "int8 / global", "mb": 2.28, "gz": 0.21, "sisdr": 9.7},
    ]

    json.dump(data, open(OUT, "w"))
    print("wrote", OUT, f"({os.path.getsize(OUT)/1024:.0f} KB)")


if __name__ == "__main__":
    main()
