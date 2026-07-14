"""Collect every input and output into one neatly labelled `audio/` folder, with a README
and a machine-readable manifest.

Layout:
    audio/
      README.md                     what everything is, plus the metrics table
      manifest.csv                  same, machine-readable
      00_hush_demo_clip/            the 5 s clip from the model repo
        input_noisy.wav
        hush_f32.wav / hush_f16.wav / hush_int8.wav
      01_munching/ .. 06_water/     the DeepFilterNet2 demo recordings
        input_noisy_16k.wav         (resampled 48 -> 16 kHz; Hush is a 16 kHz model)
        hush_f32.wav / hush_f16.wav / hush_int8.wav
        reference_deepfilternet2_48k.wav   <- a DIFFERENT model, for context only
"""
import csv
import glob
import json
import os
import shutil
import sys

import numpy as np
import soundfile as sf

sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__))
CRATE = os.path.dirname(HERE)
ROOT = os.path.dirname(CRATE)
BS = os.path.join(ROOT, "bench_samples")
CMP = os.path.join(ROOT, "comparison")
OUT = os.path.join(ROOT, "audio")

PRECS = ("f32", "f16", "int8")
NOISE = {"01": "munching", "02": "door", "03": "aircon",
         "04": "chips", "05": "street", "06": "water"}


def db(x):
    return 20 * np.log10(max(float(x), 1e-12))


def stats(path):
    x, sr = sf.read(path, dtype="float32")
    if x.ndim > 1:
        x = x.mean(1)
    return {"sr": sr, "dur": round(len(x) / sr, 1), "rms": round(db(np.sqrt(np.mean(x**2))), 1)}


def main():
    if os.path.isdir(OUT):
        shutil.rmtree(OUT)
    os.makedirs(OUT)

    res = {r["id"]: r for r in json.load(open(os.path.join(BS, "results.json")))["results"]}
    rows = []

    # ---- 00: the original demo clip from the model repo ----
    d = os.path.join(OUT, "00_hush_demo_clip")
    os.makedirs(d)
    shutil.copy(os.path.join(CMP, "0_noisy_input.wav"), os.path.join(d, "input_noisy.wav"))
    for p in PRECS:
        n = {"f32": "1", "f16": "2", "int8": "3"}[p]
        shutil.copy(os.path.join(CMP, f"{n}_enhanced_{p}.wav"), os.path.join(d, f"hush_{p}.wav"))
    rows.append({
        "folder": "00_hush_demo_clip", "source": "weya-ai/hush model repo", "noise": "mixed",
        "duration_s": stats(os.path.join(d, "input_noisy.wav"))["dur"],
        "input_snr_db": "", "f32_noise_removed_db": 42.0, "f16_noise_removed_db": 42.0,
        "int8_noise_removed_db": 39.5, "f16_error_dbfs": -99.9, "int8_error_dbfs": -59.0,
        "f16_sisdr_db": 70.6, "int8_sisdr_db": 29.8,
    })

    # ---- 01..06: DeepFilterNet2 demo recordings ----
    for sid, noise in NOISE.items():
        d = os.path.join(OUT, f"{sid}_{noise}")
        os.makedirs(d)
        shutil.copy(os.path.join(BS, "in16k", f"{sid}_noisy16k.wav"),
                    os.path.join(d, "input_noisy_16k.wav"))
        for p in PRECS:
            shutil.copy(os.path.join(BS, p, f"{sid}_{p}.wav"), os.path.join(d, f"hush_{p}.wav"))
        ref = glob.glob(os.path.join(BS, "df2_ref", f"{sid}_*.wav"))
        if ref:
            shutil.copy(ref[0], os.path.join(d, "reference_deepfilternet2_48k.wav"))
        r = res[sid]
        rows.append({
            "folder": f"{sid}_{noise}", "source": "DeepFilterNet2 demo set", "noise": noise,
            "duration_s": r["dur"], "input_snr_db": r["in_snr"],
            "f32_noise_removed_db": r["f32"]["reduction"],
            "f16_noise_removed_db": r["f16"]["reduction"],
            "int8_noise_removed_db": r["int8"]["reduction"],
            "f16_error_dbfs": r["f16"]["resid"], "int8_error_dbfs": r["int8"]["resid"],
            "f16_sisdr_db": r["f16"]["sisdr"], "int8_sisdr_db": r["int8"]["sisdr"],
        })

    # ---- manifest.csv ----
    with open(os.path.join(OUT, "manifest.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # ---- README.md ----
    dfn = [r for r in rows if r["input_snr_db"] != ""]
    avg = lambda k: sum(r[k] for r in dfn) / len(dfn)
    tbl = "\n".join(
        f"| `{r['folder']}` | {r['noise']} | {r['input_snr_db'] or '—'} | "
        f"{r['f32_noise_removed_db']:+.1f} | {r['f16_noise_removed_db']:+.1f} | "
        f"{r['int8_noise_removed_db']:+.1f} | {r['f16_error_dbfs']:.0f} | {r['int8_error_dbfs']:.0f} |"
        for r in rows)

    readme = f"""# audio/ — every input and output, labelled

The same Hush model run three times over each recording. **Only the weight precision
changes** — identical code, identical architecture.

## What's in each folder

| file | what it is |
|---|---|
| `input_noisy*.wav` | the recording fed to the model |
| `hush_f32.wav` | **reference** — full-precision weights (9.12 MB) |
| `hush_f16.wav` | **what ships** — half-precision weights (4.56 MB) |
| `hush_int8.wav` | **rejected** — 8-bit weights (2.28 MB) |
| `reference_deepfilternet2_48k.wav` | a *different* model, for context only (see note) |

## Results

Noise removed = drop in the noise floor (p10 frame energy) vs the input.
Error = level of the signal that quantisation *added* on top of the f32 output.

| folder | noise | input SNR | f32 | f16 | int8 | f16 error | int8 error |
|---|---|---|---|---|---|---|---|
{tbl}

Averaged over the six DeepFilterNet2 recordings: f32 **{avg('f32_noise_removed_db'):+.1f} dB**,
f16 **{avg('f16_noise_removed_db'):+.1f} dB**, int8 **{avg('int8_noise_removed_db'):+.1f} dB**.

## What to listen for

**f16 vs f32: you will not hear a difference — there isn't one to hear.** It tracks f32 to
within 0.1 dB on every sample, and its error sits around −94 dBFS, at or below the −96 dBFS
noise floor of a 16-bit WAV. The difference is not merely quiet; it cannot be *stored* in the
output file.

**int8 loses twice.** It adds a broadband hiss (≈ −57 dBFS, ~37 dB above the 16-bit floor —
clearly audible) *and* it removes ~2 dB less noise. The quantised weights make the model worse
at its job, then add grit on top. The gap is widest exactly where the model is strongest: on
`01_munching`, f32 strips 72.2 dB of noise and int8 manages only 63.8 dB.

The cause is the weight distribution: a handful of ±31 outliers force a per-tensor int8 scale
that wastes the range on the 99 % of weights living inside ±0.23. Per-channel scales would
likely fix it — and would unlock int8's measured 3.5× kernel speedup.

## Notes on the sources

- `00_hush_demo_clip` is the sample published with the [Hush model](https://huggingface.co/weya-ai/hush).
- `01`–`06` are from the [DeepFilterNet2 demo set](https://rikorose.github.io/DeepFilterNet2-Samples/),
  resampled 48 → 16 kHz because **Hush is a 16 kHz model**.
- `reference_deepfilternet2_48k.wav` is DeepFilterNet2's own output at full 48 kHz band. It is
  a **different model at twice the bandwidth** — included so you can hear the recordings
  handled well, *not* as a fair head-to-head. Comparing it directly against `hush_*.wav` would
  be unfair to both.

Regenerate: `python hush-vani/tools/bench_samples.py && python hush-vani/tools/organize_audio.py`
"""
    open(os.path.join(OUT, "README.md"), "w", encoding="utf-8").write(readme)

    n = sum(len(glob.glob(os.path.join(OUT, "*", "*.wav"))) for _ in [0])
    print(f"wrote {OUT}")
    for d in sorted(os.listdir(OUT)):
        p = os.path.join(OUT, d)
        if os.path.isdir(p):
            print(f"  {d}/")
            for f in sorted(os.listdir(p)):
                print(f"      {f}")
    print(f"\n{n} wav files + README.md + manifest.csv")


if __name__ == "__main__":
    main()
