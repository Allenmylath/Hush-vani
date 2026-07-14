# Hush-vani

A from-scratch Rust implementation of the [Hush](https://huggingface.co/weya-ai/hush) speech
enhancement model (a DeepFilterNet3 derivative), and the ONNX Runtime reference pipeline it
was validated against.

**One crate, weights included.**

| directory | crate | what |
|---|---|---|
| [`hush-vani/`](hush-vani/) | `hush-vani` | the whole thing: DSP, kernels, encoder, decoders, the `Hush` API, and the embedded int8 weights |
| [`ort/`](ort/) | — | the reference: Python + `onnxruntime`, bit-accurate against the model author's published audio |

Runtime-dispatched AVX2, no ONNX runtime, no BLAS, no Python at inference.

The Rust output matches the ONNX Runtime pipeline to float32 rounding (**SI-SDR 129.7 dB**),
and is faster: **1.07x on the neural network**, 1.12x on the full pipeline, single-threaded,
measured with a paired interleaved benchmark.

On the reference clip the model drops the noise floor by **42 dB** while costing speech only
−3.4 dB, at ~100x realtime on one core.

- **Using the crate** → [`hush-vani/README.md`](hush-vani/README.md)
- **How it was built, measured, and optimised** (including the experiments that lost) →
  [`hush-vani/ENGINEERING.md`](hush-vani/ENGINEERING.md)

## Quick start

```toml
[dependencies]
hush-vani = "0.1"
```

```rust
use hush_vani::Hush;

let hush = Hush::new()?;             // weights are embedded — nothing to download
let clean = hush.enhance(&noisy)?;   // mono 16 kHz f32 in [-1, 1]
```

The embedded weights are **int8** — 2.44 MB, a quarter of the original f32. They are
dequantised to f32 at load and run through the f32 kernels, so int8 is a storage format, not
a compute one: same speed, and on six real recordings it removes **+43.4 dB of noise, exactly
matching full f32 weights**. See [the weights section](hush-vani/README.md#weights) for what
it does cost (a −72.4 dBFS noise floor) and how the block scales were chosen.

## Setting up this repo

Two things are deliberately **not** vendored here: the f32/f16 weight blobs and the ONNX
bundle they are derived from. (The int8 blob *is* vendored — it is what the crate ships.)

```bash
# 1. the published ONNX bundle -> ort/bundle/
mkdir -p ort/bundle
curl -L -o ort/onnx.tar.gz \
  https://huggingface.co/weya-ai/hush/resolve/main/onnx/advanced_dfnet16k_model_best_onnx.tar.gz
tar -xzf ort/onnx.tar.gz -C ort/bundle

# 2. weight blobs -> hush-vani/assets/
pip install onnx numpy soundfile onnxruntime deepfilterlib
python hush-vani/tools/export_weights.py                       # f32,  9.12 MB
python hush-vani/tools/export_weights.py --f16  --out weights.f16
python hush-vani/tools/export_weights.py --int8 --out weights.int8   # what ships

# 3. (optional) ORT intermediates, so `hush-devtool verify` can check every tensor
python hush-vani/tools/export_fixtures.py
```

Then:

```bash
cargo build --release --features devtools
./target/release/hush-devtool verify        # vs ORT + libdf, tensor by tensor
./target/release/hush-vani hush-vani/sample_raw.wav clean.wav
```

## Benchmarks

```bash
./target/release/hush-devtool bench 5           # 1 and 2 threads
python hush-vani/tools/ab_bench.py              # paired, interleaved, Rust vs onnxruntime
python hush-vani/tools/bench_samples.py         # f32/f16/int8 over 6 real recordings
./target/release/hush-kernel-ab                 # paired old-vs-new kernel A/B
./target/release/hush-profile                   # per-kernel GFMA/s
```

Timing on this class of machine drifts ~20% between sessions, which is larger than the
Rust-vs-ORT margin. Every comparison in this repo is therefore **paired** — the two
implementations run interleaved and the ratio is taken within each turn. `tools/ab_bench.py`
reports a bootstrap CI and a Mann-Whitney U test. Unpaired medians repeatedly gave the wrong
answer during development; the write-up records those cases.

## Publishing

One crate, no dependency ordering to worry about:

```bash
cargo publish -p hush-vani --dry-run    # check it packages (2.3 MB compressed)
cargo publish -p hush-vani
```

## Licence

Apache-2.0, matching the upstream model. The embedded weights are the
[`weya-ai/hush`](https://huggingface.co/weya-ai/hush) model redistributed under its own
Apache-2.0 licence; see `hush-vani/NOTICE`. Not affiliated with or endorsed by Weya AI.
