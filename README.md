# Hush-vani

A from-scratch Rust implementation of the [Hush](https://huggingface.co/weya-ai/hush) speech
enhancement model (a DeepFilterNet3 derivative), and the ONNX Runtime reference pipeline it
was validated against.

| directory | what |
|---|---|
| [`hush-vani/`](hush-vani/) | **the crate** — pure-Rust model, runtime-dispatched AVX2 kernels, no ONNX runtime, no BLAS |
| [`ort/`](ort/) | the reference: Python + `onnxruntime`, bit-accurate against the model author's published audio |

The Rust output matches the ONNX Runtime pipeline to float32 rounding (**SI-SDR 129.7 dB**),
and is faster: **1.07x on the neural network**, 1.12x on the full pipeline, single-threaded,
measured with a paired interleaved benchmark. See [`hush-vani/README.md`](hush-vani/README.md)
for the numbers and how they were obtained.

On the reference clip the model drops the noise floor by **42 dB** while costing speech only
−3.4 dB, at ~110x realtime on one core (5 s of audio in ~45 ms).

## Quick start (the crate)

```toml
[dependencies]
hush-vani = "0.1"
```

```rust
use hush_vani::Hush;

let hush = Hush::from_paths("weights.bin", "weights.txt")?;
let clean = hush.enhance(&noisy)?;   // mono 16 kHz f32 in [-1, 1]
# Ok::<(), hush_vani::Error>(())
```

## Setting up this repo

Two things are deliberately **not** vendored here: the model weights (they belong to
`weya-ai/hush`) and the ONNX bundle they are derived from. Fetch and convert them:

```bash
# 1. the published ONNX bundle -> ort/bundle/
mkdir -p ort/bundle
curl -L -o ort/onnx.tar.gz \
  https://huggingface.co/weya-ai/hush/resolve/main/onnx/advanced_dfnet16k_model_best_onnx.tar.gz
tar -xzf ort/onnx.tar.gz -C ort/bundle

# 2. flat f32 weight arena for the Rust crate -> hush-vani/assets/
pip install onnx numpy soundfile onnxruntime deepfilterlib
python hush-vani/tools/export_weights.py

# 3. (optional) ORT intermediates, so `hush-devtool verify` can check every tensor
python hush-vani/tools/export_fixtures.py
```

Then:

```bash
cd hush-vani
cargo build --release --features devtools
./target/release/hush-devtool verify        # vs ORT + libdf, tensor by tensor
./target/release/hush-vani sample_raw.wav clean.wav --weights ./assets
```

## Benchmarks

```bash
cd hush-vani
./target/release/hush-devtool bench 5   # 1 and 2 threads
python tools/ab_bench.py                # paired, interleaved, Rust vs onnxruntime
./target/release/hush-kernel-ab         # paired old-vs-new kernel A/B
./target/release/hush-profile           # per-kernel GFMA/s
```

Timing on this class of machine drifts ~20% between sessions, which is larger than the
Rust-vs-ORT margin. Every comparison in this repo is therefore **paired** — the two
implementations run interleaved and the ratio is taken within each turn. `tools/ab_bench.py`
reports a bootstrap CI and a Mann-Whitney U test. Unpaired medians repeatedly gave the wrong
answer during development; the write-up records those cases.

## Licence

Apache-2.0, matching the upstream model. The weights are not redistributed here and remain
under the terms of [`weya-ai/hush`](https://huggingface.co/weya-ai/hush).
