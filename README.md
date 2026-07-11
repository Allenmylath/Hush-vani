# Hush-vani

A from-scratch Rust implementation of the [Hush](https://huggingface.co/weya-ai/hush) speech
enhancement model (a DeepFilterNet3 derivative), and the ONNX Runtime reference pipeline it
was validated against.

A Cargo workspace of two crates, plus the reference pipeline they were validated against:

| directory | crate | what |
|---|---|---|
| [`hush-vani-core/`](hush-vani-core/) | `hush-vani-core` | shared DSP + neural kernels + the **encoder** (first stage) |
| [`hush-vani/`](hush-vani/) | `hush-vani` | the **decoders** (second stage), the merge, and the `Hush` API — the crate users install |
| [`hush-vani-weights/`](hush-vani-weights/) | `hush-vani-weights` | the model weights as an embeddable data blob (Apache-2.0, from weya-ai/hush) |
| [`ort/`](ort/) | — | the reference: Python + `onnxruntime`, bit-accurate against the model author's published audio |

`hush-vani` depends on `hush-vani-core`. The split follows the model: the encoder produces a
shared embedding that both decoders consume, so it is a clean, dependency-free first stage.
`hush-vani-weights` is a standalone data crate, pulled in only when you enable the `bundled`
feature. Everything is runtime-dispatched AVX2, no ONNX runtime, no BLAS.

The Rust output matches the ONNX Runtime pipeline to float32 rounding (**SI-SDR 129.7 dB**),
and is faster: **1.07x on the neural network**, 1.12x on the full pipeline, single-threaded,
measured with a paired interleaved benchmark. See [`hush-vani/README.md`](hush-vani/README.md)
for the numbers and how they were obtained.

On the reference clip the model drops the noise floor by **42 dB** while costing speech only
−3.4 dB, at ~110x realtime on one core (5 s of audio in ~45 ms).

## Quick start (the crate)

Easiest — embed the weights, no files to manage:

```toml
[dependencies]
hush-vani = { version = "0.1", features = ["bundled"] }
```

```rust
let hush = hush_vani::Hush::bundled()?;      // ~4.6 MB of f16 weights baked into the binary
let clean = hush.enhance(&noisy)?;           // mono 16 kHz f32 in [-1, 1]
```

The bundled weights are **float16** — half the size, widened to f32 at load, so no speed cost.
Output differs from full-f32 weights by 75.5 dB SI-SDR, above what a 16-bit WAV can represent.
For bit-exactness against onnxruntime (129.7 dB), export f32 weights and use `from_paths`.

Or load weights from a file at runtime (no `bundled` feature, smaller binary):

```rust
use hush_vani::Hush;
let hush = Hush::from_paths("weights.bin", "weights.txt")?;
let clean = hush.enhance(&noisy)?;
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

## Publishing

Three crates, published in dependency order. `cargo publish` resolves every declared
dependency (including optional ones) against crates.io, so the leaves go first and
`hush-vani` — which depends on both — goes last:

```bash
cargo publish -p hush-vani-core      # no dependencies
cargo publish -p hush-vani-weights   # no dependencies (pure data, ~8 MB)
cargo publish -p hush-vani           # depends on both; publish after they are live
```

`hush-vani-core` and `hush-vani-weights` are independent and can go in either order. Wait a
few seconds for each to index before the next. Confirm the names are free first with
`cargo publish -p hush-vani-core --dry-run` (the dry-run on `hush-vani` will report its
unpublished deps as "not found" — expected until the leaves are live).

## Licence

Apache-2.0, matching the upstream model. The weights are not redistributed here and remain
under the terms of [`weya-ai/hush`](https://huggingface.co/weya-ai/hush).
