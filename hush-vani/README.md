# hush-vani

Real-time speech enhancement in pure Rust. Feed it noisy speech, get clean speech back.

**Weights are built in.** No model download, no ONNX runtime, no BLAS, no Python.

```toml
[dependencies]
hush-vani = "0.1"
```

```rust
use hush_vani::Hush;

let hush = Hush::new()?;              // 2.4 MB of weights, embedded in the crate
let clean = hush.enhance(&noisy)?;    // mono, 16 kHz, f32 in [-1, 1]
```

That's the whole API for the common case. On a typical clip it drops the noise floor by
**42 dB** while costing speech only −3.4 dB, and runs at **~100x realtime** on one core.

---

## Command line

```bash
cargo install hush-vani
hush-vani noisy.wav clean.wav
```

```
hush-vani <input.wav> <output.wav> [OPTIONS]

  -a, --atten <DB>      limit suppression to this many dB (e.g. 12)
  -w, --weights <DIR>   use weights from DIR instead of the built-in ones
  -h, --help
```

Input must be **mono 16 kHz** (16-bit or float wav).

## Audio format — read this first

Three things trip people up:

**1. Mono, 16 kHz, `f32` in `[-1, 1]`.** Anything else must be resampled and downmixed
first. The crate does not do it for you. Check with `Hush::SAMPLE_RATE`.

**2. The output lags the input by 160 samples (10 ms).** That is the model's algorithmic
latency — one hop — not a bug. To line the output up against the input, drop the first
`Hush::LATENCY_SAMPLES` samples:

```rust
let clean = hush.enhance(&noisy)?;
let aligned = &clean[Hush::LATENCY_SAMPLES..];   // now sample-aligned with `noisy`
```

**3. Audio is processed in whole 10 ms frames.** A trailing partial frame is dropped, so
`enhance` returns `audio.len() / 160 * 160` samples. Fewer than 160 samples in and you get
`Error::TooShort`.

## API

Loading — pick one:

| | |
|---|---|
| `Hush::new()` | the built-in int8 weights. **Use this.** |
| `Hush::from_paths(bin, manifest)` | your own `weights.bin` + `weights.txt` from disk |
| `Hush::from_bytes(bin, manifest)` | same, already in memory (e.g. `include_bytes!`) |

Running:

```rust
// Full suppression.
let clean = hush.enhance(&noisy)?;

// Cap suppression at ~12 dB by mixing the noisy signal back in. Full suppression can
// sound unnaturally dry; this is the usual fix.
let gentler = hush.enhance_with(&noisy, Some(12.0))?;

// Per-frame speech-vs-noise estimate in dB, one value per 10 ms. A cheap VAD signal —
// runs only the encoder, so it is much faster than a full enhance.
let snr: Vec<f32> = hush.local_snr(&noisy)?;
```

Constants: `Hush::SAMPLE_RATE` (16000), `Hush::LATENCY_SAMPLES` (160), `Hush::FRAME_SAMPLES`
(160). `Hush::is_accelerated()` reports whether the AVX2 kernels are live on this CPU.

`Hush` is `Send + Sync` and `enhance` takes `&self`, so one instance can be shared across
threads — each call allocates its own DSP state.

### Errors

`Error::TooShort` (under one 160-sample frame), `Error::Io`, `Error::Weights` /
`Error::MissingTensor` (a malformed or mismatched weight blob — you cannot hit these with
`Hush::new()`).

## No streaming yet

`enhance()` processes a **whole utterance**. The GRUs start from zero state on every call,
so feeding it successive chunks of a live stream would restart the recurrence each time and
degrade the output badly. Buffer the utterance, or wait for a streaming API — the GRU state
is already held in the right place, so it is a small change, just not one that's exposed.

## Performance

5 s clip, median of 15 runs, one modern x86-64 core:

| | 1 thread | 2 threads |
|---|---|---|
| total | 51 ms | 45 ms |
| per 10 ms frame | 0.10 ms | 0.09 ms |
| realtime factor | ~99x | ~112x |

Model load is a one-time ~10 ms. AVX2 + FMA kernels are selected **at runtime** — a stock
`cargo build --release` gets them, no `RUSTFLAGS` needed. Without AVX2 it falls back to
portable scalar code, several times slower.

For the last few percent, `RUSTFLAGS="-C target-cpu=native"` helps the remaining scalar code
(depthwise convs, reshapes) by ~8%.

## WebAssembly

It works, with no `wasm-bindgen` and no JS glue — the module has **zero imports**:

```toml
[dependencies]
hush-vani = { version = "0.1", default-features = false }   # drop the `cli` feature
```

```bash
cargo build --release --target wasm32-unknown-unknown
```

| | native (AVX2) | wasm |
|---|---|---|
| 5 s clip | 51 ms | **885 ms** |
| realtime factor | ~100x | **~5.7x** |
| module size | — | 2.8 MB (mostly weights) |

Correct output — it removes the same noise as native. **Fine for offline work** (denoise a
recorded clip in a browser); a live mic is tight but no longer out of reach.

Two things to know. There are no wasm SIMD kernels yet, so it runs the portable scalar path;
`-C target-feature=+simd128` does **not** help, because LLVM will not auto-vectorise a float
dot product (see [ENGINEERING.md](ENGINEERING.md) — the same reason the x86 kernels are
hand-written intrinsics). Explicit `core::arch::wasm32` v128 kernels are the obvious next
step and would slot in beside the AVX2 ones. And the 2.8 MB module is almost entirely
weights, which is exactly why they are int8 — f32 would make this a 9.5 MB download.

## Weights and precision

The built-in weights are the [`weya-ai/hush`](https://huggingface.co/weya-ai/hush) model
(Apache-2.0), quantised to **int8** — 2.44 MB, a quarter of the original float32.

int8 here is a **storage format, not a compute format**: the weights are dequantised to f32
at load and run through the same f32 kernels, so it is neither faster nor slower than f32.
It just makes the crate small. Measured over six real recordings:

| weights | noise removed | SI-SDR vs f32 | error level | size |
|---|---|---|---|---|
| f32 | +43.4 dB | *(reference)* | — | 9.12 MB |
| f16 | +43.4 dB | 67.0 dB | −94.6 dBFS | 4.56 MB |
| **int8** *(built in)* | **+43.4 dB** | 44.0 dB | −72.4 dBFS | **2.44 MB** |

**int8 removes exactly as much noise as full f32 weights** — the quantisation costs nothing
in the thing the model is actually for. What it costs is a noise floor at −72.4 dBFS:
inaudible at any normal listening level, but not below what a 16-bit WAV can store, the way
f16's is.

You almost certainly do not care. If you do — you want f16's extra 22 dB, or bit-exactness
with onnxruntime (129.7 dB SI-SDR) — export other weights and load them yourself:

```bash
pip install onnx
python tools/export_weights.py --f16 --out weights.f16   # or no flag for f32
```

```rust
let hush = Hush::from_paths("weights.f16.bin", "weights.f16.txt")?;
```

## Feature flags

| flag | default | what |
|---|---|---|
| `cli` | **on** | the `hush-vani` binary (pulls in `hound` for wav I/O) |
| `f16-kernels` | off | run the matmuls natively on f16 weights. Halves resident weight memory, ~10% *slower* — the kernels are issue-bound, not bandwidth-bound. A memory trade, not a speed one. |
| `devtools` | off | verification and benchmark binaries; needs the repo's `assets/` |

Library-only, no `hound`:

```toml
hush-vani = { version = "0.1", default-features = false }
```

## How it works

A from-scratch implementation of Hush (a DeepFilterNet3 derivative): STFT → ERB features →
encoder → ERB decoder + deep-filter decoder → deep filtering → overlap-add synthesis. The
ERB decoder predicts a coarse gain mask over 32 perceptual bands; the deep-filter decoder
predicts complex filter taps applied to the lowest 64 bins, which is what recovers detail a
plain mask would smear.

Output matches the reference ONNX Runtime pipeline to float32 rounding (**SI-SDR 129.7 dB**)
when run on f32 weights.

For how it was built and optimised — including the measurements, and the "obvious"
optimisations that lost — see **[ENGINEERING.md](ENGINEERING.md)**.

## Licence

Apache-2.0, matching the upstream model. The embedded weights are the
[`weya-ai/hush`](https://huggingface.co/weya-ai/hush) model redistributed under its own
Apache-2.0 licence; the container format changes and the values are quantised to int8 as
described above. See `LICENSE` and `NOTICE`. Not affiliated with or endorsed by Weya AI.
