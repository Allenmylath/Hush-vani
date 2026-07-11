# hush-vani-core

Core of [`hush-vani`](https://crates.io/crates/hush-vani): the shared DSP and neural
primitives, plus the **encoder** — the first stage of the Hush speech-enhancement model
(a DeepFilterNet3 derivative).

**Most users want [`hush-vani`](https://crates.io/crates/hush-vani)**, which builds the full
enhancer (`Hush`) on top of this crate. Depend on `hush-vani-core` directly only to reuse:

- `dsp` — vorbis-window STFT (`rfft/N`), 32-band ERB features, exponential normalisers,
  overlap-add ISTFT, and the spectral merge helpers (`erb_inv`, `apply_df`). Formulas
  verified numerically against DeepFilterNet's `libdf`.
- `nn` — `dot` / `matvec` / `gemm_nt`, grouped/depthwise/pointwise convolutions, depthwise
  frequency ConvTranspose, grouped linear, and the ONNX GRU (`linear_before_reset = 1`).
  AVX2 + FMA kernels are dispatched at **runtime** (`simd::has_avx2()`), with a portable
  scalar fallback.
- `Encoder` / `EncOut` — the encoder stage in isolation, e.g. to use the predicted local-SNR
  head as a voice-activity signal without running the decoders.
- `Weights` / `WeightBank` — the flat, 64-byte-aligned weight arena and per-stage access.

```rust
use std::sync::Arc;
use hush_vani_core::{Encoder, Weights};

let weights = Arc::new(Weights::from_paths("weights.bin", "weights.txt")?);
let encoder = Encoder::new(weights)?;
// let out = encoder.encode(&feat_erb, &feat_spec, t);
# Ok::<(), hush_vani_core::Error>(())
```

Licence: Apache-2.0, matching the upstream model. Weights are not redistributed here.
