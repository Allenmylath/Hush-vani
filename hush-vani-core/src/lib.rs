//! Core of [`hush-vani`](https://crates.io/crates/hush-vani): the shared DSP and neural
//! primitives, plus the **encoder** — the first stage of the Hush speech-enhancement model.
//!
//! Most users want the [`hush-vani`](https://crates.io/crates/hush-vani) crate, which builds
//! the full enhancer (`Hush`) on top of this one. Depend on `hush-vani-core` directly only to
//! reuse the kernels, the STFT/ERB DSP, or the [`Encoder`] in isolation.
//!
//! # Layout
//!
//! - [`dsp`] — vorbis-window STFT, ERB features, exponential normalisers, ISTFT, and the
//!   spectral merge helpers ([`dsp::erb_inv`], [`dsp::apply_df`]).
//! - [`nn`] — dot / matvec / GEMM, grouped/depthwise/pointwise convolutions, GRU. AVX2+FMA
//!   kernels are selected at **runtime** ([`simd::has_avx2`]); portable scalar fallback.
//! - [`Encoder`] / [`EncOut`] — the encoder stage and its outputs.
//! - [`Weights`] / [`WeightBank`] — the flat aligned weight arena and per-stage access.

#![warn(missing_docs)]

// Low-level reuse surface: documented at the module level, but individual kernels/constants
// are not forced to carry doc comments -- they are power-user API, not the curated one.
#[allow(missing_docs)]
pub mod alloc;
#[allow(missing_docs)]
pub mod dsp;
#[allow(missing_docs)]
pub mod nn;
#[allow(missing_docs)]
pub mod simd;
#[allow(missing_docs)]
pub mod weights;

pub mod bank;
pub mod encoder;
mod error;

pub use bank::WeightBank;
pub use encoder::{EncOut, Encoder};
pub use error::Error;
pub use weights::Weights;
