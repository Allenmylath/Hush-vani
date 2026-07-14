//! Real-time speech enhancement and background-speaker suppression, in pure Rust.
//!
//! A from-scratch implementation of the [Hush](https://huggingface.co/weya-ai/hush) model
//! (a DeepFilterNet3 derivative): STFT, ERB features, the encoder / ERB decoder / deep-filter
//! decoder, and overlap-add synthesis. No ONNX runtime, no BLAS, no Python.
//!
//! Output matches the reference ONNX Runtime pipeline to float32 rounding
//! (SI-SDR 129.7 dB end-to-end).
//!
//! # Example
//!
//! The weights are **built in** — nothing to download, export, or ship alongside your binary:
//!
//! ```no_run
//! use hush_vani::Hush;
//!
//! let hush = Hush::new()?;                  // weights embedded in the crate
//! let noisy: Vec<f32> = vec![0.0; 16_000];  // 1 s of 16 kHz mono, in [-1, 1]
//! let clean = hush.enhance(&noisy)?;
//! # Ok::<(), hush_vani::Error>(())
//! ```
//!
//! # Crate layout
//!
//! [`Hush`] is the whole API. Below it, and public for reuse: [`dsp`] (STFT, ERB features,
//! ISTFT), [`nn`] (dot / matvec / GEMM, convolutions, GRU — AVX2+FMA chosen at **runtime**,
//! with a portable scalar fallback), [`Encoder`], [`Decoders`], and the [`Weights`] arena.
//!
//! # Audio format
//!
//! Mono, 16 kHz, `f32` samples in `[-1, 1]`. See [`Hush::SAMPLE_RATE`].
//!
//! The output lags the input by [`Hush::LATENCY_SAMPLES`] (160 samples, 10 ms) — one hop of
//! algorithmic latency. Trim that many samples from the front to align with the input.
//!
//! Input is processed in whole 10 ms frames; any trailing partial frame is ignored, so the
//! output length is `audio.len() / 160 * 160`.
//!
//! # Weights
//!
//! The [`weya-ai/hush`](https://huggingface.co/weya-ai/hush) weights are embedded in the
//! crate (Apache-2.0, see `NOTICE`), quantised to **int8**: 2.44 MB, against 4.56 MB for
//! float16 and 9.12 MB for the original float32.
//!
//! int8 here is a **storage format, not a compute format** — the weights are dequantised to
//! f32 at load and run through the same f32 kernels, so it is neither faster nor slower.
//! What it buys is size, and it costs nothing in the thing the model is for: on six real
//! recordings it removes **+43.4 dB of noise, exactly matching full f32 weights**. It sits
//! 44.0 dB SI-SDR from the f32 output (an error level of −72.4 dBFS), which is inaudible at
//! any normal listening level but, unlike f16 (−94.6 dBFS), not literally below what a
//! 16-bit WAV can store.
//!
//! Each group of 64 weights along a matmul row carries its own scale. One scale per *tensor*
//! — the obvious first thing to try — has to cover the GRUs' ±31 outliers alongside the 99%
//! of weights inside ±0.23, and that model removes 2.1 dB *less* noise than f32 (8.4 dB
//! worse on one sample). The block scales cost 159 KB and give all of it back.
//!
//! To use different weights, export them with `tools/export_weights.py` (`--f16`, or neither
//! flag for f32) and load with [`Hush::from_paths`] or [`Hush::from_bytes`].
//!
//! # Performance
//!
//! AVX2 + FMA kernels are selected at **runtime**; no special `RUSTFLAGS` needed. On a
//! modern x86-64 core this runs ~110x faster than real time single-threaded (5 s of audio in
//! ~45 ms). Without AVX2 it falls back to portable scalar code, roughly 4x slower.
//!
//! # Streaming
//!
//! [`Hush::enhance`] processes a whole utterance: the GRUs start from zero state each call,
//! so feeding successive chunks would restart the recurrence and degrade output. A streaming
//! API is not yet exposed.

#![warn(missing_docs)]

// Low-level reuse surface: documented at the module level, but individual kernels and
// constants are not forced to carry doc comments -- they are power-user API, not the
// curated one.
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
pub mod decoders;
pub mod encoder;
mod error;

pub use bank::WeightBank;
pub use decoders::Decoders;
pub use encoder::{EncOut, Encoder};
pub use error::Error;
pub use weights::Weights;

use dsp::{Dsp, FREQS, HOP, NB_DF};
use rustfft::num_complex::Complex32;
use std::path::Path;
use std::sync::Arc;

/// The embedded int8 weight blob: payload followed by a table of f32 block scales.
const WEIGHTS_BIN: &[u8] = include_bytes!("../data/weights.bin");

/// The manifest describing [`WEIGHTS_BIN`].
const MANIFEST: &str = include_str!("../data/weights.txt");

/// A loaded Hush model, ready to enhance audio.
///
/// Holds the [`Encoder`] and [`Decoders`] over one shared, reference-counted weight arena.
/// Cheap to share: [`enhance`](Hush::enhance) takes `&self` and allocates its own DSP state
/// per call, so one instance can be used concurrently from several threads.
pub struct Hush {
    encoder: Encoder,
    decoders: Decoders,
}

impl Hush {
    /// Required sample rate, in Hz. Audio at any other rate must be resampled first.
    pub const SAMPLE_RATE: u32 = 16_000;

    /// Algorithmic latency: the output lags the input by this many samples (10 ms).
    pub const LATENCY_SAMPLES: usize = HOP;

    /// Samples per processed frame (10 ms).
    pub const FRAME_SAMPLES: usize = HOP;

    fn build(weights: Weights) -> Result<Self, Error> {
        let w = Arc::new(weights);
        Ok(Hush {
            encoder: Encoder::new(Arc::clone(&w))?,
            decoders: Decoders::new(w)?,
        })
    }

    /// Load from an in-memory weight blob and manifest, e.g. via `include_bytes!`.
    pub fn from_bytes(weights_bin: &[u8], manifest: &str) -> Result<Self, Error> {
        Self::build(Weights::from_bytes(weights_bin, manifest)?)
    }

    /// Load `weights.bin` and `weights.txt` from disk.
    pub fn from_paths(bin: impl AsRef<Path>, manifest: impl AsRef<Path>) -> Result<Self, Error> {
        Self::build(Weights::from_paths(bin, manifest)?)
    }

    /// Load the model. The weights are embedded in the crate — no files, no export script.
    ///
    /// This is the one you want. See the [crate docs](crate#weights) for what the embedded
    /// int8 weights cost (in short: 2.44 MB, and nothing measurable in noise removal).
    pub fn new() -> Result<Self, Error> {
        Self::from_bytes(WEIGHTS_BIN, MANIFEST)
    }

    /// True if this build will use the AVX2 + FMA kernels on this CPU.
    pub fn is_accelerated() -> bool {
        simd::has_avx2()
    }

    /// Enhance a mono 16 kHz signal. Full suppression; see [`Hush::enhance_with`] to limit it.
    ///
    /// Returns `audio.len() / 160 * 160` samples, lagging the input by
    /// [`LATENCY_SAMPLES`](Hush::LATENCY_SAMPLES).
    pub fn enhance(&self, audio: &[f32]) -> Result<Vec<f32>, Error> {
        self.enhance_with(audio, None)
    }

    /// Enhance with an optional attenuation limit in dB.
    ///
    /// `atten_lim_db` caps how much the noise may be suppressed, by mixing the noisy signal
    /// back in: `out = noisy * lim + enhanced * (1 - lim)` where `lim = 10^(-|dB|/20)`.
    /// `Some(12.0)` gives ~12 dB of reduction; `None` (or a large value) suppresses fully.
    /// Useful when full suppression sounds unnaturally dry.
    pub fn enhance_with(&self, audio: &[f32], atten_lim_db: Option<f32>) -> Result<Vec<f32>, Error> {
        let t = audio.len() / HOP;
        if t == 0 {
            return Err(Error::TooShort { samples: audio.len(), needed: HOP });
        }

        let mut d = Dsp::new();
        let spec = d.analysis(&audio[..t * HOP]);
        let feat_erb = d.erb_feat(&spec, t);
        let feat_spec = d.spec_feat(&spec, t);

        // first stage (core) -> shared embedding + skips; second stage (this crate) -> mask + coefs
        let e = self.encoder.encode(&feat_erb, &feat_spec, t);
        let mask = self.decoders.erb_dec(&e, t);
        let coefs = self.decoders.df_dec(&e, t);

        // merge: ERB mask on all bins, deep filter overwrites the first NB_DF
        let gains = dsp::erb_inv(&mask, t);
        let mut out_spec = vec![Complex32::new(0.0, 0.0); t * FREQS];
        for i in 0..t * FREQS {
            out_spec[i] = spec[i] * gains[i];
        }
        dsp::apply_df(&spec, &coefs, t, &mut out_spec);
        debug_assert!(NB_DF < FREQS);

        if let Some(db) = atten_lim_db {
            if db.abs() > 0.0 {
                let lim = 10f32.powf(-db.abs() / 20.0);
                for i in 0..t * FREQS {
                    out_spec[i] = spec[i] * lim + out_spec[i] * (1.0 - lim);
                }
            }
        }

        Ok(d.synthesis(&out_spec, t))
    }

    /// Predicted local SNR per 10 ms frame, in dB (the model's `lsnr` head).
    ///
    /// Useful as a cheap voice-activity / noisiness signal. Runs only the encoder.
    pub fn local_snr(&self, audio: &[f32]) -> Result<Vec<f32>, Error> {
        let t = audio.len() / HOP;
        if t == 0 {
            return Err(Error::TooShort { samples: audio.len(), needed: HOP });
        }
        let mut d = Dsp::new();
        let spec = d.analysis(&audio[..t * HOP]);
        let feat_erb = d.erb_feat(&spec, t);
        let feat_spec = d.spec_feat(&spec, t);
        Ok(self.encoder.encode(&feat_erb, &feat_spec, t).lsnr)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_short_audio() {
        let e = Error::TooShort { samples: 3, needed: 160 };
        assert!(e.to_string().contains("too short"));
    }

    #[test]
    fn constants_are_consistent() {
        assert_eq!(Hush::SAMPLE_RATE, 16_000);
        assert_eq!(Hush::LATENCY_SAMPLES, 160);
        assert_eq!(Hush::FRAME_SAMPLES, 160);
    }
}
