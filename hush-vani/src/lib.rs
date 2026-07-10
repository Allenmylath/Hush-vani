//! Real-time speech enhancement and background-speaker suppression, in pure Rust.
//!
//! A from-scratch implementation of the [Hush](https://huggingface.co/weya-ai/hush) model
//! (a DeepFilterNet3 derivative): STFT, ERB features, the encoder / ERB decoder / deep-filter
//! decoder, and overlap-add synthesis. No ONNX runtime, no BLAS, no Python. The only
//! dependency is `rustfft`.
//!
//! Output matches the reference ONNX Runtime pipeline to float32 rounding
//! (SI-SDR 129.7 dB end-to-end).
//!
//! # Example
//!
//! ```no_run
//! use hush_vani::Hush;
//!
//! let hush = Hush::from_paths("weights.bin", "weights.txt")?;
//! let noisy: Vec<f32> = vec![0.0; 16_000]; // 1 s of 16 kHz mono, in [-1, 1]
//! let clean = hush.enhance(&noisy)?;
//! # Ok::<(), hush_vani::Error>(())
//! ```
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
//! Weights are **not** bundled (9 MB, and they carry the upstream model's license). Generate
//! `weights.bin` + `weights.txt` from the published ONNX bundle with the `export_weights.py`
//! script in the repository, then load them with [`Hush::from_paths`] or
//! [`Hush::from_bytes`] (e.g. via `include_bytes!`).
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

#[doc(hidden)]
pub mod alloc;
#[doc(hidden)]
pub mod dsp;
mod error;
#[doc(hidden)]
pub mod model;
#[doc(hidden)]
pub mod nn;
#[doc(hidden)]
pub mod simd;
#[doc(hidden)]
pub mod weights;

pub use error::Error;

use dsp::{Dsp, FREQS, HOP, NB_DF};
use model::Model;
use rustfft::num_complex::Complex32;
use std::path::Path;
use weights::Weights;

/// A loaded Hush model, ready to enhance audio.
///
/// Cheap to share: [`enhance`](Hush::enhance) takes `&self` and allocates its own DSP state
/// per call, so one instance can be used concurrently from several threads.
pub struct Hush {
    model: Model,
}

impl Hush {
    /// Required sample rate, in Hz. Audio at any other rate must be resampled first.
    pub const SAMPLE_RATE: u32 = 16_000;

    /// Algorithmic latency: the output lags the input by this many samples (10 ms).
    pub const LATENCY_SAMPLES: usize = HOP;

    /// Samples per processed frame (10 ms).
    pub const FRAME_SAMPLES: usize = HOP;

    /// Load from an in-memory weight blob and manifest, e.g. via `include_bytes!`.
    pub fn from_bytes(weights_bin: &[u8], manifest: &str) -> Result<Self, Error> {
        Ok(Hush { model: Model::new(Weights::from_bytes(weights_bin, manifest)?)? })
    }

    /// Load `weights.bin` and `weights.txt` from disk.
    pub fn from_paths(bin: impl AsRef<Path>, manifest: impl AsRef<Path>) -> Result<Self, Error> {
        Ok(Hush { model: Model::new(Weights::from_paths(bin, manifest)?)? })
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

        let e = self.model.enc(&feat_erb, &feat_spec, t);
        let mask = self.model.erb_dec(&e, t);
        let coefs = self.model.df_dec(&e, t);

        let gains = dsp::erb_inv(&mask, t);
        let mut out_spec = vec![Complex32::new(0.0, 0.0); t * FREQS];
        for i in 0..t * FREQS {
            out_spec[i] = spec[i] * gains[i];
        }
        dsp::apply_df(&spec, &coefs, t, &mut out_spec); // overwrites bins < NB_DF
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
        Ok(self.model.enc(&feat_erb, &feat_spec, t).lsnr)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_short_audio() {
        // no weights needed: the length check happens first, so build a model-free probe
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
