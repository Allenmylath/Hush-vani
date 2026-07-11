//! Bundled weights for [`hush-vani`](https://crates.io/crates/hush-vani), as an embedded
//! data blob — so you don't have to ship a `weights.bin` alongside your binary or run any
//! export script.
//!
//! This is a **pure, dependency-free data crate**. The ergonomic loader lives in `hush-vani`
//! itself: enable its `bundled` feature and call `Hush::bundled()`. Depend on this crate
//! directly only if you want the raw bytes:
//!
//! ```ignore
//! // in a crate that also depends on `hush-vani`:
//! use hush_vani::Hush;
//! let hush = Hush::from_bytes(hush_vani_weights::WEIGHTS_BIN, hush_vani_weights::MANIFEST)?;
//! ```
//!
//! The weights are the [`weya-ai/hush`](https://huggingface.co/weya-ai/hush) model,
//! redistributed under Apache-2.0 (see the bundled `LICENSE` and `NOTICE`).
//!
//! # Precision
//!
//! The blob is **float16** (4.56 MB, embedded). It is widened to f32 at load and run through
//! the same f32 kernels, so there is no speed cost. End to end the output differs from full
//! float32 weights by 75.5 dB SI-SDR — above the ~67 dB a 16-bit WAV can represent, so it is
//! inaudible in a shipped file. If you need bit-exactness against the reference ONNX
//! pipeline, generate f32 weights with `tools/export_weights.py` and load them with
//! `Hush::from_paths` instead.

#![forbid(unsafe_code)]
// Pure data: no_std for real builds, but the test harness needs std.
#![cfg_attr(not(test), no_std)]

/// The little-endian f32 weight arena (~9 MB). Pair with [`MANIFEST`].
pub const WEIGHTS_BIN: &[u8] = include_bytes!("../data/weights.bin");

/// The tensor manifest describing [`WEIGHTS_BIN`].
pub const MANIFEST: &str = include_str!("../data/weights.txt");

#[cfg(test)]
mod tests {
    #[test]
    fn blob_is_present_and_sane() {
        assert!(super::WEIGHTS_BIN.len() > 4_000_000, "weights.bin looks truncated");
        assert_eq!(super::WEIGHTS_BIN.len() % 2, 0, "not a whole number of f16");
        assert!(super::MANIFEST.starts_with("#dtype f16"), "expected an f16 blob");
        assert!(super::MANIFEST.contains("enc/onnx::GRU_361"), "manifest missing a known tensor");
    }
}
