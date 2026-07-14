//! Shared weight access for one stage of the model.
//!
//! A `WeightBank` owns everything one stage needs, in the layout its kernels want:
//!
//! - **small tensors** (conv kernels, biases) stay f32 — a few thousand values, not worth
//!   converting, and their kernels are not bandwidth-bound.
//! - **matmul weights** (grouped-linear + GRU input `W`) are held as **f16**.
//! - **GRU recurrent `R`** is repacked into panels *and* held as f16.
//!
//! Halving the weight stream is a real speedup, not just a size win: the recurrent matvec
//! re-reads the whole 768x256 matrix every frame and is bandwidth-bound, so fewer bytes per
//! FMA goes straight to the bottom line. `_mm256_cvtph_ps` does the widening for free.
//!
//! The bank copies out what it needs and does **not** retain the source [`Weights`], so the
//! f32 arena is freed once the stages are built.

use crate::alloc::{AlignedU16, AlignedVec};
use crate::error::Error;
use crate::nn::{
    grouped_linear, grouped_linear_f16, gru, gru_f16, pack_rows, pack_rows_f16, to_f16_vec, PANEL,
};
use crate::simd::has_f16c;
use crate::weights::Weights;
use std::collections::HashMap;
use std::sync::Arc;

/// GRU hidden size (all five GRUs in the model are 256-wide).
pub const HIDDEN: usize = 256;

/// A weight matrix, in whichever precision this build is using.
enum Mat {
    F32(AlignedVec),
    F16(AlignedU16),
}

/// Validated, kernel-ready weights for one model stage.
pub struct WeightBank {
    small: HashMap<String, AlignedVec>, // f32: convs, biases
    mats: HashMap<String, Mat>,         // grouped-linear + GRU input weights
    packed: HashMap<String, Mat>,       // GRU recurrent weights, panel-packed
    shapes: HashMap<String, Vec<usize>>,
}

impl WeightBank {
    /// Validate every `required` tensor, then materialise:
    /// `mats` (grouped-linear + GRU `W`) and `gru_r` (recurrent, packed) in f16 when the CPU
    /// has F16C, else f32; everything else in `required` as plain f32.
    ///
    /// A bad weight file is rejected here, not mid-inference.
    pub fn new(
        w: Arc<Weights>,
        required: &[&str],
        mats: &[&str],
        gru_r: &[&str],
    ) -> Result<Self, Error> {
        // f16 *compute* is opt-in (`f16-kernels`) and off by default: measured 0.90x.
        // The weight matrices already sit in L2, so halving the bytes buys no bandwidth,
        // while `vcvtph2ps` contends with the FMA ports. f16 is a storage win, not a compute
        // win -- by default an f16 file is widened to f32 once here and runs the fast f32
        // kernels. Enable the feature to trade ~10% throughput for half the resident memory.
        let f16 = w.is_f16() && has_f16c() && cfg!(feature = "f16-kernels");
        let mut bank = WeightBank {
            small: HashMap::new(),
            mats: HashMap::new(),
            packed: HashMap::new(),
            shapes: HashMap::new(),
        };

        for name in required {
            let t = w.get(name)?;
            bank.shapes.insert((*name).to_string(), w.shape(name)?.to_vec());

            if gru_r.contains(name) {
                let s = w.shape(name)?; // [1, 3H, H]
                if s.len() != 3 || s[1] % PANEL != 0 || s[2] % 8 != 0 {
                    return Err(Error::Weights(format!("{name}: unexpected GRU shape {s:?}")));
                }
                let (rows, k) = (s[1], s[2]);
                bank.packed.insert(
                    (*name).to_string(),
                    if f16 {
                        Mat::F16(pack_rows_f16(t, rows, k))
                    } else {
                        Mat::F32(pack_rows(t, rows, k))
                    },
                );
            } else if mats.contains(name) {
                bank.mats.insert(
                    (*name).to_string(),
                    if f16 { Mat::F16(to_f16_vec(t)) } else { Mat::F32(AlignedVec::from_slice(t)) },
                );
            } else {
                bank.small.insert((*name).to_string(), AlignedVec::from_slice(t));
            }
        }
        Ok(bank)
    }

    /// A small f32 tensor (conv kernel, bias) by name.
    ///
    /// Panics if `name` is a matmul weight (those live in f16) or was not in `required` —
    /// both are programming errors, not runtime input errors.
    #[inline]
    pub fn t(&self, name: &str) -> &[f32] {
        self.small
            .get(name)
            .unwrap_or_else(|| panic!("{name} is not a small f32 tensor in this bank"))
    }

    /// Grouped linear layer (ONNX `Einsum btgi,gih->btgh`) using the named `[G, I, H]` weight.
    pub fn gl(&self, x: &[f32], t: usize, name: &str) -> Vec<f32> {
        let s = self.shapes.get(name).unwrap_or_else(|| panic!("no shape for {name}"));
        let (g, i, h) = (s[0], s[1], s[2]);
        match self.mats.get(name).unwrap_or_else(|| panic!("{name} is not a matmul weight")) {
            Mat::F16(w) => grouped_linear_f16(x, t, g, i, h, w),
            Mat::F32(w) => grouped_linear(x, t, g, i, h, w),
        }
    }

    /// A GRU with input width `i`, reading `<prefix>/<w>`, packed `<prefix>/<r>` and
    /// `<prefix>/<b>` (gate order z, r, h; `linear_before_reset = 1`).
    pub fn gru(&self, x: &[f32], t: usize, i: usize, prefix: &str, ids: (&str, &str, &str)) -> Vec<f32> {
        let wn = format!("{prefix}/{}", ids.0);
        let rn = format!("{prefix}/{}", ids.1);
        let b = self.t(&format!("{prefix}/{}", ids.2));
        let w = self.mats.get(&wn).unwrap_or_else(|| panic!("missing GRU W {wn}"));
        let r = self.packed.get(&rn).unwrap_or_else(|| panic!("missing GRU R {rn}"));
        match (w, r) {
            (Mat::F16(w), Mat::F16(r)) => gru_f16(x, t, i, w, r, b, HIDDEN),
            (Mat::F32(w), Mat::F32(r)) => gru(x, t, i, w, r, b, HIDDEN),
            _ => unreachable!("W and R precision must match"),
        }
    }

    /// True if this bank is holding its matmul weights as f16.
    pub fn is_f16(&self) -> bool {
        self.mats.values().any(|m| matches!(m, Mat::F16(_)))
    }
}
