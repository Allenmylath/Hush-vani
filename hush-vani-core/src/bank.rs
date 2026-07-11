//! Shared weight access for one stage of the model.
//!
//! A `WeightBank` bundles name-addressed tensors (from a shared [`Weights`]) with the
//! recurrent GRU matrices repacked into the panel layout. Both the [`Encoder`](crate::Encoder)
//! in this crate and the decoders in the `hush-vani` crate are built on one.

use crate::alloc::AlignedVec;
use crate::error::Error;
use crate::nn::{grouped_linear, gru, pack_rows, PANEL};
use crate::weights::Weights;
use std::collections::HashMap;
use std::sync::Arc;

/// GRU hidden size (all five GRUs in the model are 256-wide).
pub const HIDDEN: usize = 256;

/// Validated tensor access plus packed recurrent matrices for one model stage.
pub struct WeightBank {
    w: Arc<Weights>,
    packed_r: HashMap<String, AlignedVec>,
}

impl WeightBank {
    /// Validate that every `required` tensor is present, then pack each `gru_r` matrix into
    /// the panel layout. After this succeeds, [`t`](Self::t), [`gl`](Self::gl) and
    /// [`gru`](Self::gru) cannot fail — a bad weight file is rejected here, not mid-inference.
    ///
    /// `gru_r` names are the fully-qualified `"<prefix>/<id>"` recurrent-weight names; the
    /// same string is reconstructed by [`gru`](Self::gru).
    pub fn new(w: Arc<Weights>, required: &[&str], gru_r: &[&str]) -> Result<Self, Error> {
        for name in required {
            w.get(name)?;
        }
        let mut packed_r = HashMap::new();
        for name in gru_r {
            let s = w.shape(name)?; // [1, 3H, H]
            if s.len() != 3 || s[1] % PANEL != 0 || s[2] % 8 != 0 {
                return Err(Error::Weights(format!("{name}: unexpected GRU shape {s:?}")));
            }
            packed_r.insert((*name).to_string(), pack_rows(w.get(name)?, s[1], s[2]));
        }
        Ok(WeightBank { w, packed_r })
    }

    /// A validated tensor by name.
    ///
    /// Panics if `name` was not in the `required` set passed to [`new`](Self::new); it is a
    /// programming error, not a runtime input error, so it is not returned as [`Error`].
    #[inline]
    pub fn t(&self, name: &str) -> &[f32] {
        self.w.get(name).expect("tensor validated in WeightBank::new")
    }

    /// Grouped linear layer (ONNX `Einsum btgi,gih->btgh`) using the named `[G, I, H]` weight.
    pub fn gl(&self, x: &[f32], t: usize, name: &str) -> Vec<f32> {
        let s = self.w.shape(name).expect("tensor validated in WeightBank::new");
        grouped_linear(x, t, s[0], s[1], s[2], self.t(name))
    }

    /// A GRU with input width `i`, reading `<prefix>/<w>`, packed `<prefix>/<r>` and
    /// `<prefix>/<b>` (gate order z, r, h; `linear_before_reset = 1`).
    pub fn gru(&self, x: &[f32], t: usize, i: usize, prefix: &str, ids: (&str, &str, &str)) -> Vec<f32> {
        let w = self.t(&format!("{prefix}/{}", ids.0));
        let r = &self.packed_r[&format!("{prefix}/{}", ids.1)];
        let b = self.t(&format!("{prefix}/{}", ids.2));
        gru(x, t, i, w, r, b, HIDDEN)
    }
}
