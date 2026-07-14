//! The Hush decoders — the second stage of the model.
//!
//! Given the encoder's [`EncOut`], the ERB decoder produces a 32-band spectral mask and the
//! deep-filter decoder produces order-5 complex filter coefficients. Merging them into an
//! enhanced spectrum is done by the caller (see [`crate::Hush::enhance`]).

use crate::bank::{WeightBank, HIDDEN};
use crate::dsp::{NB_DF, NB_ERB};
use crate::nn::*;
use crate::simd::{sigmoid_slice, tanh_slice};
use crate::{EncOut, Error, Weights};
use std::sync::Arc;

/// Tensors the decoders read. Validated once at construction.
#[rustfmt::skip]
const REQUIRED: [&str; 37] = [
    "erb_dec/emb_gru.linear_in.0.weight", "erb_dec/emb_gru.linear_out.0.weight",
    "erb_dec/onnx::GRU_291", "erb_dec/onnx::GRU_292", "erb_dec/onnx::GRU_293",
    "erb_dec/onnx::Conv_247", "erb_dec/onnx::Conv_248",
    "erb_dec/convt3.0.weight", "erb_dec/onnx::Conv_250", "erb_dec/onnx::Conv_251",
    "erb_dec/onnx::Conv_253", "erb_dec/onnx::Conv_254",
    "erb_dec/convt2.0.weight", "erb_dec/onnx::Conv_256", "erb_dec/onnx::Conv_257",
    "erb_dec/onnx::Conv_259", "erb_dec/onnx::Conv_260",
    "erb_dec/convt1.0.weight", "erb_dec/onnx::Conv_262", "erb_dec/onnx::Conv_263",
    "erb_dec/onnx::Conv_265", "erb_dec/onnx::Conv_266",
    "erb_dec/onnx::Conv_268", "erb_dec/onnx::Conv_269",

    "df_dec/df_gru.linear_in.0.weight", "df_dec/df_out.0.weight",
    "df_dec/onnx::GRU_337", "df_dec/onnx::GRU_338", "df_dec/onnx::GRU_339",
    "df_dec/onnx::GRU_357", "df_dec/onnx::GRU_358", "df_dec/onnx::GRU_359",
    "df_dec/onnx::GRU_377", "df_dec/onnx::GRU_378", "df_dec/onnx::GRU_379",
    "df_dec/onnx::Conv_314", "df_dec/onnx::Conv_315",
];

/// Matmul weights (grouped-linear + GRU input `W`) — held as f16.
const MATS: [&str; 8] = [
    "erb_dec/emb_gru.linear_in.0.weight",
    "erb_dec/emb_gru.linear_out.0.weight",
    "erb_dec/onnx::GRU_291",
    "df_dec/df_gru.linear_in.0.weight",
    "df_dec/df_out.0.weight",
    "df_dec/onnx::GRU_337",
    "df_dec/onnx::GRU_357",
    "df_dec/onnx::GRU_377",
];

const GRU_R: [&str; 4] = [
    "erb_dec/onnx::GRU_292",
    "df_dec/onnx::GRU_338",
    "df_dec/onnx::GRU_358",
    "df_dec/onnx::GRU_378",
];

/// The ERB and deep-filter decoders. Construct once with [`Decoders::new`].
pub struct Decoders {
    bank: WeightBank,
}

impl Decoders {
    /// Build the decoders from shared weights, validating and packing what they need.
    pub fn new(weights: Arc<Weights>) -> Result<Self, Error> {
        Ok(Decoders { bank: WeightBank::new(weights, &REQUIRED, &MATS, &GRU_R)? })
    }

    /// ERB decoder: `emb` + skip connections -> a `[T, 32]` sigmoid spectral mask.
    pub fn erb_dec(&self, e: &EncOut, t: usize) -> Vec<f32> {
        let g = |n: &str| self.bank.t(&format!("erb_dec/{n}"));

        let mut x = self.bank.gl(&e.emb, t, "erb_dec/emb_gru.linear_in.0.weight");
        relu_(&mut x);
        let x = self.bank.gru(&x, t, HIDDEN, "erb_dec", ("onnx::GRU_291", "onnx::GRU_292", "onnx::GRU_293"));
        let mut x = self.bank.gl(&x, t, "erb_dec/emb_gru.linear_out.0.weight"); // [T,128]
        relu_(&mut x);

        // [T, 8, 16] -> (16, T, 8)
        let mut y = vec![0f32; 16 * t * 8];
        for ch in 0..16 {
            for ti in 0..t {
                for f in 0..8 {
                    y[ch * t * 8 + ti * 8 + f] = x[ti * 128 + f * 16 + ch];
                }
            }
        }

        let skip = |acc: &mut Vec<f32>, e: &[f32], f: usize, pw: &str, pb: &str| {
            let mut s = pointwise(e, 16, t * f, g(pw), Some(g(pb)), 16);
            relu_(&mut s);
            for (a, v) in acc.iter_mut().zip(s.iter()) {
                *a += v;
            }
        };

        skip(&mut y, &e.e3, 8, "onnx::Conv_247", "onnx::Conv_248");
        let (y, _) = dw_conv13(&y, 16, t, 8, g("convt3.0.weight"), 1);
        let mut y = pointwise(&y, 16, t * 8, g("onnx::Conv_250"), Some(g("onnx::Conv_251")), 16);
        relu_(&mut y);

        skip(&mut y, &e.e2, 8, "onnx::Conv_253", "onnx::Conv_254");
        let (y, f) = convtranspose_dw(&y, 16, t, 8, g("convt2.0.weight")); // 8 -> 16
        let mut y = pointwise(&y, 16, t * f, g("onnx::Conv_256"), Some(g("onnx::Conv_257")), 16);
        relu_(&mut y);

        skip(&mut y, &e.e1, 16, "onnx::Conv_259", "onnx::Conv_260");
        let (y, f) = convtranspose_dw(&y, 16, t, 16, g("convt1.0.weight")); // 16 -> 32
        let mut y = pointwise(&y, 16, t * f, g("onnx::Conv_262"), Some(g("onnx::Conv_263")), 16);
        relu_(&mut y);

        skip(&mut y, &e.e0, 32, "onnx::Conv_265", "onnx::Conv_266");
        let mut m = conv1x3(&y, 16, t, NB_ERB, g("onnx::Conv_268"), g("onnx::Conv_269"), 1);
        sigmoid_slice(&mut m);
        m // [T, 32]
    }

    /// Deep-filter decoder: `emb` + `c0` -> `[T, 64, 10]` order-5 complex coefficients.
    pub fn df_dec(&self, e: &EncOut, t: usize) -> Vec<f32> {
        let g = |n: &str| self.bank.t(&format!("df_dec/{n}"));

        let mut x = self.bank.gl(&e.emb, t, "df_dec/df_gru.linear_in.0.weight"); // [T,256]
        relu_(&mut x);
        let mut h = x;
        for ids in [
            ("onnx::GRU_337", "onnx::GRU_338", "onnx::GRU_339"),
            ("onnx::GRU_357", "onnx::GRU_358", "onnx::GRU_359"),
            ("onnx::GRU_377", "onnx::GRU_378", "onnx::GRU_379"),
        ] {
            h = self.bank.gru(&h, t, HIDDEN, "df_dec", ids);
        }

        let mut coefs = self.bank.gl(&h, t, "df_dec/df_out.0.weight"); // [T, 640]
        tanh_slice(&mut coefs);

        // df_convp: 1x1 conv on c0, 16 -> 10, then (10,T,64) added as [T][64][10]
        let mut cp = pointwise(&e.c0, 16, t * NB_DF, g("onnx::Conv_314"), Some(g("onnx::Conv_315")), 10);
        relu_(&mut cp);
        for k in 0..10 {
            for ti in 0..t {
                for f in 0..NB_DF {
                    coefs[ti * 640 + f * 10 + k] += cp[k * t * NB_DF + ti * NB_DF + f];
                }
            }
        }
        coefs
    }
}
