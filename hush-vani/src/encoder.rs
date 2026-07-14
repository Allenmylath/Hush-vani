//! The Hush encoder — the first stage of the model.
//!
//! Takes the ERB and complex-spectrum features and produces the shared embedding, the four
//! encoder skip connections, the deep-filter feature `c0`, and the predicted local SNR.
//! Both decoders (in the `hush-vani` crate) consume this [`EncOut`]; the encoder itself has
//! no dependency on them, which is why it lives in the core crate.

use crate::bank::{WeightBank, HIDDEN};
use crate::dsp::NB_DF;
use crate::error::Error;
use crate::nn::*;
use crate::simd::sigmoid;
use crate::weights::Weights;
use std::sync::Arc;

/// Tensors the encoder reads. Validated once at construction.
#[rustfmt::skip]
const REQUIRED: [&str; 25] = [
    "enc/onnx::Conv_310", "enc/onnx::Conv_311",
    "enc/erb_conv1.0.weight", "enc/onnx::Conv_313", "enc/onnx::Conv_314",
    "enc/erb_conv2.0.weight", "enc/onnx::Conv_316", "enc/onnx::Conv_317",
    "enc/erb_conv3.0.weight", "enc/onnx::Conv_319", "enc/onnx::Conv_320",
    "enc/df_conv0.1.weight", "enc/onnx::Conv_322", "enc/onnx::Conv_323",
    "enc/df_conv1.0.weight", "enc/onnx::Conv_325", "enc/onnx::Conv_326",
    "enc/df_fc_emb.0.weight",
    "enc/emb_gru.linear_in.0.weight", "enc/emb_gru.linear_out.0.weight",
    "enc/onnx::GRU_360", "enc/onnx::GRU_361", "enc/onnx::GRU_362",
    "enc/onnx::MatMul_365", "enc/lsnr_fc.0.bias",
];

/// Matmul weights (grouped-linear + GRU input `W`) — held as f16, since these are the
/// bandwidth-bound streams. Everything else in [`REQUIRED`] stays f32.
const MATS: [&str; 4] = [
    "enc/df_fc_emb.0.weight",
    "enc/emb_gru.linear_in.0.weight",
    "enc/emb_gru.linear_out.0.weight",
    "enc/onnx::GRU_360",
];

const GRU_R: [&str; 1] = ["enc/onnx::GRU_361"];

/// Encoder outputs, consumed by the decoders.
///
/// `e0`..`e3` are the encoder skip connections (channels-major `[16, T, F]`), `emb` is the
/// `[T, 128]` embedding, `c0` is the `[16, T, 64]` deep-filter feature, and `lsnr` is the
/// per-frame predicted local SNR in dB.
pub struct EncOut {
    /// Skip connection, `[16, T, 32]`.
    pub e0: Vec<f32>,
    /// Skip connection, `[16, T, 16]`.
    pub e1: Vec<f32>,
    /// Skip connection, `[16, T, 8]`.
    pub e2: Vec<f32>,
    /// Skip connection, `[16, T, 8]`.
    pub e3: Vec<f32>,
    /// Shared embedding, `[T, 128]`.
    pub emb: Vec<f32>,
    /// Deep-filter feature, `[16, T, 64]`.
    pub c0: Vec<f32>,
    /// Predicted local SNR per frame, in dB, `[T]`.
    pub lsnr: Vec<f32>,
}

/// The encoder stage. Construct once with [`Encoder::new`], then call [`encode`](Encoder::encode).
pub struct Encoder {
    bank: WeightBank,
}

impl Encoder {
    /// Build the encoder from shared weights, validating and packing what it needs.
    pub fn new(weights: Arc<Weights>) -> Result<Self, Error> {
        Ok(Encoder { bank: WeightBank::new(weights, &REQUIRED, &MATS, &GRU_R)? })
    }

    /// Run the encoder over `t` frames of features.
    ///
    /// `feat_erb` is `[1, T, 32]`, `feat_spec` is `[2, T, 64]` (real then imag), both as
    /// produced by [`Dsp`](crate::dsp::Dsp).
    pub fn encode(&self, feat_erb: &[f32], feat_spec: &[f32], t: usize) -> EncOut {
        let g = |n: &str| self.bank.t(&format!("enc/{n}"));

        // erb_conv0: 3x3 causal, 1 -> 16 channels, F = 32
        let mut e0 = conv3x3_causal(feat_erb, 1, t, 32, g("onnx::Conv_310"), Some(g("onnx::Conv_311")), 16, 1);
        relu_(&mut e0);

        let erb_block = |x: &[f32], f: usize, dw: &str, pw: &str, pb: &str, stride: usize| {
            let (y, fo) = dw_conv13(x, 16, t, f, g(dw), stride);
            let mut z = pointwise(&y, 16, t * fo, g(pw), Some(g(pb)), 16);
            relu_(&mut z);
            (z, fo)
        };
        let (e1, _) = erb_block(&e0, 32, "erb_conv1.0.weight", "onnx::Conv_313", "onnx::Conv_314", 2);
        let (e2, _) = erb_block(&e1, 16, "erb_conv2.0.weight", "onnx::Conv_316", "onnx::Conv_317", 2);
        let (e3, _) = erb_block(&e2, 8, "erb_conv3.0.weight", "onnx::Conv_319", "onnx::Conv_320", 1);

        // df_conv0: 3x3 causal, 2 groups (one per re/im plane), no bias
        let c = conv3x3_causal(feat_spec, 2, t, NB_DF, g("df_conv0.1.weight"), None, 16, 2);
        let mut c0 = pointwise(&c, 16, t * NB_DF, g("onnx::Conv_322"), Some(g("onnx::Conv_323")), 16);
        relu_(&mut c0);

        let (cx, fo) = dw_conv13(&c0, 16, t, NB_DF, g("df_conv1.0.weight"), 2); // 64 -> 32
        let mut cx = pointwise(&cx, 16, t * fo, g("onnx::Conv_325"), Some(g("onnx::Conv_326")), 16);
        relu_(&mut cx);

        // (C,T,F) -> [T, F*C]  (numpy: transpose(0,2,3,1).reshape)
        let mut cf = vec![0f32; t * fo * 16];
        for ch in 0..16 {
            for ti in 0..t {
                for f in 0..fo {
                    cf[ti * fo * 16 + f * 16 + ch] = cx[ch * t * fo + ti * fo + f];
                }
            }
        }
        let mut emb = self.bank.gl(&cf, t, "enc/df_fc_emb.0.weight"); // [T,128]
        relu_(&mut emb);

        // e3 (16,T,8) -> [T,128], added in
        for ch in 0..16 {
            for ti in 0..t {
                for f in 0..8 {
                    emb[ti * 128 + f * 16 + ch] += e3[ch * t * 8 + ti * 8 + f];
                }
            }
        }

        let mut x = self.bank.gl(&emb, t, "enc/emb_gru.linear_in.0.weight"); // [T,256]
        relu_(&mut x);
        let x = self.bank.gru(&x, t, HIDDEN, "enc", ("onnx::GRU_360", "onnx::GRU_361", "onnx::GRU_362"));
        let mut emb = self.bank.gl(&x, t, "enc/emb_gru.linear_out.0.weight"); // [T,128]
        relu_(&mut emb);

        let wl = g("onnx::MatMul_365");
        let bl = g("lsnr_fc.0.bias")[0];
        let lsnr = (0..t)
            .map(|ti| sigmoid(dot(&emb[ti * 128..(ti + 1) * 128], wl) + bl) * 50.0 - 15.0)
            .collect();

        EncOut { e0, e1, e2, e3, emb, c0, lsnr }
    }

    /// Access to the underlying weight bank, for stages built in a downstream crate.
    #[doc(hidden)]
    pub fn bank(&self) -> &WeightBank {
        &self.bank
    }
}
