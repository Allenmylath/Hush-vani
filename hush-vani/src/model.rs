//! The three Hush graphs, hand-written. Mirrors tools/{enc,erb_dec,df_dec}.txt.

use crate::alloc::AlignedVec;
use crate::dsp::{NB_DF, NB_ERB};
use crate::error::Error;
use crate::nn::*;
use crate::simd::sigmoid;
use crate::weights::Weights;
use std::collections::HashMap;

/// The five recurrent matrices, repacked once at load into the PANEL-row layout.
const GRU_R: [&str; 5] = [
    "enc/onnx::GRU_361",
    "erb_dec/onnx::GRU_292",
    "df_dec/onnx::GRU_338",
    "df_dec/onnx::GRU_358",
    "df_dec/onnx::GRU_378",
];

/// Every tensor the forward pass reads. Checked once at load so a truncated or mismatched
/// weight file reports a clean error instead of panicking halfway through inference.
#[rustfmt::skip]
const REQUIRED: [&str; 62] = [
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

pub struct Model {
    w: Weights,
    packed_r: HashMap<String, AlignedVec>,
}

pub struct EncOut {
    pub e0: Vec<f32>,
    pub e1: Vec<f32>,
    pub e2: Vec<f32>,
    pub e3: Vec<f32>,
    pub emb: Vec<f32>,
    pub c0: Vec<f32>,
    pub lsnr: Vec<f32>,
}

impl Model {
    /// Validates that every tensor the forward pass touches is present, then repacks the
    /// recurrent matrices. After this succeeds the internal getters cannot fail.
    pub fn new(w: Weights) -> Result<Self, Error> {
        for name in REQUIRED {
            w.get(name)?;
        }
        let mut packed_r = HashMap::new();
        for name in GRU_R {
            let s = w.shape(name)?; // [1, 3H, H]
            if s.len() != 3 || s[1] % PANEL != 0 || s[2] % 8 != 0 {
                return Err(Error::Weights(format!("{name}: unexpected GRU shape {s:?}")));
            }
            packed_r.insert(name.to_string(), pack_rows(w.get(name)?, s[1], s[2]));
        }
        Ok(Model { w, packed_r })
    }

    /// Infallible after `Model::new` validated every name.
    fn t(&self, name: &str) -> &[f32] {
        self.w.get(name).expect("tensor validated in Model::new")
    }

    fn gl(&self, x: &[f32], t: usize, name: &str) -> Vec<f32> {
        let s = self.w.shape(name).expect("tensor validated in Model::new"); // [G, I, H]
        grouped_linear(x, t, s[0], s[1], s[2], self.t(name))
    }

    fn gru_(&self, x: &[f32], t: usize, i: usize, prefix: &str, ids: (&str, &str, &str)) -> Vec<f32> {
        let w = self.t(&format!("{prefix}/{}", ids.0));
        let r = &self.packed_r[&format!("{prefix}/{}", ids.1)];
        let b = self.t(&format!("{prefix}/{}", ids.2));
        gru(x, t, i, w, r, b, 256)
    }

    // ----------------------------------------------------------------- encoder
    pub fn enc(&self, feat_erb: &[f32], feat_spec: &[f32], t: usize) -> EncOut {
        let g = |n: &str| self.t(&format!("enc/{n}"));

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
        let mut emb = self.gl(&cf, t, "enc/df_fc_emb.0.weight"); // [T,128]
        relu_(&mut emb);

        // e3 (16,T,8) -> [T,128], added in
        for ch in 0..16 {
            for ti in 0..t {
                for f in 0..8 {
                    emb[ti * 128 + f * 16 + ch] += e3[ch * t * 8 + ti * 8 + f];
                }
            }
        }

        let mut x = self.gl(&emb, t, "enc/emb_gru.linear_in.0.weight"); // [T,256]
        relu_(&mut x);
        let x = self.gru_(&x, t, 256, "enc", ("onnx::GRU_360", "onnx::GRU_361", "onnx::GRU_362"));
        let mut emb = self.gl(&x, t, "enc/emb_gru.linear_out.0.weight"); // [T,128]
        relu_(&mut emb);

        let wl = g("onnx::MatMul_365");
        let bl = g("lsnr_fc.0.bias")[0];
        let lsnr = (0..t)
            .map(|ti| sigmoid(dot(&emb[ti * 128..(ti + 1) * 128], wl) + bl) * 50.0 - 15.0)
            .collect();

        EncOut { e0, e1, e2, e3, emb, c0, lsnr }
    }

    // ------------------------------------------------------------- erb decoder
    pub fn erb_dec(&self, e: &EncOut, t: usize) -> Vec<f32> {
        let g = |n: &str| self.t(&format!("erb_dec/{n}"));

        let mut x = self.gl(&e.emb, t, "erb_dec/emb_gru.linear_in.0.weight");
        relu_(&mut x);
        let x = self.gru_(&x, t, 256, "erb_dec", ("onnx::GRU_291", "onnx::GRU_292", "onnx::GRU_293"));
        let mut x = self.gl(&x, t, "erb_dec/emb_gru.linear_out.0.weight"); // [T,128]
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
        crate::simd::sigmoid_slice(&mut m);
        m // [T, 32]
    }

    // -------------------------------------------------------------- df decoder
    pub fn df_dec(&self, e: &EncOut, t: usize) -> Vec<f32> {
        let g = |n: &str| self.t(&format!("df_dec/{n}"));

        let mut x = self.gl(&e.emb, t, "df_dec/df_gru.linear_in.0.weight"); // [T,256]
        relu_(&mut x);
        let mut h = x;
        for ids in [
            ("onnx::GRU_337", "onnx::GRU_338", "onnx::GRU_339"),
            ("onnx::GRU_357", "onnx::GRU_358", "onnx::GRU_359"),
            ("onnx::GRU_377", "onnx::GRU_378", "onnx::GRU_379"),
        ] {
            h = self.gru_(&h, t, 256, "df_dec", ids);
        }

        let mut coefs = self.gl(&h, t, "df_dec/df_out.0.weight"); // [T, 640]
        crate::simd::tanh_slice(&mut coefs);

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

/// erb mask is produced as [T, NB_ERB]; helper keeps naming honest at the call site.
pub const _MASK_STRIDE: usize = NB_ERB;
