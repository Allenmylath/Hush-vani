//! Streaming enhancement: one 10 ms frame at a time, with the model's state carried over.
//!
//! [`Hush::enhance`] starts its GRUs, and the STFT's normalisation, from scratch on every call,
//! so re-running it over live chunks (even with a lookback window) never lets them settle: the
//! ERB and spectrum normalisations have a 1 s time constant. But the model is causal end to end:
//! the two 3x3 convs look two frames back, the deep filter four, every other conv is
//! frequency-only. So a frame needs only that history, the five GRU hidden states and the
//! [`Dsp`](crate::dsp::Dsp) state, which [`HushStream`] keeps. The output matches
//! [`Hush::enhance`] over the same audio to float rounding, with the same 160-sample lag.
//!
//! ```no_run
//! use hush_vani::{Hush, HushStream};
//! use std::sync::Arc;
//!
//! let hush = Arc::new(Hush::new()?);
//! let mut stream = HushStream::new(Arc::clone(&hush));
//! let mut clean = vec![0.0; 160];
//! stream.process_frame(&[0.0; 160], &mut clean); // 10 ms of 16 kHz mono in [-1, 1]
//! let more = stream.process(&[0.0; 1000]);        // any length: whole frames out, the rest kept
//! # Ok::<(), hush_vani::Error>(())
//! ```

use crate::alloc::AlignedVec;
use crate::bank::{WeightBank, HIDDEN};
use crate::dsp::{erb_inv, Dsp, DF_ORDER, FREQS, HOP, NB_DF, NB_ERB};
use crate::nn::{conv1x3, conv3x3_causal, convtranspose_dw, dot, dw_conv13, pointwise, relu_};
use crate::simd::{sigmoid, sigmoid_slice, tanh_slice};
use crate::Hush;
use rustfft::num_complex::Complex32;
use std::sync::Arc;

/// Frames the causal 3x3 convs see: t-2, t-1, t.
const CONV_T: usize = 3;

/// The GRUs, by weight-name prefix and tensor ids, in the order [`HushStream`] keeps their state.
const GRUS: [(&str, (&str, &str, &str)); 5] = [
    ("enc", ("onnx::GRU_360", "onnx::GRU_361", "onnx::GRU_362")),
    ("erb_dec", ("onnx::GRU_291", "onnx::GRU_292", "onnx::GRU_293")),
    ("df_dec", ("onnx::GRU_337", "onnx::GRU_338", "onnx::GRU_339")),
    ("df_dec", ("onnx::GRU_357", "onnx::GRU_358", "onnx::GRU_359")),
    ("df_dec", ("onnx::GRU_377", "onnx::GRU_378", "onnx::GRU_379")),
];

/// The last frame of a channels-major `(C, T, F)` map, as `(C, 1, F)`.
fn last_frame(x: &[f32], c: usize, t: usize, f: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(c * f);
    for ch in 0..c {
        let s = ch * t * f + (t - 1) * f;
        out.extend_from_slice(&x[s..s + f]);
    }
    out
}

/// Shift a `(C, T, F)` history one frame back and write `frame` (`(C, 1, F)`) as the newest.
fn push_frame(hist: &mut [f32], frame: &[f32], c: usize, t: usize, f: usize) {
    for ch in 0..c {
        let block = &mut hist[ch * t * f..(ch + 1) * t * f];
        block.copy_within(f.., 0);
        block[(t - 1) * f..].copy_from_slice(&frame[ch * f..(ch + 1) * f]);
    }
}

/// [`Hush`] as a stream. Feed 16 kHz mono in `[-1, 1]`; output lags input by
/// [`Hush::LATENCY_SAMPLES`], exactly as [`Hush::enhance`]'s does. One per audio stream; the
/// [`Hush`] (the weights) can be shared between any number of them.
pub struct HushStream {
    hush: Arc<Hush>,
    dsp: Dsp,
    gru: [AlignedVec; 5],
    erb_hist: Vec<f32>,          // (1, 3, NB_ERB)
    spec_hist: Vec<f32>,         // (2, 3, NB_DF)
    spec_ring: Vec<Complex32>,   // the last DF_ORDER spectra, oldest first
    pending: Vec<f32>,           // input short of a whole frame ([`process`](Self::process))
    atten_lim: Option<f32>,
    lsnr: f32,
}

impl HushStream {
    /// A stream over `hush`'s weights, starting from silence.
    pub fn new(hush: Arc<Hush>) -> Self {
        Self {
            hush,
            dsp: Dsp::new(),
            gru: std::array::from_fn(|_| AlignedVec::zeros(HIDDEN)),
            erb_hist: vec![0.0; CONV_T * NB_ERB],
            spec_hist: vec![0.0; 2 * CONV_T * NB_DF],
            spec_ring: vec![Complex32::new(0.0, 0.0); DF_ORDER * FREQS],
            pending: Vec::new(),
            atten_lim: None,
            lsnr: -15.0,
        }
    }

    /// Cap suppression at `db` decibels by mixing the noisy signal back in (`None` or 0: full),
    /// as [`Hush::enhance_with`] does.
    pub fn set_atten_lim_db(&mut self, db: Option<f32>) {
        self.atten_lim = db.filter(|d| d.abs() > 0.0).map(|d| 10f32.powf(-d.abs() / 20.0));
    }

    /// The model's predicted local SNR for the last frame, in dB.
    pub fn lsnr(&self) -> f32 {
        self.lsnr
    }

    /// Forget everything heard, as if new.
    pub fn reset(&mut self) {
        let hush = Arc::clone(&self.hush);
        let atten_lim = self.atten_lim;
        *self = Self::new(hush);
        self.atten_lim = atten_lim;
    }

    /// Enhance `audio` of any length: whole frames come out (lagging by
    /// [`Hush::LATENCY_SAMPLES`]); a partial frame is kept for the next call.
    pub fn process(&mut self, audio: &[f32]) -> Vec<f32> {
        self.pending.extend_from_slice(audio);
        let frames = self.pending.len() / HOP;
        let mut out = vec![0f32; frames * HOP];
        let input = std::mem::take(&mut self.pending);
        for (i, o) in input.chunks_exact(HOP).zip(out.chunks_exact_mut(HOP)) {
            self.process_frame(i, o);
        }
        self.pending = input[frames * HOP..].to_vec();
        out
    }

    /// Enhance one 10 ms frame ([`Hush::FRAME_SAMPLES`] samples) into `out`.
    pub fn process_frame(&mut self, input: &[f32], out: &mut [f32]) {
        assert_eq!(input.len(), HOP, "a frame is {HOP} samples");
        let spec = self.dsp.analysis(input);
        let feat_erb = self.dsp.erb_feat(&spec, 1);
        let feat_spec = self.dsp.spec_feat(&spec, 1);
        push_frame(&mut self.erb_hist, &feat_erb, 1, CONV_T, NB_ERB);
        push_frame(&mut self.spec_hist, &feat_spec, 2, CONV_T, NB_DF);
        self.spec_ring.copy_within(FREQS.., 0);
        self.spec_ring[(DF_ORDER - 1) * FREQS..].copy_from_slice(&spec);

        let hush = Arc::clone(&self.hush);
        let enc: &WeightBank = hush.encoder.bank();
        let dec: &WeightBank = hush.decoders.bank();
        let [g_enc, g_erb, g_df0, g_df1, g_df2] = &mut self.gru;

        // ---- encoder (see `Encoder::encode`)
        let g = |n: &str| enc.t(&format!("enc/{n}"));
        let e0 = conv3x3_causal(&self.erb_hist, 1, CONV_T, 32, g("onnx::Conv_310"), Some(g("onnx::Conv_311")), 16, 1);
        let mut e0 = last_frame(&e0, 16, CONV_T, 32);
        relu_(&mut e0);
        let erb_block = |x: &[f32], f: usize, dw: &str, pw: &str, pb: &str, stride: usize| {
            let (y, fo) = dw_conv13(x, 16, 1, f, g(dw), stride);
            let mut z = pointwise(&y, 16, fo, g(pw), Some(g(pb)), 16);
            relu_(&mut z);
            z
        };
        let e1 = erb_block(&e0, 32, "erb_conv1.0.weight", "onnx::Conv_313", "onnx::Conv_314", 2);
        let e2 = erb_block(&e1, 16, "erb_conv2.0.weight", "onnx::Conv_316", "onnx::Conv_317", 2);
        let e3 = erb_block(&e2, 8, "erb_conv3.0.weight", "onnx::Conv_319", "onnx::Conv_320", 1);

        let c = conv3x3_causal(&self.spec_hist, 2, CONV_T, NB_DF, g("df_conv0.1.weight"), None, 16, 2);
        let c = last_frame(&c, 16, CONV_T, NB_DF);
        let mut c0 = pointwise(&c, 16, NB_DF, g("onnx::Conv_322"), Some(g("onnx::Conv_323")), 16);
        relu_(&mut c0);
        let (cx, fo) = dw_conv13(&c0, 16, 1, NB_DF, g("df_conv1.0.weight"), 2);
        let mut cx = pointwise(&cx, 16, fo, g("onnx::Conv_325"), Some(g("onnx::Conv_326")), 16);
        relu_(&mut cx);
        let mut cf = vec![0f32; fo * 16];
        for ch in 0..16 {
            for f in 0..fo {
                cf[f * 16 + ch] = cx[ch * fo + f];
            }
        }
        let mut emb = enc.gl(&cf, 1, "enc/df_fc_emb.0.weight");
        relu_(&mut emb);
        for ch in 0..16 {
            for f in 0..8 {
                emb[f * 16 + ch] += e3[ch * 8 + f];
            }
        }
        let mut x = enc.gl(&emb, 1, "enc/emb_gru.linear_in.0.weight");
        relu_(&mut x);
        enc.gru_step(&x, HIDDEN, GRUS[0].0, GRUS[0].1, g_enc);
        let mut emb = enc.gl(g_enc, 1, "enc/emb_gru.linear_out.0.weight");
        relu_(&mut emb);
        self.lsnr = sigmoid(dot(&emb, g("onnx::MatMul_365")) + g("lsnr_fc.0.bias")[0]) * 50.0 - 15.0;

        // ---- ERB decoder (see `Decoders::erb_dec`)
        let g = |n: &str| dec.t(&format!("erb_dec/{n}"));
        let mut x = dec.gl(&emb, 1, "erb_dec/emb_gru.linear_in.0.weight");
        relu_(&mut x);
        dec.gru_step(&x, HIDDEN, GRUS[1].0, GRUS[1].1, g_erb);
        let mut x = dec.gl(g_erb, 1, "erb_dec/emb_gru.linear_out.0.weight");
        relu_(&mut x);
        let mut y = vec![0f32; 16 * 8];
        for ch in 0..16 {
            for f in 0..8 {
                y[ch * 8 + f] = x[f * 16 + ch];
            }
        }
        let skip = |acc: &mut Vec<f32>, e: &[f32], f: usize, pw: &str, pb: &str| {
            let mut s = pointwise(e, 16, f, g(pw), Some(g(pb)), 16);
            relu_(&mut s);
            for (a, v) in acc.iter_mut().zip(s.iter()) {
                *a += v;
            }
        };
        skip(&mut y, &e3, 8, "onnx::Conv_247", "onnx::Conv_248");
        let (y, _) = dw_conv13(&y, 16, 1, 8, g("convt3.0.weight"), 1);
        let mut y = pointwise(&y, 16, 8, g("onnx::Conv_250"), Some(g("onnx::Conv_251")), 16);
        relu_(&mut y);
        skip(&mut y, &e2, 8, "onnx::Conv_253", "onnx::Conv_254");
        let (y, f) = convtranspose_dw(&y, 16, 1, 8, g("convt2.0.weight"));
        let mut y = pointwise(&y, 16, f, g("onnx::Conv_256"), Some(g("onnx::Conv_257")), 16);
        relu_(&mut y);
        skip(&mut y, &e1, 16, "onnx::Conv_259", "onnx::Conv_260");
        let (y, f) = convtranspose_dw(&y, 16, 1, 16, g("convt1.0.weight"));
        let mut y = pointwise(&y, 16, f, g("onnx::Conv_262"), Some(g("onnx::Conv_263")), 16);
        relu_(&mut y);
        skip(&mut y, &e0, 32, "onnx::Conv_265", "onnx::Conv_266");
        let mut mask = conv1x3(&y, 16, 1, NB_ERB, g("onnx::Conv_268"), g("onnx::Conv_269"), 1);
        sigmoid_slice(&mut mask);

        // ---- deep-filter decoder (see `Decoders::df_dec`)
        let mut x = dec.gl(&emb, 1, "df_dec/df_gru.linear_in.0.weight");
        relu_(&mut x);
        dec.gru_step(&x, HIDDEN, GRUS[2].0, GRUS[2].1, g_df0);
        dec.gru_step(g_df0, HIDDEN, GRUS[3].0, GRUS[3].1, g_df1);
        dec.gru_step(g_df1, HIDDEN, GRUS[4].0, GRUS[4].1, g_df2);
        let mut coefs = dec.gl(g_df2, 1, "df_dec/df_out.0.weight");
        tanh_slice(&mut coefs);
        let mut cp = pointwise(&c0, 16, NB_DF, dec.t("df_dec/onnx::Conv_314"), Some(dec.t("df_dec/onnx::Conv_315")), 10);
        relu_(&mut cp);
        for k in 0..10 {
            for f in 0..NB_DF {
                coefs[f * 10 + k] += cp[k * NB_DF + f];
            }
        }

        // ---- merge (see `Hush::enhance_with`)
        let gains = erb_inv(&mask, 1);
        let mut out_spec: Vec<Complex32> = spec.iter().zip(gains.iter()).map(|(s, g)| s * g).collect();
        for f in 0..NB_DF {
            let mut acc = Complex32::new(0.0, 0.0);
            for o in 0..DF_ORDER {
                let s = self.spec_ring[o * FREQS + f];
                let c = coefs[f * 2 * DF_ORDER + o * 2];
                let d = coefs[f * 2 * DF_ORDER + o * 2 + 1];
                acc.re += s.re * c - s.im * d;
                acc.im += s.im * c + s.re * d;
            }
            out_spec[f] = acc;
        }
        if let Some(lim) = self.atten_lim {
            for (o, s) in out_spec.iter_mut().zip(spec.iter()) {
                *o = s * lim + *o * (1.0 - lim);
            }
        }
        out.copy_from_slice(&self.dsp.synthesis(&out_spec, 1));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Harmonics with a wobbling pitch, gated like syllables, under broadband noise.
    fn noisy_speech(n: usize) -> Vec<f32> {
        let mut seed = 0x1234_5678u32;
        (0..n)
            .map(|i| {
                let t = i as f32 / 16_000.0;
                let f0 = 140.0 + 30.0 * (2.0 * std::f32::consts::PI * 3.0 * t).sin();
                let gate = if (t * 4.0) as u32 % 3 == 2 { 0.0 } else { 1.0 };
                let voice: f32 = (1..8).map(|k| (2.0 * std::f32::consts::PI * f0 * k as f32 * t).sin() / k as f32).sum();
                seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                let noise = (seed >> 8) as f32 / (1u32 << 24) as f32 - 0.5;
                0.2 * gate * voice + 0.15 * noise
            })
            .collect()
    }

    fn snr_db(reference: &[f32], x: &[f32]) -> f32 {
        let err: f32 = reference.iter().zip(x).map(|(a, b)| (a - b) * (a - b)).sum();
        let sig: f32 = reference.iter().map(|a| a * a).sum();
        10.0 * (sig / err.max(1e-30)).log10()
    }

    /// The stream is the batch model, frame for frame.
    #[test]
    fn matches_enhance() {
        let hush = Arc::new(Hush::new().unwrap());
        let audio = noisy_speech(16_000 * 3);
        let batch = hush.enhance(&audio).unwrap();
        let mut s = HushStream::new(Arc::clone(&hush));
        let mut streamed = vec![0f32; batch.len()];
        for (i, o) in audio.chunks_exact(HOP).zip(streamed.chunks_exact_mut(HOP)) {
            s.process_frame(i, o);
        }
        let snr = snr_db(&batch, &streamed);
        assert!(snr > 60.0, "stream drifts from enhance: {snr:.1} dB");
    }

    /// Odd-sized chunks come out the same as whole frames; reset starts over.
    #[test]
    fn chunks_and_reset() {
        let hush = Arc::new(Hush::new().unwrap());
        let audio = noisy_speech(16_000);
        let mut a = HushStream::new(Arc::clone(&hush));
        let whole = a.process(&audio);
        let mut b = HushStream::new(Arc::clone(&hush));
        let mut chunked = Vec::new();
        for c in audio.chunks(333) {
            chunked.extend(b.process(c));
        }
        assert_eq!(whole, chunked);
        a.reset();
        assert_eq!(a.process(&audio), whole);
    }
}
