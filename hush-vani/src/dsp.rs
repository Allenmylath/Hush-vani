//! STFT / ERB feature extraction / synthesis, matching libdf (DeepFilterNet) exactly.
//! Formulas verified numerically against libdf 0.5.6 -- see tools/probe_dsp.py.

use rustfft::{num_complex::Complex32, Fft, FftPlanner};
use std::sync::Arc;

pub const SR: usize = 16000;
pub const FFT: usize = 320;
pub const HOP: usize = 160;
pub const FREQS: usize = FFT / 2 + 1; // 161
pub const NB_ERB: usize = 32;
pub const NB_DF: usize = 64;
pub const DF_ORDER: usize = 5;
pub const ERB_WIDTHS: [usize; NB_ERB] = [
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 6, 7, 7, 8, 8, 10, 12, 12, 14,
    16, 18,
];

pub struct Dsp {
    window: Vec<f32>,
    fwd: Arc<dyn Fft<f32>>,
    inv: Arc<dyn Fft<f32>>,
    alpha: f32,
    erb_state: [f32; NB_ERB],
    unit_state: [f32; NB_DF],
    ana_buf: Vec<f32>,
    syn_buf: Vec<f32>,
    scratch: Vec<Complex32>,
}

impl Dsp {
    pub fn new() -> Self {
        let mut p = FftPlanner::new();
        // vorbis window: sin(pi/2 * sin^2(pi (i+0.5) / N))
        let window = (0..FFT)
            .map(|i| {
                let s = (std::f64::consts::PI * (i as f64 + 0.5) / FFT as f64).sin();
                ((std::f64::consts::PI / 2.0) * s * s).sin() as f32
            })
            .collect();
        let alpha = (-(HOP as f64 / SR as f64) / 1.0).exp() as f32;

        let mut erb_state = [0f32; NB_ERB];
        for (i, s) in erb_state.iter_mut().enumerate() {
            *s = -60.0 + (-90.0 - -60.0) * (i as f32) / (NB_ERB as f32 - 1.0);
        }
        let mut unit_state = [0f32; NB_DF];
        for (i, s) in unit_state.iter_mut().enumerate() {
            *s = 0.001 + (0.0001 - 0.001) * (i as f32) / (NB_DF as f32 - 1.0);
        }

        Dsp {
            window,
            fwd: p.plan_fft_forward(FFT),
            inv: p.plan_fft_inverse(FFT),
            alpha,
            erb_state,
            unit_state,
            ana_buf: vec![0.0; FFT],
            syn_buf: vec![0.0; FFT],
            scratch: vec![Complex32::new(0.0, 0.0); FFT],
        }
    }

    /// audio -> spec [T][FREQS] complex, scaled by 1/FFT
    pub fn analysis(&mut self, audio: &[f32]) -> Vec<Complex32> {
        let t = audio.len() / HOP;
        let mut spec = vec![Complex32::new(0.0, 0.0); t * FREQS];
        let inv_n = 1.0 / FFT as f32;
        for ti in 0..t {
            self.ana_buf.copy_within(HOP.., 0);
            self.ana_buf[FFT - HOP..].copy_from_slice(&audio[ti * HOP..(ti + 1) * HOP]);
            for i in 0..FFT {
                self.scratch[i] = Complex32::new(self.ana_buf[i] * self.window[i], 0.0);
            }
            self.fwd.process(&mut self.scratch);
            for f in 0..FREQS {
                spec[ti * FREQS + f] = self.scratch[f] * inv_n;
            }
        }
        spec
    }

    /// spec [T][FREQS] -> audio, overlap-add. Introduces one hop (10 ms) of delay.
    pub fn synthesis(&mut self, spec: &[Complex32], t: usize) -> Vec<f32> {
        let mut out = vec![0f32; t * HOP];
        for ti in 0..t {
            for f in 0..FREQS {
                self.scratch[f] = spec[ti * FREQS + f];
            }
            for f in FREQS..FFT {
                self.scratch[f] = spec[ti * FREQS + (FFT - f)].conj();
            }
            self.inv.process(&mut self.scratch);
            // rustfft's inverse is unnormalised => equals numpy.irfft * FFT, which is what libdf uses
            for i in 0..FFT {
                self.syn_buf[i] += self.scratch[i].re * self.window[i];
            }
            out[ti * HOP..(ti + 1) * HOP].copy_from_slice(&self.syn_buf[..HOP]);
            self.syn_buf.copy_within(HOP.., 0);
            self.syn_buf[FFT - HOP..].fill(0.0);
        }
        out
    }

    /// ERB band energies in dB, then exponential mean normalisation. -> [T][NB_ERB]
    pub fn erb_feat(&mut self, spec: &[Complex32], t: usize) -> Vec<f32> {
        let mut out = vec![0f32; t * NB_ERB];
        for ti in 0..t {
            let row = &spec[ti * FREQS..(ti + 1) * FREQS];
            let mut f0 = 0;
            for (b, &wdt) in ERB_WIDTHS.iter().enumerate() {
                let mut e = 0f32;
                for f in f0..f0 + wdt {
                    e += row[f].re * row[f].re + row[f].im * row[f].im;
                }
                f0 += wdt;
                let db = 10.0 * (e / wdt as f32 + 1e-10).log10();
                let s = &mut self.erb_state[b];
                *s = db * (1.0 - self.alpha) + *s * self.alpha;
                out[ti * NB_ERB + b] = (db - *s) / 40.0;
            }
        }
        out
    }

    /// Unit-norm the first NB_DF bins. -> [2][T][NB_DF] (re plane, then im plane)
    pub fn spec_feat(&mut self, spec: &[Complex32], t: usize) -> Vec<f32> {
        let mut out = vec![0f32; 2 * t * NB_DF];
        for ti in 0..t {
            for f in 0..NB_DF {
                let x = spec[ti * FREQS + f];
                let mag = (x.re * x.re + x.im * x.im).sqrt();
                let s = &mut self.unit_state[f];
                *s = mag * (1.0 - self.alpha) + *s * self.alpha;
                let d = s.sqrt();
                out[ti * NB_DF + f] = x.re / d;
                out[t * NB_DF + ti * NB_DF + f] = x.im / d;
            }
        }
        out
    }
}

/// Expand a 32-band ERB mask across the 161 frequency bins (libdf erb_inv).
pub fn erb_inv(m: &[f32], t: usize) -> Vec<f32> {
    let mut g = vec![0f32; t * FREQS];
    for ti in 0..t {
        let mut f0 = 0;
        for (b, &wdt) in ERB_WIDTHS.iter().enumerate() {
            let v = m[ti * NB_ERB + b];
            for f in f0..f0 + wdt {
                g[ti * FREQS + f] = v;
            }
            f0 += wdt;
        }
    }
    g
}

/// Causal order-5 complex FIR across frames, applied to the first NB_DF bins.
/// coefs [T][NB_DF][2*DF_ORDER], laid out (order, re/im) in the last axis.
pub fn apply_df(spec: &[Complex32], coefs: &[f32], t: usize, out: &mut [Complex32]) {
    for ti in 0..t {
        for f in 0..NB_DF {
            let mut acc = Complex32::new(0.0, 0.0);
            for o in 0..DF_ORDER {
                let tt = ti as isize + o as isize - (DF_ORDER as isize - 1);
                if tt < 0 {
                    continue;
                }
                let s = spec[tt as usize * FREQS + f];
                let c = coefs[(ti * NB_DF + f) * 2 * DF_ORDER + o * 2];
                let d = coefs[(ti * NB_DF + f) * 2 * DF_ORDER + o * 2 + 1];
                acc.re += s.re * c - s.im * d;
                acc.im += s.im * c + s.re * d;
            }
            out[ti * FREQS + f] = acc;
        }
    }
}
