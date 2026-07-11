//! Development tool: correctness against the ONNX Runtime fixtures, and benchmarks.
//! Requires `assets/` (weights + fixtures); not shipped in the published crate.

use hush_vani::{Decoders, Encoder, Hush};
use hush_vani_core::dsp::*;
use hush_vani_core::Weights;
use rustfft::num_complex::Complex32;
use std::sync::Arc;
use std::time::Instant;

fn assets(name: &str) -> String {
    format!("{}/assets/{}", env!("CARGO_MANIFEST_DIR"), name)
}

fn load_bin(name: &str) -> Vec<f32> {
    let raw = std::fs::read(assets(name)).unwrap_or_else(|_| panic!("missing fixture {name}"));
    raw.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

/// The two stages, built over one shared weight arena.
struct Net {
    enc: Encoder,
    dec: Decoders,
}

fn load_model() -> Net {
    let w = Arc::new(
        Weights::from_paths(assets("weights.bin"), assets("weights.txt")).expect("weights"),
    );
    Net {
        enc: Encoder::new(Arc::clone(&w)).expect("encoder"),
        dec: Decoders::new(w).expect("decoders"),
    }
}

fn read_wav(path: &str) -> (Vec<f32>, u32) {
    let mut r = hound::WavReader::open(path).expect("open wav");
    let spec = r.spec();
    let s: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Int => r.samples::<i16>().map(|x| x.unwrap() as f32 / 32768.0).collect(),
        hound::SampleFormat::Float => r.samples::<f32>().map(|x| x.unwrap()).collect(),
    };
    (s, spec.sample_rate)
}

struct Timing {
    dsp_ms: f64,
    enc_ms: f64,
    dec_ms: f64,
    post_ms: f64,
}

/// `par`: run erb_dec and df_dec concurrently. They share only `emb`.
fn enhance(m: &Net, audio: &[f32], par: bool) -> (Vec<f32>, Timing) {
    let mut d = Dsp::new();
    let t = audio.len() / HOP;

    let t0 = Instant::now();
    let spec = d.analysis(audio);
    let feat_erb = d.erb_feat(&spec, t);
    let feat_spec = d.spec_feat(&spec, t);
    let dsp_ms = t0.elapsed().as_secs_f64() * 1e3;

    let t0 = Instant::now();
    let e = m.enc.encode(&feat_erb, &feat_spec, t);
    let enc_ms = t0.elapsed().as_secs_f64() * 1e3;

    let t0 = Instant::now();
    let (mask, coefs) = if par {
        std::thread::scope(|s| {
            let h = s.spawn(|| m.dec.erb_dec(&e, t));
            let c = m.dec.df_dec(&e, t);
            (h.join().unwrap(), c)
        })
    } else {
        (m.dec.erb_dec(&e, t), m.dec.df_dec(&e, t))
    };
    let dec_ms = t0.elapsed().as_secs_f64() * 1e3;

    let t0 = Instant::now();
    let gains = erb_inv(&mask, t);
    let mut out_spec = vec![Complex32::new(0.0, 0.0); t * FREQS];
    for i in 0..t * FREQS {
        out_spec[i] = spec[i] * gains[i];
    }
    apply_df(&spec, &coefs, t, &mut out_spec);
    let out = d.synthesis(&out_spec, t);
    let post_ms = t0.elapsed().as_secs_f64() * 1e3;

    (out, Timing { dsp_ms, enc_ms, dec_ms, post_ms })
}

fn maxdiff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "length mismatch");
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max)
}

fn si_sdr(est: &[f32], reference: &[f32]) -> f64 {
    let n = est.len().min(reference.len());
    let (e, r) = (&est[..n], &reference[..n]);
    let me = e.iter().sum::<f32>() as f64 / n as f64;
    let mr = r.iter().sum::<f32>() as f64 / n as f64;
    let (mut er, mut rr) = (0.0f64, 0.0f64);
    for i in 0..n {
        er += (e[i] as f64 - me) * (r[i] as f64 - mr);
        rr += (r[i] as f64 - mr).powi(2);
    }
    let a = er / rr;
    let (mut sp, mut sn) = (0.0f64, 0.0f64);
    for i in 0..n {
        let p = a * (r[i] as f64 - mr);
        sp += p * p;
        sn += (e[i] as f64 - me - p).powi(2);
    }
    10.0 * (sp / (sn + 1e-20)).log10()
}

fn cmd_verify() {
    let m = load_model();
    let (audio, _) = read_wav(&assets("../sample_raw.wav"));
    let t = audio.len() / HOP;

    let mut d = Dsp::new();
    let spec = d.analysis(&audio);
    let feat_erb = d.erb_feat(&spec, t);
    let feat_spec = d.spec_feat(&spec, t);

    let re: Vec<f32> = spec.iter().map(|c| c.re).collect();
    let im: Vec<f32> = spec.iter().map(|c| c.im).collect();
    println!("avx2 = {}", Hush::is_accelerated());
    println!("DSP vs libdf:");
    println!("  spec.re     max|diff| = {:.3e}", maxdiff(&re, &load_bin("spec_re.bin")));
    println!("  spec.im     max|diff| = {:.3e}", maxdiff(&im, &load_bin("spec_im.bin")));
    println!("  feat_erb    max|diff| = {:.3e}", maxdiff(&feat_erb, &load_bin("feat_erb.bin")));
    println!("  feat_spec   max|diff| = {:.3e}", maxdiff(&feat_spec, &load_bin("feat_spec.bin")));

    let e = m.enc.encode(&feat_erb, &feat_spec, t);
    println!("NN vs onnxruntime:");
    for (n, v) in [("e0", &e.e0), ("e1", &e.e1), ("e2", &e.e2), ("e3", &e.e3), ("c0", &e.c0),
                   ("emb", &e.emb), ("lsnr", &e.lsnr)] {
        println!("  {n:11} max|diff| = {:.3e}", maxdiff(v, &load_bin(&format!("{n}.bin"))));
    }
    println!("  {:11} max|diff| = {:.3e}", "mask", maxdiff(&m.dec.erb_dec(&e, t), &load_bin("mask.bin")));
    println!("  {:11} max|diff| = {:.3e}", "coefs", maxdiff(&m.dec.df_dec(&e, t), &load_bin("coefs.bin")));

    let (out, _) = enhance(&m, &audio, false);
    let refa = load_bin("enhanced.bin");
    println!("end-to-end audio vs onnxruntime:");
    println!("  max|diff| = {:.3e}   SI-SDR = {:.1} dB", maxdiff(&out, &refa), si_sdr(&out, &refa));
}

fn tile(audio: &[f32], sr: u32, secs: usize) -> Vec<f32> {
    let mut a = Vec::new();
    while a.len() < secs * sr as usize {
        a.extend_from_slice(audio);
    }
    a.truncate(secs * sr as usize);
    a
}

fn cmd_bench(secs: usize) {
    let m = load_model();
    let (audio, sr) = read_wav(&assets("../sample_raw.wav"));
    let a = tile(&audio, sr, secs);
    let dur = a.len() as f64 / sr as f64;
    let t = a.len() / HOP;

    println!("audio {dur:.2}s ({t} frames), median of 15 runs\n");
    for (label, par) in [("1 thread", false), ("2 threads", true)] {
        for _ in 0..3 {
            enhance(&m, &a, par);
        }
        let n = 15;
        let mut times = Vec::new();
        let mut last = None;
        for _ in 0..n {
            let t0 = Instant::now();
            let (_, tm) = enhance(&m, &a, par);
            times.push(t0.elapsed().as_secs_f64() * 1e3);
            last = Some(tm);
        }
        times.sort_by(|x, y| x.partial_cmp(y).unwrap());
        let med = times[n / 2];
        let tm = last.unwrap();
        println!("{label}:");
        println!("  total      {med:8.2} ms   RTF={:.5}  {:.0}x realtime", med / 1e3 / dur, 1e3 * dur / med);
        println!("  min/max    {:8.2} / {:.2} ms", times[0], times[n - 1]);
        println!("  per frame  {:8.4} ms", med / t as f64);
        println!("  breakdown: dsp={:.2} enc={:.2} decoders={:.2} post={:.2}",
                 tm.dsp_ms, tm.enc_ms, tm.dec_ms, tm.post_ms);
    }
}

/// One wall-clock ms per line, for external statistical comparison.
fn cmd_raw(secs: usize, reps: usize, threads: usize, nn_only: bool) {
    let m = load_model();
    let (audio, sr) = read_wav(&assets("../sample_raw.wav"));
    let a = tile(&audio, sr, secs);
    let par = threads > 1;

    if nn_only {
        let t = a.len() / HOP;
        let mut d = Dsp::new();
        let spec = d.analysis(&a);
        let feat_erb = d.erb_feat(&spec, t);
        let feat_spec = d.spec_feat(&spec, t);
        let nn = |par: bool| {
            let e = m.enc.encode(&feat_erb, &feat_spec, t);
            if par {
                std::thread::scope(|s| {
                    let h = s.spawn(|| m.dec.erb_dec(&e, t));
                    let c = m.dec.df_dec(&e, t);
                    (h.join().unwrap(), c)
                })
            } else {
                (m.dec.erb_dec(&e, t), m.dec.df_dec(&e, t))
            }
        };
        for _ in 0..3 {
            nn(par);
        }
        for _ in 0..reps {
            let t0 = Instant::now();
            let o = nn(par);
            println!("{:.4}", t0.elapsed().as_secs_f64() * 1e3);
            std::hint::black_box(&o);
        }
        return;
    }

    for _ in 0..3 {
        enhance(&m, &a, par);
    }
    for _ in 0..reps {
        let t0 = Instant::now();
        let (o, _) = enhance(&m, &a, par);
        println!("{:.4}", t0.elapsed().as_secs_f64() * 1e3);
        std::hint::black_box(&o);
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let n = |i: usize, d: usize| args.get(i).and_then(|s| s.parse().ok()).unwrap_or(d);
    match args.get(1).map(|s| s.as_str()) {
        Some("verify") => cmd_verify(),
        Some("bench") => cmd_bench(n(2, 5)),
        Some("raw") => cmd_raw(n(2, 5), n(3, 50), n(4, 1), args.get(5).map(|s| s == "nn").unwrap_or(false)),
        _ => eprintln!("usage: hush-devtool [verify | bench [secs] | raw <secs> <reps> <threads> [nn]]"),
    }
}

