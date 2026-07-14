//! `hush-vani` — denoise a mono 16 kHz wav file.

use hush_vani::Hush;
use std::process::ExitCode;

const USAGE: &str = "\
hush-vani — speech enhancement / background-speaker suppression

USAGE:
    hush-vani <input.wav> <output.wav> [OPTIONS]

OPTIONS:
    -w, --weights <DIR>   load weights.bin + weights.txt from DIR instead of the
                          int8 weights built into this binary [env: HUSH_WEIGHTS]
    -a, --atten <DB>      limit suppression to this many dB (e.g. 12); default: unlimited
    -h, --help            print this help

Input must be mono 16 kHz. Output lags the input by 160 samples (10 ms).";

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() || args.iter().any(|a| a == "-h" || a == "--help") {
        println!("{USAGE}");
        return Ok(());
    }

    let mut positional = Vec::new();
    // None => use the weights embedded in the binary. --weights (or HUSH_WEIGHTS) overrides,
    // which is how you run f32/f16 weights, or your own.
    let mut weights_dir: Option<String> = std::env::var("HUSH_WEIGHTS").ok();
    let mut atten: Option<f32> = None;
    let mut it = args.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "-w" | "--weights" => {
                weights_dir = Some(it.next().ok_or("--weights needs a directory")?.clone())
            }
            "-a" | "--atten" => atten = Some(it.next().ok_or("--atten needs a number")?.parse()?),
            _ if a.starts_with('-') => return Err(format!("unknown flag {a}").into()),
            _ => positional.push(a.clone()),
        }
    }
    if positional.len() != 2 {
        return Err("expected <input.wav> <output.wav>; see --help".into());
    }

    let hush = match &weights_dir {
        Some(d) => Hush::from_paths(format!("{d}/weights.bin"), format!("{d}/weights.txt"))?,
        None => Hush::new()?,
    };

    let mut rd = hound::WavReader::open(&positional[0])?;
    let spec = rd.spec();
    if spec.channels != 1 {
        return Err(format!("expected mono, got {} channels", spec.channels).into());
    }
    if spec.sample_rate != Hush::SAMPLE_RATE {
        return Err(format!("expected {} Hz, got {} Hz", Hush::SAMPLE_RATE, spec.sample_rate).into());
    }
    let audio: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Int => {
            let scale = 1.0 / (1i64 << (spec.bits_per_sample - 1)) as f32;
            rd.samples::<i32>().map(|s| s.map(|v| v as f32 * scale)).collect::<Result<_, _>>()?
        }
        hound::SampleFormat::Float => rd.samples::<f32>().collect::<Result<_, _>>()?,
    };

    let t0 = std::time::Instant::now();
    let out = hush.enhance_with(&audio, atten)?;
    let ms = t0.elapsed().as_secs_f64() * 1e3;

    let mut wr = hound::WavWriter::create(
        &positional[1],
        hound::WavSpec {
            channels: 1,
            sample_rate: Hush::SAMPLE_RATE,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        },
    )?;
    for &v in &out {
        wr.write_sample((v.clamp(-1.0, 1.0) * 32767.0).round() as i16)?;
    }
    wr.finalize()?;

    let dur = out.len() as f64 / Hush::SAMPLE_RATE as f64;
    eprintln!(
        "{:.2}s audio in {ms:.1} ms ({:.0}x realtime, avx2={})",
        dur,
        dur * 1e3 / ms,
        Hush::is_accelerated()
    );
    Ok(())
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}
