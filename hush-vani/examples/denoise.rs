//! The whole API: load the model (weights are embedded) and denoise a wav.
//! Run with: cargo run --release --example denoise
use std::io::Write;

fn main() {
    let hush = hush_vani::Hush::new().expect("load model");
    println!("model loaded; avx2={}", hush_vani::Hush::is_accelerated());

    // read the sample wav shipped in the crate dir
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/sample_raw.wav");
    let mut rd = hound::WavReader::open(path).expect("open sample");
    let audio: Vec<f32> = rd.samples::<i16>().map(|s| s.unwrap() as f32 / 32768.0).collect();

    let out = hush.enhance(&audio).expect("enhance");
    let rms = (out.iter().map(|x| x * x).sum::<f32>() / out.len() as f32).sqrt();
    println!("enhanced {} samples, output rms {:.1} dBFS", out.len(), 20.0 * rms.log10());

    // write it so it can be listened to / compared
    let outp = concat!(env!("CARGO_MANIFEST_DIR"), "/sample_enhanced.wav");
    let spec = hound::WavSpec { channels: 1, sample_rate: 16000, bits_per_sample: 16, sample_format: hound::SampleFormat::Int };
    let mut w = hound::WavWriter::create(outp, spec).unwrap();
    for &v in &out {
        w.write_sample((v.clamp(-1.0, 1.0) * 32767.0).round() as i16).unwrap();
    }
    w.finalize().unwrap();
    print!("wrote {outp}");
    std::io::stdout().flush().ok();
}
