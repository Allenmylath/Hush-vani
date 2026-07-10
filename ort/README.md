# Hush — ONNX Runtime inference

ORT inference for [`weya-ai/hush`](https://huggingface.co/weya-ai/hush), a DeepFilterNet3-derived
speech enhancement / background-speaker suppression model (16 kHz, causal).

Output is validated **bit-accurately** against the reference denoised audio published
in the model repo (SI-SDR 71.7 dB, corr 1.000000 — matched to the int16 quantization
step of the reference WAV).

## Setup

```
pip install onnxruntime numpy soundfile deepfilterlib
python run_report.py
```

`deepfilterlib` provides `libdf`, the same Rust DSP used at training time (STFT, ERB
filterbank, exponential normalization, synthesis).

## What the ONNX bundle actually contains

`onnx/advanced_dfnet16k_model_best_onnx.tar.gz` unpacks to three graphs — the neural
network only. Everything else is host-side DSP.

| graph | opset | nodes | params | inputs | outputs |
|---|---|---|---|---|---|
| `enc.onnx` | 14 | 163 | 0.466M | `feat_erb[1,1,S,32]`, `feat_spec[1,2,S,64]` | `e0,e1,e2,e3,emb,c0,lsnr` |
| `erb_dec.onnx` | 14 | 104 | 0.462M | `emb[1,S,128]`, `e3,e2,e1,e0` | `m[1,1,S,32]` (sigmoid mask) |
| `df_dec.onnx` | 14 | 92 | 1.352M | `emb[1,S,128]`, `c0[1,16,S,64]` | `coefs[1,S,64,10]` |

2.28M params total, 9.18 MB on disk. `config.ini`: `sr=16000 fft=320 hop=160 nb_erb=32
nb_df=64 df_order=5 lookahead=0 norm_tau=1.0`.

## Pipeline (implemented in `hush_ort.py`)

1. `libdf` STFT → `spec [1,T,161]` complex. Verified perfect-reconstruction with
   **160-sample (10 ms) delay** (`probe_delay.py`, relerr 1.5e-14).
2. `feat_erb = erb_norm(erb(spec), alpha)`, `feat_spec = unit_norm(spec[...,:64], alpha)`
   as real/imag channels. `alpha = exp(-hop/sr/tau) = 0.990050`.
3. `enc` → `erb_dec` (mask) and `df_dec` (filter coefficients).
4. Mask path: `erb_inv(m)` expands 32 bands → 161 bins; applied to bins `64:`.
5. DF path: `coefs → [B,T,64,5,2]`, causal order-5 complex FIR over frames `t-4..t`;
   applied to bins `:64`.
6. `libdf` synthesis → waveform.

## Results (5.00 s sample, `sample_00006_raw.wav`, 1 thread)

Validation against the repo's own `sample_00006_denoised.wav`:

- SI-SDR **71.74 dB**, correlation **1.000000**, max abs diff **3.1e-5** (= int16 LSB)
- The reference file is delay-compensated; our raw output lags it by exactly 160 samples.

Enhancement effect:

- output RMS −29.38 dBFS vs input −23.87 dBFS (−5.51 dB)
- **noise floor reduced 42.0 dB** (p10 frame energy: −28.99 → −71.02 dB)
- speech level (p90) −3.40 dB; dynamic range 8.3 dB → 46.9 dB
- predicted `lsnr`: mean −4.05 dB, range [−14.7, +14.3]
- ERB mask mean 0.304; 15% of mask bins below 0.1

Speed (whole-utterance, `intra_op_num_threads=1`):

- median **74.2 ms** for 5.00 s audio → **RTF 0.0148**, ~67x real time
- **0.148 ms per 10 ms frame** — comfortably inside the model card's "<1 ms per 10 ms"
- stage split: `df_dec` 49%, `erb_dec` 25%, `enc` 20%, DSP 7%
- thread scaling is flat (2 threads ≈ 65 ms, 4 threads ≈ 76 ms — the graphs are too
  small to parallelize well)

`atten_lim_db` blends dry/wet (`spec*lim + enh*(1-lim)`, `lim = 10^(-|dB|/20)`):

| atten_lim | out RMS | noise reduction |
|---|---|---|
| none / 100 dB | −29.38 dBFS | +42.0 dB |
| 20 dB | −28.98 dBFS | +18.5 dB |
| 12 dB | −28.20 dBFS | +10.8 dB |
| 6 dB | −26.73 dBFS | +5.4 dB |

## Important: these graphs cannot be streamed frame-by-frame

The GRUs (1 in `enc`, 1 in `erb_dec`, 3 in `df_dec`) are baked in with **no exposed
`h0`/`hn` tensors**. Every `session.run()` restarts the recurrence from zero, so short
chunks both degrade output and cost more per frame (`chunk_test.py`):

| chunk (frames) | ms/frame | SI-SDR vs full-sequence | mask MAE |
|---|---|---|---|
| 1 | 1.86 | 4.04 dB | 0.0558 |
| 10 | 0.41 | 7.42 dB | 0.0450 |
| 50 | 0.26 | 13.04 dB | 0.0298 |
| 100 | 0.26 | 17.73 dB | 0.0225 |
| 500 (full) | 0.28 | exact | 0.0000 |

So ORT here is an **offline / full-utterance** path. For real-time streaming use either
the vendor's `weya_nc` C library (it keeps state internally, which is why its API is
`weya_nc_process_frame`), or re-export the graphs with GRU hidden states as explicit
inputs and outputs.

## Files

- `hush_ort.py` — the pipeline (`HushORT.enhance()`)
- `run_report.py` — validation + benchmark + signal stats (produces the numbers above)
- `chunk_test.py` — chunked/streaming degradation study
- `inspect_io.py`, `inspect_graph.py`, `probe_delay.py` — graph signatures, op counts, STFT delay
- `sample_enhanced_ort.wav` — our output
