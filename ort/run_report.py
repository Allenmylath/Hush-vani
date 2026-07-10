"""Run Hush ORT inference on the HF sample, validate against the reference
denoised file, benchmark, and print a report."""

import json
import os
import time

import numpy as np
import onnxruntime as ort
import soundfile as sf

from hush_ort import HERE, DELAY, HushORT


def db(x):
    return 20 * np.log10(max(float(x), 1e-12))


def rms(x):
    return float(np.sqrt(np.mean(np.square(x)) + 1e-20))


def best_delay(y, x, max_d=512):
    """delay of y relative to x, by normalised cross-correlation"""
    best = (0, -2.0)
    for d in range(max_d + 1):
        n = min(len(x), len(y) - d)
        if n < 1000:
            break
        a, b = y[d : d + n], x[:n]
        c = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
        if c > best[1]:
            best = (d, c)
    return best


def si_sdr(est, ref):
    ref = ref - ref.mean()
    est = est - est.mean()
    a = np.dot(est, ref) / (np.dot(ref, ref) + 1e-12)
    proj = a * ref
    noise = est - proj
    return 10 * np.log10((np.dot(proj, proj) + 1e-12) / (np.dot(noise, noise) + 1e-12))


def main():
    noisy, sr = sf.read(os.path.join(HERE, "sample_raw.wav"), dtype="float32")
    ref, sr2 = sf.read(os.path.join(HERE, "sample_denoised_ref.wav"), dtype="float32")
    assert sr == sr2 == 16000
    dur = len(noisy) / sr

    model = HushORT(intra_threads=1, inter_threads=1)

    print("=" * 78)
    print("HUSH — ONNX Runtime inference report")
    print("=" * 78)
    print(f"onnxruntime {ort.__version__} | providers: {ort.get_available_providers()}")
    print(f"config: sr={model.sr} fft={model.fft_size} hop={model.hop_size} "
          f"nb_erb={model.nb_erb} nb_df={model.nb_df} df_order={model.df_order} "
          f"lookahead={model.df_lookahead} alpha={model.alpha:.6f}")
    sizes = {n: os.path.getsize(os.path.join(model.bundle_dir, n)) / 1e6
             for n in ["enc.onnx", "erb_dec.onnx", "df_dec.onnx"]}
    print("graphs: " + "  ".join(f"{k}={v:.2f}MB" for k, v in sizes.items())
          + f"  total={sum(sizes.values()):.2f}MB")
    print(f"input : sample_raw.wav  {len(noisy)} samples  {dur:.2f}s  "
          f"rms={db(rms(noisy)):.1f} dBFS  peak={db(np.abs(noisy).max()):.1f} dBFS")
    print()

    # ---------------- inference ----------------
    out, aux = model.enhance(noisy, collect=True)
    sf.write(os.path.join(HERE, "sample_enhanced_ort.wav"), out, sr)

    T = aux["timing"]["frames"]
    print("--- inference -------------------------------------------------------")
    print(f"frames: {T}  ({T} x 10ms)   output: {len(out)} samples")
    print(f"lsnr   : mean={aux['lsnr'].mean():+6.2f} dB  "
          f"p10={np.percentile(aux['lsnr'],10):+.2f}  "
          f"p50={np.percentile(aux['lsnr'],50):+.2f}  "
          f"p90={np.percentile(aux['lsnr'],90):+.2f}  "
          f"range=[{aux['lsnr'].min():+.2f},{aux['lsnr'].max():+.2f}]")
    m = aux["mask"]
    print(f"erb mask: mean={m.mean():.4f} min={m.min():.4f} max={m.max():.4f}  "
          f"frac<0.1={float((m<0.1).mean()):.3f}  frac>0.9={float((m>0.9).mean()):.3f}")
    c = aux["coefs"]
    print(f"df coefs: shape={c.shape} mean|c|={np.abs(c).mean():.4f} max|c|={np.abs(c).max():.4f}")
    print()

    # ---------------- validation vs reference ----------------
    print("--- validation vs HF reference (sample_00006_denoised.wav) -----------")
    d_out, c_out = best_delay(out, ref)
    print(f"our output vs reference: best delay={d_out} samples, corr={c_out:.6f}")
    d2, c2 = best_delay(ref, out)
    print(f"reference vs our output: best delay={d2} samples, corr={c2:.6f}")

    # align on whichever direction won
    if c_out >= c2:
        a, b = out[d_out:], ref[: len(out) - d_out]
    else:
        a, b = out[: len(ref) - d2], ref[d2:]
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    print(f"aligned on {n} samples ({n/sr:.2f}s)")
    print(f"  SI-SDR(ours vs ref) : {si_sdr(a, b):7.2f} dB")
    print(f"  corr                : {np.corrcoef(a,b)[0,1]:.6f}")
    print(f"  rms ours={db(rms(a)):.2f} dBFS   rms ref={db(rms(b)):.2f} dBFS   "
          f"delta={db(rms(a))-db(rms(b)):+.2f} dB")
    print(f"  max abs diff        : {np.abs(a-b).max():.6f}")
    print()

    # ---------------- what the model did to the signal ----------------
    print("--- enhancement effect ----------------------------------------------")
    nz = noisy[: len(out) - DELAY]
    en = out[DELAY:][: len(nz)]
    print(f"input  rms: {db(rms(nz)):7.2f} dBFS")
    print(f"output rms: {db(rms(en)):7.2f} dBFS   ({db(rms(en))-db(rms(nz)):+.2f} dB)")
    # noise floor = 10th pct of frame energy; speech = 90th
    fl = 320
    nf = len(nz) // fl
    fn = nz[: nf * fl].reshape(nf, fl)
    fe = en[: nf * fl].reshape(nf, fl)
    en_n = 10 * np.log10(np.mean(fn**2, 1) + 1e-12)
    en_e = 10 * np.log10(np.mean(fe**2, 1) + 1e-12)
    print(f"noise floor (p10 frame energy): in={np.percentile(en_n,10):7.2f} dB  "
          f"out={np.percentile(en_e,10):7.2f} dB  "
          f"reduction={np.percentile(en_n,10)-np.percentile(en_e,10):+.2f} dB")
    print(f"speech level (p90 frame energy): in={np.percentile(en_n,90):7.2f} dB  "
          f"out={np.percentile(en_e,90):7.2f} dB  "
          f"change={np.percentile(en_e,90)-np.percentile(en_n,90):+.2f} dB")
    print(f"dynamic range (p90-p10): in={np.percentile(en_n,90)-np.percentile(en_n,10):.2f} dB  "
          f"out={np.percentile(en_e,90)-np.percentile(en_e,10):.2f} dB")
    print()

    # ---------------- benchmark ----------------
    print("--- benchmark (1 thread, whole-utterance) ---------------------------")
    for _ in range(3):
        model.enhance(noisy)  # warmup
    runs = []
    N = 15
    for _ in range(N):
        t0 = time.perf_counter()
        model.enhance(noisy)
        runs.append((time.perf_counter() - t0) * 1e3)
        runs[-1] = runs[-1]
    parts = model.last_timing
    runs = np.array(runs)
    print(f"end-to-end over {N} runs on {dur:.2f}s audio:")
    print(f"  mean={runs.mean():.2f} ms  median={np.median(runs):.2f} ms  "
          f"min={runs.min():.2f} ms  max={runs.max():.2f} ms  std={runs.std():.2f}")
    rtf = np.median(runs) / 1e3 / dur
    print(f"  RTF = {rtf:.5f}   ({1/rtf:.0f}x faster than real time)")
    print(f"  per 10ms frame = {np.median(runs)/T:.4f} ms")
    print()
    print("  stage breakdown (last run, ms):")
    tot = parts["total_ms"]
    for k in ["features_ms", "enc_ms", "erb_dec_ms", "df_dec_ms", "post_synth_ms"]:
        print(f"    {k:16s} {parts[k]:7.2f}  ({parts[k]/tot*100:5.1f}%)")
    print(f"    {'nn total':16s} {parts['nn_ms']:7.2f}  ({parts['nn_ms']/tot*100:5.1f}%)")
    print(f"    {'TOTAL':16s} {tot:7.2f}")
    print()

    # thread scaling
    print("--- thread scaling ---------------------------------------------------")
    for th in [1, 2, 4]:
        mt = HushORT(intra_threads=th, inter_threads=1)
        for _ in range(2):
            mt.enhance(noisy)
        r = []
        for _ in range(7):
            t0 = time.perf_counter()
            mt.enhance(noisy)
            r.append((time.perf_counter() - t0) * 1e3)
        med = float(np.median(r))
        print(f"  intra_op_threads={th}: median={med:7.2f} ms   RTF={med/1e3/dur:.5f}   "
              f"per-frame={med/T:.4f} ms")
    print()

    # ---------------- attenuation limit sweep ----------------
    print("--- attenuation limit sweep -----------------------------------------")
    for lim in [None, 100.0, 20.0, 12.0, 6.0]:
        o = model.enhance(noisy, atten_lim_db=lim)
        e = o[DELAY:][: len(nz)]
        fe = e[: nf * fl].reshape(nf, fl)
        ee = 10 * np.log10(np.mean(fe**2, 1) + 1e-12)
        tag = "none" if lim is None else f"{lim:g} dB"
        print(f"  atten_lim={tag:>7s}: out rms={db(rms(e)):7.2f} dBFS  "
              f"noise floor={np.percentile(ee,10):7.2f} dB  "
              f"NR={np.percentile(en_n,10)-np.percentile(ee,10):+6.2f} dB")
    print()
    print(f"wrote: {os.path.join(HERE, 'sample_enhanced_ort.wav')}")


if __name__ == "__main__":
    main()
