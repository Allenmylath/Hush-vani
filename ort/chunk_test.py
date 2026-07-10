"""Quantify the cost of the missing GRU state: run the ONNX graphs on chunks of the
feature sequence (features/DSP kept full-context) and compare against full-sequence
output. Any degradation is purely the recurrent state being reset per Run()."""

import os
import time

import numpy as np
import soundfile as sf

from hush_ort import HERE, HushORT


def si_sdr(est, ref):
    ref = ref - ref.mean()
    est = est - est.mean()
    a = np.dot(est, ref) / (np.dot(ref, ref) + 1e-12)
    proj = a * ref
    noise = est - proj
    return 10 * np.log10((np.dot(proj, proj) + 1e-12) / (np.dot(noise, noise) + 1e-12))


def nn_chunked(model, feat_erb, feat_spec, chunk):
    T = feat_erb.shape[2]
    ms, cs = [], []
    lat = []
    for s in range(0, T, chunk):
        e = min(s + chunk, T)
        fe = np.ascontiguousarray(feat_erb[:, :, s:e])
        fs = np.ascontiguousarray(feat_spec[:, :, s:e])
        t0 = time.perf_counter()
        e0, e1, e2, e3, emb, c0, lsnr = model.enc.run(None, {"feat_erb": fe, "feat_spec": fs})
        (m,) = model.erb_dec.run(None, {"emb": emb, "e3": e3, "e2": e2, "e1": e1, "e0": e0})
        (coefs,) = model.df_dec.run(None, {"emb": emb, "c0": c0})
        lat.append((time.perf_counter() - t0) * 1e3)
        ms.append(m)
        cs.append(coefs)
    return np.concatenate(ms, axis=2), np.concatenate(cs, axis=1), np.array(lat)


def synth(model, spec, m, coefs):
    from libdf import erb_inv

    gains = erb_inv(np.ascontiguousarray(m[:, 0]), model.erb_widths)
    spec_e = (spec * gains).copy()
    spec_e[..., : model.nb_df] = model.apply_df(spec, coefs)
    return spec_e


def main():
    noisy, sr = sf.read(os.path.join(HERE, "sample_raw.wav"), dtype="float32")
    model = HushORT(intra_threads=1, inter_threads=1)

    spec, feat_erb, feat_spec = model.features(noisy)
    T = feat_erb.shape[2]

    # full-sequence reference (what run_report produced, validated vs HF)
    e0, e1, e2, e3, emb, c0, lsnr = model.enc.run(
        None, {"feat_erb": feat_erb, "feat_spec": feat_spec}
    )
    (m_full,) = model.erb_dec.run(None, {"emb": emb, "e3": e3, "e2": e2, "e1": e1, "e0": e0})
    (c_full,) = model.df_dec.run(None, {"emb": emb, "c0": c0})
    spec_full = synth(model, spec, m_full, c_full)
    model.df_state.reset()
    ref_out = model.df_state.synthesis(np.ascontiguousarray(spec_full))[0]

    print("=" * 78)
    print("Chunked / streaming behaviour (GRU state is NOT exposed in the graphs)")
    print("=" * 78)
    print(f"{'chunk':>8} {'chunk_ms':>10} {'per-frame':>11} {'RTF':>9} "
          f"{'SI-SDR vs full':>15} {'mask MAE':>10}")
    print("-" * 78)

    for chunk in [1, 2, 5, 10, 25, 50, 100, 250, 500]:
        for _ in range(2):
            nn_chunked(model, feat_erb, feat_spec, chunk)
        m_c, c_c, lat = nn_chunked(model, feat_erb, feat_spec, chunk)
        spec_c = synth(model, spec, m_c, c_c)
        model.df_state.reset()
        out_c = model.df_state.synthesis(np.ascontiguousarray(spec_c))[0]

        n = min(len(out_c), len(ref_out))
        s = si_sdr(out_c[:n], ref_out[:n])
        mae = float(np.abs(m_c - m_full).mean())
        # wall time per chunk -> per 10ms frame; RTF vs audio covered by chunk
        per_chunk = float(np.median(lat))
        per_frame = per_chunk / chunk
        rtf = per_frame / 10.0
        sd = f"{s:.2f}" if np.isfinite(s) else "inf"
        print(f"{chunk:>8} {per_chunk:>10.3f} {per_frame:>11.4f} {rtf:>9.4f} "
              f"{sd:>15} {mae:>10.5f}")

    print()
    print("chunk=500 is the whole 5s utterance (single Run) => the validated path.")
    print("Small chunks reset every GRU each call: output degrades AND gets slower")
    print("per frame (fixed per-Run overhead is amortised over fewer frames).")
    print()
    print("Real-time streaming therefore needs either:")
    print("  (a) the vendor's weya_nc C library (manages state internally), or")
    print("  (b) re-exporting the graphs with GRU h0/hn as explicit inputs/outputs.")


if __name__ == "__main__":
    main()
