"""ONNX Runtime inference for weya-ai/hush (DeepFilterNet3-style speech enhancement).

The exported bundle ships three graphs (enc / erb_dec / df_dec) that cover only the
neural net. STFT, ERB feature extraction, exponential normalisation, mask application,
deep-filter application and synthesis all live outside the graphs and are done here
(DSP via `libdf`, the same Rust library used during training).

The GRUs are baked into the graphs with no exposed hidden-state tensors, so every
Run() call restarts the recurrence: inference must be done on a whole sequence.
"""

import configparser
import os
import time

import numpy as np
import onnxruntime as ort
from libdf import DF, erb, erb_inv, erb_norm, unit_norm
from numpy.lib.stride_tricks import sliding_window_view

HERE = os.path.dirname(os.path.abspath(__file__))


class HushORT:
    def __init__(self, bundle_dir=None, intra_threads=1, inter_threads=1):
        bundle_dir = bundle_dir or os.path.join(HERE, "bundle")
        self.bundle_dir = bundle_dir

        cfg = configparser.ConfigParser()
        cfg.read(os.path.join(bundle_dir, "config.ini"))
        self.sr = cfg.getint("df", "sr")
        self.fft_size = cfg.getint("df", "fft_size")
        self.hop_size = cfg.getint("df", "hop_size")
        self.nb_erb = cfg.getint("df", "nb_erb")
        self.nb_df = cfg.getint("df", "nb_df")
        self.min_nb_erb_freqs = cfg.getint("df", "min_nb_erb_freqs")
        self.norm_tau = cfg.getfloat("df", "norm_tau")
        self.df_order = cfg.getint("deepfilternet", "df_order")
        self.df_lookahead = cfg.getint("deepfilternet", "df_lookahead")

        # exponential normalisation decay, matches df.utils.get_norm_alpha
        self.alpha = float(np.exp(-(self.hop_size / self.sr) / self.norm_tau))

        self.df_state = DF(
            sr=self.sr,
            fft_size=self.fft_size,
            hop_size=self.hop_size,
            nb_bands=self.nb_erb,
            min_nb_erb_freqs=self.min_nb_erb_freqs,
        )
        self.erb_widths = self.df_state.erb_widths()

        so = ort.SessionOptions()
        so.intra_op_num_threads = intra_threads
        so.inter_op_num_threads = inter_threads
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        mk = lambda n: ort.InferenceSession(
            os.path.join(bundle_dir, n), so, providers=["CPUExecutionProvider"]
        )
        self.enc = mk("enc.onnx")
        self.erb_dec = mk("erb_dec.onnx")
        self.df_dec = mk("df_dec.onnx")

        self.last_timing = {}

    # ---- feature extraction -------------------------------------------------
    def features(self, audio):
        """audio: float32 [N] -> spec [1,T,F] complex64, feat_erb, feat_spec"""
        self.df_state.reset()
        spec = self.df_state.analysis(audio[None, :].copy())  # [1,T,161] complex64
        feat_erb = erb_norm(erb(spec, self.erb_widths), self.alpha)  # [1,T,32]
        feat_erb = feat_erb[:, None]  # [1,1,T,32]
        sf = unit_norm(spec[..., : self.nb_df], self.alpha)  # [1,T,64] complex
        feat_spec = np.stack([sf.real, sf.imag], axis=1)  # [1,2,T,64]
        return spec, np.ascontiguousarray(feat_erb), np.ascontiguousarray(feat_spec)

    # ---- deep filter --------------------------------------------------------
    def apply_df(self, spec, coefs):
        """spec [1,T,F] complex, coefs [1,T,64,2*O] -> filtered first nb_df bins."""
        B, T, F = spec.shape
        O = self.df_order
        c = coefs.reshape(B, T, self.nb_df, O, 2)
        c = np.transpose(c, (0, 3, 1, 2, 4))  # [B,O,T,F,2]
        c = np.transpose(c, (0, 2, 1, 3, 4))  # [B,T,O,F,2]

        sd = spec[..., : self.nb_df]
        sd = np.stack([sd.real, sd.imag], axis=-1)  # [B,T,64,2]
        pad_l = O - 1 - self.df_lookahead
        padded = np.pad(sd, ((0, 0), (pad_l, self.df_lookahead), (0, 0), (0, 0)))
        win = sliding_window_view(padded, O, axis=1)  # [B,T,64,2,O]
        win = np.moveaxis(win, -1, 2)  # [B,T,O,64,2]

        re = win[..., 0] * c[..., 0] - win[..., 1] * c[..., 1]
        im = win[..., 1] * c[..., 0] + win[..., 0] * c[..., 1]
        return (re.sum(2) + 1j * im.sum(2)).astype(np.complex64)  # [B,T,64]

    # ---- full enhancement ---------------------------------------------------
    def enhance(self, audio, atten_lim_db=None, collect=False):
        audio = np.ascontiguousarray(audio.astype(np.float32))
        t = {}

        t0 = time.perf_counter()
        spec, feat_erb, feat_spec = self.features(audio)
        t["features_ms"] = (time.perf_counter() - t0) * 1e3

        t0 = time.perf_counter()
        e0, e1, e2, e3, emb, c0, lsnr = self.enc.run(
            None, {"feat_erb": feat_erb, "feat_spec": feat_spec}
        )
        t["enc_ms"] = (time.perf_counter() - t0) * 1e3

        t0 = time.perf_counter()
        (m,) = self.erb_dec.run(None, {"emb": emb, "e3": e3, "e2": e2, "e1": e1, "e0": e0})
        t["erb_dec_ms"] = (time.perf_counter() - t0) * 1e3

        t0 = time.perf_counter()
        (coefs,) = self.df_dec.run(None, {"emb": emb, "c0": c0})
        t["df_dec_ms"] = (time.perf_counter() - t0) * 1e3

        t0 = time.perf_counter()
        gains = erb_inv(np.ascontiguousarray(m[:, 0]), self.erb_widths)  # [1,T,161]
        spec_m = spec * gains
        spec_e = spec_m.copy()
        spec_e[..., : self.nb_df] = self.apply_df(spec, coefs)

        if atten_lim_db is not None and abs(atten_lim_db) > 0:
            lim = 10 ** (-abs(atten_lim_db) / 20)
            spec_e = spec * lim + spec_e * (1 - lim)

        out = self.df_state.synthesis(np.ascontiguousarray(spec_e))[0]
        t["post_synth_ms"] = (time.perf_counter() - t0) * 1e3

        t["total_ms"] = sum(t.values())
        t["nn_ms"] = t["enc_ms"] + t["erb_dec_ms"] + t["df_dec_ms"]
        t["frames"] = spec.shape[1]
        self.last_timing = t

        if collect:
            return out, {
                "spec": spec, "spec_e": spec_e, "mask": m, "gains": gains,
                "lsnr": lsnr, "coefs": coefs, "emb": emb, "timing": t,
            }
        return out


DELAY = 160  # samples of algorithmic delay (1 hop), verified by STFT roundtrip
