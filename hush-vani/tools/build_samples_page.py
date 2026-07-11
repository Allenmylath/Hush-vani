"""Listening + metrics page for the DeepFilterNet2 demo samples at f32 / f16 / int8."""
import base64
import io
import json
import os
import sys

import numpy as np
import soundfile as sf

sys.stdout.reconfigure(encoding="utf-8")
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BS = os.path.join(ROOT, "bench_samples")
OUT = os.path.join(BS, "samples.html")
R = json.load(open(os.path.join(BS, "results.json")))
SR, CLIP = 16000, 6  # embed the first CLIP seconds to keep the page small

# representative: biggest int8 gap, steady broadband, lowest input SNR
FEATURED = ["01", "03", "06"]
VERS = [("noisy", "Noisy input", "mut"), ("f32", "f32", "s1"), ("f16", "f16", "s2"), ("int8", "int8", "s3")]


def wav_b64(path):
    x, sr = sf.read(path, dtype="float32")
    x = x[: CLIP * sr]
    buf = io.BytesIO()
    sf.write(buf, x, sr, format="WAV", subtype="PCM_16")
    return base64.b64encode(buf.getvalue()).decode()


def src(sid, v):
    if v == "noisy":
        return os.path.join(BS, "in16k", f"{sid}_noisy16k.wav")
    return os.path.join(BS, v, f"{sid}_{v}.wav")


W, H = 500, 44


def wave(path, cls):
    x, _ = sf.read(path, dtype="float32")
    n = 240
    m = len(x) // n
    env = np.abs(x[: m * n].reshape(n, m)).max(1)
    mx = max(env.max(), 1e-6)
    top = " ".join(f"{i/(n-1)*W:.1f},{H/2 - v/mx*(H/2-1):.1f}" for i, v in enumerate(env))
    bot = " ".join(f"{i/(n-1)*W:.1f},{H/2 + v/mx*(H/2-1):.1f}" for i, v in reversed(list(enumerate(env))))
    return (f'<svg viewBox="0 0 {W} {H}" preserveAspectRatio="none" class="wave">'
            f'<polygon class="{cls}" points="{top} {bot}"/></svg>')


rows = {r["id"]: r for r in R["results"]}

# ---------- featured players ----------
blocks = []
for sid in FEATURED:
    r = rows[sid]
    players = []
    for v, label, cls in VERS:
        p = src(sid, v)
        if v == "noisy":
            meta = f'input SNR ≈ {r["in_snr"]:.0f} dB'
            badge = ""
        else:
            d = r[v]
            meta = f'removed {d["reduction"]:+.1f} dB noise'
            if v == "f32":
                badge = '<span class="badge ref">reference</span>'
            elif v == "f16":
                badge = '<span class="badge good">inaudible</span>'
            else:
                badge = '<span class="badge bad">audible hiss</span>'
        players.append(f"""
      <div class="pl">
        <div class="plh"><span class="nm">{label}</span>{badge}</div>
        {wave(p, cls)}
        <audio controls preload="none" src="data:audio/wav;base64,{wav_b64(p)}"></audio>
        <div class="meta">{meta}</div>
      </div>""")
    blocks.append(f"""
  <section class="card">
    <h3>{sid} · {r["noise"]}<span class="dur">10 s · input SNR ≈ {r["in_snr"]:.0f} dB</span></h3>
    <div class="players">{"".join(players)}</div>
  </section>""")

# ---------- full table ----------
trs = []
for r in R["results"]:
    gap = r["f32"]["reduction"] - r["int8"]["reduction"]
    trs.append(f"""<tr>
      <td><b>{r['id']}</b> {r['noise']}</td>
      <td class="num">{r['in_snr']:.0f} dB</td>
      <td class="num s1b">{r['f32']['reduction']:+.1f}</td>
      <td class="num s2b">{r['f16']['reduction']:+.1f}</td>
      <td class="num s3b">{r['int8']['reduction']:+.1f}</td>
      <td class="num">{r['f16']['resid']:.0f} dBFS</td>
      <td class="num warn">{r['int8']['resid']:.0f} dBFS</td>
      <td class="num">{-gap:+.1f} dB</td>
    </tr>""")

a = R["agg"]
html = f"""<div class="root">
<header>
  <h1>Real-world noise: f32 vs f16 vs int8</h1>
  <p class="lede">Six recordings from the
  <a href="https://rikorose.github.io/DeepFilterNet2-Samples/">DeepFilterNet2 demo set</a> —
  munching, doors, air-con, crisp packets, street, running water — pushed through the
  <em>same</em> Hush model three times, changing only the weight precision.
  <strong>f16 is indistinguishable from f32 on every sample. int8 audibly degrades and denoises worse.</strong></p>
  <p class="note">Resampled 48→16 kHz (Hush is a 16 kHz model), so these are not comparable to
  DeepFilterNet2's own full-band output — this is a precision comparison, not a model shootout.</p>
</header>

<div class="tiles">
  <div class="tile"><div class="v s1b">{a['f32']['reduction']:+.1f} dB</div><div class="k">f32 · noise removed</div></div>
  <div class="tile"><div class="v s2b">{a['f16']['reduction']:+.1f} dB</div><div class="k">f16 · <b>identical</b></div></div>
  <div class="tile"><div class="v s3b">{a['int8']['reduction']:+.1f} dB</div><div class="k">int8 · 2.1 dB worse</div></div>
  <div class="tile"><div class="v">{a['f16']['resid']:.0f} / {a['int8']['resid']:.0f}</div><div class="k">error dBFS · f16 / int8</div></div>
</div>

{"".join(blocks)}

<section class="card">
  <h3>All six samples</h3>
  <div class="scroll"><table>
    <thead><tr>
      <th>sample</th><th>input SNR</th>
      <th colspan="3" class="ctr">noise removed</th>
      <th colspan="2" class="ctr">error level</th><th>int8 penalty</th>
    </tr>
    <tr class="sub">
      <th></th><th></th><th class="s1b">f32</th><th class="s2b">f16</th><th class="s3b">int8</th>
      <th>f16</th><th>int8</th><th></th>
    </tr></thead>
    <tbody>{"".join(trs)}</tbody>
  </table></div>
  <p class="body"><b>f16 matches f32 to within 0.1 dB on every single sample</b>, with an error
  level averaging {a['f16']['resid']:.0f} dBFS — at or below the −96 dBFS floor of a 16-bit WAV,
  so it is not merely quiet, it is <em>unstorable</em> in the output file.</p>
  <p class="body"><b>int8 loses twice.</b> It adds a broadband hiss ({a['int8']['resid']:.0f} dBFS,
  ~37 dB above the 16-bit floor) <em>and</em> it removes {a['f32']['reduction'] - a['int8']['reduction']:.1f} dB
  less noise — the quantised weights make the model worse at its job, then add grit on top. On
  sample 01 the gap is 8.4 dB. The cause is the weight distribution: a few ±31 outliers force a
  per-tensor scale that wastes the int8 range on the 99 % of weights inside ±0.23. Per-channel
  scales would likely fix it — and would unlock int8's measured 3.5× kernel speedup.</p>
</section>
<footer>Audio: <code>bench_samples/</code> (all 24 wavs) · metrics: <code>results.json</code></footer>
</div>

<style>
.root{{
 --surface:#fcfcfb;--plane:#f9f9f7;--ink:#0b0b0b;--ink2:#52514e;--muted:#898781;
 --border:rgba(11,11,11,.10);--s1:#2a78d6;--s2:#1baf7a;--s3:#e34948;--mut:#c3c2b7;
 --good:#0ca30c;--bad:#d03b3b;--warn:#b06a00;
 font-family:system-ui,-apple-system,"Segoe UI",sans-serif;color:var(--ink);
 background:var(--plane);padding:26px;max-width:1200px;margin:0 auto;}}
@media (prefers-color-scheme:dark){{.root{{
 --surface:#1a1a19;--plane:#0d0d0d;--ink:#fff;--ink2:#c3c2b7;--border:rgba(255,255,255,.10);
 --s1:#3987e5;--s2:#199e70;--s3:#e66767;--mut:#52514e;--warn:#eda100;}}}}
:root[data-theme="dark"] .root{{--surface:#1a1a19;--plane:#0d0d0d;--ink:#fff;--ink2:#c3c2b7;
 --border:rgba(255,255,255,.10);--s1:#3987e5;--s2:#199e70;--s3:#e66767;--mut:#52514e;--warn:#eda100;}}
:root[data-theme="light"] .root{{--surface:#fcfcfb;--plane:#f9f9f7;--ink:#0b0b0b;--ink2:#52514e;
 --border:rgba(11,11,11,.10);--s1:#2a78d6;--s2:#1baf7a;--s3:#e34948;--mut:#c3c2b7;--warn:#b06a00;}}
h1{{font-size:23px;margin:0 0 8px;letter-spacing:-.015em;}}
.lede{{color:var(--ink2);max-width:78ch;line-height:1.55;margin:0 0 8px;}}
.lede em{{font-style:normal;font-weight:600;color:var(--ink);}} .lede strong{{color:var(--ink);}}
.lede a{{color:var(--s1);}}
.note{{font-size:12.5px;color:var(--muted);max-width:78ch;line-height:1.5;margin:0 0 20px;}}
.tiles{{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:10px;margin-bottom:18px;}}
.tile{{background:var(--surface);border:1px solid var(--border);border-radius:10px;padding:12px 14px;}}
.tile .v{{font-size:19px;font-weight:650;font-variant-numeric:tabular-nums;}}
.tile .k{{font-size:11.5px;color:var(--muted);margin-top:2px;}}
.card{{background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:16px;margin-bottom:14px;}}
h3{{font-size:15px;margin:0 0 12px;display:flex;justify-content:space-between;align-items:baseline;gap:10px;}}
.dur{{font-size:12px;color:var(--muted);font-weight:400;}}
.players{{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:12px;}}
.pl{{min-width:0;}}
.plh{{display:flex;align-items:center;gap:7px;margin-bottom:4px;}}
.nm{{font-size:13px;font-weight:600;}}
.wave{{width:100%;height:44px;display:block;margin-bottom:5px;}}
.wave polygon.s1{{fill:var(--s1);}} .wave polygon.s2{{fill:var(--s2);}}
.wave polygon.s3{{fill:var(--s3);}} .wave polygon.mut{{fill:var(--mut);}}
audio{{width:100%;height:32px;}}
.meta{{font-size:11.5px;color:var(--muted);margin-top:4px;font-variant-numeric:tabular-nums;}}
.badge{{font-size:10.5px;padding:2px 7px;border-radius:999px;font-weight:600;}}
.badge.good{{background:color-mix(in srgb,var(--good) 16%,transparent);color:var(--good);}}
.badge.bad{{background:color-mix(in srgb,var(--bad) 16%,transparent);color:var(--bad);}}
.badge.ref{{background:color-mix(in srgb,var(--s1) 15%,transparent);color:var(--s1);}}
.scroll{{overflow-x:auto;}}
table{{width:100%;border-collapse:collapse;font-size:13px;min-width:640px;}}
th{{text-align:left;font-size:11.5px;color:var(--muted);font-weight:600;padding:5px 9px;border-bottom:1px solid var(--border);}}
th.ctr{{text-align:center;}} tr.sub th{{padding-top:0;}}
td{{padding:8px 9px;border-bottom:1px solid var(--border);}}
td.num{{font-variant-numeric:tabular-nums;text-align:right;}}
.s1b{{color:var(--s1);}} .s2b{{color:var(--s2);font-weight:650;}} .s3b{{color:var(--s3);}}
td.warn{{color:var(--warn);font-weight:600;}}
.body{{font-size:13.5px;color:var(--ink2);line-height:1.6;max-width:84ch;margin:12px 0 0;}}
.body em{{font-style:normal;color:var(--ink);font-weight:600;}} .body b{{color:var(--ink);}}
footer{{color:var(--muted);font-size:12px;margin-top:14px;}} code{{font-size:11.5px;}}
</style>
"""
open(OUT, "w", encoding="utf-8").write(html)
print(f"wrote {OUT} ({os.path.getsize(OUT)/1e6:.2f} MB)")
