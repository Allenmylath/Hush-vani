"""Self-contained listening page: the same clip through f32 / f16 / int8, side by side."""
import base64
import json
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CMP = os.path.join(ROOT, "comparison")
OUT = os.path.join(CMP, "listen.html")
M = json.load(open(os.path.join(CMP, "metrics.json")))

FILES = {
    "noisy": "0_noisy_input.wav",
    "f32": "1_enhanced_f32.wav",
    "f16": "2_enhanced_f16.wav",
    "int8": "3_enhanced_int8.wav",
}


def b64(f):
    with open(os.path.join(CMP, f), "rb") as fh:
        return base64.b64encode(fh.read()).decode()


W, H = 560, 54


def wave(env, cls):
    n = len(env)
    mx = max(max(env), 1e-6)
    top = " ".join(f"{i / (n - 1) * W:.1f},{H / 2 - v / mx * (H / 2 - 2):.1f}" for i, v in enumerate(env))
    bot = " ".join(f"{i / (n - 1) * W:.1f},{H / 2 + v / mx * (H / 2 - 2):.1f}" for i, v in reversed(list(enumerate(env))))
    return (f'<svg viewBox="0 0 {W} {H}" preserveAspectRatio="none" class="wave" role="img" aria-label="waveform">'
            f'<polygon class="{cls}" points="{top} {bot}"/></svg>')


CARDS = [
    ("noisy", "Noisy input", "the original recording", "muted", None),
    ("f32", "f32 weights", "9.12 MB · the reference", "s1", None),
    ("f16", "f16 weights", "4.56 MB · <b>shipped</b>", "s2", "good"),
    ("int8", "int8 weights", "2.28 MB · not shipped", "s3", "bad"),
]

cards = []
for key, title, sub, cls, badge in CARDS:
    r = M["rows"].get(key, {})
    if key == "noisy":
        stat = '<div class="stat"><span class="k">the problem</span><span class="v">background noise</span></div>'
        b = ""
    elif key == "f32":
        stat = ('<div class="stat"><span class="k">error vs reference</span><span class="v">— none —</span></div>'
                f'<div class="stat"><span class="k">noise removed</span><span class="v">{r["reduction"]:+.1f} dB</span></div>')
        b = ""
    else:
        verdict = "below the 16-bit floor" if r["residual"] < -96 else "audible"
        stat = (f'<div class="stat"><span class="k">vs f32</span><span class="v">{r["sisdr"]:.1f} dB SI-SDR</span></div>'
                f'<div class="stat"><span class="k">error level</span><span class="v">{r["residual"]:.1f} dBFS <em>{verdict}</em></span></div>'
                f'<div class="stat"><span class="k">noise removed</span><span class="v">{r["reduction"]:+.1f} dB</span></div>')
        b = f'<span class="badge {badge}">{"inaudible" if badge == "good" else "degraded"}</span>'
    cards.append(f"""
    <article class="card">
      <header><div><h3>{title} {b}</h3><p>{sub}</p></div></header>
      {wave(M["env"][key], cls)}
      <audio controls preload="none" src="data:audio/wav;base64,{b64(FILES[key])}"></audio>
      <div class="stats">{stat}</div>
    </article>""")

html = f"""<div class="root">
<header class="page">
  <h1>Hear the difference</h1>
  <p class="lede">The same 5-second clip, denoised three times — identical model, identical
  code, only the <em>weight precision</em> changes. Play them back to back.
  <strong>f16 is indistinguishable from f32; int8 adds a hiss you can hear.</strong></p>
</header>

<section class="grid">{"".join(cards)}</section>

<section class="card wide">
  <h3>Why f16 is safe and int8 is not</h3>
  <p class="body">Quantisation doesn't change the speech — it adds an <em>error signal</em> on
  top. What matters is how loud that error is:</p>
  <table>
    <thead><tr><th>weights</th><th>error level</th><th>vs a 16-bit WAV's noise floor (−96 dBFS)</th></tr></thead>
    <tbody>
      <tr><td><b>f16</b></td><td class="num">−99.9 dBFS</td>
          <td><span class="badge good">3 dB below it</span> — the error cannot even be stored in the file, let alone heard</td></tr>
      <tr><td><b>int8</b></td><td class="num">−59.0 dBFS</td>
          <td><span class="badge bad">37 dB above it</span> — a broadband layer sitting ~27 dB under the speech</td></tr>
    </tbody>
  </table>
  <p class="body">int8 also denoises slightly <em>worse</em>: it leaves the noise floor 2.5 dB
  higher (−68.5 dB vs −71.0 dB) while adding its own grit. The cause is the weight
  distribution — a handful of ±31 outliers force a per-tensor scale that wastes the whole
  int8 range on the 99 % of weights living inside ±0.23. Per-channel scales would fix it.</p>
  <p class="body foot">Files: <code>comparison/</code> · f32 129.7 dB vs onnxruntime · f16 75.5 dB · int8 29.8 dB (all end-to-end)</p>
</section>
</div>

<style>
.root {{
  --surface:#fcfcfb; --plane:#f9f9f7; --ink:#0b0b0b; --ink2:#52514e; --muted:#898781;
  --border:rgba(11,11,11,0.10);
  --s1:#2a78d6; --s2:#1baf7a; --s3:#e34948; --mut:#c3c2b7;
  --good:#0ca30c; --bad:#d03b3b;
  font-family:system-ui,-apple-system,"Segoe UI",sans-serif;
  color:var(--ink); background:var(--plane); padding:26px; max-width:1180px; margin:0 auto;
}}
@media (prefers-color-scheme:dark){{ .root{{
  --surface:#1a1a19; --plane:#0d0d0d; --ink:#fff; --ink2:#c3c2b7; --border:rgba(255,255,255,0.10);
  --s1:#3987e5; --s2:#199e70; --s3:#e66767; --mut:#52514e;
}}}}
:root[data-theme="dark"] .root{{
  --surface:#1a1a19; --plane:#0d0d0d; --ink:#fff; --ink2:#c3c2b7; --border:rgba(255,255,255,0.10);
  --s1:#3987e5; --s2:#199e70; --s3:#e66767; --mut:#52514e;
}}
:root[data-theme="light"] .root{{
  --surface:#fcfcfb; --plane:#f9f9f7; --ink:#0b0b0b; --ink2:#52514e; --border:rgba(11,11,11,0.10);
  --s1:#2a78d6; --s2:#1baf7a; --s3:#e34948; --mut:#c3c2b7;
}}
h1{{ font-size:24px; margin:0 0 8px; letter-spacing:-0.015em; }}
.lede{{ color:var(--ink2); max-width:74ch; line-height:1.55; margin:0 0 22px; }}
.lede em{{ font-style:normal; font-weight:600; color:var(--ink); }}
.lede strong{{ color:var(--ink); }}
.grid{{ display:grid; grid-template-columns:repeat(auto-fit,minmax(300px,1fr)); gap:14px; }}
.card{{ background:var(--surface); border:1px solid var(--border); border-radius:12px; padding:15px; }}
.card.wide{{ margin-top:16px; }}
.card header{{ margin-bottom:9px; }}
h3{{ font-size:15px; margin:0 0 2px; display:flex; align-items:center; gap:8px; }}
.card p{{ font-size:12.5px; color:var(--muted); margin:0; }}
.wave{{ width:100%; height:54px; display:block; margin:6px 0 9px; }}
.wave polygon.s1{{ fill:var(--s1); }} .wave polygon.s2{{ fill:var(--s2); }}
.wave polygon.s3{{ fill:var(--s3); }} .wave polygon.muted{{ fill:var(--mut); }}
audio{{ width:100%; height:34px; }}
.stats{{ margin-top:10px; display:flex; flex-direction:column; gap:5px; }}
.stat{{ display:flex; justify-content:space-between; font-size:12.5px; gap:10px; }}
.stat .k{{ color:var(--muted); }}
.stat .v{{ font-variant-numeric:tabular-nums; font-weight:600; text-align:right; }}
.stat .v em{{ font-style:normal; font-weight:400; color:var(--muted); display:block; font-size:11.5px; }}
.badge{{ font-size:11px; padding:2px 8px; border-radius:999px; font-weight:600; }}
.badge.good{{ background:color-mix(in srgb,var(--good) 16%,transparent); color:var(--good); }}
.badge.bad{{ background:color-mix(in srgb,var(--bad) 16%,transparent); color:var(--bad); }}
table{{ width:100%; border-collapse:collapse; margin:12px 0; font-size:13.5px; }}
th{{ text-align:left; font-size:12px; color:var(--muted); font-weight:600; padding:6px 10px; border-bottom:1px solid var(--border); }}
td{{ padding:9px 10px; border-bottom:1px solid var(--border); vertical-align:middle; }}
td.num{{ font-variant-numeric:tabular-nums; font-weight:600; }}
.body{{ font-size:13.5px; color:var(--ink2); line-height:1.6; max-width:82ch; margin:8px 0; }}
.body em{{ font-style:normal; color:var(--ink); font-weight:600; }}
.foot{{ color:var(--muted); font-size:12px; margin-top:14px; }}
code{{ font-size:12px; }}
</style>
"""
open(OUT, "w", encoding="utf-8").write(html)
print(f"wrote {OUT}  ({os.path.getsize(OUT)/1024:.0f} KB)")
