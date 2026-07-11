"""Render the weight-distribution analysis as a self-contained, theme-aware HTML page."""
import json
import math
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__))
D = json.load(open(os.path.join(HERE, "hist.json")))
OUT = os.path.join(HERE, "weights_viz.html")

W, H = 620, 300
ML, MR, MT, MB = 52, 16, 16, 40  # margins
PW, PH = W - ML - MR, H - MT - MB


def x_map(v, lo, hi):
    return ML + (v - lo) / (hi - lo) * PW


def esc(s):
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def axes(xticks, yticks, xlab, ylab, x2px, y2px):
    s = []
    for v, lab in yticks:
        y = y2px(v)
        s.append(f'<line class="grid" x1="{ML}" y1="{y:.1f}" x2="{ML+PW}" y2="{y:.1f}"/>')
        s.append(f'<text class="tick" x="{ML-6}" y="{y+3:.1f}" text-anchor="end">{esc(lab)}</text>')
    for v, lab in xticks:
        x = x2px(v)
        s.append(f'<text class="tick" x="{x:.1f}" y="{MT+PH+16}" text-anchor="middle">{esc(lab)}</text>')
    s.append(f'<line class="axis" x1="{ML}" y1="{MT+PH}" x2="{ML+PW}" y2="{MT+PH}"/>')
    s.append(f'<text class="axlab" x="{ML+PW/2:.0f}" y="{H-6}" text-anchor="middle">{esc(xlab)}</text>')
    s.append(f'<text class="axlab" transform="translate(13,{MT+PH/2:.0f}) rotate(-90)" text-anchor="middle">{esc(ylab)}</text>')
    return "".join(s)


def fig(title, sub, svg):
    return (f'<figure class="card"><figcaption><h3>{esc(title)}</h3>'
            f'<p>{sub}</p></figcaption><div class="plot">{svg}</div></figure>')


def hist_log():
    e, c = D["full"]["edges"], D["full"]["counts"]
    lo, hi = e[0], e[-1]
    ymax = math.log10(max(c) + 1)
    y2 = lambda v: MT + PH - v / ymax * PH  # v is log10(count+1)
    x2 = lambda v: x_map(v, lo, hi)
    bars = []
    for i, cnt in enumerate(c):
        if cnt <= 0:
            continue
        x0, x1 = x2(e[i]), x2(e[i + 1])
        y = y2(math.log10(cnt + 1))
        bars.append(f'<rect class="s1" x="{x0:.1f}" y="{y:.1f}" width="{max(x1-x0-0.6,0.6):.1f}" '
                    f'height="{MT+PH-y:.1f}"><title>[{e[i]:.2f}, {e[i+1]:.2f}]  {cnt:,} weights</title></rect>')
    yt = [(k, ("1" if k == 0 else f"1e{k}")) for k in range(0, int(ymax) + 1)]
    xt = [(v, str(v)) for v in (-30, -20, -10, 0, 10, 20, 30)]
    svg = axes(xt, yt, "weight value", "count (log)", x2, y2) + "".join(bars)
    return f'<svg viewBox="0 0 {W} {H}" role="img">{svg}</svg>'


def hist_central():
    e, d = D["central"]["edges"], D["central"]["density"]
    lo, hi = e[0], e[-1]
    ymax = max(d) * 1.05
    y2 = lambda v: MT + PH - v / ymax * PH
    x2 = lambda v: x_map(v, lo, hi)
    bars = []
    for i, dv in enumerate(d):
        x0, x1 = x2(e[i]), x2(e[i + 1])
        y = y2(dv)
        bars.append(f'<rect class="s1" x="{x0:.1f}" y="{y:.1f}" width="{max(x1-x0-0.6,0.6):.1f}" '
                    f'height="{MT+PH-y:.1f}"><title>{e[i]:+.3f}  density {dv:.2f}</title></rect>')
    xt = [(v, f"{v:+.1f}" if v else "0") for v in (-0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3)]
    yt = [(v, f"{v:.0f}") for v in (0, ymax * 0.25, ymax * 0.5, ymax * 0.75) ]
    svg = axes(xt, yt, "weight value", "density", x2, y2) + "".join(bars)
    return f'<svg viewBox="0 0 {W} {H}" role="img">{svg}</svg>'


def per_kind():
    lo, hi = -0.3, 0.3
    allmax = max(max(D["per_kind"][k]["density"]) for k in D["per_kind"])
    ymax = allmax * 1.05
    y2 = lambda v: MT + PH - v / ymax * PH
    x2 = lambda v: x_map(v, lo, hi)
    cls = {"gru": "s1", "linear": "s2", "conv": "s3"}
    lab = {"gru": "GRU", "linear": "linear", "conv": "conv"}
    paths = []
    for k in ("conv", "linear", "gru"):
        e, d = D["per_kind"][k]["edges"], D["per_kind"][k]["density"]
        pts = " ".join(f"{x2((e[i]+e[i+1])/2):.1f},{y2(d[i]):.1f}" for i in range(len(d)))
        paths.append(f'<polyline class="line {cls[k]}" points="{pts}"/>')
        # direct label at the peak
        pk = d.index(max(d))
        px, py = x2((e[pk] + e[pk + 1]) / 2), y2(d[pk])
        dx = 8 if k == "gru" else (-6 if k == "conv" else 8)
        anch = "start" if dx > 0 else "end"
        paths.append(f'<text class="dlab {cls[k]}" x="{px+dx:.1f}" y="{py-4:.1f}" '
                     f'text-anchor="{anch}">{lab[k]}</text>')
    xt = [(v, f"{v:+.1f}" if v else "0") for v in (-0.3, -0.15, 0, 0.15, 0.3)]
    yt = [(v, f"{v:.0f}") for v in (0, ymax * 0.33, ymax * 0.66)]
    svg = axes(xt, yt, "weight value", "density", x2, y2) + "".join(paths)
    return f'<svg viewBox="0 0 {W} {H}" role="img">{svg}</svg>'


def cdf():
    xs, ys = D["cdf_abs"]["x"], D["cdf_abs"]["y"]
    lo, hi = 0, 0.5
    y2 = lambda v: MT + PH - v * PH
    x2 = lambda v: x_map(v, lo, hi)
    pts = " ".join(f"{x2(min(x,hi)):.1f},{y2(y):.1f}" for x, y in zip(xs, ys) if x <= hi + 1e-9)
    marks = []
    for thr in (0.01, 0.05, 0.1):
        # nearest sample
        y = min((yy for xx, yy in zip(xs, ys) if xx >= thr), default=ys[-1])
        mx, my = x2(thr), y2(y)
        marks.append(f'<circle class="dot s1" cx="{mx:.1f}" cy="{my:.1f}" r="4"/>')
        marks.append(f'<text class="dlab s1" x="{mx+6:.1f}" y="{my+4:.1f}" text-anchor="start">'
                     f'|w|&lt;{thr:g}: {y*100:.0f}%</text>')
    xt = [(v, f"{v:.2f}") for v in (0, 0.1, 0.2, 0.3, 0.4, 0.5)]
    yt = [(v, f"{int(v*100)}%") for v in (0, 0.25, 0.5, 0.75, 1.0)]
    svg = (axes(xt, yt, "|weight|", "cumulative", x2, y2)
           + f'<polyline class="line s1" points="{pts}"/>' + "".join(marks))
    return f'<svg viewBox="0 0 {W} {H}" role="img">{svg}</svg>'


def compress_table():
    rows = []
    for r in D["compress"]:
        q = "lossless-grade" if r["sisdr"] >= 67 else ("audible" if r["sisdr"] >= 40 else "degraded")
        badge = {"lossless-grade": "good", "audible": "warn", "degraded": "bad"}[q]
        rows.append(
            f'<tr><td>{esc(r["scheme"])}</td><td class="num">{r["gz"]:.2f} MB</td>'
            f'<td class="num">{D["compress"][0]["gz"]/r["gz"]:.1f}×</td>'
            f'<td class="num">{r["sisdr"]:.1f} dB</td>'
            f'<td><span class="badge {badge}">{q}</span></td></tr>')
    return ('<table class="ctab"><thead><tr><th>storage</th><th>gzipped</th><th>vs f32</th>'
            '<th>audio SI-SDR</th><th>quality</th></tr></thead><tbody>'
            + "".join(rows) + "</tbody></table>"
            '<p class="note">SI-SDR is end-to-end output vs onnxruntime. A 16-bit WAV tops out '
            'near 67&nbsp;dB, so anything above that line is indistinguishable in a shipped file — '
            'which is why <strong>f16 is effectively free</strong>.</p>')


s = D["stats"]
tiles = [
    ("2.28 M", "weights (f32)"),
    (f'{s["median"]:.3f}', "median (exactly 0)"),
    (f'{s["kurt_excess"]:,.0f}', "excess kurtosis"),
    ("85.8%", "within |w| &lt; 0.1"),
    (f'[{s["p1"]:.2f}, {s["p99"]:.2f}]', "1st–99th percentile"),
    (f'[{s["min"]:.1f}, {s["max"]:.1f}]', "full range (outliers)"),
]
tiles_html = "".join(f'<div class="tile"><div class="v">{v}</div><div class="k">{k}</div></div>'
                     for v, k in tiles)

html = f"""<div class="viz-root">
<header>
  <h1>Hush weights — distribution</h1>
  <p class="lede">2.28&nbsp;million float32 across 62 tensors. Zero-centered and
  <em>extremely</em> peaked: a needle at 0 with rare far outliers — a shape that compresses
  well once you quantize.</p>
  <div class="tiles">{tiles_html}</div>
</header>

<section class="grid2">
  {fig("Full range, log scale", "Almost everything is a spike at 0; the bars out to &plusmn;31 are a handful of outlier weights (note the log y-axis).", hist_log())}
  {fig("Central region, linear scale", "The same data zoomed to [&minus;0.3, 0.3]: a sharp Laplacian cusp, not a Gaussian bell.", hist_central())}
  {fig("By layer type", "Normalized density. GRU weights are the tightest (they are 87% of all weights); conv the widest.", per_kind())}
  {fig("How concentrated", "Cumulative fraction of weights below a given magnitude. Most of the mass is in a tiny band.", cdf())}
</section>

<section class="card wide">
  <figcaption><h3>What compression buys</h3>
  <p>Measured end-to-end, not just per-tensor error.</p></figcaption>
  {compress_table()}
</section>

<footer>Generated from <code>hush-vani-weights/data/weights.bin</code> · analysis in <code>tools/analyze_weights.py</code></footer>
</div>

<style>
.viz-root {{
  --surface-1:#fcfcfb; --plane:#f9f9f7; --ink:#0b0b0b; --ink2:#52514e; --muted:#898781;
  --grid:#e1e0d9; --axis:#c3c2b7; --border:rgba(11,11,11,0.10);
  --s1:#2a78d6; --s2:#1baf7a; --s3:#eda100;
  --good:#0ca30c; --warn:#eda100; --bad:#d03b3b;
  font-family:system-ui,-apple-system,"Segoe UI",sans-serif; color:var(--ink);
  background:var(--plane); padding:24px; max-width:1320px; margin:0 auto;
}}
@media (prefers-color-scheme:dark){{ .viz-root{{
  --surface-1:#1a1a19; --plane:#0d0d0d; --ink:#fff; --ink2:#c3c2b7; --muted:#898781;
  --grid:#2c2c2a; --axis:#383835; --border:rgba(255,255,255,0.10);
  --s1:#3987e5; --s2:#199e70; --s3:#c98500;
}}}}
:root[data-theme="dark"] .viz-root{{
  --surface-1:#1a1a19; --plane:#0d0d0d; --ink:#fff; --ink2:#c3c2b7;
  --grid:#2c2c2a; --axis:#383835; --border:rgba(255,255,255,0.10);
  --s1:#3987e5; --s2:#199e70; --s3:#c98500;
}}
:root[data-theme="light"] .viz-root{{
  --surface-1:#fcfcfb; --plane:#f9f9f7; --ink:#0b0b0b; --ink2:#52514e;
  --grid:#e1e0d9; --axis:#c3c2b7; --border:rgba(11,11,11,0.10);
  --s1:#2a78d6; --s2:#1baf7a; --s3:#eda100;
}}
.viz-root h1{{ font-size:22px; margin:0 0 6px; letter-spacing:-0.01em; }}
.lede{{ color:var(--ink2); max-width:70ch; margin:0 0 18px; line-height:1.5; }}
.lede em{{ font-style:normal; font-weight:600; color:var(--ink); }}
.tiles{{ display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:10px; margin-bottom:22px; }}
.tile{{ background:var(--surface-1); border:1px solid var(--border); border-radius:10px; padding:12px 14px; }}
.tile .v{{ font-size:19px; font-weight:650; letter-spacing:-0.01em; }}
.tile .k{{ font-size:12px; color:var(--muted); margin-top:2px; }}
.grid2{{ display:grid; grid-template-columns:repeat(auto-fit,minmax(400px,1fr)); gap:16px; }}
.card{{ background:var(--surface-1); border:1px solid var(--border); border-radius:12px; padding:16px; }}
.card.wide{{ margin-top:16px; }}
figcaption h3{{ font-size:15px; margin:0 0 3px; }}
figcaption p{{ font-size:12.5px; color:var(--ink2); margin:0 0 8px; line-height:1.45; }}
.plot{{ width:100%; overflow-x:auto; }}
.plot svg{{ width:100%; height:auto; display:block; }}
.grid{{ stroke:var(--grid); stroke-width:1; }}
.axis{{ stroke:var(--axis); stroke-width:1; }}
.tick{{ fill:var(--muted); font-size:11px; font-variant-numeric:tabular-nums; }}
.axlab{{ fill:var(--ink2); font-size:11.5px; }}
.s1{{ fill:var(--s1); }} .s2{{ fill:var(--s2); }} .s3{{ fill:var(--s3); }}
rect.s1{{ transition:opacity .1s; }} rect.s1:hover{{ opacity:.55; }}
.line{{ fill:none; stroke-width:2; }}
.line.s1{{ stroke:var(--s1); }} .line.s2{{ stroke:var(--s2); }} .line.s3{{ stroke:var(--s3); }}
.dot.s1{{ fill:var(--s1); stroke:var(--surface-1); stroke-width:1.5; }}
.dlab{{ font-size:11.5px; font-weight:600; }}
.dlab.s1{{ fill:var(--s1); }} .dlab.s2{{ fill:var(--s2); }} .dlab.s3{{ fill:var(--s3); }}
.ctab{{ width:100%; border-collapse:collapse; margin-top:10px; font-size:13.5px; }}
.ctab th{{ text-align:left; color:var(--muted); font-weight:600; font-size:12px; padding:6px 10px; border-bottom:1px solid var(--border); }}
.ctab td{{ padding:8px 10px; border-bottom:1px solid var(--border); }}
.ctab td.num{{ font-variant-numeric:tabular-nums; }}
.badge{{ font-size:11.5px; padding:2px 8px; border-radius:999px; font-weight:600; }}
.badge.good{{ background:color-mix(in srgb,var(--good) 16%,transparent); color:var(--good); }}
.badge.warn{{ background:color-mix(in srgb,var(--warn) 20%,transparent); color:var(--warn); }}
.badge.bad{{ background:color-mix(in srgb,var(--bad) 16%,transparent); color:var(--bad); }}
.note{{ font-size:12.5px; color:var(--ink2); line-height:1.5; margin:12px 2px 0; max-width:80ch; }}
footer{{ color:var(--muted); font-size:12px; margin-top:20px; }}
footer code,.note code{{ font-size:11.5px; }}
</style>
"""

open(OUT, "w", encoding="utf-8").write(html)
print("wrote", OUT, f"({os.path.getsize(OUT)/1024:.0f} KB)")
