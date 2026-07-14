"""Build audio/index.html and serve it, so every version can be A/B'd side by side.

    python hush-vani/tools/serve_audio.py            # build + serve + open browser
    python hush-vani/tools/serve_audio.py --build    # just write index.html

The page keeps ONE audio element per sample and swaps its source in place, preserving the
playhead — so switching f32 -> int8 mid-word is instant and at the same instant of audio,
which is the only way the difference is actually audible.
"""
import csv
import functools
import http.server
import json
import os
import socketserver
import sys
import threading
import webbrowser

import numpy as np
import soundfile as sf

sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
AUDIO = os.path.join(ROOT, "audio")
PORT = 8000

VERSIONS = [
    ("input", "Noisy input", "the recording", "mut"),
    ("f32", "f32", "reference · 9.12 MB", "s1"),
    ("f16", "f16", "ships · 4.56 MB", "s2"),
    ("int8", "int8", "rejected · 2.28 MB", "s3"),
    ("dfn2", "DeepFilterNet2", "different model, 48 kHz", "s4"),
]


def files_for(folder):
    d = os.path.join(AUDIO, folder)
    f = {}
    for n in os.listdir(d):
        if not n.endswith(".wav"):
            continue
        if n.startswith("input"):
            f["input"] = n
        elif n == "hush_f32.wav":
            f["f32"] = n
        elif n == "hush_f16.wav":
            f["f16"] = n
        elif n == "hush_int8.wav":
            f["int8"] = n
        elif n.startswith("reference_deepfilternet2"):
            f["dfn2"] = n
    return f


def envelope(path, n=420):
    x, _ = sf.read(path, dtype="float32")
    if x.ndim > 1:
        x = x.mean(1)
    m = max(len(x) // n, 1)
    e = np.abs(x[: n * m].reshape(-1, m)).max(1)
    if len(e) < n:
        e = np.pad(e, (0, n - len(e)))
    return (e / max(e.max(), 1e-6)).round(3).tolist()


def build():
    rows = list(csv.DictReader(open(os.path.join(AUDIO, "manifest.csv"), encoding="utf-8")))
    data = []
    for r in rows:
        folder = r["folder"]
        f = files_for(folder)
        envs = {k: envelope(os.path.join(AUDIO, folder, v)) for k, v in f.items()}
        data.append({
            "folder": folder,
            "noise": r["noise"],
            "snr": r["input_snr_db"],
            "dur": float(r["duration_s"]),
            "files": f,
            "env": envs,
            "m": {
                "f32": {"red": float(r["f32_noise_removed_db"])},
                "f16": {"red": float(r["f16_noise_removed_db"]),
                        "err": float(r["f16_error_dbfs"]),
                        "sisdr": float(r["f16_sisdr_db"])},
                "int8": {"red": float(r["int8_noise_removed_db"]),
                         "err": float(r["int8_error_dbfs"]),
                         "sisdr": float(r["int8_sisdr_db"])},
            },
        })

    html = INDEX.replace("__DATA__", json.dumps(data)).replace(
        "__VERSIONS__", json.dumps([[k, n, s, c] for k, n, s, c in VERSIONS]))
    p = os.path.join(AUDIO, "index.html")
    open(p, "w", encoding="utf-8").write(html)
    print(f"built {p}")
    return p


INDEX = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Hush — f32 / f16 / int8, side by side</title>
<style>
:root{
 --surface:#fcfcfb;--plane:#f9f9f7;--ink:#0b0b0b;--ink2:#52514e;--muted:#898781;
 --border:rgba(11,11,11,.10);--s1:#2a78d6;--s2:#1baf7a;--s3:#e34948;--s4:#7a5af0;--mut:#b9b8b1;
 --good:#0ca30c;--bad:#d03b3b;--track:#e1e0d9;
}
@media(prefers-color-scheme:dark){:root{
 --surface:#1a1a19;--plane:#0d0d0d;--ink:#fff;--ink2:#c3c2b7;--muted:#898781;
 --border:rgba(255,255,255,.10);--s1:#3987e5;--s2:#199e70;--s3:#e66767;--s4:#9085e9;--mut:#4a4a46;
 --track:#2c2c2a;
}}
*{box-sizing:border-box}
body{margin:0;background:var(--plane);color:var(--ink);
 font-family:system-ui,-apple-system,"Segoe UI",sans-serif;padding:26px 20px 60px;}
.wrap{max-width:1080px;margin:0 auto;}
h1{font-size:24px;margin:0 0 8px;letter-spacing:-.015em;}
.lede{color:var(--ink2);line-height:1.55;max-width:76ch;margin:0 0 6px;}
.lede b{color:var(--ink);}
.hint{font-size:12.5px;color:var(--muted);margin:0 0 22px;}
kbd{background:var(--surface);border:1px solid var(--border);border-bottom-width:2px;
 border-radius:4px;padding:1px 5px;font-size:11px;font-family:inherit;}
.card{background:var(--surface);border:1px solid var(--border);border-radius:12px;
 padding:15px 16px;margin-bottom:14px;}
.top{display:flex;justify-content:space-between;align-items:baseline;gap:12px;margin-bottom:10px;}
.name{font-size:15px;font-weight:650;}
.name span{color:var(--muted);font-weight:400;font-size:12.5px;margin-left:8px;}
.tabs{display:flex;gap:6px;flex-wrap:wrap;margin-bottom:10px;}
.tab{border:1px solid var(--border);background:transparent;color:var(--ink2);
 border-radius:999px;padding:5px 12px;font:inherit;font-size:12.5px;cursor:pointer;
 display:flex;align-items:center;gap:6px;transition:.12s;}
.tab:hover{border-color:var(--muted);}
.tab .dot{width:8px;height:8px;border-radius:50%;flex:none;}
.tab[aria-selected=true]{color:#fff;border-color:transparent;font-weight:600;}
.tab[data-v=input][aria-selected=true]{background:var(--mut);color:var(--ink);}
.tab[data-v=f32][aria-selected=true]{background:var(--s1);}
.tab[data-v=f16][aria-selected=true]{background:var(--s2);}
.tab[data-v=int8][aria-selected=true]{background:var(--s3);}
.tab[data-v=dfn2][aria-selected=true]{background:var(--s4);}
.tab:focus-visible{outline:2px solid var(--s1);outline-offset:2px;}
.viz{position:relative;height:74px;margin-bottom:8px;cursor:pointer;}
canvas{width:100%;height:74px;display:block;border-radius:6px;background:var(--track);}
.play{display:flex;align-items:center;gap:12px;}
button.pp{width:38px;height:38px;border-radius:50%;border:none;cursor:pointer;flex:none;
 background:var(--ink);color:var(--plane);font-size:14px;display:grid;place-items:center;}
button.pp:focus-visible{outline:2px solid var(--s1);outline-offset:2px;}
.time{font-size:12px;color:var(--muted);font-variant-numeric:tabular-nums;min-width:82px;}
.meta{font-size:12.5px;color:var(--ink2);margin-left:auto;text-align:right;
 font-variant-numeric:tabular-nums;}
.meta b{color:var(--ink);}
.badge{font-size:10.5px;padding:2px 7px;border-radius:999px;font-weight:600;margin-left:6px;}
.badge.good{background:color-mix(in srgb,var(--good) 16%,transparent);color:var(--good);}
.badge.bad{background:color-mix(in srgb,var(--bad) 16%,transparent);color:var(--bad);}
footer{color:var(--muted);font-size:12px;margin-top:20px;line-height:1.6;}
footer code{font-size:11.5px;}
</style></head><body><div class="wrap">
<h1>Hear it: f32 vs f16 vs int8</h1>
<p class="lede">The same model over each recording — <b>only the weight precision changes</b>.
Switching version <b>keeps the playhead</b>, so you hear the same instant of audio each way.</p>
<p class="hint">Click the waveform to seek · <kbd>space</kbd> play/pause · <kbd>1</kbd>–<kbd>5</kbd>
switch version on the last-used player</p>
<div id="app"></div>
<footer>
  <b>f16 is indistinguishable from f32</b> — its error lands near −94 dBFS, at or below the
  −96 dBFS floor of a 16-bit WAV, so it cannot even be stored in the file.<br>
  <b>int8 loses twice</b> — an audible hiss (≈ −57 dBFS) <em>and</em> ~2 dB less noise removed.
  Best A/B: <code>01_munching</code>, f32 → int8.<br>
  DeepFilterNet2 is a <em>different</em> model at 48 kHz — context, not a fair head-to-head.
</footer>
</div>
<script>
const DATA = __DATA__, VERSIONS = __VERSIONS__;
const COL = {input:'--mut',f32:'--s1',f16:'--s2',int8:'--s3',dfn2:'--s4'};
const cssv = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
let last = null;

const app = document.getElementById('app');
DATA.forEach((d, i) => {
  const avail = VERSIONS.filter(([k]) => d.files[k]);
  const card = document.createElement('div');
  card.className = 'card';
  card.innerHTML = `
    <div class="top">
      <div class="name">${d.folder.replace(/_/g,' ')}
        <span>${d.dur}s${d.snr ? ' · input SNR ≈ '+d.snr+' dB' : ''}</span></div>
    </div>
    <div class="tabs" role="tablist">
      ${avail.map(([k,n,s]) => `<button class="tab" data-v="${k}" role="tab"
         aria-selected="${k==='f32'}" title="${s}">
         <i class="dot" style="background:var(${COL[k]})"></i>${n}</button>`).join('')}
    </div>
    <div class="viz"><canvas></canvas></div>
    <div class="play">
      <button class="pp" aria-label="Play">▶</button>
      <span class="time">0:00 / 0:00</span>
      <span class="meta"></span>
    </div>`;
  app.appendChild(card);

  const cv = card.querySelector('canvas'), ctx = cv.getContext('2d');
  const au = new Audio(); au.preload = 'metadata';
  const pp = card.querySelector('.pp'), tm = card.querySelector('.time');
  const meta = card.querySelector('.meta');
  let cur = 'f32';

  const fmt = t => `${Math.floor(t/60)}:${String(Math.floor(t%60)).padStart(2,'0')}`;

  function draw() {
    const env = d.env[cur] || [];
    const W = cv.width = cv.clientWidth * devicePixelRatio;
    const H = cv.height = 74 * devicePixelRatio;
    ctx.clearRect(0,0,W,H);
    const prog = au.duration ? au.currentTime / au.duration : 0;
    const c = cssv(COL[cur]);
    for (let i=0;i<env.length;i++){
      const x = i/env.length*W, w = Math.max(W/env.length-1*devicePixelRatio, 1);
      const h = Math.max(env[i]*(H*0.86), 1.5*devicePixelRatio);
      ctx.fillStyle = c;
      ctx.globalAlpha = (i/env.length) <= prog ? 1 : 0.32;
      ctx.fillRect(x, (H-h)/2, w, h);
    }
    ctx.globalAlpha = 1;
  }

  function setMeta(){
    if (cur==='input'){ meta.innerHTML = 'the recording fed to the model'; return; }
    if (cur==='dfn2'){ meta.innerHTML = 'different model · 48 kHz · context only'; return; }
    const m = d.m[cur];
    if (cur==='f32'){ meta.innerHTML = `<b>${m.red>0?'+':''}${m.red.toFixed(1)} dB</b> noise removed · reference`; return; }
    const bad = cur==='int8';
    meta.innerHTML = `<b>${m.red>0?'+':''}${m.red.toFixed(1)} dB</b> noise removed ·
      error ${m.err.toFixed(0)} dBFS
      <span class="badge ${bad?'bad':'good'}">${bad?'audible':'inaudible'}</span>`;
  }

  function select(k, keep=true){
    if (!d.files[k] || k===cur) return;
    const t = au.currentTime, playing = !au.paused;
    cur = k;
    card.querySelectorAll('.tab').forEach(b =>
      b.setAttribute('aria-selected', b.dataset.v===k));
    au.src = encodeURI(d.folder + '/' + d.files[k]);
    au.addEventListener('loadedmetadata', () => {
      if (keep) au.currentTime = Math.min(t, au.duration||t);
      if (playing) au.play();
      draw();
    }, {once:true});
    setMeta();
  }

  card.querySelectorAll('.tab').forEach(b =>
    b.onclick = () => { last = api; select(b.dataset.v); });

  pp.onclick = () => { last = api; au.paused ? au.play() : au.pause(); };
  au.onplay  = () => { pp.textContent = '❚❚'; pp.setAttribute('aria-label','Pause'); };
  au.onpause = () => { pp.textContent = '▶';  pp.setAttribute('aria-label','Play'); };
  au.ontimeupdate = () => {
    tm.textContent = `${fmt(au.currentTime)} / ${fmt(au.duration||d.dur)}`;
    draw();
  };
  au.onended = () => { pp.textContent='▶'; };
  card.querySelector('.viz').onclick = e => {
    last = api;
    const r = e.currentTarget.getBoundingClientRect();
    if (au.duration) { au.currentTime = (e.clientX-r.left)/r.width * au.duration; draw(); }
  };

  const api = { select, toggle: () => au.paused ? au.play() : au.pause(), avail };
  au.src = encodeURI(d.folder + '/' + d.files.f32);
  setMeta();
  au.addEventListener('loadedmetadata', () => {
    tm.textContent = `0:00 / ${fmt(au.duration)}`; draw();
  }, {once:true});
  addEventListener('resize', draw);
  if (i===0) last = api;
});

addEventListener('keydown', e => {
  if (!last || e.target.tagName==='BUTTON' && e.key===' ') {}
  if (e.key===' ') { e.preventDefault(); last && last.toggle(); return; }
  const n = parseInt(e.key,10);
  if (n>=1 && n<=5 && last) {
    const v = last.avail[n-1];
    if (v) last.select(v[0]);
  }
});
</script></body></html>
"""


def serve(path):
    os.chdir(AUDIO)
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=AUDIO)
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("127.0.0.1", PORT), handler) as httpd:
        url = f"http://127.0.0.1:{PORT}/index.html"
        print(f"\n  serving {AUDIO}")
        print(f"  →  {url}\n\n  Ctrl-C to stop")
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nstopped")


if __name__ == "__main__":
    p = build()
    if "--build" not in sys.argv:
        serve(p)
