# hush-vani — engineering notes

How this implementation was built, measured, and where it is still wrong. None of this is
needed to *use* the crate — see [README.md](README.md) for that. It is kept because the
measurements are the whole reason the code looks the way it does, and because several
"obvious" optimisations lost.

Every reverted change is recorded rather than quietly dropped, including one (weight
packing) that was reverted **in error** and reinstated once measured properly.

## Correctness

`hush-devtool verify` — everything matches to float32 rounding:

| tensor | max abs diff |
|---|---|
| `spec.re` / `spec.im` (STFT vs libdf) | 4.5e-9 / 3.7e-9 |
| `feat_erb`, `feat_spec` | 5.7e-7, 8.9e-8 |
| `e0`…`e3`, `c0` | 3.6e-7 … 3.9e-6 |
| `emb`, `lsnr` | 4.2e-6, 1.0e-5 |
| `mask`, `coefs` | 1.0e-6, 7.7e-7 |
| **output audio vs ORT** | **1.2e-7 (SI-SDR 129.7 dB)** |

## Speed vs ONNX Runtime
<a id="speed-vs-onnx-runtime"></a>

`python tools/ab_bench.py` — 3 replicates × 50 samples per side, the two implementations
**interleaved in blocks of 10** with alternating order. The reported statistic is the ratio
*within* each interleaved turn, bootstrapped over turns: the two sides are measured adjacent
in time, so thermal drift cancels. (Pooling raw samples does not work here — the machine
warms ~20% across three replicates, and that between-replicate variance swamps a 5% effect.)

5 s clip, 500 frames, n=15 paired turns per scope:

| scope | threads | speedup | 95% CI | turns won |
|---|---|---|---|---|
| **NN only** (no DSP) | 1 | **1.066x** | [1.046, 1.083] | 14/15 |
| full pipeline | 1 | **1.120x** | [1.092, 1.150] | 15/15 |
| full pipeline | 2 | **1.333x** | [1.293, 1.375] | 15/15 |

All three significant. Median wall-clock, 5 s of audio: rust 55.9 ms (NN), 62.9 ms
(pipeline, 1 thread), 57.6 ms (pipeline, 2 threads) — RTF 0.0112 / 0.0126 / 0.0115.

The three wins have different causes, and only the first is a kernel result:

- **NN, 1.066x** — from fixing what the profiler pointed at: `grouped_linear` and
  `pointwise` were running at 4–12 GFMA/s while the GRU kernels hit 19–25, and the
  recurrent matvec gained 1.16x from panel-packing its weights.
- **Pipeline, +5% on top** — the Rust STFT/ERB/ISTFT costs 0.55 ms against `libdf`'s
  ~4.9 ms. That says more about `libdf` than about either NN.
- **2 threads, 1.333x — read with care.** ORT is *slower* with 2 intra-op threads (76.2 ms)
  than with 1 (69.6 ms): it parallelises inside ops far too small to amortise the sync.
  Against ORT's own best configuration (1 thread, 69.6 ms), the 2-thread Rust build
  (57.6 ms) is **1.21x** faster — that is the honest figure. The gain is decomposition, not
  kernels: `erb_dec` and `df_dec` are independent given `emb`, so they run concurrently.

Absolute ms drift between sessions by ~20%; on a cold machine `hush bench` reports ~68 ms
(1 thread) / ~55 ms (2 threads). Only the paired ratios above are stable.

## Optimisation history

Each step is a measured median, correctness re-verified after every change:

Medians from `hush bench 5` on an otherwise idle machine, correctness re-verified after
every change. These are unpaired and drift between sessions — use them for the *ratios*
between steps, not to compare against ORT (see the A/B table above for that).

| step | total | vs prev |
|---|---|---|
| naive scalar (LLVM auto-vectorisation only) | 198 ms | — |
| explicit AVX2 `dot` + 8-row `matvec` | 152 ms | 1.30x |
| 64-byte aligned weight arena + `vmovaps` | 96 ms | 1.58x |
| branch-free convs on zero-padded rows | 92 ms | 1.04x |
| tiled GEMM for the GRU input projection | 80.5 ms | 1.14x |
| vectorised exp/tanh (Cephes, 8-wide) | 68.0 ms | 1.18x |
| register-blocked `grouped_linear` + `pointwise`, 12-row packed recurrent matvec | — | 1.08x (paired, NN only) |
| + `erb_dec` ∥ `df_dec` on 2 threads | — | 1.16x (paired, full) |
| | **~3.6x total** | |

The last two rows are measured as paired ratios (see below) rather than absolute medians,
because by then the differences were smaller than this machine's run-to-run drift.

### What actually mattered

**LLVM will not auto-vectorise a float dot product.** Strict IEEE ordering forbids
reassociating the adds, so `iter().sum()` — and every accumulator-array trick I tried —
compiled to scalar FMA: 2.9 GFMA/s, i.e. exactly scalar throughput. Explicit
`_mm256_fmadd_ps` gets 12 GFMA/s. This was the single biggest win and the least obvious.

**Alignment is worth ~30%.** `Vec<f32>` is 16-byte aligned on Windows, so half of every
32-byte AVX2 load straddles a cache line. Padding each tensor to a 64-byte boundary in the
exporter and allocating the arena with `alloc_zeroed(align=64)` lets the kernel use
`vmovaps`. Measured on the same data, same kernel (`micro.rs`):

| load | GFMA/s |
|---|---|
| `loadu`, base + 4 B | 12.9 |
| `loadu`, 64 B-aligned base | 14.5 |
| `load` (aligned) | 16.5 |

**A matvec is load-bound; a GEMM isn't.** The 8-row `matvec` issues ~9 loads per 8 FMAs
because each weight row is used exactly once — that caps it near 16 GFMA/s regardless of
FMA capacity. The GRU's input projection `x@Wᵀ` has no recurrence, so it can be tiled
(4 weight rows × 2 timesteps, 8 accumulators) down to ~6 loads per 8 FMAs. The recurrent
`h@Rᵀ` cannot be tiled — `h` changes every frame — and that is the hard floor of this
model: ~6 ms per GRU, ~30 ms for all five.

**Scalar libm was a quarter of the runtime.** The gates evaluate ~1.28 M `expf` and
~640 K `tanhf` per 5 s clip, at 7.4 ns and 17.9 ns each (`micro.rs`) — about 27 ms,
including the `tanh` over the 320 K deep-filter coefficients. Replacing them with 8-wide
Cephes approximations (`simd.rs`) and vectorising the whole gate update took 80.5 → 68.0 ms
in one step: MLAS has vectorised transcendentals, and until this change we were paying
scalar for all of them. Cost: end-to-end SI-SDR vs ORT drops from 131.1 dB to 129.7 dB —
irrelevant at that margin.

**Profile before you optimise the thing you assume is slow.** After the GRU work I assumed
the GRUs were all that was left. `profile.rs` said otherwise: the GRU kernels ran at 19–25
GFMA/s while `grouped_linear` sat at 6–12 and `pointwise` at 3.9 — roughly 20 ms of the
83 ms NN running at a third of achievable speed. Both were axpy loops that stream the whole
output row per input element (3 memory ops per vector-FMA); holding a 64-wide output slab
in registers drops that to ~1.1. Two caveats found only by measuring:

- `pointwise` needs its output-block size as a **const generic**. With a runtime bound
  (`let ob = 8.min(cout-o0)`) LLVM spills the accumulators to the stack and the kernel runs
  at *half* the speed of the naive version.
- It also needs spatial tiling. The `cin` input planes sit `tf` floats apart, so sweeping
  channels for each position touches 16 streams 128 KB apart; the prefetcher gives up and
  the blocked version loses. A 4096-element tile keeps a slab of all planes in L2.

**Weight packing works — I dismissed it once on bad evidence.** Repacking the recurrent
matrix into 12-row panels (so the kernel walks one sequential stream instead of 12 pointers
1 KB apart) is worth **1.16x**, bit-identically. An earlier attempt was judged "slower" from
a noisy full-pipeline timing and reverted; an in-process paired A/B (`kernel_ab.rs`) shows
it clearly winning. 12 rows beats 8 because 12 accumulators plus the broadcast vector still
fit the 16 ymm registers, giving 13 memory ops per 12 vector-FMAs instead of 9 per 8.

### What didn't work (measured, reverted)

- **More accumulator chains in scalar code.** 4×8 accumulators to hide FMA latency: no
  change (2.9 → 3.0 GFMA/s). The code was scalar, so latency was never the bottleneck.
  Chasing the wrong bottleneck cost two rounds of regressions (198 → 228 → 277 ms).
- **Cache-blocking the weight matrix.** Sweeping the matvec footprint from 8 KB to 3 MB
  changed nothing (2.8–3.3 GFMA/s), proving it was issue-bound, not memory-bound. Worth
  measuring before optimising: the "obviously memory-bound" reading was wrong.
- **Register-blocked `pointwise` without tiling / without const-generic blocks.** 0.5x and
  0.9x respectively. Both fixed above; the naive-looking axpy beat a half-finished kernel.
- **f16 *compute* kernels.** The reasoning looked airtight: the recurrent matvec re-reads the
  whole 768×256 matrix every frame, so holding weights as f16 halves the bytes per FMA, and
  `vcvtph2ps` widens 8 lanes in one instruction. Measured end-to-end (paired, interleaved):
  **0.904x — 10% *slower***, CI [0.814, 0.971].

  The premise was simply false, and one number shows why (`hush-kernel-ab`):

  | kernel (f32) | throughput |
  |---|---|
  | input GEMM `x@Wᵀ` | **55.2 GFMA/s** |
  | recurrent matvec `h@Rᵀ` | 37.2 GFMA/s |

  AVX2 peak is 2 FMA units × 8 lanes ≈ **56 GFMA/s**. The GEMM is at ~99% of the hardware
  issue ceiling — **you cannot be memory-starved while executing at peak FMA rate.** Nothing
  was ever waiting on memory, so there is no stall for the conversions to hide in: every
  `vcvtph2ps` is an extra uop that displaces real work. The kernel closest to peak lost the
  *most* (GEMM 0.80x, matvec 0.89x) — backwards for a bandwidth story, exactly right for an
  issue-bound one. (A first attempt to explain this by weight *reuse* — 1 convert per 2 FMAs
  in the GEMM vs 1-per-1 in the matvec — predicted the GEMM would be fine. It was worse. The
  data corrected that too.)

  Kept behind the `f16-kernels` feature: they do halve resident weight memory
  (9.1 → 4.6 MB), a real trade for a memory-tight target, just not a speed one.

### The int8 control experiment
<a id="the-int8-control-experiment"></a>

If the kernels are issue-bound, the lever is **fewer instructions**, not fewer bytes. AVX-VNNI's
`vpdpbusd` does **32 int8 MACs in one instruction** against an f32 FMA's 8 — 4× fewer
instructions for identical work. Prediction: int8 should be *much* faster where f16 was slower.

On the recurrent matvec (`hush-kernel-ab`, paired, real weights):

| weights | kernel throughput | vs f32 | model size |
|---|---|---|---|
| **f32** | 41.3 GFMA/s | 1.00x | 9.12 MB |
| **f16** | 32.7 GFMA/s | **0.85x** 🔻 | 4.56 MB |
| **int8** (VNNI) | **144.5 GFMA/s** | **3.50x** 🔺 | 2.28 MB |

int8 exceeds the f32 FMA *peak* (~56 GFMA/s) — impossible for a bandwidth story, inevitable
for an instruction-count one. Same data volume argument, opposite result, because the
instruction count went the other way. That settles it: **these kernels are bound by issue
rate, and the only currency that buys speed is instructions.**

**This kernel is not what ships, and the 3.5× is not claimed anywhere above.** It quantises
the *activations* too (that is what `vpdpbusd` requires), it needs AVX-VNNI (Ice Lake / Zen 4
and later), and it lives in the benchmark binary. The shipped int8 is a *storage* format:
dequantised at load, run on the f32 kernels, exactly the same speed as f32. Turning this
control experiment into a real compute path — measuring what activation quantisation costs,
and writing an AVX2 fallback for the CPUs without VNNI — is the open next step.

The first version of this experiment used one scale per tensor and reported "30.0 dB
end-to-end, audible, not shippable". That verdict was about the *scaling scheme*, not about
int8: with 64-wide block scales the same 8-bit weights remove exactly as much noise as f32
(see [Weights](#weights)). What made the difference was measuring the right thing — SI-SDR
said "noisier", but the number that mattered was **noise removed**, where per-tensor int8 was
2.1 dB *worse at the job* while merely sounding grittier.

Every reverted change is listed rather than quietly dropped — including one, weight
packing, that was reverted *in error* and later reinstated once measured properly.

### `f32::mul_add` made every non-AVX2 target 3x slower

Found by asking why the wasm build ran at 1.8x realtime when it should have been ~10x.

The scalar fallbacks — the ones that run on wasm, on aarch64, and on pre-Haswell x86 — were
all written with `a.mul_add(b, c)`. That is a *fused* multiply-add: one rounding instead of
two, and free **if the target has an FMA instruction at compile time**. Neither condition
held. This crate dispatches AVX2 at *runtime*, so it is built without `-C target-feature=+fma`;
and wasm has no FMA instruction at all. On both, `mul_add` cannot lower to hardware, so it
calls the software `fmaf`, which computes the exact double-width product and rounds once.

Measured on a 256-wide f32 dot product, same source both sides:

| | plain `a*b + c` | `f32::mul_add` | |
|---|---|---|---|
| native (no `+fma`) | 5.9 GFLOP/s | 2.2 GFLOP/s | **2.6x slower** |
| wasm32 | 4.3 GFLOP/s | 0.53 GFLOP/s | **8.0x slower** |

Nine call sites, every one of them in a hot kernel: `dot_scalar`, `axpy`, `matvec_packed`,
`gemm_nt`, `grouped_linear`, `pointwise`, and the GRU gate update. Replacing them with plain
`a * b + c` took a 5 s clip on wasm from **2802 ms to 885 ms (3.2x)**, with bit-identical
output quality (35.8 dB of noise removed, unchanged).

The AVX2 kernels in `mod x86` still use `mul_add` and should: they live inside
`#[target_feature(enable = "avx2,fma")]`, where it *is* a single `vfmadd` and the extra
precision really is free. The cost of the change is one extra rounding per MAC in the scalar
path — which already sums in a different order from the vector path anyway. End-to-end
against onnxruntime is unchanged at 129.7 dB SI-SDR.

The general lesson, and it is the same one as the f16 kernels above: **an intrinsic that is
free on the target you benchmark can be catastrophic on the target you don't.** `mul_add` is
strictly better on paper — fewer roundings, fewer instructions — and it was silently costing
8x everywhere the benchmark never looked.

### What wasm costs after that

Still ~5.7x realtime against ~100x native. What remains is the honest cost of no SIMD: the
portable path does one f32 at a time where AVX2 does eight. `-C target-feature=+simd128` buys
**nothing** (2802 → 2869 ms, i.e. noise), for exactly the reason the x86 kernels are
hand-written: LLVM will not auto-vectorise a float dot product, because IEEE ordering forbids
reassociating the adds. Explicit `core::arch::wasm32` v128 kernels would slot in beside the
AVX2 ones; 4 lanes instead of 8 suggests ~3-4x, i.e. ~20x realtime.

### A bug this found: subnormal f16 decoded 2× small

Storing block scales as f16 exposed a latent bug in `f16_to_f32`: for **2046 of the 2048
subnormal** bit patterns it returned exactly half the right value (the renormalising shift
was off by one and left the implicit leading 1 in the mantissa field). Normals were all
correct, and the round-trip test stepped `x` by 0.001 — never landing on a subnormal — so it
had always passed. The hardware `vcvtph2ps` used by the f16 *kernels* gets this right, so
the scalar loader and the SIMD kernels had been quietly disagreeing.

It barely moved the f16 weights (those values are tiny by definition: SI-SDR 66.9 → 67.0 dB),
but a bad *scale* corrupts all 64 weights in its block, which is how it finally showed up.
Fixed, and now checked exhaustively over all 65,536 bit patterns against the definition
rather than a round-trip.

### Where the remaining time goes

Per GRU at T=500 (`profile.rs`, idle machine): recurrent matvec 5.1 ms, input GEMM 3.9 ms,
gates 0.4 ms. Five GRUs ≈ 47 ms of a ~68 ms pipeline. The recurrent `h@Rᵀ` streams all
768×256 weights every frame and `h` changes each step, so it cannot be tiled over time; at
~19 GFMA/s it is close to the L2 bandwidth roofline for that access pattern. Going
meaningfully faster from here means changing the numerics (bf16/int8 weights halve the
traffic) or the model, not the schedule.

### Benchmarking method (two levels)

1. **Kernel level** — `cargo run --release --bin kernel_ab`: runs the old and new kernel in
   the *same process*, interleaved, and reports the paired ratio. Required, because this
   machine throttles ~1.9x under sustained load: untouched code "regressed" from 9.4 ms to
   17.9 ms between two runs.
2. **System level** — `python tools/ab_bench.py`: interleaves Rust and ORT in blocks,
   alternates which goes first, and takes the ratio *within* each turn before bootstrapping
   over turns.

Three separate ways the naive approach lied here, all corrected above:

- An **unpaired** comparison "showed" Rust winning by 3% when a paired one showed ORT ahead.
- A **50-sample** run called a difference significant (p=4e-11) that a 100-sample run put
  at p=0.24.
- **Pooling paired samples across replicates** put the NN result at "not significant, CI
  straddles 0" while every individual replicate had Rust ahead (1.026x, 1.100x, 1.062x) —
  the machine warmed 20% between replicates and that variance drowned the effect. Ratio-
  within-turn gives 1.066x, CI [1.046, 1.083], 14/15 turns.

Never trust a single median from this machine at the <10% scale. `tools/ab_bench.py` takes
`n=50` by default; set `AB_TOTAL=100` for a tighter CI.

## Layout

**Published crate** (`cargo package` → 27 files, 2.3 MB compressed, nearly all of it weights):

- `src/lib.rs` — public API: `Hush`, `Error`.
- `data/weights.{bin,txt}` — the embedded int8 weights and their manifest.
- `src/dsp.rs` — vorbis window, STFT (`rfft/N`), ERB bands, exponential normalisers, ISTFT,
  deep-filter application. Formulas reverse-engineered from `libdf` (see `tools/probe_dsp.py`).
- `src/nn.rs` — AVX2 `dot`/`matvec`/`gemm_nt`, grouped/depthwise/pointwise convs, depthwise
  freq ConvTranspose, grouped linear, ONNX GRU (`linear_before_reset=1`). Runtime-dispatched.
- `src/simd.rs` — 8-wide `exp`/`sigmoid`/`tanh` (Cephes polynomials, ~1e-7 relative), f16
  conversion, and the cached AVX2 capability check.
- `src/encoder.rs`, `src/decoders.rs`, `src/bank.rs` — the three graphs, mirroring
  `tools/{enc,erb_dec,df_dec}.txt`, over a shared weight bank.
- `src/weights.rs`, `src/alloc.rs` — 64-byte-aligned weight arena, name-addressed; f32 / f16 /
  int8 blobs all decode to the same f32 arena here.
- `src/bin/cli.rs` — the `hush-vani` binary.

**Repo only** (excluded from the crate):

- `assets/` — f32/f16 weight blobs and ORT fixtures for `verify`.
- `src/bin/{devtool,profile,kernel_ab,micro,features}.rs` — gated behind `--features devtools`.
- `tools/export_weights.py` — ONNX initializers → `weights.{bin,txt}` (`--f16`, `--int8`).
- `tools/quant_schemes.py`, `tools/quant_sweep.py` — how the int8 block size and scale dtype
  were chosen: quantise in numpy, run the real kernels, measure the audio.
- `tools/verify_int8.py` — asserts the Rust int8 loader reproduces that simulation exactly.
- `tools/bench_samples.py` — f32 / f16 / int8 over six real DeepFilterNet2 demo recordings.
- `tools/export_fixtures.py` — ORT intermediates → `assets/*.bin` (for `verify`).
- `tools/dump_graph.py` — human-readable ONNX graph dump.
- `tools/probe_dsp.py`, `tools/probe_synth.py` — how the `libdf` DSP was reverse-engineered.
- `tools/ab_bench.py` — the paired Rust-vs-ORT benchmark.
- `tools/compare_audio.py` — checks the produced wav against ORT and the published reference.

Python is only ever a build/verification step; it never runs at inference time.

## Notes

- **Runtime dispatch, not compile-time.** An earlier version gated the AVX2 kernels on
  `#[cfg(target_feature = "avx2")]`, which silently degrades to scalar (~4x slower) for
  anyone who builds without `-C target-cpu=native`. A published crate cannot assume that,
  so the kernels are now `#[target_feature]` functions selected by `is_x86_feature_detected!`.
- **License**: Apache-2.0, matching the upstream Hush model. The embedded weights are the
  [`weya-ai/hush`](https://huggingface.co/weya-ai/hush) model redistributed under its own
  Apache-2.0 licence — the container format changes, no weight values are altered beyond the
  int8 quantisation described above. See `LICENSE` and `NOTICE`. Not affiliated with or
  endorsed by Weya AI.

