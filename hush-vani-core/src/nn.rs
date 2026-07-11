//! Hand-written NN primitives. Feature maps are laid out (C, T, F) row-major, so the
//! frequency axis is contiguous -- every conv here is frequency-local.
//!
//! Every kernel has a portable scalar path and an AVX2+FMA path selected at runtime by
//! [`crate::simd::has_avx2`]. On x86-64 with AVX2 the fast path is ~4x the scalar one.

use crate::alloc::AlignedVec;
use crate::simd::{has_avx2, sigmoid};

/// Rows per panel in the packed layout. 12 accumulators + 1 broadcast vector still fit the
/// 16 ymm registers, and 13 memory ops per 12 vector-FMAs beats 9-per-8 (measured 1.16x).
pub const PANEL: usize = 12;

// ---------------------------------------------------------------- scalar helpers

/// out += s * x
#[inline]
pub fn axpy(out: &mut [f32], s: f32, x: &[f32]) {
    #[cfg(target_arch = "x86_64")]
    if has_avx2() {
        unsafe { x86::axpy(out, s, x) };
        return;
    }
    for (o, v) in out.iter_mut().zip(x.iter()) {
        *o = s.mul_add(*v, *o);
    }
}

pub fn relu_(x: &mut [f32]) {
    for v in x.iter_mut() {
        if *v < 0.0 {
            *v = 0.0;
        }
    }
}

#[inline]
fn dot_scalar(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = [0f32; 8];
    let mut ca = a.chunks_exact(8);
    let mut cb = b.chunks_exact(8);
    for (x, y) in ca.by_ref().zip(cb.by_ref()) {
        for k in 0..8 {
            acc[k] = x[k].mul_add(y[k], acc[k]);
        }
    }
    let mut s = (acc[0] + acc[1]) + (acc[2] + acc[3]) + ((acc[4] + acc[5]) + (acc[6] + acc[7]));
    for (x, y) in ca.remainder().iter().zip(cb.remainder().iter()) {
        s += x * y;
    }
    s
}

/// Dot product.
///
/// LLVM will not auto-vectorise a float reduction -- strict IEEE ordering forbids
/// reassociating the adds, and no accumulator-array arrangement convinces it (measured:
/// 2.7-4.5 GFMA/s, i.e. scalar FMA throughput). Explicit AVX2 reaches ~12 GFMA/s.
#[inline]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    #[cfg(target_arch = "x86_64")]
    if has_avx2() {
        return unsafe { x86::dot(a, b) };
    }
    dot_scalar(a, b)
}

// ---------------------------------------------------------------- matvec / gemm

/// out[j] = dot(m[j], x) + bias[j], for j in 0..rows.
pub fn matvec(m: &[f32], rows: usize, k: usize, x: &[f32], bias: Option<&[f32]>, out: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    if has_avx2() {
        // rows are contiguous with stride k; aligned iff base and stride are 32B-aligned
        let ok = crate::alloc::is_aligned32(m.as_ptr())
            && crate::alloc::is_aligned32(x.as_ptr())
            && (k * 4) % 32 == 0;
        unsafe {
            if ok {
                x86::matvec::<true>(m, rows, k, x, bias, out)
            } else {
                x86::matvec::<false>(m, rows, k, x, bias, out)
            }
        }
        return;
    }
    for j in 0..rows {
        out[j] = dot_scalar(&m[j * k..(j + 1) * k], x) + bias.map_or(0.0, |b| b[j]);
    }
}

/// Repack [rows,K] row-major into `PANEL`-row panels: within each panel the rows' 8-float
/// chunks are interleaved, so the kernel walks one sequential stream instead of `PANEL`
/// pointers K*4 bytes apart. Requires rows % PANEL == 0 and K % 8 == 0.
pub fn pack_rows(m: &[f32], rows: usize, k: usize) -> AlignedVec {
    assert!(rows % PANEL == 0 && k % 8 == 0, "pack_rows: bad shape {rows}x{k}");
    let mut p = AlignedVec::zeros(rows * k);
    for jb in 0..rows / PANEL {
        for c in 0..k / 8 {
            for l in 0..PANEL {
                let s = (jb * PANEL + l) * k + c * 8;
                let d = jb * PANEL * k + c * 8 * PANEL + l * 8;
                p[d..d + 8].copy_from_slice(&m[s..s + 8]);
            }
        }
    }
    p
}

/// matvec against a [`pack_rows`] panel layout. Bit-identical to [`matvec`] (same summation
/// order per row); 1.16x faster on the recurrent shape.
pub fn matvec_packed(p: &[f32], rows: usize, k: usize, x: &[f32], out: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    if has_avx2() && crate::alloc::is_aligned32(p.as_ptr()) && crate::alloc::is_aligned32(x.as_ptr()) {
        unsafe { x86::matvec_packed(p, rows, k, x, out) };
        return;
    }
    for j in 0..rows {
        let (jb, l) = (j / PANEL, j % PANEL);
        let mut s = 0.0;
        for c in 0..k / 8 {
            for e in 0..8 {
                s = p[jb * PANEL * k + c * 8 * PANEL + l * 8 + e].mul_add(x[c * 8 + e], s);
            }
        }
        out[j] = s;
    }
}

/// out[t][j] = dot(w[j], x[t]) + b[j]   -- x [T,K] row-major, w [rows,K] row-major.
///
/// A matvec is load-bound (~9 loads per 8 FMAs) because each weight row is used once. This
/// is a GEMM: tiling 4 weight rows x 2 timesteps drops it to 6 loads per 8 FMAs.
pub fn gemm_nt(x: &[f32], t: usize, k: usize, w: &[f32], rows: usize, b: &[f32], out: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    if has_avx2()
        && (k * 4) % 32 == 0
        && crate::alloc::is_aligned32(w.as_ptr())
        && crate::alloc::is_aligned32(x.as_ptr())
    {
        unsafe { x86::gemm_nt(x, t, k, w, rows, b, out) };
        return;
    }
    for ti in 0..t {
        matvec(w, rows, k, &x[ti * k..(ti + 1) * k], Some(b), &mut out[ti * rows..(ti + 1) * rows]);
    }
}

// ---------------------------------------------------------------- convolutions

/// Zero-pad the frequency axis by 1 on each side: (C,T,F) -> (C,T,F+2).
fn pad_freq(x: &[f32], c: usize, t: usize, f: usize) -> Vec<f32> {
    let fp = f + 2;
    let mut out = vec![0f32; c * t * fp];
    for ch in 0..c {
        for ti in 0..t {
            let s = ch * t * f + ti * f;
            let d = ch * t * fp + ti * fp + 1;
            out[d..d + f].copy_from_slice(&x[s..s + f]);
        }
    }
    out
}

/// 1x1 conv over (C,T,F): out[o] = b[o] + sum_c w[o][c] * x[c]
///
/// The naive form (`for o { for c { out[o] += w * x[c] } }`) reads and writes each output
/// plane once per input channel. Instead walk 8 spatial positions at a time and accumulate
/// all outputs in registers, so each output element is written exactly once.
pub fn pointwise(x: &[f32], cin: usize, tf: usize, w: &[f32], b: Option<&[f32]>, cout: usize) -> Vec<f32> {
    let mut out = vec![0f32; cout * tf];

    #[cfg(target_arch = "x86_64")]
    if has_avx2() {
        unsafe { x86::pointwise(x, cin, tf, w, b, cout, &mut out) };
        return out;
    }

    for o in 0..cout {
        let dst = &mut out[o * tf..(o + 1) * tf];
        if let Some(b) = b {
            dst.fill(b[o]);
        }
        for c in 0..cin {
            let s = w[o * cin + c];
            if s != 0.0 {
                axpy(dst, s, &x[c * tf..(c + 1) * tf]);
            }
        }
    }
    out
}

/// Depthwise conv, kernel (1,3) over frequency, zero-pad 1 each side, given stride.
pub fn dw_conv13(x: &[f32], c: usize, t: usize, f: usize, w: &[f32], stride: usize) -> (Vec<f32>, usize) {
    let fo_n = (f + 2 - 3) / stride + 1;
    let mut out = vec![0f32; c * t * fo_n];
    for ch in 0..c {
        let k = [w[ch * 3], w[ch * 3 + 1], w[ch * 3 + 2]];
        for ti in 0..t {
            let src = &x[ch * t * f + ti * f..ch * t * f + (ti + 1) * f];
            let dst = &mut out[ch * t * fo_n + ti * fo_n..ch * t * fo_n + (ti + 1) * fo_n];
            for (fo, d) in dst.iter_mut().enumerate() {
                let base = (fo * stride) as isize - 1;
                let mut s = 0.0;
                for (kk, &kv) in k.iter().enumerate() {
                    let idx = base + kk as isize;
                    if idx >= 0 && (idx as usize) < f {
                        s = kv.mul_add(src[idx as usize], s);
                    }
                }
                *d = s;
            }
        }
    }
    (out, fo_n)
}

/// Depthwise ConvTranspose over frequency: kernel (1,3), stride (1,2), pads [0,1,0,1],
/// output_padding [0,1]  =>  F_out = 2 * F_in.
///
/// With pad_begin = 1, output index n maps to full-conv index m = n+1, so even n takes a
/// single kernel tap and odd n takes two. No intermediate is materialised.
pub fn convtranspose_dw(x: &[f32], c: usize, t: usize, f: usize, w: &[f32]) -> (Vec<f32>, usize) {
    let fo = 2 * f;
    let mut out = vec![0f32; c * t * fo];
    for ch in 0..c {
        let (k0, k1, k2) = (w[ch * 3], w[ch * 3 + 1], w[ch * 3 + 2]);
        for ti in 0..t {
            let src = &x[ch * t * f + ti * f..ch * t * f + (ti + 1) * f];
            let dst = &mut out[ch * t * fo + ti * fo..ch * t * fo + (ti + 1) * fo];
            for i in 0..f {
                dst[2 * i] = src[i] * k1;
                let mut odd = src[i] * k2;
                if i + 1 < f {
                    odd = k0.mul_add(src[i + 1], odd);
                }
                dst[2 * i + 1] = odd;
            }
        }
    }
    (out, fo)
}

/// 3x3 conv with causal time padding (2 frames of history, zero lookahead) and zero-pad 1
/// on frequency. `groups` splits input and output channels evenly.
///
/// Written as accumulate-by-tap (axpy over frequency) on a pre-padded input, so the inner
/// loop is branch-free and vectorises.
pub fn conv3x3_causal(
    x: &[f32],
    cin: usize,
    t: usize,
    f: usize,
    w: &[f32],
    b: Option<&[f32]>,
    cout: usize,
    groups: usize,
) -> Vec<f32> {
    let cg = cin / groups;
    let og = cout / groups;
    let fp = f + 2;
    let xp = pad_freq(x, cin, t, f);
    let mut out = vec![0f32; cout * t * f];
    for o in 0..cout {
        let g = o / og;
        for ti in 0..t {
            let dst = &mut out[o * t * f + ti * f..o * t * f + (ti + 1) * f];
            if let Some(b) = b {
                dst.fill(b[o]);
            }
            for ci in 0..cg {
                let c = g * cg + ci;
                for kt in 0..3 {
                    let tt = ti as isize + kt as isize - 2; // taps at t-2, t-1, t
                    if tt < 0 {
                        continue;
                    }
                    let row = &xp[c * t * fp + tt as usize * fp..c * t * fp + (tt as usize + 1) * fp];
                    for kf in 0..3 {
                        let s = w[((o * cg + ci) * 3 + kt) * 3 + kf];
                        if s != 0.0 {
                            axpy(dst, s, &row[kf..kf + f]);
                        }
                    }
                }
            }
        }
    }
    out
}

/// Conv with kernel (1,3) over frequency, full channel mixing, zero-pad 1.
pub fn conv1x3(x: &[f32], cin: usize, t: usize, f: usize, w: &[f32], b: &[f32], cout: usize) -> Vec<f32> {
    let fp = f + 2;
    let xp = pad_freq(x, cin, t, f);
    let mut out = vec![0f32; cout * t * f];
    for o in 0..cout {
        for ti in 0..t {
            let dst = &mut out[o * t * f + ti * f..o * t * f + (ti + 1) * f];
            dst.fill(b[o]);
            for c in 0..cin {
                let row = &xp[c * t * fp + ti * fp..c * t * fp + (ti + 1) * fp];
                for kf in 0..3 {
                    let s = w[(o * cin + c) * 3 + kf];
                    if s != 0.0 {
                        axpy(dst, s, &row[kf..kf + f]);
                    }
                }
            }
        }
    }
    out
}

// ---------------------------------------------------------------- linear / GRU

/// One output row: dst[0..h] = sum_i xv[i] * w[i][0..h].
///
/// Holds a 64-wide slab of the output in 8 ymm registers while sweeping i. The axpy form
/// reloads and restores the whole output row once per i: 3 memory ops per vector-FMA
/// against ~1.1 here. Inputs are post-ReLU so ~half the xv are zero; skipping them pays.
fn linear_row(xv: &[f32], w: &[f32], h: usize, dst: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    if has_avx2() {
        unsafe { x86::linear_row(xv, w, h, dst) };
        return;
    }
    dst.fill(0.0);
    for (ii, &xs) in xv.iter().enumerate() {
        if xs != 0.0 {
            axpy(dst, xs, &w[ii * h..(ii + 1) * h]);
        }
    }
}

/// x: [T, G*I], w: [G, I, H]  ->  [T, G*H]   (ONNX Einsum btgi,gih->btgh)
pub fn grouped_linear(x: &[f32], t: usize, g: usize, i: usize, h: usize, w: &[f32]) -> Vec<f32> {
    let mut out = vec![0f32; t * g * h];
    for ti in 0..t {
        for gi in 0..g {
            let xv = &x[ti * g * i + gi * i..ti * g * i + (gi + 1) * i];
            let dst = &mut out[ti * g * h + gi * h..ti * g * h + (gi + 1) * h];
            linear_row(xv, &w[gi * i * h..(gi + 1) * i * h], h, dst);
        }
    }
    out
}

/// z = sig(xw_z + hr_z + rb_z); r = sig(xw_r + hr_r + rb_r)
/// hh = tanh(xw_h + r * (hr_h + rb_h)); h = (1-z)*hh + z*h
fn gate_update(xwt: &[f32], hr: &[f32], rb: &[f32], h: &mut [f32], hs: usize) {
    #[cfg(target_arch = "x86_64")]
    if has_avx2() && crate::alloc::is_aligned32(hr.as_ptr()) && crate::alloc::is_aligned32(h.as_ptr()) {
        unsafe { x86::gate_update(xwt, hr, rb, h, hs) };
        return;
    }
    for k in 0..hs {
        let z = sigmoid(xwt[k] + hr[k] + rb[k]);
        let rr = sigmoid(xwt[hs + k] + hr[hs + k] + rb[hs + k]);
        let hh = (xwt[2 * hs + k] + rr * (hr[2 * hs + k] + rb[2 * hs + k])).tanh();
        h[k] = (1.0 - z).mul_add(hh, z * h[k]);
    }
}

/// ONNX GRU, single direction, `linear_before_reset = 1`.
/// x [T,I], w [3H,I], `r_packed` = [`pack_rows`](r [3H,H]), b [6H]  (gate order z, r, h)
///
/// The input projection x@W^T is a real GEMM (no recurrence), so it is hoisted out of the
/// timestep loop and tiled. Only h@R^T stays sequential -- it is the model's hard floor,
/// ~5 ms per GRU at T=500, sitting near the L2 bandwidth roofline.
pub fn gru(x: &[f32], t: usize, i: usize, w: &[f32], r_packed: &[f32], b: &[f32], h_size: usize) -> Vec<f32> {
    let h3 = 3 * h_size;

    // aligned scratch: x rows and the h vector are the second operand of every kernel
    let mut xa = AlignedVec::zeros(t * i);
    xa.copy_from_slice(x);
    let mut xw = AlignedVec::zeros(t * h3);
    gemm_nt(&xa, t, i, w, h3, &b[..h3], &mut xw);

    let mut h = AlignedVec::zeros(h_size);
    let mut hr = AlignedVec::zeros(h3);
    let mut y = vec![0f32; t * h_size];
    for ti in 0..t {
        matvec_packed(r_packed, h3, h_size, &h, &mut hr);
        gate_update(&xw[ti * h3..(ti + 1) * h3], &hr, &b[h3..], &mut h, h_size);
        y[ti * h_size..(ti + 1) * h_size].copy_from_slice(&h);
    }
    y
}

// ---------------------------------------------------------------- AVX2 kernels

#[cfg(target_arch = "x86_64")]
mod x86 {
    use std::arch::x86_64::*;

    #[target_feature(enable = "avx2,fma")]
    #[inline]
    unsafe fn hsum(s: __m256) -> f32 {
        let mut v = _mm_add_ps(_mm256_castps256_ps128(s), _mm256_extractf128_ps(s, 1));
        v = _mm_hadd_ps(v, v);
        v = _mm_hadd_ps(v, v);
        _mm_cvtss_f32(v)
    }

    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn axpy(out: &mut [f32], s: f32, x: &[f32]) {
        let n = out.len().min(x.len());
        let sv = _mm256_set1_ps(s);
        let (po, px) = (out.as_mut_ptr(), x.as_ptr());
        let mut i = 0;
        while i + 8 <= n {
            let o = _mm256_loadu_ps(po.add(i));
            _mm256_storeu_ps(po.add(i), _mm256_fmadd_ps(sv, _mm256_loadu_ps(px.add(i)), o));
            i += 8;
        }
        for j in i..n {
            out[j] = s.mul_add(x[j], out[j]);
        }
    }

    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn dot(a: &[f32], b: &[f32]) -> f32 {
        let n = a.len();
        let (pa, pb) = (a.as_ptr(), b.as_ptr());
        let mut a0 = _mm256_setzero_ps();
        let mut a1 = _mm256_setzero_ps();
        let mut i = 0;
        while i + 16 <= n {
            a0 = _mm256_fmadd_ps(_mm256_loadu_ps(pa.add(i)), _mm256_loadu_ps(pb.add(i)), a0);
            a1 = _mm256_fmadd_ps(_mm256_loadu_ps(pa.add(i + 8)), _mm256_loadu_ps(pb.add(i + 8)), a1);
            i += 16;
        }
        while i + 8 <= n {
            a0 = _mm256_fmadd_ps(_mm256_loadu_ps(pa.add(i)), _mm256_loadu_ps(pb.add(i)), a0);
            i += 8;
        }
        let mut r = hsum(_mm256_add_ps(a0, a1));
        while i < n {
            r += a[i] * b[i];
            i += 1;
        }
        r
    }

    #[target_feature(enable = "avx2,fma")]
    #[inline]
    unsafe fn ld<const A: bool>(p: *const f32) -> __m256 {
        if A {
            _mm256_load_ps(p)
        } else {
            _mm256_loadu_ps(p)
        }
    }

    /// 8 rows per pass: the `x` block is loaded once and reused 8 times (9 loads per 8
    /// FMAs instead of 2 per 1), and 8 accumulator chains cover the ~4-cycle FMA latency.
    /// `A` selects aligned loads; a straddling 32-byte load costs an extra cycle.
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn matvec<const A: bool>(
        m: &[f32], rows: usize, k: usize, x: &[f32], bias: Option<&[f32]>, out: &mut [f32],
    ) {
        let (px, pm) = (x.as_ptr(), m.as_ptr());
        let mut j = 0;
        while j + 8 <= rows {
            let mut acc = [_mm256_setzero_ps(); 8];
            let base = pm.add(j * k);
            let mut i = 0;
            while i + 8 <= k {
                let xv = ld::<A>(px.add(i));
                for (l, a) in acc.iter_mut().enumerate() {
                    *a = _mm256_fmadd_ps(ld::<A>(base.add(l * k + i)), xv, *a);
                }
                i += 8;
            }
            for l in 0..8 {
                let mut r = hsum(acc[l]);
                for ii in i..k {
                    r += m[(j + l) * k + ii] * x[ii];
                }
                out[j + l] = r + bias.map_or(0.0, |b| b[j + l]);
            }
            j += 8;
        }
        while j < rows {
            out[j] = dot(&m[j * k..(j + 1) * k], x) + bias.map_or(0.0, |b| b[j]);
            j += 1;
        }
    }

    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn matvec_packed(p: &[f32], rows: usize, k: usize, x: &[f32], out: &mut [f32]) {
        const PANEL: usize = super::PANEL;
        let px = x.as_ptr();
        for jb in 0..rows / PANEL {
            let mut acc = [_mm256_setzero_ps(); PANEL];
            let panel = p.as_ptr().add(jb * PANEL * k);
            for c in 0..k / 8 {
                let xv = _mm256_load_ps(px.add(c * 8));
                let base = panel.add(c * 8 * PANEL);
                for (l, a) in acc.iter_mut().enumerate() {
                    *a = _mm256_fmadd_ps(_mm256_load_ps(base.add(l * 8)), xv, *a);
                }
            }
            for l in 0..PANEL {
                out[jb * PANEL + l] = hsum(acc[l]);
            }
        }
    }

    /// 4 weight rows x 2 timesteps, 8 accumulators: each weight vector serves 2 timesteps
    /// and each x vector 4 rows -> 6 loads per 8 FMAs.
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn gemm_nt(
        x: &[f32], t: usize, k: usize, w: &[f32], rows: usize, b: &[f32], out: &mut [f32],
    ) {
        let (pw, px) = (w.as_ptr(), x.as_ptr());
        let mut ti = 0;
        while ti + 2 <= t {
            let (x0, x1) = (px.add(ti * k), px.add((ti + 1) * k));
            let mut j = 0;
            while j + 4 <= rows {
                let mut a = [_mm256_setzero_ps(); 8]; // [row][t]
                let base = pw.add(j * k);
                let mut i = 0;
                while i + 8 <= k {
                    let v0 = _mm256_load_ps(x0.add(i));
                    let v1 = _mm256_load_ps(x1.add(i));
                    for l in 0..4 {
                        let m = _mm256_load_ps(base.add(l * k + i));
                        a[l * 2] = _mm256_fmadd_ps(m, v0, a[l * 2]);
                        a[l * 2 + 1] = _mm256_fmadd_ps(m, v1, a[l * 2 + 1]);
                    }
                    i += 8;
                }
                for l in 0..4 {
                    for (u, tt) in [ti, ti + 1].iter().enumerate() {
                        let mut r = hsum(a[l * 2 + u]);
                        for ii in i..k {
                            r += w[(j + l) * k + ii] * x[tt * k + ii];
                        }
                        out[tt * rows + j + l] = r + b[j + l];
                    }
                }
                j += 4;
            }
            while j < rows {
                for tt in [ti, ti + 1] {
                    out[tt * rows + j] = dot(&w[j * k..(j + 1) * k], &x[tt * k..(tt + 1) * k]) + b[j];
                }
                j += 1;
            }
            ti += 2;
        }
        while ti < t {
            matvec::<true>(w, rows, k, &x[ti * k..(ti + 1) * k], Some(b), &mut out[ti * rows..(ti + 1) * rows]);
            ti += 1;
        }
    }

    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn linear_row(xv: &[f32], w: &[f32], h: usize, dst: &mut [f32]) {
        let pw = w.as_ptr();
        let mut hb = 0;
        while hb + 64 <= h {
            let mut acc = [_mm256_setzero_ps(); 8];
            for (ii, &xs) in xv.iter().enumerate() {
                if xs == 0.0 {
                    continue;
                }
                let bc = _mm256_set1_ps(xs);
                let base = pw.add(ii * h + hb);
                for (l, a) in acc.iter_mut().enumerate() {
                    *a = _mm256_fmadd_ps(_mm256_loadu_ps(base.add(l * 8)), bc, *a);
                }
            }
            for l in 0..8 {
                _mm256_storeu_ps(dst.as_mut_ptr().add(hb + l * 8), acc[l]);
            }
            hb += 64;
        }
        while hb + 8 <= h {
            let mut a = _mm256_setzero_ps();
            for (ii, &xs) in xv.iter().enumerate() {
                if xs != 0.0 {
                    a = _mm256_fmadd_ps(_mm256_loadu_ps(pw.add(ii * h + hb)), _mm256_set1_ps(xs), a);
                }
            }
            _mm256_storeu_ps(dst.as_mut_ptr().add(hb), a);
            hb += 8;
        }
        for hh in hb..h {
            let mut s = 0.0;
            for (ii, &xs) in xv.iter().enumerate() {
                s = xs.mul_add(w[ii * h + hh], s);
            }
            dst[hh] = s;
        }
    }

    /// `OB` must be a const so the accumulator array stays in ymm registers. With a runtime
    /// bound LLVM spills it to the stack and the kernel runs at *half* the speed of the
    /// naive axpy version (measured 0.52x).
    #[target_feature(enable = "avx2,fma")]
    unsafe fn pw_block<const OB: usize>(
        x: &[f32], cin: usize, tf: usize, w: &[f32], b: Option<&[f32]>, o0: usize,
        out: &mut [f32], p0: usize, pn: usize,
    ) {
        let (px, pw, po) = (x.as_ptr(), w.as_ptr(), out.as_mut_ptr());
        let mut p = p0;
        while p + 8 <= pn {
            let mut acc = [_mm256_setzero_ps(); OB];
            if let Some(b) = b {
                for l in 0..OB {
                    acc[l] = _mm256_set1_ps(b[o0 + l]);
                }
            }
            for c in 0..cin {
                let xv = _mm256_loadu_ps(px.add(c * tf + p));
                for l in 0..OB {
                    acc[l] = _mm256_fmadd_ps(_mm256_set1_ps(*pw.add((o0 + l) * cin + c)), xv, acc[l]);
                }
            }
            for l in 0..OB {
                _mm256_storeu_ps(po.add((o0 + l) * tf + p), acc[l]);
            }
            p += 8;
        }
        for pp in p..pn {
            for l in 0..OB {
                let mut s = b.map_or(0.0, |b| b[o0 + l]);
                for c in 0..cin {
                    s = w[(o0 + l) * cin + c].mul_add(x[c * tf + pp], s);
                }
                out[(o0 + l) * tf + pp] = s;
            }
        }
    }

    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn pointwise(
        x: &[f32], cin: usize, tf: usize, w: &[f32], b: Option<&[f32]>, cout: usize, out: &mut [f32],
    ) {
        // Tile the spatial axis: the cin input planes are `tf` floats apart, so sweeping c
        // for every p touches `cin` streams far apart and the prefetcher falls over. With
        // PT=4096 a tile of all cin planes is ~256 KB and stays in L2 across the o-blocks.
        const PT: usize = 4096;
        let mut p0 = 0;
        while p0 < tf {
            let pn = (p0 + PT).min(tf);
            let mut o0 = 0;
            while o0 + 8 <= cout {
                pw_block::<8>(x, cin, tf, w, b, o0, out, p0, pn);
                o0 += 8;
            }
            while o0 + 2 <= cout {
                pw_block::<2>(x, cin, tf, w, b, o0, out, p0, pn);
                o0 += 2;
            }
            while o0 < cout {
                pw_block::<1>(x, cin, tf, w, b, o0, out, p0, pn);
                o0 += 1;
            }
            p0 = pn;
        }
    }

    /// 8 lanes at a time with vectorised sigmoid/tanh -- this is where the model's ~1.9M
    /// transcendental calls live.
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn gate_update(xwt: &[f32], hr: &[f32], rb: &[f32], h: &mut [f32], hs: usize) {
        use crate::simd::x86::{sigmoid8, tanh8};
        let mut k = 0;
        while k + 8 <= hs {
            let z = sigmoid8(_mm256_add_ps(
                _mm256_add_ps(_mm256_loadu_ps(xwt.as_ptr().add(k)), _mm256_load_ps(hr.as_ptr().add(k))),
                _mm256_loadu_ps(rb.as_ptr().add(k)),
            ));
            let rr = sigmoid8(_mm256_add_ps(
                _mm256_add_ps(
                    _mm256_loadu_ps(xwt.as_ptr().add(hs + k)),
                    _mm256_load_ps(hr.as_ptr().add(hs + k)),
                ),
                _mm256_loadu_ps(rb.as_ptr().add(hs + k)),
            ));
            let inner = _mm256_add_ps(
                _mm256_load_ps(hr.as_ptr().add(2 * hs + k)),
                _mm256_loadu_ps(rb.as_ptr().add(2 * hs + k)),
            );
            let hh = tanh8(_mm256_fmadd_ps(rr, inner, _mm256_loadu_ps(xwt.as_ptr().add(2 * hs + k))));
            let hold = _mm256_load_ps(h.as_ptr().add(k));
            let nh = _mm256_fmadd_ps(_mm256_sub_ps(_mm256_set1_ps(1.0), z), hh, _mm256_mul_ps(z, hold));
            _mm256_store_ps(h.as_mut_ptr().add(k), nh);
            k += 8;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rnd(n: usize, seed: u64) -> Vec<f32> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
                ((s >> 33) as f32 / (1u64 << 31) as f32) - 0.5
            })
            .collect()
    }

    #[test]
    fn dot_matches_scalar() {
        let a = rnd(256, 1);
        let b = rnd(256, 2);
        assert!((dot(&a, &b) - dot_scalar(&a, &b)).abs() < 1e-4);
    }

    #[test]
    fn packed_matvec_matches_matvec() {
        let (rows, k) = (PANEL * 4, 32);
        let m = AlignedVec::from_slice(&rnd(rows * k, 3));
        let x = AlignedVec::from_slice(&rnd(k, 4));
        let p = pack_rows(&m, rows, k);
        let (mut a, mut b) = (vec![0f32; rows], vec![0f32; rows]);
        matvec(&m, rows, k, &x, None, &mut a);
        matvec_packed(&p, rows, k, &x, &mut b);
        for (u, v) in a.iter().zip(&b) {
            assert!((u - v).abs() < 1e-5, "{u} vs {v}");
        }
    }

    #[test]
    fn gemm_nt_matches_matvec() {
        let (t, k, rows) = (7, 32, 16);
        let x = AlignedVec::from_slice(&rnd(t * k, 5));
        let w = AlignedVec::from_slice(&rnd(rows * k, 6));
        let bias = rnd(rows, 7);
        let mut g = vec![0f32; t * rows];
        gemm_nt(&x, t, k, &w, rows, &bias, &mut g);
        for ti in 0..t {
            let mut r = vec![0f32; rows];
            matvec(&w, rows, k, &x[ti * k..(ti + 1) * k], Some(&bias), &mut r);
            for j in 0..rows {
                assert!((g[ti * rows + j] - r[j]).abs() < 1e-4);
            }
        }
    }

    #[test]
    fn convtranspose_doubles_frequency() {
        let (c, t, f) = (2, 3, 4);
        let x = rnd(c * t * f, 8);
        let w = rnd(c * 3, 9);
        let (y, fo) = convtranspose_dw(&x, c, t, f, &w);
        assert_eq!(fo, 2 * f);
        assert_eq!(y.len(), c * t * 2 * f);
    }
}
