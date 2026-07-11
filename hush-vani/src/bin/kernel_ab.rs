//! In-process, interleaved A/B of the old (axpy) vs new (register-blocked) kernels.
//!
//! The machine throttles hard under sustained benchmarking -- untouched code measured 1.9x
//! slower across runs. Absolute ms are meaningless here; only the paired ratio is.

use hush_vani_core::nn::{axpy, grouped_linear, pointwise};
use std::hint::black_box;
use std::time::Instant;

const T: usize = 500;

fn rnd(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
            let v = ((s >> 33) as f32 / (1u64 << 31) as f32) - 0.5;
            if v < 0.0 { 0.0 } else { v } // post-ReLU: ~half zeros, like the real inputs
        })
        .collect()
}

// ---- reference (previous) implementations ----
fn grouped_linear_axpy(x: &[f32], t: usize, g: usize, i: usize, h: usize, w: &[f32]) -> Vec<f32> {
    let mut out = vec![0f32; t * g * h];
    for ti in 0..t {
        for gi in 0..g {
            let xv = &x[ti * g * i + gi * i..ti * g * i + (gi + 1) * i];
            let dst = &mut out[ti * g * h + gi * h..ti * g * h + (gi + 1) * h];
            for (ii, &xs) in xv.iter().enumerate() {
                if xs != 0.0 {
                    axpy(dst, xs, &w[gi * i * h + ii * h..gi * i * h + (ii + 1) * h]);
                }
            }
        }
    }
    out
}

fn pointwise_axpy(x: &[f32], cin: usize, tf: usize, w: &[f32], b: Option<&[f32]>, cout: usize) -> Vec<f32> {
    let mut out = vec![0f32; cout * tf];
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

fn med(v: &mut Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

/// interleaved: alternate old/new, block of 3, 8 turns each
fn ab<R>(name: &str, fmas: f64, mut old: impl FnMut() -> R, mut new: impl FnMut() -> R) {
    for _ in 0..2 {
        black_box(old());
        black_box(new());
    }
    let (mut to, mut tn) = (Vec::new(), Vec::new());
    for turn in 0..8 {
        for _ in 0..3 {
            if turn % 2 == 0 {
                let t = Instant::now();
                black_box(old());
                to.push(t.elapsed().as_secs_f64() * 1e3);
                let t = Instant::now();
                black_box(new());
                tn.push(t.elapsed().as_secs_f64() * 1e3);
            } else {
                let t = Instant::now();
                black_box(new());
                tn.push(t.elapsed().as_secs_f64() * 1e3);
                let t = Instant::now();
                black_box(old());
                to.push(t.elapsed().as_secs_f64() * 1e3);
            }
        }
    }
    let (mo, mn) = (med(&mut to), med(&mut tn));
    println!(
        "  {name:26} old {mo:6.2} ms [{:5.1} GFMA/s]   new {mn:6.2} ms [{:5.1} GFMA/s]   {:.2}x",
        fmas / mo / 1e6,
        fmas / mn / 1e6,
        mo / mn
    );
}

/// 12 output rows per pass instead of 8: 13 memory ops per 12 vector-FMAs rather than
/// 9 per 8. 12 accumulators + 1 xv still fits the 16 ymm registers.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn matvec12(m: &[f32], rows: usize, k: usize, x: &[f32], out: &mut [f32]) {
    use std::arch::x86_64::*;
    let px = x.as_ptr();
    let pm = m.as_ptr();
    let mut j = 0;
    while j + 12 <= rows {
        let mut acc = [_mm256_setzero_ps(); 12];
        let base = pm.add(j * k);
        let mut i = 0;
        while i + 8 <= k {
            let xv = _mm256_load_ps(px.add(i));
            for (l, a) in acc.iter_mut().enumerate() {
                *a = _mm256_fmadd_ps(_mm256_load_ps(base.add(l * k + i)), xv, *a);
            }
            i += 8;
        }
        for l in 0..12 {
            let s = acc[l];
            let mut v = _mm_add_ps(_mm256_castps256_ps128(s), _mm256_extractf128_ps(s, 1));
            v = _mm_hadd_ps(v, v);
            v = _mm_hadd_ps(v, v);
            out[j + l] = _mm_cvtss_f32(v);
        }
        j += 12;
    }
    for jj in j..rows {
        out[jj] = hush_vani_core::nn::dot(&m[jj * k..(jj + 1) * k], x);
    }
}

/// Pack [rows,k] into 8-row panels: each 8-float chunk of k has the 8 rows contiguous,
/// so the kernel streams one sequential run instead of 8 pointers 1 KB apart.
fn pack8(m: &[f32], rows: usize, k: usize) -> hush_vani_core::alloc::AlignedVec {
    let mut p = hush_vani_core::alloc::AlignedVec::zeros(rows * k);
    for jb in 0..rows / 8 {
        for c in 0..k / 8 {
            for l in 0..8 {
                let s = (jb * 8 + l) * k + c * 8;
                let d = jb * 8 * k + c * 64 + l * 8;
                p[d..d + 8].copy_from_slice(&m[s..s + 8]);
            }
        }
    }
    p
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn matvec_packed(p: &[f32], rows: usize, k: usize, x: &[f32], out: &mut [f32]) {
    use std::arch::x86_64::*;
    let px = x.as_ptr();
    for jb in 0..rows / 8 {
        let mut acc = [_mm256_setzero_ps(); 8];
        let panel = p.as_ptr().add(jb * 8 * k);
        for c in 0..k / 8 {
            let xv = _mm256_load_ps(px.add(c * 8));
            let base = panel.add(c * 64);
            for (l, a) in acc.iter_mut().enumerate() {
                *a = _mm256_fmadd_ps(_mm256_load_ps(base.add(l * 8)), xv, *a);
            }
        }
        for l in 0..8 {
            let s = acc[l];
            let mut v = _mm_add_ps(_mm256_castps256_ps128(s), _mm256_extractf128_ps(s, 1));
            v = _mm_hadd_ps(v, v);
            v = _mm_hadd_ps(v, v);
            out[jb * 8 + l] = _mm_cvtss_f32(v);
        }
    }
}

fn pack_n(m: &[f32], rows: usize, k: usize, nb: usize) -> hush_vani_core::alloc::AlignedVec {
    let mut p = hush_vani_core::alloc::AlignedVec::zeros(rows * k);
    for jb in 0..rows / nb {
        for c in 0..k / 8 {
            for l in 0..nb {
                let s = (jb * nb + l) * k + c * 8;
                let d = jb * nb * k + c * 8 * nb + l * 8;
                p[d..d + 8].copy_from_slice(&m[s..s + 8]);
            }
        }
    }
    p
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn matvec_packed12(p: &[f32], rows: usize, k: usize, x: &[f32], out: &mut [f32]) {
    use std::arch::x86_64::*;
    let px = x.as_ptr();
    for jb in 0..rows / 12 {
        let mut acc = [_mm256_setzero_ps(); 12];
        let panel = p.as_ptr().add(jb * 12 * k);
        for c in 0..k / 8 {
            let xv = _mm256_load_ps(px.add(c * 8));
            let base = panel.add(c * 96);
            for (l, a) in acc.iter_mut().enumerate() {
                *a = _mm256_fmadd_ps(_mm256_load_ps(base.add(l * 8)), xv, *a);
            }
        }
        for l in 0..12 {
            let s = acc[l];
            let mut v = _mm_add_ps(_mm256_castps256_ps128(s), _mm256_extractf128_ps(s, 1));
            v = _mm_hadd_ps(v, v);
            v = _mm_hadd_ps(v, v);
            out[jb * 12 + l] = _mm_cvtss_f32(v);
        }
    }
}

fn matvec_ab() {
    use hush_vani_core::alloc::AlignedVec;
    use hush_vani_core::nn::matvec;
    if !hush_vani_core::simd::has_avx2() {
        println!("\n(no AVX2 on this CPU; skipping matvec variants)");
        return;
    }
    let (rows, k) = (768usize, 256usize);
    let r = AlignedVec::from_slice(&rnd(rows * k, 21));
    let rp = pack8(&r, rows, k);
    let rp12 = pack_n(&r, rows, k, 12);
    let h = AlignedVec::from_slice(&rnd(k, 22));
    let mut o1 = vec![0f32; rows];
    let mut o2 = vec![0f32; rows];
    let f = (rows * k * T) as f64;

    println!("\nrecurrent matvec (768x256, x{T} = one GRU's h@R^T):");
    ab("8-row  vs 12-row", f,
       || for _ in 0..T { matvec(&r, rows, k, &h, None, &mut o1) },
       || for _ in 0..T { unsafe { matvec12(&r, rows, k, &h, &mut o2) } });
    ab("8-row  vs packed8", f,
       || for _ in 0..T { matvec(&r, rows, k, &h, None, &mut o1) },
       || for _ in 0..T { unsafe { matvec_packed(&rp, rows, k, &h, &mut o2) } });
    ab("8-row  vs packed12", f,
       || for _ in 0..T { matvec(&r, rows, k, &h, None, &mut o1) },
       || for _ in 0..T { unsafe { matvec_packed12(&rp12, rows, k, &h, &mut o2) } });
    ab("packed8 vs packed12", f,
       || for _ in 0..T { unsafe { matvec_packed(&rp, rows, k, &h, &mut o1) } },
       || for _ in 0..T { unsafe { matvec_packed12(&rp12, rows, k, &h, &mut o2) } });

    // sanity: all variants agree
    unsafe { matvec_packed12(&rp12, rows, k, &h, &mut o2) };
    matvec(&r, rows, k, &h, None, &mut o1);
    let d = o1.iter().zip(&o2).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
    println!("  max|diff| between 8-row and packed12: {d:.2e}");
}

fn main() {
    println!("interleaved old-vs-new, medians (ratio >1 = new is faster)\n");

    println!("grouped_linear:");
    for (g, i, h, name) in [
        (1usize, 256usize, 640usize, "df_out     [1,256,640]"),
        (1, 128, 256, "linear_in  [1,128,256]"),
        (1, 256, 128, "linear_out [1,256,128]"),
        (16, 32, 8, "df_fc_emb  [16,32,8]"),
    ] {
        let x = rnd(T * g * i, 7);
        let w = rnd(g * i * h, 8);
        let f = (T * g * i * h) as f64;
        ab(name, f, || grouped_linear_axpy(&x, T, g, i, h, &w), || grouped_linear(&x, T, g, i, h, &w));
    }

    println!("\npointwise:");
    for (cin, cout, tf, name) in [
        (16usize, 16usize, T * 64, "c0        16->16 @T*64"),
        (16, 16, T * 32, "erb_dec   16->16 @T*32"),
        (16, 10, T * 64, "df_convp  16->10 @T*64"),
    ] {
        let x = rnd(cin * tf, 9);
        let w = rnd(cout * cin, 10);
        let b = rnd(cout, 11);
        let f = (cin * cout * tf) as f64;
        ab(name, f, || pointwise_axpy(&x, cin, tf, &w, Some(&b), cout),
                    || pointwise(&x, cin, tf, &w, Some(&b), cout));
    }

    matvec_ab();
}

