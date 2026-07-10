//! Micro-benchmarks to find where the GRU time actually goes.
use std::hint::black_box;
use std::time::Instant;

#[inline(always)]
fn dot_1acc(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = [0f32; 8];
    let mut ca = a.chunks_exact(8);
    let mut cb = b.chunks_exact(8);
    for (x, y) in ca.by_ref().zip(cb.by_ref()) {
        for k in 0..8 {
            acc[k] = x[k].mul_add(y[k], acc[k]);
        }
    }
    (acc[0] + acc[1]) + (acc[2] + acc[3]) + ((acc[4] + acc[5]) + (acc[6] + acc[7]))
}

#[inline(always)]
fn dot_4acc(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = [[0f32; 8]; 4];
    let mut ca = a.chunks_exact(32);
    let mut cb = b.chunks_exact(32);
    for (x, y) in ca.by_ref().zip(cb.by_ref()) {
        for l in 0..4 {
            for k in 0..8 {
                acc[l][k] = x[l * 8 + k].mul_add(y[l * 8 + k], acc[l][k]);
            }
        }
    }
    let mut v = [0f32; 8];
    for k in 0..8 {
        v[k] = (acc[0][k] + acc[1][k]) + (acc[2][k] + acc[3][k]);
    }
    (v[0] + v[1]) + (v[2] + v[3]) + ((v[4] + v[5]) + (v[6] + v[7]))
}

#[inline]
fn dot_avx2(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    if hush_vani::simd::has_avx2() {
        return unsafe { dot_avx2_impl(a, b) };
    }
    dot_1acc(a, b)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_avx2_impl(a: &[f32], b: &[f32]) -> f32 {
    {
        use std::arch::x86_64::*;
        let n = a.len();
        let (pa, pb) = (a.as_ptr(), b.as_ptr());
        let mut a0 = _mm256_setzero_ps();
        let mut a1 = _mm256_setzero_ps();
        let mut a2 = _mm256_setzero_ps();
        let mut a3 = _mm256_setzero_ps();
        let mut i = 0;
        while i + 32 <= n {
            a0 = _mm256_fmadd_ps(_mm256_loadu_ps(pa.add(i)), _mm256_loadu_ps(pb.add(i)), a0);
            a1 = _mm256_fmadd_ps(_mm256_loadu_ps(pa.add(i + 8)), _mm256_loadu_ps(pb.add(i + 8)), a1);
            a2 = _mm256_fmadd_ps(_mm256_loadu_ps(pa.add(i + 16)), _mm256_loadu_ps(pb.add(i + 16)), a2);
            a3 = _mm256_fmadd_ps(_mm256_loadu_ps(pa.add(i + 24)), _mm256_loadu_ps(pb.add(i + 24)), a3);
            i += 32;
        }
        while i + 8 <= n {
            a0 = _mm256_fmadd_ps(_mm256_loadu_ps(pa.add(i)), _mm256_loadu_ps(pb.add(i)), a0);
            i += 8;
        }
        let s = _mm256_add_ps(_mm256_add_ps(a0, a1), _mm256_add_ps(a2, a3));
        let lo = _mm256_castps256_ps128(s);
        let hi = _mm256_extractf128_ps(s, 1);
        let mut v = _mm_add_ps(lo, hi);
        v = _mm_hadd_ps(v, v);
        v = _mm_hadd_ps(v, v);
        let mut r = _mm_cvtss_f32(v);
        while i < n {
            r += a[i] * b[i];
            i += 1;
        }
        r
    }
}

/// matvec: out[j] = dot(m[j], x), m is [rows, k]
fn matvec<F: Fn(&[f32], &[f32]) -> f32>(m: &[f32], rows: usize, k: usize, x: &[f32], out: &mut [f32], f: F) {
    for j in 0..rows {
        out[j] = f(&m[j * k..(j + 1) * k], x);
    }
}

/// Allocate a f32 buffer whose base address is 64-byte aligned.
fn aligned(n: usize, f: impl Fn(usize) -> f32) -> (*mut f32, usize) {
    unsafe {
        let layout = std::alloc::Layout::from_size_align(n * 4, 64).unwrap();
        let p = std::alloc::alloc(layout) as *mut f32;
        for i in 0..n {
            *p.add(i) = f(i);
        }
        (p, n)
    }
}

/// 8-row matvec, matching nn::matvec, parameterised on aligned vs unaligned loads.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn matvec8(m: *const f32, rows: usize, k: usize, x: *const f32, out: &mut [f32], aligned_ld: bool) {
    use std::arch::x86_64::*;
    let mut j = 0;
    while j + 8 <= rows {
        let mut acc = [_mm256_setzero_ps(); 8];
        let base = m.add(j * k);
        let mut i = 0;
        while i + 8 <= k {
            let xv = if aligned_ld { _mm256_load_ps(x.add(i)) } else { _mm256_loadu_ps(x.add(i)) };
            for (l, a) in acc.iter_mut().enumerate() {
                let p = base.add(l * k + i);
                let mv = if aligned_ld { _mm256_load_ps(p) } else { _mm256_loadu_ps(p) };
                *a = _mm256_fmadd_ps(mv, xv, *a);
            }
            i += 8;
        }
        for l in 0..8 {
            let s = acc[l];
            let mut v = _mm_add_ps(_mm256_castps256_ps128(s), _mm256_extractf128_ps(s, 1));
            v = _mm_hadd_ps(v, v);
            v = _mm_hadd_ps(v, v);
            out[j + l] = _mm_cvtss_f32(v);
        }
        j += 8;
    }
}

fn bench_align() {
    let (rows, k, t) = (768usize, 256usize, 500usize);
    let (ma, _) = aligned(rows * k, |i| (i % 17) as f32 * 0.01);
    let (xa, _) = aligned(k, |i| (i % 7) as f32 * 0.1);
    let mut out = vec![0f32; rows];
    let fmas = (rows * k * t) as f64;

    println!("\n8-row matvec, load kind (same data, 64B-aligned base):");
    for (name, al) in [("loadu (unaligned)", false), ("load  (aligned)", true)] {
        unsafe {
            for _ in 0..2 {
                matvec8(ma, rows, k, xa, &mut out, al);
            }
            let t0 = Instant::now();
            for _ in 0..t {
                matvec8(black_box(ma), rows, k, black_box(xa), &mut out, al);
            }
            let el = t0.elapsed().as_secs_f64();
            black_box(&out);
            println!("  {name:20} {:6.2} ms => {:5.2} GFMA/s", el * 1e3, fmas / el / 1e9);
        }
    }
    // deliberately misalign the base by 4 bytes (what Vec<f32> can give you)
    unsafe {
        let mm = ma.add(1);
        let xx = xa.add(1);
        for _ in 0..2 {
            matvec8(mm, rows, k, xx, &mut out, false);
        }
        let t0 = Instant::now();
        for _ in 0..t {
            matvec8(black_box(mm), rows, k, black_box(xx), &mut out, false);
        }
        let el = t0.elapsed().as_secs_f64();
        black_box(&out);
        println!("  {:20} {:6.2} ms => {:5.2} GFMA/s", "loadu (base+4B)", el * 1e3, fmas / el / 1e9);
    }
}

fn main() {
    bench_align();
    let (rows, k, t) = (768usize, 256usize, 500usize);
    let m: Vec<f32> = (0..rows * k).map(|i| (i % 17) as f32 * 0.01).collect();
    let x: Vec<f32> = (0..k).map(|i| (i % 7) as f32 * 0.1).collect();
    let mut out = vec![0f32; rows];

    let fmas = (rows * k * t) as f64;

    for (name, f) in [
        ("dot 1x8 acc", dot_1acc as fn(&[f32], &[f32]) -> f32),
        ("dot 4x8 acc", dot_4acc as fn(&[f32], &[f32]) -> f32),
        ("dot avx2", dot_avx2 as fn(&[f32], &[f32]) -> f32),
    ] {
        for _ in 0..2 {
            matvec(&m, rows, k, &x, &mut out, f);
        }
        let t0 = Instant::now();
        for _ in 0..t {
            matvec(black_box(&m), rows, k, black_box(&x), &mut out, f);
        }
        let el = t0.elapsed().as_secs_f64();
        black_box(&out);
        println!("{name:14} {:7.2} ms for {t} matvecs  => {:5.2} GFMA/s", el * 1e3, fmas / el / 1e9);
    }

    // Is dot memory-bound or issue-bound? Shrink the matrix until it lives in L1.
    // If GFMA/s jumps, we are bandwidth-bound on weights, not scalar.
    println!("\nmatvec vs weight-matrix footprint (dot 1x8 acc):");
    for &r in &[8usize, 32, 128, 768, 3072] {
        let mm: Vec<f32> = (0..r * k).map(|i| (i % 17) as f32 * 0.01).collect();
        let mut o = vec![0f32; r];
        let reps = (500 * 768 / r).max(1);
        for _ in 0..2 {
            matvec(&mm, r, k, &x, &mut o, dot_1acc);
        }
        let t0 = Instant::now();
        for _ in 0..reps {
            matvec(black_box(&mm), r, k, black_box(&x), &mut o, dot_1acc);
        }
        let el = t0.elapsed().as_secs_f64();
        black_box(&o);
        let f = (r * k * reps) as f64;
        println!("  rows={r:5} ({:6.0} KB)  {:5.2} GFMA/s", (r * k * 4) as f64 / 1024.0, f / el / 1e9);
    }

    // transcendentals: one GRU step does 2H sigmoid + H tanh; 500 steps x 5 GRUs
    let n = 500 * 5 * 512; // sigmoids
    let v: Vec<f32> = (0..1024).map(|i| (i as f32 - 512.0) * 0.01).collect();
    let t0 = Instant::now();
    let mut s = 0f32;
    for i in 0..n {
        s += 1.0 / (1.0 + (-black_box(v[i % 1024])).exp());
    }
    let el = t0.elapsed().as_secs_f64();
    black_box(s);
    println!("\n{n} sigmoid (expf)  {:7.2} ms  ({:.2} ns each)", el * 1e3, el * 1e9 / n as f64);

    let n2 = 500 * 5 * 256;
    let t0 = Instant::now();
    let mut s = 0f32;
    for i in 0..n2 {
        s += black_box(v[i % 1024]).tanh();
    }
    let el = t0.elapsed().as_secs_f64();
    black_box(s);
    println!("{n2} tanh          {:7.2} ms  ({:.2} ns each)", el * 1e3, el * 1e9 / n2 as f64);
}

