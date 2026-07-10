//! Vectorised exp / sigmoid / tanh, plus the runtime AVX2 capability check.
//!
//! The GRU gates call expf ~1.28M times and tanhf ~640K times per 5 s clip. Scalar libm
//! costs ~7 ns and ~18 ns per call -- roughly a quarter of total NN time. These are 8-wide
//! Cephes-style approximations, accurate to ~1e-7 relative.
//!
//! Feature detection is at *runtime*: a published crate cannot assume the user builds with
//! `-C target-cpu=native`, and silently falling back to scalar would cost ~4x.

use std::sync::OnceLock;

/// True if AVX2 + FMA are available on this CPU. Cached after the first call.
#[inline]
pub fn has_avx2() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        static OK: OnceLock<bool> = OnceLock::new();
        *OK.get_or_init(|| is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma"))
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

#[inline]
pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

pub fn sigmoid_slice(v: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    if has_avx2() {
        unsafe { x86::sigmoid_slice(v) };
        return;
    }
    for x in v.iter_mut() {
        *x = sigmoid(*x);
    }
}

pub fn tanh_slice(v: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    if has_avx2() {
        unsafe { x86::tanh_slice(v) };
        return;
    }
    for x in v.iter_mut() {
        *x = x.tanh();
    }
}

#[cfg(target_arch = "x86_64")]
pub mod x86 {
    use std::arch::x86_64::*;

    const EXP_HI: f32 = 88.376_26;
    const EXP_LO: f32 = -88.376_26;
    const LOG2EF: f32 = 1.442_695;
    const C1: f32 = 0.693_359_4;
    const C2: f32 = -2.121_944_4e-4;

    /// exp(x), Cephes polynomial. |rel err| < ~1e-7 over the finite range.
    ///
    /// # Safety
    /// Caller must ensure AVX2 + FMA are available (see [`super::has_avx2`]).
    #[target_feature(enable = "avx2,fma")]
    #[inline]
    pub unsafe fn exp8(x: __m256) -> __m256 {
        let x = _mm256_min_ps(_mm256_set1_ps(EXP_HI), _mm256_max_ps(_mm256_set1_ps(EXP_LO), x));

        // n = round(x / ln2), reduce x into [-ln2/2, ln2/2]
        let fx = _mm256_floor_ps(_mm256_fmadd_ps(x, _mm256_set1_ps(LOG2EF), _mm256_set1_ps(0.5)));
        let x = _mm256_fnmadd_ps(fx, _mm256_set1_ps(C1), x);
        let x = _mm256_fnmadd_ps(fx, _mm256_set1_ps(C2), x);

        let z = _mm256_mul_ps(x, x);
        let mut y = _mm256_set1_ps(1.987_569_1e-4);
        y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(1.398_199_9e-3));
        y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(8.333_452e-3));
        y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(4.166_579_6e-2));
        y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(1.666_666_5e-1));
        y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(5.000_000_6e-1));
        y = _mm256_fmadd_ps(y, z, x);
        y = _mm256_add_ps(y, _mm256_set1_ps(1.0));

        // scale by 2^n
        let imm0 = _mm256_cvttps_epi32(fx);
        let imm0 = _mm256_add_epi32(imm0, _mm256_set1_epi32(0x7f));
        let pow2n = _mm256_castsi256_ps(_mm256_slli_epi32(imm0, 23));
        _mm256_mul_ps(y, pow2n)
    }

    /// # Safety
    /// Caller must ensure AVX2 + FMA are available.
    #[target_feature(enable = "avx2,fma")]
    #[inline]
    pub unsafe fn sigmoid8(x: __m256) -> __m256 {
        let e = exp8(_mm256_sub_ps(_mm256_setzero_ps(), x));
        _mm256_div_ps(_mm256_set1_ps(1.0), _mm256_add_ps(_mm256_set1_ps(1.0), e))
    }

    /// tanh(x) = 1 - 2/(e^{2x} + 1).
    /// Near zero that form cancels, so fall back to the (exact in f32) identity tanh(x) ~ x.
    ///
    /// # Safety
    /// Caller must ensure AVX2 + FMA are available.
    #[target_feature(enable = "avx2,fma")]
    #[inline]
    pub unsafe fn tanh8(x: __m256) -> __m256 {
        let e = exp8(_mm256_add_ps(x, x));
        let t = _mm256_sub_ps(
            _mm256_set1_ps(1.0),
            _mm256_div_ps(_mm256_set1_ps(2.0), _mm256_add_ps(e, _mm256_set1_ps(1.0))),
        );
        let absmask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fff_ffff));
        let small = _mm256_cmp_ps(_mm256_and_ps(x, absmask), _mm256_set1_ps(2.44e-4), _CMP_LT_OQ);
        _mm256_blendv_ps(t, x, small)
    }

    /// # Safety
    /// Caller must ensure AVX2 + FMA are available.
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn sigmoid_slice(v: &mut [f32]) {
        let n = v.len();
        let mut i = 0;
        while i + 8 <= n {
            let p = v.as_mut_ptr().add(i);
            _mm256_storeu_ps(p, sigmoid8(_mm256_loadu_ps(p)));
            i += 8;
        }
        for x in &mut v[i..] {
            *x = super::sigmoid(*x);
        }
    }

    /// # Safety
    /// Caller must ensure AVX2 + FMA are available.
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn tanh_slice(v: &mut [f32]) {
        let n = v.len();
        let mut i = 0;
        while i + 8 <= n {
            let p = v.as_mut_ptr().add(i);
            _mm256_storeu_ps(p, tanh8(_mm256_loadu_ps(p)));
            i += 8;
        }
        for x in &mut v[i..] {
            *x = x.tanh();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vectorised_matches_libm() {
        if !has_avx2() {
            return;
        }
        let xs: Vec<f32> = (-800..800).map(|i| i as f32 * 0.05).collect();

        let mut s = xs.clone();
        sigmoid_slice(&mut s);
        for (a, x) in s.iter().zip(&xs) {
            assert!((a - sigmoid(*x)).abs() < 2e-7, "sigmoid({x}) = {a}");
        }

        let mut t = xs.clone();
        tanh_slice(&mut t);
        for (a, x) in t.iter().zip(&xs) {
            assert!((a - x.tanh()).abs() < 2e-7, "tanh({x}) = {a}");
        }
    }
}
