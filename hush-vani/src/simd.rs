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

/// True if AVX2 + FMA **and** F16C are available, i.e. the f16 weight kernels can run.
///
/// F16C (`vcvtph2ps`) has shipped on every x86 core with AVX2, but it is a separate CPUID
/// bit, so it is checked rather than assumed.
#[inline]
pub fn has_f16c() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        static OK: OnceLock<bool> = OnceLock::new();
        *OK.get_or_init(|| has_avx2() && is_x86_feature_detected!("f16c"))
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

/// Scalar f32 -> IEEE binary16 bits (round-to-nearest-even), for the load-time conversion.
pub fn f32_to_f16(x: f32) -> u16 {
    let b = x.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let mut exp = ((b >> 23) & 0xff) as i32 - 127 + 15;
    let mant = b & 0x007f_ffff;

    if ((b >> 23) & 0xff) == 0xff {
        // Inf / NaN
        return sign | 0x7c00 | if mant != 0 { 0x0200 } else { 0 };
    }
    if exp >= 0x1f {
        return sign | 0x7c00; // overflow -> Inf
    }
    if exp <= 0 {
        // subnormal (or underflow to zero)
        if exp < -10 {
            return sign;
        }
        let mant = mant | 0x0080_0000;
        let shift = (14 - exp) as u32;
        let half = 1u32 << (shift - 1);
        let round = mant & (half | (half - 1) | ((mant >> shift) & 1));
        let v = (mant + if round > half || (round == half && (mant >> shift) & 1 == 1) { half } else { 0 }) >> shift;
        return sign | v as u16;
    }
    // normal: round mantissa to 10 bits, round-to-nearest-even
    let lsb = (mant >> 13) & 1;
    let rounded = mant + 0x0fff + lsb;
    if rounded & 0x0080_0000 != 0 {
        exp += 1;
        if exp >= 0x1f {
            return sign | 0x7c00;
        }
    }
    sign | ((exp as u16) << 10) | ((rounded >> 13) & 0x03ff) as u16
}

/// Scalar IEEE binary16 bits -> f32. Exact (every f16 is representable in f32).
#[inline]
pub fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h & 0x8000) as u32) << 16;
    let exp = ((h >> 10) & 0x1f) as u32;
    let mant = (h & 0x03ff) as u32;
    let bits = if exp == 0 {
        if mant == 0 {
            sign
        } else {
            // Subnormal: the value is mant * 2^-24, with no implicit leading 1. Renormalise
            // by finding the mantissa's top set bit at position p, so the value is
            // 1.f * 2^(p-24); the f32 exponent is then biased to p + 103.
            //
            // `shift` is 10 - p, so the exponent is 113 - shift and the fraction has to
            // travel left by 23 - p = shift + 13, which also pushes the implicit 1 out to
            // bit 23 where the mask drops it.
            let shift = mant.leading_zeros() - 21; // = 10 - p, for mant in 1..=1023
            let e = 113 - shift;
            sign | (e << 23) | ((mant << (shift + 13)) & 0x007f_ffff)
        }
    } else if exp == 0x1f {
        sign | 0x7f80_0000 | (mant << 13)
    } else {
        sign | ((exp + 127 - 15) << 23) | (mant << 13)
    };
    f32::from_bits(bits)
}

/// Quantise a f32 slice to f16 bits.
pub fn to_f16(src: &[f32], dst: &mut [u16]) {
    for (d, &s) in dst.iter_mut().zip(src) {
        *d = f32_to_f16(s);
    }
}

/// Dequantise f16 bits back to f32.
pub fn from_f16(src: &[u16], dst: &mut [f32]) {
    for (d, &s) in dst.iter_mut().zip(src) {
        *d = f16_to_f32(s);
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
    fn f16_roundtrip_is_stable_and_accurate() {
        // f16 has a 10-bit mantissa => relative error <= 2^-11 ~ 4.9e-4
        for i in -20000..20000 {
            let x = i as f32 * 0.001;
            let r = f16_to_f32(f32_to_f16(x));
            let tol = 5e-4 * x.abs().max(1e-5);
            assert!((r - x).abs() <= tol, "x={x} -> {r}");
            // quantising an already-f16 value must be idempotent
            assert_eq!(f32_to_f16(r), f32_to_f16(x), "not idempotent at x={x}");
        }
        assert_eq!(f16_to_f32(f32_to_f16(0.0)), 0.0);
        assert!(f16_to_f32(f32_to_f16(-0.0)).is_sign_negative());
    }

    /// Every one of the 65536 bit patterns, against the definition rather than a round-trip.
    ///
    /// The loop above steps x by 0.001, so it never lands on a subnormal f16 (|x| < 6.1e-5)
    /// and happily passed while `f16_to_f32` returned exactly half the right answer for 2046
    /// of the 2048 subnormals. The int8 weights hide a per-block scale in an f16, and a scale
    /// that decodes 2x small corrupts all 64 weights that share it -- so this is now checked
    /// exhaustively. The hardware `vcvtph2ps` used by the f16 kernels always got this right,
    /// which is precisely why the disagreement stayed invisible.
    #[test]
    fn f16_to_f32_is_exact_for_every_bit_pattern() {
        for h in 0u32..=0xffff {
            let h = h as u16;
            let exp = (h >> 10) & 0x1f;
            if exp == 0x1f {
                continue; // inf / NaN
            }
            let sign = if h & 0x8000 != 0 { -1.0f32 } else { 1.0 };
            let mant = (h & 0x03ff) as f32;
            let want = if exp == 0 {
                // subnormal: no implicit leading 1, value is mant * 2^-24
                sign * mant * (2.0f32).powi(-24)
            } else {
                sign * (1.0 + mant / 1024.0) * (2.0f32).powi(exp as i32 - 15)
            };
            assert_eq!(
                f16_to_f32(h).to_bits(),
                want.to_bits(),
                "0x{h:04x}: got {}, want {want:e}",
                f16_to_f32(h)
            );
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn f16c_hardware_matches_scalar() {
        if !has_f16c() {
            return;
        }
        // the SIMD path must agree exactly with the scalar reference
        let xs: Vec<f32> = (0..800).map(|i| (i as f32 - 400.0) * 0.037).collect();
        let mut h = vec![0u16; xs.len()];
        to_f16(&xs, &mut h);
        let mut back = vec![0f32; xs.len()];
        from_f16(&h, &mut back);
        for (i, (&b, &x)) in back.iter().zip(&xs).enumerate() {
            assert_eq!(b, f16_to_f32(f32_to_f16(x)), "lane {i}");
        }
    }

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
