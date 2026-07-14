fn main() {
    println!("compile-time target features:");
    for f in ["sse2", "avx", "avx2", "fma", "avx512f"] {
        let on = match f {
            "sse2" => cfg!(target_feature = "sse2"),
            "avx" => cfg!(target_feature = "avx"),
            "avx2" => cfg!(target_feature = "avx2"),
            "fma" => cfg!(target_feature = "fma"),
            "avx512f" => cfg!(target_feature = "avx512f"),
            _ => false,
        };
        println!("  {f:8} {}", if on { "YES" } else { "no" });
    }
    println!("runtime detected:");
    println!("  avx2 {}", is_x86_feature_detected!("avx2"));
    println!("  fma  {}", is_x86_feature_detected!("fma"));
    println!("  f16c {}", is_x86_feature_detected!("f16c"));
    println!("  avx512f {}", is_x86_feature_detected!("avx512f"));
    // int8 dot-product instructions: vpdpbusd = 32 MACs in ONE instruction (vs 8 for an f32
    // FMA). On an issue-bound kernel that is the only thing that could make int8 faster.
    println!("  avxvnni      {}", is_x86_feature_detected!("avxvnni"));
    println!("  avx512vnni   {}", is_x86_feature_detected!("avx512vnni"));
    println!("kernels:");
    println!("  avx2 path  {}", hush_vani::simd::has_avx2());
    println!("  f16 path   {}", hush_vani::simd::has_f16c());
}
