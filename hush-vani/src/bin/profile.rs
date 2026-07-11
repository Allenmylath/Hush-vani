//! Where does NN time actually go? Times each kernel at the shapes the model uses.
use hush_vani_core::alloc::AlignedVec;
use hush_vani_core::nn::*;
use std::hint::black_box;
use std::time::Instant;

const T: usize = 500;

fn ms<R>(mut f: impl FnMut() -> R, reps: usize) -> f64 {
    for _ in 0..2 {
        black_box(f());
    }
    let t0 = Instant::now();
    for _ in 0..reps {
        black_box(f());
    }
    t0.elapsed().as_secs_f64() * 1e3 / reps as f64
}

fn rnd(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((s >> 33) as f32 / (1u64 << 31) as f32) - 0.5
        })
        .collect()
}

fn arnd(n: usize, seed: u64) -> AlignedVec {
    AlignedVec::from_slice(&rnd(n, seed))
}

fn main() {
    let (h, i) = (256usize, 256usize);
    let h3 = 3 * h;

    // ---- GRU pieces ----
    let w = arnd(h3 * i, 1);
    let r = arnd(h3 * h, 2);
    let b = rnd(6 * h, 3);
    let x = arnd(T * i, 4);

    let mut xw = AlignedVec::zeros(T * h3);
    let t_xw = ms(|| gemm_nt(&x, T, i, &w, h3, &b[..h3], black_box(&mut xw)), 20);

    let hv = arnd(h, 5);
    let mut hr = AlignedVec::zeros(h3);
    let t_hr = ms(|| {
        for _ in 0..T {
            matvec(&r, h3, h, &hv, None, black_box(&mut hr));
        }
    }, 10);

    let t_gru = ms(|| gru(&x, T, i, &w, &r, &b, h), 10);

    println!("per-GRU at T={T}, H=256:");
    println!("  input proj (gemm_nt)   {t_xw:6.2} ms   [{:.1} GFMA/s]", (T * h3 * i) as f64 / t_xw / 1e6);
    println!("  recurrent  (matvec xT) {t_hr:6.2} ms   [{:.1} GFMA/s]", (T * h3 * h) as f64 / t_hr / 1e6);
    println!("  gates + overhead       {:6.2} ms", t_gru - t_xw - t_hr);
    println!("  full gru()             {t_gru:6.2} ms   -> x5 = {:.1} ms", t_gru * 5.0);

    // ---- grouped linears ----
    let cases: [(usize, usize, usize, &str); 4] = [
        (1, 256, 640, "df_out      [1,256,640]"),
        (1, 128, 256, "linear_in   [1,128,256]"),
        (1, 256, 128, "linear_out  [1,256,128]"),
        (16, 32, 8, "df_fc_emb   [16,32,8]"),
    ];
    println!("\ngrouped_linear:");
    let mut gl_total = 0.0;
    for (g, ii, hh, name) in cases {
        let xx = rnd(T * g * ii, 7);
        let ww = rnd(g * ii * hh, 8);
        let t = ms(|| grouped_linear(&xx, T, g, ii, hh, &ww), 20);
        let f = (T * g * ii * hh) as f64;
        println!("  {name}  {t:6.2} ms  [{:.1} GFMA/s]", f / t / 1e6);
        gl_total += t;
    }
    // enc has linear_in+linear_out, erb_dec has both, df_dec has df_out + its own linear_in
    println!("  (model uses these ~6x total)");

    // ---- convs ----
    println!("\nconvs:");
    let c0 = rnd(16 * T * 64, 9);
    let pw = rnd(16 * 16, 10);
    let pb = rnd(16, 11);
    let t = ms(|| pointwise(&c0, 16, T * 64, &pw, Some(&pb), 16), 20);
    println!("  pointwise 16->16 @T*64  {t:6.2} ms  [{:.1} GFMA/s]", (16 * 16 * T * 64) as f64 / t / 1e6);

    let fs = rnd(2 * T * 64, 12);
    let w33 = rnd(16 * 1 * 3 * 3, 13);
    let t = ms(|| conv3x3_causal(&fs, 2, T, 64, &w33, None, 16, 2), 20);
    println!("  conv3x3_causal df_conv0 {t:6.2} ms");

    let fe = rnd(1 * T * 32, 14);
    let w33b = rnd(16 * 1 * 3 * 3, 15);
    let bb = rnd(16, 16);
    let t = ms(|| conv3x3_causal(&fe, 1, T, 32, &w33b, Some(&bb), 16, 1), 20);
    println!("  conv3x3_causal erb_conv0 {t:5.2} ms");

    let y = rnd(16 * T * 32, 17);
    let w13 = rnd(1 * 16 * 3, 18);
    let b13 = rnd(1, 19);
    let t = ms(|| conv1x3(&y, 16, T, 32, &w13, &b13, 1), 20);
    println!("  conv1x3 conv0_out        {t:5.2} ms");

    println!("\n(5 GRUs dominate: {:.1} ms of a ~83 ms NN)", t_gru * 5.0);
    println!("gl_total for one pass of the 4 shapes above: {gl_total:.2} ms");
}

