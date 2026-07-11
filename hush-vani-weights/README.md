# hush-vani-weights

Bundled weights for [`hush-vani`](https://crates.io/crates/hush-vani), so you don't have to
ship a `weights.bin` next to your binary or run any export script.

**Most users don't depend on this crate directly.** Enable the `bundled` feature on
`hush-vani` and it pulls these weights in for you:

```toml
[dependencies]
hush-vani = { version = "0.1", features = ["bundled"] }
```

```rust
let hush = hush_vani::Hush::bundled()?;   // one line, weights embedded
let clean = hush.enhance(&noisy)?;
```

This crate itself is a **pure, dependency-free data blob** exposing the raw bytes, for when
you want to load them yourself:

```rust
use hush_vani::Hush;
let hush = Hush::from_bytes(hush_vani_weights::WEIGHTS_BIN, hush_vani_weights::MANIFEST)?;
# Ok::<(), hush_vani::Error>(())
```

The blob adds **~8 MB** to your binary (embedded via `include_bytes!`). To load weights from
a file at runtime instead, use
[`Hush::from_paths`](https://docs.rs/hush-vani/latest/hush_vani/struct.Hush.html#method.from_paths)
and don't depend on this crate at all.

## Licence and attribution

These weights are the [`weya-ai/hush`](https://huggingface.co/weya-ai/hush) model,
redistributed under **Apache-2.0** (the model's own license). The container format is
changed (flat f32 arena); no weight values are altered. See `LICENSE` and `NOTICE`. This
crate is not affiliated with or endorsed by Weya AI.
