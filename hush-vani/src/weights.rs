use crate::alloc::AlignedVec;
use crate::error::Error;
use std::collections::HashMap;
use std::path::Path;

/// What the blob on disk holds. Every format is decoded to f32 here; this only says what
/// had to be undone to get there.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Dtype {
    F32,
    F16,
    /// Symmetric int8, one f16 scale per block of weights along the contraction axis.
    Int8 {
        /// Where the f16 scale table starts, in *elements* (= bytes, int8 codes being 1 byte).
        payload: usize,
    },
}

/// A flat, 64-byte-aligned arena of every tensor in the model, addressed by name.
///
/// The manifest is a text file with one line per tensor:
/// `"<graph>/<name> <shape,comma,sep> <offset_in_elements> <numel>"`, with two extra fields
/// for int8 (see [`Weights::from_bytes`]). Offsets are padded so each tensor -- and each of
/// its rows -- starts on a 64-byte boundary, which lets the AVX2 kernels use aligned loads.
/// The padding is counted in elements, so a manifest is only valid for its own blob.
pub struct Weights {
    data: AlignedVec,
    map: HashMap<String, (Vec<usize>, usize, usize)>, // shape, offset, len
    f16: bool,
}

impl Weights {
    /// Load from a raw little-endian weight blob and its manifest text.
    ///
    /// The blob is `f32` unless the manifest's first line says otherwise:
    ///
    /// - `#dtype f16` — half the size, widened to f32 here; the stages then re-narrow their
    ///   matmul weights back to f16 for the kernels.
    /// - `#dtype int8 payload <n>` — a quarter the size. The first `n` bytes are int8 codes;
    ///   the rest is a table of f32 scales. Each manifest line carries two extra fields,
    ///   `<scale_off> <block>`, and a weight is recovered as
    ///   `q[i] * scale[scale_off + i / block]`.
    ///
    /// int8 is **decoded to f32 here and run through the f32 kernels**, exactly as f16 is. It
    /// is a storage format, not a compute mode, so it costs no accuracy beyond the rounding
    /// already baked into the file.
    ///
    /// A block spans part of one matmul row and never straddles two, which is the whole
    /// reason int8 is usable at all: a single scale per tensor has to cover the GRUs' ±31
    /// outliers alongside the 99% of weights inside ±0.23, and the resulting model removes
    /// 2.1 dB less noise on average (8.4 dB worse on one sample). Block-wise scales cost
    /// 79 KB and give all of that back.
    pub fn from_bytes(bin: &[u8], manifest: &str) -> Result<Self, Error> {
        let head = manifest.lines().next().unwrap_or("").trim();
        let dtype = if let Some(rest) = head.strip_prefix("#dtype int8") {
            let payload = rest
                .split_whitespace()
                .nth(1)
                .and_then(|v| v.parse().ok())
                .ok_or_else(|| Error::Weights("int8 header needs `payload <n_elements>`".into()))?;
            if payload > bin.len() {
                return Err(Error::Weights("int8 payload runs past end of blob".into()));
            }
            Dtype::Int8 { payload }
        } else if head.starts_with("#dtype f16") {
            Dtype::F16
        } else {
            Dtype::F32
        };

        // For int8 the arena is the payload; the tensors are dequantised into it below, once
        // the manifest has told us each one's scales. For f32/f16 the whole blob decodes here.
        let floats: Vec<f32> = match dtype {
            Dtype::Int8 { payload } => vec![0.0; payload],
            Dtype::F16 => {
                if bin.len() % 2 != 0 {
                    return Err(Error::Weights("f16 blob length is not a multiple of 2".into()));
                }
                bin.chunks_exact(2)
                    .map(|c| crate::simd::f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
                    .collect()
            }
            Dtype::F32 => {
                if bin.len() % 4 != 0 {
                    return Err(Error::Weights("f32 blob length is not a multiple of 4".into()));
                }
                bin.chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect()
            }
        };
        let mut data = AlignedVec::from_slice(&floats);

        // int8 scale table: f32, one per block. f16 would save 80 KB but a scale is amax/127,
        // and the blocks whose weights all sit around 1e-6 flush to zero under f16's smallest
        // subnormal -- silently zeroing the 64 weights each one covers.
        let scales: &[u8] = match dtype {
            Dtype::Int8 { payload } => &bin[payload..],
            _ => &[],
        };
        if scales.len() % 4 != 0 {
            return Err(Error::Weights("int8 scale table is not a multiple of 4 bytes".into()));
        }
        let n_scales = scales.len() / 4;
        let scale = |i: usize| -> f32 {
            let b = &scales[4 * i..4 * i + 4];
            f32::from_le_bytes([b[0], b[1], b[2], b[3]])
        };

        let mut map = HashMap::new();
        for (n, line) in manifest.lines().enumerate() {
            if line.trim().is_empty() || line.trim_start().starts_with('#') {
                continue;
            }
            let bad = || Error::Weights(format!("malformed manifest line {}: {line:?}", n + 1));

            // f32/f16: `name shape off len`.  int8: `name shape off len scale_off block`.
            let (head, off, len, q) = if let Dtype::Int8 { .. } = dtype {
                let mut it = line.rsplitn(5, ' ');
                let block: usize = it.next().ok_or_else(bad)?.parse().map_err(|_| bad())?;
                let soff: usize = it.next().ok_or_else(bad)?.parse().map_err(|_| bad())?;
                let len: usize = it.next().ok_or_else(bad)?.parse().map_err(|_| bad())?;
                let off: usize = it.next().ok_or_else(bad)?.parse().map_err(|_| bad())?;
                (it.next().ok_or_else(bad)?, off, len, Some((soff, block)))
            } else {
                let mut it = line.rsplitn(3, ' ');
                let len: usize = it.next().ok_or_else(bad)?.parse().map_err(|_| bad())?;
                let off: usize = it.next().ok_or_else(bad)?.parse().map_err(|_| bad())?;
                (it.next().ok_or_else(bad)?, off, len, None)
            };

            let (name, shape_s) = head.rsplit_once(' ').ok_or_else(bad)?;
            let shape: Vec<usize> = if shape_s.is_empty() {
                Vec::new()
            } else {
                shape_s.split(',').map(|d| d.parse().map_err(|_| bad())).collect::<Result<_, _>>()?
            };
            if off + len > data.len() {
                return Err(Error::Weights(format!("tensor {name} runs past end of blob")));
            }

            if let Some((soff, block)) = q {
                if block == 0 {
                    return Err(Error::Weights(format!("tensor {name} has a zero block length")));
                }
                let nb = (len + block - 1) / block; // div_ceil is 1.73+; the crate pins 1.70
                if soff + nb > n_scales {
                    return Err(Error::Weights(format!("tensor {name} runs past the scale table")));
                }
                for i in 0..len {
                    let code = bin[off + i] as i8;
                    data[off + i] = code as f32 * scale(soff + i / block);
                }
            }
            map.insert(name.to_string(), (shape, off, len));
        }
        if map.is_empty() {
            return Err(Error::Weights("manifest contained no tensors".into()));
        }
        Ok(Weights { data, map, f16: dtype == Dtype::F16 })
    }

    /// Load from `weights.bin` + `weights.txt` on disk.
    pub fn from_paths(bin: impl AsRef<Path>, manifest: impl AsRef<Path>) -> Result<Self, Error> {
        let raw = std::fs::read(bin)?;
        let txt = std::fs::read_to_string(manifest)?;
        Self::from_bytes(&raw, &txt)
    }

    /// True if these weights came from an f16 blob.
    ///
    /// This selects the kernel precision: an f16 file runs the f16 kernels (half the weight
    /// bandwidth, ~75 dB end-to-end), an f32 file runs the exact f32 kernels (~130 dB).
    /// Precision is therefore a property of the weights you ship, never a silent downgrade.
    pub fn is_f16(&self) -> bool {
        self.f16
    }

    /// A tensor's data by name, or [`Error::MissingTensor`].
    pub fn get(&self, name: &str) -> Result<&[f32], Error> {
        let (_, off, len) = self.map.get(name).ok_or_else(|| Error::MissingTensor(name.into()))?;
        Ok(&self.data[*off..*off + *len])
    }

    /// A tensor's shape by name, or [`Error::MissingTensor`].
    pub fn shape(&self, name: &str) -> Result<&[usize], Error> {
        Ok(&self.map.get(name).ok_or_else(|| Error::MissingTensor(name.into()))?.0)
    }
}
