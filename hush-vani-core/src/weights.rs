use crate::alloc::AlignedVec;
use crate::error::Error;
use std::collections::HashMap;
use std::path::Path;

/// A flat, 64-byte-aligned arena of every tensor in the model, addressed by name.
///
/// The manifest is a text file with one line per tensor:
/// `"<graph>/<name> <shape,comma,sep> <offset_in_floats> <numel>"`.
/// Offsets are padded to 16 floats so each tensor -- and each of its rows -- starts on a
/// 64-byte boundary, which lets the AVX2 kernels use aligned loads.
pub struct Weights {
    data: AlignedVec,
    map: HashMap<String, (Vec<usize>, usize, usize)>, // shape, offset, len
    f16: bool,
}

impl Weights {
    /// Load from a raw little-endian weight blob and its manifest text.
    ///
    /// The blob is `f32` by default. A manifest whose first line is `#dtype f16` marks an
    /// **f16** blob (half the size), which is widened to f32 here; the stages then re-narrow
    /// their matmul weights back to f16 for the kernels. Offsets and lengths in the manifest
    /// are always in *elements*, so they are unchanged between the two formats.
    pub fn from_bytes(bin: &[u8], manifest: &str) -> Result<Self, Error> {
        let is_f16 = manifest
            .lines()
            .next()
            .map(|l| l.trim_start().starts_with("#dtype f16"))
            .unwrap_or(false);

        let floats: Vec<f32> = if is_f16 {
            if bin.len() % 2 != 0 {
                return Err(Error::Weights("f16 blob length is not a multiple of 2".into()));
            }
            bin.chunks_exact(2)
                .map(|c| crate::simd::f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
                .collect()
        } else {
            if bin.len() % 4 != 0 {
                return Err(Error::Weights("f32 blob length is not a multiple of 4".into()));
            }
            bin.chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect()
        };
        let data = AlignedVec::from_slice(&floats);

        let mut map = HashMap::new();
        for (n, line) in manifest.lines().enumerate() {
            if line.trim().is_empty() || line.trim_start().starts_with('#') {
                continue;
            }
            let bad = || Error::Weights(format!("malformed manifest line {}: {line:?}", n + 1));
            let mut it = line.rsplitn(3, ' ');
            let len: usize = it.next().ok_or_else(bad)?.parse().map_err(|_| bad())?;
            let off: usize = it.next().ok_or_else(bad)?.parse().map_err(|_| bad())?;
            let head = it.next().ok_or_else(bad)?;
            let (name, shape_s) = head.rsplit_once(' ').ok_or_else(bad)?;
            let shape: Vec<usize> = if shape_s.is_empty() {
                Vec::new()
            } else {
                shape_s.split(',').map(|d| d.parse().map_err(|_| bad())).collect::<Result<_, _>>()?
            };
            if off + len > data.len() {
                return Err(Error::Weights(format!("tensor {name} runs past end of blob")));
            }
            map.insert(name.to_string(), (shape, off, len));
        }
        if map.is_empty() {
            return Err(Error::Weights("manifest contained no tensors".into()));
        }
        Ok(Weights { data, map, f16: is_f16 })
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
