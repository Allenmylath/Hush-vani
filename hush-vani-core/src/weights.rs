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
}

impl Weights {
    /// Load from a raw little-endian f32 blob and its manifest text.
    pub fn from_bytes(bin: &[u8], manifest: &str) -> Result<Self, Error> {
        if bin.len() % 4 != 0 {
            return Err(Error::Weights("weights blob length is not a multiple of 4".into()));
        }
        let floats: Vec<f32> = bin
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let data = AlignedVec::from_slice(&floats);

        let mut map = HashMap::new();
        for (n, line) in manifest.lines().enumerate() {
            if line.trim().is_empty() {
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
        Ok(Weights { data, map })
    }

    /// Load from `weights.bin` + `weights.txt` on disk.
    pub fn from_paths(bin: impl AsRef<Path>, manifest: impl AsRef<Path>) -> Result<Self, Error> {
        let raw = std::fs::read(bin)?;
        let txt = std::fs::read_to_string(manifest)?;
        Self::from_bytes(&raw, &txt)
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
