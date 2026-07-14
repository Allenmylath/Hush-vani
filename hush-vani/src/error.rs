use std::fmt;

/// Errors returned when loading a model or enhancing audio.
#[derive(Debug)]
#[non_exhaustive]
pub enum Error {
    /// Reading the weight blob or manifest from disk failed.
    Io(std::io::Error),
    /// The weight blob or manifest is malformed.
    Weights(String),
    /// A tensor the model needs is absent from the manifest.
    MissingTensor(String),
    /// Input audio was too short to produce a single 10 ms frame.
    TooShort {
        /// Number of samples supplied.
        samples: usize,
        /// Minimum number of samples required (one frame).
        needed: usize,
    },
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Io(e) => write!(f, "io error: {e}"),
            Error::Weights(m) => write!(f, "invalid weights: {m}"),
            Error::MissingTensor(n) => write!(f, "weights are missing tensor {n:?}"),
            Error::TooShort { samples, needed } => {
                write!(f, "audio too short: {samples} samples, need at least {needed}")
            }
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Error::Io(e) => Some(e),
            _ => None,
        }
    }
}

impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Error::Io(e)
    }
}
