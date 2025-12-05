use thiserror::Error;

#[derive(Debug, Error)]
pub enum Error {
    #[error("invalid input: {0}")]
    InvalidInput(String),
    #[error("Conversion failure")]
    ConversionFailure,
    #[error("Dimension mismatch: {0}")]
    DimensionMismatch(String),
}

pub type Result<T> = std::result::Result<T, Error>;
