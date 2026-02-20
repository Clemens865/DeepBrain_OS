use rvf_types::RvfError;
use std::fmt;

/// Errors from the DeepBrain storage layer.
#[derive(Debug)]
pub enum DeepBrainError {
    /// Error from the underlying RvfStore.
    Rvf(RvfError),
    /// SQLite error (id_map or legacy persistence).
    Sqlite(rusqlite::Error),
    /// IO error (file operations).
    Io(std::io::Error),
    /// The store file was not found at the expected path.
    StoreNotFound(String),
    /// A vector with this string ID was not found in the id_map.
    IdNotFound(String),
    /// Dimension mismatch between query vector and store.
    DimensionMismatch { expected: u16, got: usize },
    /// The store is in read-only mode.
    ReadOnly,
    /// Migration error.
    Migration(String),
}

impl fmt::Display for DeepBrainError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Rvf(e) => write!(f, "RVF error: {:?}", e),
            Self::Sqlite(e) => write!(f, "SQLite error: {}", e),
            Self::Io(e) => write!(f, "IO error: {}", e),
            Self::StoreNotFound(path) => write!(f, "Store not found: {}", path),
            Self::IdNotFound(id) => write!(f, "ID not found: {}", id),
            Self::DimensionMismatch { expected, got } => {
                write!(f, "Dimension mismatch: expected {}, got {}", expected, got)
            }
            Self::ReadOnly => write!(f, "Store is read-only"),
            Self::Migration(msg) => write!(f, "Migration error: {}", msg),
        }
    }
}

impl std::error::Error for DeepBrainError {}

impl From<RvfError> for DeepBrainError {
    fn from(e: RvfError) -> Self {
        Self::Rvf(e)
    }
}

impl From<rusqlite::Error> for DeepBrainError {
    fn from(e: rusqlite::Error) -> Self {
        Self::Sqlite(e)
    }
}

impl From<std::io::Error> for DeepBrainError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

impl From<DeepBrainError> for String {
    fn from(e: DeepBrainError) -> String {
        e.to_string()
    }
}
