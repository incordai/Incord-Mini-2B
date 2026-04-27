//! redb-backed persistence: API keys + usage records.
//!
//! Tables:
//!   keys_by_id: key_id (UUID str)        -> bincode(StoredKey)
//!   id_by_hmac: HMAC-SHA256(secret, raw) -> key_id
//!   usage:      auto-inc u64             -> bincode(StoredUsage)
//!
//! Server secret comes from env at startup; never persisted.

use anyhow::Context;
use redb::{Database, ReadableTable, TableDefinition};
use std::path::Path;
use std::sync::Arc;

pub const KEYS_BY_ID: TableDefinition<&str, &[u8]> = TableDefinition::new("keys_by_id");
pub const ID_BY_HMAC: TableDefinition<&[u8], &str> = TableDefinition::new("id_by_hmac");
pub const USAGE: TableDefinition<u64, &[u8]> = TableDefinition::new("usage");

/// Opens (or creates) the database file. Initializes the table set on first run.
pub fn open(path: &Path) -> anyhow::Result<Arc<Database>> {
    let db = Database::create(path).with_context(|| format!("open redb at {}", path.display()))?;

    // Touch every table once so they exist before first read attempt.
    let write = db.begin_write()?;
    {
        let _ = write.open_table(KEYS_BY_ID)?;
        let _ = write.open_table(ID_BY_HMAC)?;
        let _ = write.open_table(USAGE)?;
    }
    write.commit()?;
    Ok(Arc::new(db))
}

/// Allocate the next monotonically-increasing usage id.
/// Uses `last()` on the USAGE table to find the current max key, defaulting to 0 if empty.
/// (For a write-heavy production deployment, switch to a dedicated counter table.)
pub fn next_usage_id(db: &Database) -> anyhow::Result<u64> {
    let read = db.begin_read()?;
    let table = read.open_table(USAGE)?;
    let last = table.last()?.map(|(k, _)| k.value()).unwrap_or(0);
    Ok(last + 1)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn open_creates_fresh_db() {
        let dir = tempdir().unwrap();
        let _db = open(&dir.path().join("test.redb")).unwrap();
    }

    #[test]
    fn open_is_idempotent() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.redb");
        let db1 = open(&path).unwrap();
        drop(db1);
        let _db2 = open(&path).unwrap();
    }

    #[test]
    fn next_usage_id_starts_at_one() {
        let dir = tempdir().unwrap();
        let db = open(&dir.path().join("test.redb")).unwrap();
        assert_eq!(next_usage_id(&db).unwrap(), 1);
    }
}
