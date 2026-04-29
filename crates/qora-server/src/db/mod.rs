//! redb-backed persistence: API keys + usage records + chat history + users + sessions + guests.
//!
//! Tables:
//!   keys_by_id:         key_id (UUID str)        -> bincode(StoredKey)
//!   id_by_hmac:         HMAC-SHA256(secret, raw) -> key_id
//!   usage:              auto-inc u64             -> bincode(StoredUsage)
//!   conversations:      conv_id (UUID str)       -> bincode(Conversation)
//!   messages:           (conv_id || u64 big-endian idx) -> bincode(StoredMessage)
//!   user_conversations: user_id                  -> multimap(conv_id)
//!   users:              user_id (UUID str)       -> bincode(StoredUser)
//!   sessions:           session_token            -> bincode(Session)
//!   guest_quotas:       guest_id                 -> bincode(GuestQuota)
//!   email_to_user:      email                    -> user_id
//!
//! Server secret comes from env at startup; never persisted.

use anyhow::Context;
use redb::{Database, MultimapTableDefinition, ReadableTable, TableDefinition};
use std::path::Path;
use std::sync::Arc;

pub const KEYS_BY_ID: TableDefinition<&str, &[u8]> = TableDefinition::new("keys_by_id");
pub const ID_BY_HMAC: TableDefinition<&[u8], &str> = TableDefinition::new("id_by_hmac");
pub const USAGE: TableDefinition<u64, &[u8]> = TableDefinition::new("usage");

/// conversations: conv_id (UUID str) → bincode(Conversation)
pub const CONVERSATIONS: TableDefinition<&str, &[u8]> = TableDefinition::new("conversations");

/// messages: (conv_id + msg_idx_8bytes) → bincode(StoredMessage)
/// Composite key: conv_id bytes (36 bytes for UUID) + 8 bytes big-endian u64.
pub const MESSAGES: TableDefinition<&[u8], &[u8]> = TableDefinition::new("messages");

/// user_conversations: user_id → conv_ids (multimap)
pub const USER_CONVERSATIONS: MultimapTableDefinition<&str, &str> =
    MultimapTableDefinition::new("user_conversations");

/// users: user_id (UUID str) → bincode(StoredUser)
pub const USERS: TableDefinition<&str, &[u8]> = TableDefinition::new("users");

/// sessions: session_token → bincode(Session)
pub const SESSIONS: TableDefinition<&str, &[u8]> = TableDefinition::new("sessions");

/// guest_quotas: guest_id → bincode(GuestQuota)
pub const GUEST_QUOTAS: TableDefinition<&str, &[u8]> = TableDefinition::new("guest_quotas");

/// email_to_user: email → user_id (lookup by email at login)
pub const EMAIL_TO_USER: TableDefinition<&str, &str> = TableDefinition::new("email_to_user");

/// Opens (or creates) the database file. Initializes the table set on first run.
pub fn open(path: &Path) -> anyhow::Result<Arc<Database>> {
    let db = Database::create(path).with_context(|| format!("open redb at {}", path.display()))?;

    // Touch every table once so they exist before first read attempt.
    let write = db.begin_write()?;
    {
        let _ = write.open_table(KEYS_BY_ID)?;
        let _ = write.open_table(ID_BY_HMAC)?;
        let _ = write.open_table(USAGE)?;
        let _ = write.open_table(CONVERSATIONS)?;
        let _ = write.open_table(MESSAGES)?;
        let _ = write.open_multimap_table(USER_CONVERSATIONS)?;
        let _ = write.open_table(USERS)?;
        let _ = write.open_table(SESSIONS)?;
        let _ = write.open_table(GUEST_QUOTAS)?;
        let _ = write.open_table(EMAIL_TO_USER)?;
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
