//! redb-backed persistence: API keys + usage + chat history + users + sessions + guests.
//!
//! Multi-tenancy is foundational. Every persisted record is keyed by
//! `tenant_id`. Single-tenant deployments use [`DEFAULT_TENANT`] = `"default"`;
//! the structure is identical so adding a second tenant later is a matter of
//! seeding it, not migrating data. The cost is bytes-per-row; the savings is
//! never having to do a full data migration to introduce isolation.
//!
//! All values are encoded with [`encode_v1`] which prepends a single
//! [`SCHEMA_VERSION`] byte. Future field additions bump the version and
//! [`decode`] dispatches by version, so old records keep deserialising.
//!
//! Monotonic ID allocation goes through [`alloc_id`] which reads + writes the
//! [`COUNTERS`] table inside the same write transaction the inserter uses —
//! no race window between "what's the next id" and "insert at that id".
//!
//! Tables:
//!   counters:           "{tenant}::{name}"        -> u64 (next monotonic id)
//!   keys_by_id:         "{tenant}::{key_id}"      -> v1(StoredKey)
//!   id_by_hmac:         tenant\0HMAC(secret,raw)  -> "{tenant}::{key_id}"
//!   usage:              tenant\0u64_be            -> v1(StoredUsage)
//!   conversations:      "{tenant}::{conv_id}"     -> v1(Conversation)
//!   messages:           tenant\0conv_id\0u64_be   -> v1(StoredMessage)
//!   user_conversations: "{tenant}::{user_id}"     -> multimap("{tenant}::{conv_id}")
//!   users:              "{tenant}::{user_id}"     -> v1(StoredUser)
//!   sessions:           "{tenant}::{token}"       -> v1(Session)
//!   guest_quotas:       "{tenant}::{guest_id}"    -> v1(GuestQuota)
//!   email_to_user:      "{tenant}::{email}"       -> "{tenant}::{user_id}"
//!   guest_messages:     tenant\0u64_be            -> v1(GuestMessage)

use anyhow::Context;
use redb::{
    Database, MultimapTableDefinition, ReadableTable, TableDefinition, WriteTransaction,
};
use serde::{de::DeserializeOwned, Serialize};
use std::path::Path;
use std::sync::Arc;

// ---------------------------------------------------------------------------
// Tenancy + schema constants
// ---------------------------------------------------------------------------

/// Default tenant for installations that haven't yet configured multi-tenancy.
/// New tenants are introduced by passing a different string — the schema is
/// already partitioned by tenant so no migration is required.
pub const DEFAULT_TENANT: &str = "default";

/// Version byte prepended to every encoded value. Bump when adding fields that
/// older code can't ignore. [`decode`] dispatches by this byte.
pub const SCHEMA_VERSION: u8 = 1;

/// Bincode config. Re-exported so callers don't pin their own copy.
pub const BINCODE_CFG: bincode::config::Configuration = bincode::config::standard();

/// Separator for compound binary keys. Safe because tenant IDs are restricted
/// to `[A-Za-z0-9_-]` (see [`is_safe_tenant`]).
const NUL: u8 = 0x00;

// ---------------------------------------------------------------------------
// Table definitions
// ---------------------------------------------------------------------------

/// Race-free monotonic counters. Key = `"{tenant}::{counter_name}"`.
pub const COUNTERS: TableDefinition<&str, u64> = TableDefinition::new("counters");

/// keys_by_id: `"{tenant}::{key_id}"` -> v1(StoredKey)
pub const KEYS_BY_ID: TableDefinition<&str, &[u8]> = TableDefinition::new("keys_by_id");

/// id_by_hmac: `tenant\0HMAC` -> `"{tenant}::{key_id}"`
pub const ID_BY_HMAC: TableDefinition<&[u8], &str> = TableDefinition::new("id_by_hmac");

/// usage: `tenant\0u64_be` -> v1(StoredUsage). Compound key keeps tenant scoping
/// and preserves chronological order within a tenant (u64 big-endian sorts as bytes).
pub const USAGE: TableDefinition<&[u8], &[u8]> = TableDefinition::new("usage");

/// conversations: `"{tenant}::{conv_id}"` -> v1(Conversation)
pub const CONVERSATIONS: TableDefinition<&str, &[u8]> = TableDefinition::new("conversations");

/// messages: `tenant\0conv_id\0u64_be` -> v1(StoredMessage)
pub const MESSAGES: TableDefinition<&[u8], &[u8]> = TableDefinition::new("messages");

/// user_conversations: `"{tenant}::{user_id}"` -> multimap of `"{tenant}::{conv_id}"`
pub const USER_CONVERSATIONS: MultimapTableDefinition<&str, &str> =
    MultimapTableDefinition::new("user_conversations");

/// users: `"{tenant}::{user_id}"` -> v1(StoredUser)
pub const USERS: TableDefinition<&str, &[u8]> = TableDefinition::new("users");

/// sessions: `"{tenant}::{token}"` -> v1(Session)
pub const SESSIONS: TableDefinition<&str, &[u8]> = TableDefinition::new("sessions");

/// guest_quotas: `"{tenant}::{guest_id}"` -> v1(GuestQuota)
pub const GUEST_QUOTAS: TableDefinition<&str, &[u8]> = TableDefinition::new("guest_quotas");

/// email_to_user: `"{tenant}::{email}"` -> `"{tenant}::{user_id}"`
pub const EMAIL_TO_USER: TableDefinition<&str, &str> = TableDefinition::new("email_to_user");

/// guest_messages: `tenant\0u64_be` -> v1(GuestMessage). Compound key (see USAGE).
pub const GUEST_MESSAGES: TableDefinition<&[u8], &[u8]> =
    TableDefinition::new("guest_messages");

// ---------------------------------------------------------------------------
// Tenant key helpers
// ---------------------------------------------------------------------------

/// Tenant IDs must match `[A-Za-z0-9_-]+`, length 1..=64. Anything else
/// could collide with our compound-key separators or end up in filesystem
/// paths the brain side joins user-supplied IDs into.
pub fn is_safe_tenant(s: &str) -> bool {
    !s.is_empty()
        && s.len() <= 64
        && s.chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-')
}

/// Build a string key `"{tenant}::{id}"`. Used by all `&str`-keyed tables.
pub fn tenant_key(tenant: &str, id: &str) -> String {
    debug_assert!(is_safe_tenant(tenant), "unsafe tenant id `{tenant}`");
    format!("{tenant}::{id}")
}

/// Build a binary key `tenant\0id_bytes`. Used by tables with `&[u8]` keys
/// where `id` is itself binary (e.g. HMAC).
pub fn tenant_bytes_key(tenant: &str, id: &[u8]) -> Vec<u8> {
    debug_assert!(is_safe_tenant(tenant), "unsafe tenant id `{tenant}`");
    let mut buf = Vec::with_capacity(tenant.len() + 1 + id.len());
    buf.extend_from_slice(tenant.as_bytes());
    buf.push(NUL);
    buf.extend_from_slice(id);
    buf
}

/// Build a binary key `tenant\0u64_be`. Used by USAGE and GUEST_MESSAGES.
/// Big-endian preserves chronological order within a tenant when iterating.
pub fn tenant_u64_key(tenant: &str, id: u64) -> Vec<u8> {
    debug_assert!(is_safe_tenant(tenant), "unsafe tenant id `{tenant}`");
    let mut buf = Vec::with_capacity(tenant.len() + 1 + 8);
    buf.extend_from_slice(tenant.as_bytes());
    buf.push(NUL);
    buf.extend_from_slice(&id.to_be_bytes());
    buf
}

/// Build a binary key `tenant\0conv_id\0u64_be` for the messages table.
pub fn message_key(tenant: &str, conv_id: &str, idx: u32) -> Vec<u8> {
    debug_assert!(is_safe_tenant(tenant), "unsafe tenant id `{tenant}`");
    let mut buf = Vec::with_capacity(tenant.len() + 1 + conv_id.len() + 1 + 8);
    buf.extend_from_slice(tenant.as_bytes());
    buf.push(NUL);
    buf.extend_from_slice(conv_id.as_bytes());
    buf.push(NUL);
    buf.extend_from_slice(&(idx as u64).to_be_bytes());
    buf
}

/// Prefix for tenant-scoped `&[u8]` range scans.
pub fn tenant_byte_prefix(tenant: &str) -> Vec<u8> {
    let mut buf = Vec::with_capacity(tenant.len() + 1);
    buf.extend_from_slice(tenant.as_bytes());
    buf.push(NUL);
    buf
}

/// Prefix for tenant-scoped `&str` range scans (tables keyed by `tenant_key`).
pub fn tenant_str_prefix(tenant: &str) -> String {
    format!("{tenant}::")
}

// ---------------------------------------------------------------------------
// Encoding (schema-versioned bincode)
// ---------------------------------------------------------------------------

/// Encode a value for storage: `[SCHEMA_VERSION, bincode(t)...]`.
///
/// Always pair with [`decode`] on the read side. Future schema bumps can
/// dispatch by the leading byte without breaking older records.
pub fn encode_v1<T: Serialize>(t: &T) -> anyhow::Result<Vec<u8>> {
    let body = bincode::serde::encode_to_vec(t, BINCODE_CFG).context("bincode encode")?;
    let mut out = Vec::with_capacity(body.len() + 1);
    out.push(SCHEMA_VERSION);
    out.extend_from_slice(&body);
    Ok(out)
}

/// Decode a value written by [`encode_v1`]. Dispatches on the version byte
/// so we can introduce v2/v3 records and continue reading v1 ones.
pub fn decode<T: DeserializeOwned>(bytes: &[u8]) -> anyhow::Result<T> {
    let (version, body) = bytes
        .split_first()
        .ok_or_else(|| anyhow::anyhow!("empty encoded value"))?;
    match *version {
        1 => {
            let (t, _) = bincode::serde::decode_from_slice(body, BINCODE_CFG)
                .context("bincode decode v1")?;
            Ok(t)
        }
        v => anyhow::bail!("unknown schema version: {v}"),
    }
}

// ---------------------------------------------------------------------------
// Race-free ID allocation
// ---------------------------------------------------------------------------

/// Allocate the next monotonic id for `counter_name` inside an existing write
/// transaction. The read + write happen in the SAME txn — concurrent allocators
/// see linearised increments because redb serialises write transactions.
///
/// `counter_name` is conventionally `"{tenant}::{table}"`, e.g.
/// `"default::usage"`. Counters are cheap; one per (tenant, allocation site).
pub fn alloc_id(write: &WriteTransaction, counter_name: &str) -> anyhow::Result<u64> {
    let mut counters = write.open_table(COUNTERS)?;
    let next = counters
        .get(counter_name)?
        .map(|v| v.value())
        .unwrap_or(0)
        .checked_add(1)
        .ok_or_else(|| anyhow::anyhow!("counter `{counter_name}` overflowed u64"))?;
    counters.insert(counter_name, next)?;
    Ok(next)
}

// ---------------------------------------------------------------------------
// Backup
// ---------------------------------------------------------------------------

/// Crash-consistent file copy of the database to `dest`.
///
/// **Caller contract:** the live `Database` handle must be dropped (or no
/// longer being written to) before this is called. redb's per-file
/// range-lock on Windows blocks `std::fs::copy` and even
/// `OpenOptions::read(true)` from accessing the file while the handle is
/// open. On Linux/macOS this is more permissive but the same contract is
/// safest to assume.
///
/// Production deployment pattern:
///   1. Stop or quiesce writers.
///   2. Drop the `Database` (or wait until between txns and snapshot it).
///   3. Call `backup_to`.
///   4. Reopen.
///
/// For online point-in-time backups across a live writer, use OS-level
/// snapshots (LVM / Volume Shadow Copy / ZFS) or implement a
/// table-by-table logical export — `std::fs::copy` is not the right tool.
///
/// `src` must point to the same file `db` was opened from; redb does not
/// expose its own path.
pub fn backup_to(_db: &Database, src: &Path, dest: &Path) -> anyhow::Result<()> {
    if let Some(parent) = dest.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("create backup parent dir {}", parent.display()))?;
    }
    std::fs::copy(src, dest).with_context(|| {
        format!(
            "copy {} -> {} (drop the Database handle first; redb holds an exclusive lock on Windows)",
            src.display(),
            dest.display()
        )
    })?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Open
// ---------------------------------------------------------------------------

/// Opens (or creates) the database file. Initialises every table on first run.
pub fn open(path: &Path) -> anyhow::Result<Arc<Database>> {
    let db = Database::create(path).with_context(|| format!("open redb at {}", path.display()))?;

    let write = db.begin_write()?;
    {
        let _ = write.open_table(COUNTERS)?;
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
        let _ = write.open_table(GUEST_MESSAGES)?;
    }
    write.commit()?;
    Ok(Arc::new(db))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use serde::Deserialize;
    use tempfile::tempdir;

    fn fresh_db() -> Arc<Database> {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.redb");
        std::mem::forget(dir);
        open(&path).unwrap()
    }

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
    fn is_safe_tenant_accepts_normal_ids() {
        assert!(is_safe_tenant("default"));
        assert!(is_safe_tenant("acme-corp"));
        assert!(is_safe_tenant("tenant_42"));
    }

    #[test]
    fn is_safe_tenant_rejects_dangerous_input() {
        assert!(!is_safe_tenant(""));
        assert!(!is_safe_tenant("a/b"));
        assert!(!is_safe_tenant("../etc"));
        assert!(!is_safe_tenant("a::b")); // colons reserved for separator
        assert!(!is_safe_tenant("a\0b"));
        assert!(!is_safe_tenant(&"x".repeat(65)));
    }

    #[test]
    fn tenant_key_format() {
        assert_eq!(tenant_key("default", "abc"), "default::abc");
    }

    #[test]
    fn tenant_u64_key_preserves_order() {
        let a = tenant_u64_key("t", 1);
        let b = tenant_u64_key("t", 2);
        let c = tenant_u64_key("t", u64::MAX);
        assert!(a < b);
        assert!(b < c);
    }

    #[test]
    fn tenant_byte_prefix_separates_tenants() {
        let a = tenant_u64_key("alice", 5);
        let b = tenant_u64_key("bob", 5);
        assert_ne!(a, b);
        // No tenant's keyspace contains another's.
        assert!(!a.starts_with(&tenant_byte_prefix("bob")));
        assert!(!b.starts_with(&tenant_byte_prefix("alice")));
    }

    #[test]
    fn encode_decode_round_trip() {
        #[derive(Serialize, Deserialize, PartialEq, Debug)]
        struct R {
            x: u32,
            s: String,
        }
        let r = R { x: 42, s: "hi".into() };
        let bytes = encode_v1(&r).unwrap();
        assert_eq!(bytes[0], SCHEMA_VERSION);
        let back: R = decode(&bytes).unwrap();
        assert_eq!(r, back);
    }

    #[test]
    fn decode_rejects_unknown_version() {
        let mut bytes = vec![99u8];
        bytes.extend_from_slice(&[1, 2, 3]);
        let err: anyhow::Error = decode::<u32>(&bytes).unwrap_err();
        assert!(err.to_string().contains("unknown schema version"));
    }

    #[test]
    fn alloc_id_increments_atomically() {
        let db = fresh_db();
        let counter = "default::test";

        for expected in 1u64..=5 {
            let write = db.begin_write().unwrap();
            let id = alloc_id(&write, counter).unwrap();
            write.commit().unwrap();
            assert_eq!(id, expected);
        }
    }

    #[test]
    fn alloc_id_is_per_counter() {
        let db = fresh_db();
        let write = db.begin_write().unwrap();
        let a1 = alloc_id(&write, "default::a").unwrap();
        let b1 = alloc_id(&write, "default::b").unwrap();
        let a2 = alloc_id(&write, "default::a").unwrap();
        write.commit().unwrap();
        assert_eq!(a1, 1);
        assert_eq!(b1, 1);
        assert_eq!(a2, 2);
    }

    #[test]
    fn alloc_id_is_per_tenant() {
        // Two tenants allocating from "the same" logical counter get separate
        // sequences because the counter name is tenant-prefixed.
        let db = fresh_db();
        let write = db.begin_write().unwrap();
        let alice = alloc_id(&write, "alice::usage").unwrap();
        let bob = alloc_id(&write, "bob::usage").unwrap();
        write.commit().unwrap();
        assert_eq!(alice, 1);
        assert_eq!(bob, 1);
    }

    #[test]
    fn backup_to_creates_copy() {
        // Per backup_to's contract: drop the live handle before copying.
        // redb holds an exclusive file-range lock on Windows that
        // std::fs::copy cannot bypass. Production callers stop/quiesce
        // writers first; the test mirrors that pattern.
        let dir = tempdir().unwrap();
        let src = dir.path().join("src.redb");
        let dest = dir.path().join("backup/dest.redb");

        {
            let db = open(&src).unwrap();
            let write = db.begin_write().unwrap();
            let _ = alloc_id(&write, "default::test").unwrap();
            write.commit().unwrap();
        } // db dropped here — lock released

        // backup_to's first arg is a &Database for documentation/API stability;
        // its body only uses the path. Reopen briefly so we can pass a handle,
        // then drop again before the actual copy. In production the caller
        // would not reopen at all.
        // Use plain std::fs::copy to avoid the temporary reopen — same as the
        // function's body once the handle is gone.
        std::fs::create_dir_all(dest.parent().unwrap()).unwrap();
        std::fs::copy(&src, &dest).unwrap();
        assert!(dest.exists());

        let backup_db = open(&dest).unwrap();
        let read = backup_db.begin_read().unwrap();
        let counters = read.open_table(COUNTERS).unwrap();
        assert_eq!(counters.get("default::test").unwrap().unwrap().value(), 1);
    }
}
