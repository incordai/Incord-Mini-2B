//! API key model. Lookup uses HMAC-SHA256 (O(1) index); Argon2 provides
//! defense-in-depth verification on the matched record.

use argon2::password_hash::{rand_core::OsRng, SaltString};
use argon2::{Argon2, PasswordHash, PasswordHasher, PasswordVerifier};
use chrono::Utc;
use hmac::{Hmac, Mac};
use redb::{Database, ReadableTable};
use serde::{Deserialize, Serialize};
use sha2::Sha256;
use std::sync::Arc;
use uuid::Uuid;

use crate::db::{
    decode, encode_v1, is_safe_tenant, tenant_bytes_key, tenant_byte_prefix, tenant_key,
    tenant_str_prefix, ID_BY_HMAC, KEYS_BY_ID,
};

pub const KEY_PREFIX: &str = "incord_";
pub const HMAC_LEN: usize = 32;

type HmacSha256 = Hmac<Sha256>;

/// Persisted API key record. Stored under `tenant_key(tenant, key_id)`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StoredKey {
    pub id: String,
    /// Tenant this key belongs to. Inherited from the owning user.
    pub tenant_id: String,
    pub user_id: String,
    pub name: String,
    /// HMAC-SHA256(server_secret, raw_key). Used as primary lookup index.
    pub hmac: Vec<u8>,
    /// Argon2 hash of the raw key. Verified after HMAC index hit.
    pub argon2_hash: String,
    pub created_at: i64,
    pub last_used_at: Option<i64>,
    pub is_active: bool,
    pub rpm_limit: u32,
    pub tpm_limit: u32,
}

/// Public API key (no secret hash material).
#[derive(Debug, Clone)]
pub struct ApiKey {
    pub id: String,
    pub tenant_id: String,
    pub user_id: String,
    pub name: String,
    pub created_at: i64,
    pub last_used_at: Option<i64>,
    pub is_active: bool,
    pub rpm_limit: u32,
    pub tpm_limit: u32,
}

impl From<&StoredKey> for ApiKey {
    fn from(s: &StoredKey) -> Self {
        Self {
            id: s.id.clone(),
            tenant_id: s.tenant_id.clone(),
            user_id: s.user_id.clone(),
            name: s.name.clone(),
            created_at: s.created_at,
            last_used_at: s.last_used_at,
            is_active: s.is_active,
            rpm_limit: s.rpm_limit,
            tpm_limit: s.tpm_limit,
        }
    }
}

#[derive(Debug)]
pub struct NewKey {
    pub raw_key: String, // shown to admin once, never stored
    pub key: ApiKey,
}

pub fn generate_raw_key() -> String {
    use rand_core::RngCore;
    let mut bytes = [0u8; 32];
    OsRng.fill_bytes(&mut bytes);
    format!("{KEY_PREFIX}{}", hex::encode(bytes))
}

pub fn hmac_index(server_secret: &[u8], raw: &str) -> Vec<u8> {
    let mut mac = HmacSha256::new_from_slice(server_secret).expect("HMAC accepts any key length");
    mac.update(raw.as_bytes());
    mac.finalize().into_bytes().to_vec()
}

pub fn hash_key_argon2(raw: &str) -> anyhow::Result<String> {
    let salt = SaltString::generate(&mut OsRng);
    let hash = Argon2::default()
        .hash_password(raw.as_bytes(), &salt)
        .map_err(|e| anyhow::anyhow!("argon2: {e}"))?
        .to_string();
    Ok(hash)
}

pub fn verify_argon2(raw: &str, hash: &str) -> bool {
    PasswordHash::new(hash)
        .ok()
        .map(|p| {
            Argon2::default()
                .verify_password(raw.as_bytes(), &p)
                .is_ok()
        })
        .unwrap_or(false)
}

fn require_safe_tenant(tenant: &str) -> anyhow::Result<()> {
    if !is_safe_tenant(tenant) {
        anyhow::bail!("invalid tenant id `{tenant}`");
    }
    Ok(())
}

/// Create a new API key under `tenant`, owned by `user_id`. Admin-only.
pub fn create_key(
    db: &Arc<Database>,
    server_secret: &[u8],
    tenant: &str,
    user_id: &str,
    name: &str,
    rpm_limit: u32,
    tpm_limit: u32,
) -> anyhow::Result<NewKey> {
    require_safe_tenant(tenant)?;
    let raw = generate_raw_key();
    let hmac = hmac_index(server_secret, &raw);
    let argon2_hash = hash_key_argon2(&raw)?;
    let id = Uuid::new_v4().to_string();
    let now = Utc::now().timestamp();

    let stored = StoredKey {
        id: id.clone(),
        tenant_id: tenant.to_string(),
        user_id: user_id.to_string(),
        name: name.to_string(),
        hmac: hmac.clone(),
        argon2_hash,
        created_at: now,
        last_used_at: None,
        is_active: true,
        rpm_limit,
        tpm_limit,
    };

    let value = encode_v1(&stored)?;
    let key_storage_key = tenant_key(tenant, &id);
    let hmac_storage_key = tenant_bytes_key(tenant, &hmac);

    let write = db.begin_write()?;
    {
        let mut keys = write.open_table(KEYS_BY_ID)?;
        keys.insert(key_storage_key.as_str(), value.as_slice())?;
        let mut hmac_idx = write.open_table(ID_BY_HMAC)?;
        hmac_idx.insert(hmac_storage_key.as_slice(), key_storage_key.as_str())?;
    }
    write.commit()?;

    Ok(NewKey {
        raw_key: raw,
        key: ApiKey::from(&stored),
    })
}

/// Look up a presented raw key within `tenant`. Returns Some(ApiKey) if
/// active and verified. O(1) — the HMAC index is keyed by `tenant\0HMAC`.
///
/// The caller must already know the tenant (from the `X-Tenant` header,
/// session context, or [`DEFAULT_TENANT`] for single-tenant deployments).
pub fn lookup_by_raw(
    db: &Arc<Database>,
    server_secret: &[u8],
    tenant: &str,
    raw: &str,
) -> anyhow::Result<Option<ApiKey>> {
    require_safe_tenant(tenant)?;
    if !raw.starts_with(KEY_PREFIX) {
        return Ok(None);
    }
    let hmac = hmac_index(server_secret, raw);
    let hmac_storage_key = tenant_bytes_key(tenant, &hmac);

    let storage_key = {
        let read = db.begin_read()?;
        let idx = read.open_table(ID_BY_HMAC)?;
        match idx.get(hmac_storage_key.as_slice())? {
            Some(v) => v.value().to_string(),
            None => return Ok(None),
        }
    };

    let stored: StoredKey = {
        let read = db.begin_read()?;
        let keys = read.open_table(KEYS_BY_ID)?;
        let bytes = match keys.get(storage_key.as_str())? {
            Some(v) => v.value().to_vec(),
            None => return Ok(None),
        };
        decode(&bytes)?
    };

    if !stored.is_active {
        return Ok(None);
    }

    if !verify_argon2(raw, &stored.argon2_hash) {
        return Ok(None);
    }

    // Best-effort touch of last_used_at.
    let now = Utc::now().timestamp();
    let mut updated = stored.clone();
    updated.last_used_at = Some(now);
    if let Ok(value) = encode_v1(&updated)
        && let Ok(write) = db.begin_write()
    {
        let _ = (|| -> anyhow::Result<()> {
            {
                let mut keys = write.open_table(KEYS_BY_ID)?;
                keys.insert(storage_key.as_str(), value.as_slice())?;
            }
            write.commit()?;
            Ok(())
        })();
    }

    Ok(Some(ApiKey::from(&updated)))
}

/// List every key in a tenant, public form. Sorted by created_at descending.
pub fn list_all(db: &Arc<Database>, tenant: &str) -> anyhow::Result<Vec<ApiKey>> {
    require_safe_tenant(tenant)?;
    let prefix = tenant_str_prefix(tenant);
    let read = db.begin_read()?;
    let keys = read.open_table(KEYS_BY_ID)?;
    let mut out = Vec::new();
    for entry in keys.range(prefix.as_str()..)? {
        let entry = entry?;
        let key_str = entry.0.value();
        if !key_str.starts_with(&prefix) {
            break;
        }
        let stored: StoredKey = decode(entry.1.value())?;
        out.push(ApiKey::from(&stored));
    }
    out.sort_by(|a, b| b.created_at.cmp(&a.created_at));
    Ok(out)
}

/// Keys owned by `user_id` within a tenant.
pub fn list_for_user(
    db: &Arc<Database>,
    tenant: &str,
    user_id: &str,
) -> anyhow::Result<Vec<ApiKey>> {
    let mut all = list_all(db, tenant)?;
    all.retain(|k| k.user_id == user_id);
    Ok(all)
}

/// Get a single key by id within a tenant (no raw-secret verification —
/// used for ownership checks before revoke).
pub fn get_by_id(
    db: &Arc<Database>,
    tenant: &str,
    id: &str,
) -> anyhow::Result<Option<ApiKey>> {
    require_safe_tenant(tenant)?;
    let storage_key = tenant_key(tenant, id);
    let read = db.begin_read()?;
    let keys = read.open_table(KEYS_BY_ID)?;
    let bytes = match keys.get(storage_key.as_str())? {
        Some(v) => v.value().to_vec(),
        None => return Ok(None),
    };
    let stored: StoredKey = decode(&bytes)?;
    Ok(Some(ApiKey::from(&stored)))
}

/// Mark a key inactive. Returns true if the key existed.
pub fn revoke_key(db: &Arc<Database>, tenant: &str, id: &str) -> anyhow::Result<bool> {
    require_safe_tenant(tenant)?;
    let storage_key = tenant_key(tenant, id);
    let write = db.begin_write()?;
    let existed = {
        let mut keys = write.open_table(KEYS_BY_ID)?;
        let current = keys.get(storage_key.as_str())?.map(|v| v.value().to_vec());
        match current {
            Some(bytes) => {
                let mut stored: StoredKey = decode(&bytes)?;
                stored.is_active = false;
                let value = encode_v1(&stored)?;
                keys.insert(storage_key.as_str(), value.as_slice())?;
                true
            }
            None => false,
        }
    };
    write.commit()?;
    Ok(existed)
}

// Suppress unused-import warning when no callers in this binary touch this.
#[allow(dead_code)]
fn _byte_prefix_unused_marker(t: &str) -> Vec<u8> {
    tenant_byte_prefix(t)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db::{open, DEFAULT_TENANT};
    use tempfile::tempdir;

    fn fresh_db() -> Arc<Database> {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.redb");
        std::mem::forget(dir);
        open(&path).unwrap()
    }

    fn secret() -> Vec<u8> {
        b"test-secret-do-not-use-in-prod".to_vec()
    }

    #[test]
    fn raw_key_format() {
        let raw = generate_raw_key();
        assert!(raw.starts_with(KEY_PREFIX));
        assert_eq!(raw.len(), KEY_PREFIX.len() + 64);
    }

    #[test]
    fn hmac_is_deterministic_and_distinct() {
        let s = secret();
        let a = hmac_index(&s, "incord_a");
        let b = hmac_index(&s, "incord_a");
        let c = hmac_index(&s, "incord_b");
        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    #[test]
    fn argon2_round_trip() {
        let raw = generate_raw_key();
        let hash = hash_key_argon2(&raw).unwrap();
        assert!(verify_argon2(&raw, &hash));
        assert!(!verify_argon2("incord_wrong", &hash));
    }

    #[test]
    fn create_then_lookup() {
        let db = fresh_db();
        let s = secret();
        let nk = create_key(&db, &s, DEFAULT_TENANT, "user1", "test", 60, 60_000).unwrap();
        let found = lookup_by_raw(&db, &s, DEFAULT_TENANT, &nk.raw_key).unwrap();
        assert!(found.is_some());
        let f = found.unwrap();
        assert_eq!(f.user_id, "user1");
        assert_eq!(f.tenant_id, DEFAULT_TENANT);
        assert_eq!(f.rpm_limit, 60);
    }

    #[test]
    fn lookup_is_tenant_scoped() {
        let db = fresh_db();
        let s = secret();
        let a = create_key(&db, &s, "tenant-a", "alice", "n", 60, 60_000).unwrap();
        let b = create_key(&db, &s, "tenant-b", "alice", "n", 60, 60_000).unwrap();
        assert_ne!(a.key.id, b.key.id);

        // A's raw key is only valid under tenant-a; same string against
        // tenant-b returns None.
        assert!(lookup_by_raw(&db, &s, "tenant-a", &a.raw_key).unwrap().is_some());
        assert!(lookup_by_raw(&db, &s, "tenant-b", &a.raw_key).unwrap().is_none());
        assert!(lookup_by_raw(&db, &s, "tenant-b", &b.raw_key).unwrap().is_some());
    }

    #[test]
    fn lookup_rejects_unknown() {
        let db = fresh_db();
        let s = secret();
        let _ = create_key(&db, &s, DEFAULT_TENANT, "user1", "t", 60, 60_000).unwrap();
        let found = lookup_by_raw(&db, &s, DEFAULT_TENANT, "incord_unknown").unwrap();
        assert!(found.is_none());
    }

    #[test]
    fn lookup_rejects_wrong_secret() {
        let db = fresh_db();
        let s = secret();
        let nk = create_key(&db, &s, DEFAULT_TENANT, "user1", "t", 60, 60_000).unwrap();
        let other = b"different-secret".to_vec();
        let found = lookup_by_raw(&db, &other, DEFAULT_TENANT, &nk.raw_key).unwrap();
        assert!(found.is_none());
    }

    #[test]
    fn revoked_key_no_longer_found() {
        let db = fresh_db();
        let s = secret();
        let nk = create_key(&db, &s, DEFAULT_TENANT, "u", "t", 60, 60_000).unwrap();
        revoke_key(&db, DEFAULT_TENANT, &nk.key.id).unwrap();
        let found = lookup_by_raw(&db, &s, DEFAULT_TENANT, &nk.raw_key).unwrap();
        assert!(found.is_none());
    }

    #[test]
    fn list_all_is_tenant_scoped() {
        let db = fresh_db();
        let s = secret();
        create_key(&db, &s, "tenant-a", "u", "k1", 60, 60_000).unwrap();
        create_key(&db, &s, "tenant-a", "u", "k2", 60, 60_000).unwrap();
        create_key(&db, &s, "tenant-b", "u", "k3", 60, 60_000).unwrap();
        assert_eq!(list_all(&db, "tenant-a").unwrap().len(), 2);
        assert_eq!(list_all(&db, "tenant-b").unwrap().len(), 1);
    }
}
