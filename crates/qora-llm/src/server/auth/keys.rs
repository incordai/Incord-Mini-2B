//! API key model. Lookup uses HMAC-SHA256 (O(1) index); Argon2 provides
//! defense-in-depth verification on the matched record.

use anyhow::Context;
use argon2::password_hash::{SaltString, rand_core::OsRng};
use argon2::{Argon2, PasswordHash, PasswordHasher, PasswordVerifier};
use chrono::Utc;
use hmac::{Hmac, Mac};
use redb::{Database, ReadableTable};
use serde::{Deserialize, Serialize};
use sha2::Sha256;
use std::sync::Arc;
use uuid::Uuid;

use crate::server::db::{ID_BY_HMAC, KEYS_BY_ID};

pub const KEY_PREFIX: &str = "qora_";
pub const HMAC_LEN: usize = 32;

type HmacSha256 = Hmac<Sha256>;

/// Persisted API key record (bincoded into the value bytes).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StoredKey {
    pub id: String,
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

/// Public API key (without the secret hash material).
#[derive(Debug, Clone)]
pub struct ApiKey {
    pub id: String,
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

const BINCODE_CFG: bincode::config::Configuration = bincode::config::standard();

/// Create a new API key. Admin-only entry point.
pub fn create_key(
    db: &Arc<Database>,
    server_secret: &[u8],
    user_id: &str,
    name: &str,
    rpm_limit: u32,
    tpm_limit: u32,
) -> anyhow::Result<NewKey> {
    let raw = generate_raw_key();
    let hmac = hmac_index(server_secret, &raw);
    let argon2_hash = hash_key_argon2(&raw)?;
    let id = Uuid::new_v4().to_string();
    let now = Utc::now().timestamp();

    let stored = StoredKey {
        id: id.clone(),
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

    let value = bincode::serde::encode_to_vec(&stored, BINCODE_CFG).context("bincode StoredKey")?;

    let write = db.begin_write()?;
    {
        let mut keys = write.open_table(KEYS_BY_ID)?;
        keys.insert(id.as_str(), value.as_slice())?;
        let mut hmac_idx = write.open_table(ID_BY_HMAC)?;
        hmac_idx.insert(hmac.as_slice(), id.as_str())?;
    }
    write.commit()?;

    Ok(NewKey {
        raw_key: raw,
        key: ApiKey::from(&stored),
    })
}

/// Look up a presented raw key. Returns Some(ApiKey) if active and verified.
pub fn lookup_by_raw(
    db: &Arc<Database>,
    server_secret: &[u8],
    raw: &str,
) -> anyhow::Result<Option<ApiKey>> {
    if !raw.starts_with(KEY_PREFIX) {
        return Ok(None);
    }
    let hmac = hmac_index(server_secret, raw);

    // Phase 1: O(1) HMAC -> id lookup
    let id = {
        let read = db.begin_read()?;
        let idx = read.open_table(ID_BY_HMAC)?;
        match idx.get(hmac.as_slice())? {
            Some(v) => v.value().to_string(),
            None => return Ok(None),
        }
    };

    // Phase 2: fetch StoredKey
    let stored: StoredKey = {
        let read = db.begin_read()?;
        let keys = read.open_table(KEYS_BY_ID)?;
        let bytes = match keys.get(id.as_str())? {
            Some(v) => v.value().to_vec(),
            None => return Ok(None),
        };
        let (decoded, _): (StoredKey, usize) =
            bincode::serde::decode_from_slice(&bytes, BINCODE_CFG)
                .context("bincode StoredKey decode")?;
        decoded
    };

    if !stored.is_active {
        return Ok(None);
    }

    // Phase 3: defense-in-depth Argon2 verify
    if !verify_argon2(raw, &stored.argon2_hash) {
        return Ok(None);
    }

    // Phase 4: touch last_used_at (best-effort write; ignore failure)
    let now = Utc::now().timestamp();
    let mut updated = stored.clone();
    updated.last_used_at = Some(now);
    if let Ok(value) = bincode::serde::encode_to_vec(&updated, BINCODE_CFG)
        && let Ok(write) = db.begin_write()
    {
        let _ = (|| -> anyhow::Result<()> {
            {
                let mut keys = write.open_table(KEYS_BY_ID)?;
                keys.insert(stored.id.as_str(), value.as_slice())?;
            }
            write.commit()?;
            Ok(())
        })();
    }

    Ok(Some(ApiKey::from(&updated)))
}

/// Mark a key inactive. Returns true if the key existed.
pub fn revoke_key(db: &Arc<Database>, id: &str) -> anyhow::Result<bool> {
    let write = db.begin_write()?;
    let existed = {
        let mut keys = write.open_table(KEYS_BY_ID)?;
        let current = keys.get(id)?.map(|v| v.value().to_vec());
        match current {
            Some(bytes) => {
                let (mut stored, _): (StoredKey, usize) =
                    bincode::serde::decode_from_slice(&bytes, BINCODE_CFG)?;
                stored.is_active = false;
                let value = bincode::serde::encode_to_vec(&stored, BINCODE_CFG)?;
                keys.insert(id, value.as_slice())?;
                true
            }
            None => false,
        }
    };
    write.commit()?;
    Ok(existed)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::server::db::open;
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
        let a = hmac_index(&s, "qora_a");
        let b = hmac_index(&s, "qora_a");
        let c = hmac_index(&s, "qora_b");
        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    #[test]
    fn argon2_round_trip() {
        let raw = generate_raw_key();
        let hash = hash_key_argon2(&raw).unwrap();
        assert!(verify_argon2(&raw, &hash));
        assert!(!verify_argon2("qora_wrong", &hash));
    }

    #[test]
    fn create_then_lookup() {
        let db = fresh_db();
        let s = secret();
        let nk = create_key(&db, &s, "user1", "test", 60, 60_000).unwrap();
        let found = lookup_by_raw(&db, &s, &nk.raw_key).unwrap();
        assert!(found.is_some());
        let f = found.unwrap();
        assert_eq!(f.user_id, "user1");
        assert_eq!(f.rpm_limit, 60);
    }

    #[test]
    fn lookup_rejects_unknown() {
        let db = fresh_db();
        let s = secret();
        let _ = create_key(&db, &s, "user1", "t", 60, 60_000).unwrap();
        let found = lookup_by_raw(&db, &s, "qora_unknown").unwrap();
        assert!(found.is_none());
    }

    #[test]
    fn lookup_rejects_wrong_secret() {
        let db = fresh_db();
        let s = secret();
        let nk = create_key(&db, &s, "user1", "t", 60, 60_000).unwrap();
        let other = b"different-secret".to_vec();
        let found = lookup_by_raw(&db, &other, &nk.raw_key).unwrap();
        assert!(found.is_none());
    }

    #[test]
    fn revoked_key_no_longer_found() {
        let db = fresh_db();
        let s = secret();
        let nk = create_key(&db, &s, "u", "t", 60, 60_000).unwrap();
        revoke_key(&db, &nk.key.id).unwrap();
        let found = lookup_by_raw(&db, &s, &nk.raw_key).unwrap();
        assert!(found.is_none());
    }
}
