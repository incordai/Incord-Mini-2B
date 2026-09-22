use std::sync::Arc;

use chrono::Utc;
use rand_core::OsRng;
use redb::Database;
use serde::{Deserialize, Serialize};

use crate::db::{decode, encode_v1, is_safe_tenant, tenant_key, tenant_str_prefix, SESSIONS};

const SESSION_TTL_SECS: i64 = 30 * 24 * 3600;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Session {
    pub token: String,
    /// Tenant this session belongs to. Inherited from the user.
    pub tenant_id: String,
    pub user_id: String,
    pub created_at: i64,
    pub expires_at: i64,
}

pub fn generate_token() -> String {
    use rand_core::RngCore;
    let mut bytes = [0u8; 32];
    OsRng.fill_bytes(&mut bytes);
    format!("sess_{}", hex::encode(bytes))
}

fn require_safe_tenant(tenant: &str) -> anyhow::Result<()> {
    if !is_safe_tenant(tenant) {
        anyhow::bail!("invalid tenant id `{tenant}`");
    }
    Ok(())
}

pub fn create_session(
    db: &Arc<Database>,
    tenant: &str,
    user_id: &str,
) -> anyhow::Result<Session> {
    require_safe_tenant(tenant)?;
    let now = Utc::now().timestamp();
    let session = Session {
        token: generate_token(),
        tenant_id: tenant.to_string(),
        user_id: user_id.to_string(),
        created_at: now,
        expires_at: now + SESSION_TTL_SECS,
    };
    let bytes = encode_v1(&session)?;
    let storage_key = tenant_key(tenant, &session.token);
    let write = db.begin_write()?;
    {
        let mut sessions = write.open_table(SESSIONS)?;
        sessions.insert(storage_key.as_str(), bytes.as_slice())?;
    }
    write.commit()?;
    Ok(session)
}

/// Look up a session by token within `tenant`. O(1).
pub fn lookup_session(
    db: &Arc<Database>,
    tenant: &str,
    token: &str,
) -> anyhow::Result<Option<Session>> {
    require_safe_tenant(tenant)?;
    let storage_key = tenant_key(tenant, token);
    let read = db.begin_read()?;
    let sessions = read.open_table(SESSIONS)?;
    let bytes = match sessions.get(storage_key.as_str())? {
        Some(v) => v.value().to_vec(),
        None => return Ok(None),
    };
    let session: Session = decode(&bytes)?;
    if session.expires_at < Utc::now().timestamp() {
        return Ok(None);
    }
    Ok(Some(session))
}

pub fn revoke_session(
    db: &Arc<Database>,
    tenant: &str,
    token: &str,
) -> anyhow::Result<bool> {
    require_safe_tenant(tenant)?;
    let storage_key = tenant_key(tenant, token);
    let write = db.begin_write()?;
    let existed;
    {
        let mut sessions = write.open_table(SESSIONS)?;
        existed = sessions.remove(storage_key.as_str())?.is_some();
    }
    write.commit()?;
    Ok(existed)
}

/// List all sessions in a tenant (admin view). O(tenant size).
pub fn list_for_tenant(db: &Arc<Database>, tenant: &str) -> anyhow::Result<Vec<Session>> {
    require_safe_tenant(tenant)?;
    let prefix = tenant_str_prefix(tenant);
    let read = db.begin_read()?;
    let sessions = read.open_table(SESSIONS)?;
    let mut out = Vec::new();
    for entry in sessions.range(prefix.as_str()..)? {
        let entry = entry?;
        let key = entry.0.value();
        if !key.starts_with(&prefix) {
            break;
        }
        let session: Session = decode(entry.1.value())?;
        out.push(session);
    }
    Ok(out)
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

    #[test]
    fn create_and_lookup() {
        let db = fresh_db();
        let sess = create_session(&db, DEFAULT_TENANT, "user-abc").unwrap();
        assert!(sess.token.starts_with("sess_"));
        assert_eq!(sess.tenant_id, DEFAULT_TENANT);

        let found = lookup_session(&db, DEFAULT_TENANT, &sess.token).unwrap();
        assert!(found.is_some());
        assert_eq!(found.unwrap().user_id, "user-abc");
    }

    #[test]
    fn lookup_is_tenant_scoped() {
        let db = fresh_db();
        let a = create_session(&db, "tenant-a", "alice").unwrap();
        // Looking up under the wrong tenant returns None.
        assert!(lookup_session(&db, "tenant-b", &a.token).unwrap().is_none());
        assert!(lookup_session(&db, "tenant-a", &a.token).unwrap().is_some());
    }

    #[test]
    fn revoke_removes_session() {
        let db = fresh_db();
        let sess = create_session(&db, DEFAULT_TENANT, "user-xyz").unwrap();
        let existed = revoke_session(&db, DEFAULT_TENANT, &sess.token).unwrap();
        assert!(existed);

        let found = lookup_session(&db, DEFAULT_TENANT, &sess.token).unwrap();
        assert!(found.is_none());
    }

    #[test]
    fn unknown_token_returns_none() {
        let db = fresh_db();
        let found = lookup_session(&db, DEFAULT_TENANT, "sess_nonexistent").unwrap();
        assert!(found.is_none());
    }
}
