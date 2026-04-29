use std::sync::Arc;

use chrono::Utc;
use rand_core::OsRng;
use redb::Database;
use serde::{Deserialize, Serialize};

use crate::db::SESSIONS;

const BINCODE_CFG: bincode::config::Configuration = bincode::config::standard();

/// Default session lifetime: 30 days.
const SESSION_TTL_SECS: i64 = 30 * 24 * 3600;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Session {
    pub token: String,
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

pub fn create_session(db: &Arc<Database>, user_id: &str) -> anyhow::Result<Session> {
    let now = Utc::now().timestamp();
    let session = Session {
        token: generate_token(),
        user_id: user_id.to_string(),
        created_at: now,
        expires_at: now + SESSION_TTL_SECS,
    };
    let bytes = bincode::serde::encode_to_vec(&session, BINCODE_CFG)?;
    let write = db.begin_write()?;
    {
        let mut sessions = write.open_table(SESSIONS)?;
        sessions.insert(session.token.as_str(), bytes.as_slice())?;
    }
    write.commit()?;
    Ok(session)
}

pub fn lookup_session(db: &Arc<Database>, token: &str) -> anyhow::Result<Option<Session>> {
    let read = db.begin_read()?;
    let sessions = read.open_table(SESSIONS)?;
    let bytes = match sessions.get(token)? {
        Some(v) => v.value().to_vec(),
        None => return Ok(None),
    };
    let (session, _): (Session, usize) =
        bincode::serde::decode_from_slice(&bytes, BINCODE_CFG)?;
    if session.expires_at < Utc::now().timestamp() {
        return Ok(None);
    }
    Ok(Some(session))
}

pub fn revoke_session(db: &Arc<Database>, token: &str) -> anyhow::Result<bool> {
    let write = db.begin_write()?;
    let existed;
    {
        let mut sessions = write.open_table(SESSIONS)?;
        existed = sessions.remove(token)?.is_some();
    }
    write.commit()?;
    Ok(existed)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db::open;
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
        let sess = create_session(&db, "user-abc").unwrap();
        assert!(sess.token.starts_with("sess_"));

        let found = lookup_session(&db, &sess.token).unwrap();
        assert!(found.is_some());
        assert_eq!(found.unwrap().user_id, "user-abc");
    }

    #[test]
    fn revoke_removes_session() {
        let db = fresh_db();
        let sess = create_session(&db, "user-xyz").unwrap();
        let existed = revoke_session(&db, &sess.token).unwrap();
        assert!(existed);

        let found = lookup_session(&db, &sess.token).unwrap();
        assert!(found.is_none());
    }

    #[test]
    fn unknown_token_returns_none() {
        let db = fresh_db();
        let found = lookup_session(&db, "sess_nonexistent").unwrap();
        assert!(found.is_none());
    }
}
