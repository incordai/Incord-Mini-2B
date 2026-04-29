use std::sync::Arc;

use argon2::{Argon2, PasswordHash, PasswordHasher, PasswordVerifier};
use argon2::password_hash::{rand_core::OsRng, SaltString};
use chrono::Utc;
use redb::{Database, ReadableTable};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::db::{EMAIL_TO_USER, USERS};

const BINCODE_CFG: bincode::config::Configuration = bincode::config::standard();

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StoredUser {
    pub id: String,
    pub email: String,
    pub password_hash: String, // Argon2
    pub tier: String,          // "free" | "paid"
    pub created_at: i64,
    pub last_login_at: Option<i64>,
}

#[derive(Debug, Clone)]
pub struct User {
    pub id: String,
    pub email: String,
    pub tier: String,
    pub created_at: i64,
    pub last_login_at: Option<i64>,
}

impl From<&StoredUser> for User {
    fn from(s: &StoredUser) -> Self {
        Self {
            id: s.id.clone(),
            email: s.email.clone(),
            tier: s.tier.clone(),
            created_at: s.created_at,
            last_login_at: s.last_login_at,
        }
    }
}

fn hash_password(password: &str) -> anyhow::Result<String> {
    let salt = SaltString::generate(&mut OsRng);
    Argon2::default()
        .hash_password(password.as_bytes(), &salt)
        .map(|h| h.to_string())
        .map_err(|e| anyhow::anyhow!("argon2 hash: {e}"))
}

fn verify_password(password: &str, hash: &str) -> bool {
    PasswordHash::new(hash)
        .ok()
        .map(|p| Argon2::default().verify_password(password.as_bytes(), &p).is_ok())
        .unwrap_or(false)
}

pub fn register_user(db: &Arc<Database>, email: &str, password: &str) -> anyhow::Result<User> {
    if !email.contains('@') || email.len() < 3 {
        anyhow::bail!("invalid email");
    }
    if password.len() < 8 {
        anyhow::bail!("password must be at least 8 characters");
    }

    let id = Uuid::new_v4().to_string();
    let now = Utc::now().timestamp();
    let stored = StoredUser {
        id: id.clone(),
        email: email.to_string(),
        password_hash: hash_password(password)?,
        tier: "free".to_string(),
        created_at: now,
        last_login_at: None,
    };

    let bytes = bincode::serde::encode_to_vec(&stored, BINCODE_CFG)?;

    let write = db.begin_write()?;
    {
        let mut email_idx = write.open_table(EMAIL_TO_USER)?;
        if email_idx.get(email)?.is_some() {
            anyhow::bail!("email already registered");
        }
        email_idx.insert(email, id.as_str())?;
        let mut users = write.open_table(USERS)?;
        users.insert(id.as_str(), bytes.as_slice())?;
    }
    write.commit()?;
    Ok(User::from(&stored))
}

pub fn login(db: &Arc<Database>, email: &str, password: &str) -> anyhow::Result<Option<User>> {
    let read = db.begin_read()?;
    let email_idx = read.open_table(EMAIL_TO_USER)?;
    let user_id = match email_idx.get(email)? {
        Some(v) => v.value().to_string(),
        None => return Ok(None),
    };
    let users = read.open_table(USERS)?;
    let bytes = match users.get(user_id.as_str())? {
        Some(v) => v.value().to_vec(),
        None => return Ok(None),
    };
    drop(users);
    drop(email_idx);
    drop(read);

    let (mut stored, _): (StoredUser, usize) =
        bincode::serde::decode_from_slice(&bytes, BINCODE_CFG)?;

    if !verify_password(password, &stored.password_hash) {
        return Ok(None);
    }

    // Touch last_login_at
    let now = Utc::now().timestamp();
    stored.last_login_at = Some(now);
    let updated = bincode::serde::encode_to_vec(&stored, BINCODE_CFG)?;
    let write = db.begin_write()?;
    {
        let mut users = write.open_table(USERS)?;
        users.insert(stored.id.as_str(), updated.as_slice())?;
    }
    write.commit()?;

    Ok(Some(User::from(&stored)))
}

pub fn get_user(db: &Arc<Database>, user_id: &str) -> anyhow::Result<Option<User>> {
    let read = db.begin_read()?;
    let users = read.open_table(USERS)?;
    let bytes = match users.get(user_id)? {
        Some(v) => v.value().to_vec(),
        None => return Ok(None),
    };
    let (stored, _): (StoredUser, usize) =
        bincode::serde::decode_from_slice(&bytes, BINCODE_CFG)?;
    Ok(Some(User::from(&stored)))
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
    fn register_and_login() {
        let db = fresh_db();
        let user = register_user(&db, "test@example.com", "password123").unwrap();
        assert_eq!(user.email, "test@example.com");
        assert_eq!(user.tier, "free");

        let logged_in = login(&db, "test@example.com", "password123").unwrap();
        assert!(logged_in.is_some());
        let logged_in = logged_in.unwrap();
        assert_eq!(logged_in.id, user.id);
    }

    #[test]
    fn wrong_password_returns_none() {
        let db = fresh_db();
        register_user(&db, "x@x.com", "password123").unwrap();
        let result = login(&db, "x@x.com", "wrongpassword").unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn duplicate_email_fails() {
        let db = fresh_db();
        register_user(&db, "dup@x.com", "password123").unwrap();
        let err = register_user(&db, "dup@x.com", "password456");
        assert!(err.is_err());
    }

    #[test]
    fn short_password_rejected() {
        let db = fresh_db();
        let err = register_user(&db, "x@x.com", "short");
        assert!(err.is_err());
    }
}
