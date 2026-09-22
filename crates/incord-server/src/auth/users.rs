use std::sync::Arc;

use argon2::password_hash::{rand_core::OsRng, SaltString};
use argon2::{Argon2, PasswordHash, PasswordHasher, PasswordVerifier};
use chrono::Utc;
use redb::{Database, ReadableTable};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::db::{
    decode, encode_v1, is_safe_tenant, tenant_key, tenant_str_prefix, EMAIL_TO_USER, USERS,
};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StoredUser {
    pub id: String,
    /// Tenant the user belongs to. All keys for this user are scoped under it.
    pub tenant_id: String,
    pub email: String,
    pub password_hash: String, // Argon2
    pub tier: String,          // "free" | "paid"
    pub created_at: i64,
    pub last_login_at: Option<i64>,
}

#[derive(Debug, Clone)]
pub struct User {
    pub id: String,
    pub tenant_id: String,
    pub email: String,
    pub tier: String,
    pub created_at: i64,
    pub last_login_at: Option<i64>,
}

impl From<&StoredUser> for User {
    fn from(s: &StoredUser) -> Self {
        Self {
            id: s.id.clone(),
            tenant_id: s.tenant_id.clone(),
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

fn require_safe_tenant(tenant: &str) -> anyhow::Result<()> {
    if !is_safe_tenant(tenant) {
        anyhow::bail!("invalid tenant id `{tenant}` (alnum + `_-`, length 1..=64)");
    }
    Ok(())
}

pub fn register_user(
    db: &Arc<Database>,
    tenant: &str,
    email: &str,
    password: &str,
) -> anyhow::Result<User> {
    require_safe_tenant(tenant)?;
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
        tenant_id: tenant.to_string(),
        email: email.to_string(),
        password_hash: hash_password(password)?,
        tier: "free".to_string(),
        created_at: now,
        last_login_at: None,
    };

    let bytes = encode_v1(&stored)?;
    let user_key = tenant_key(tenant, &id);
    let email_key = tenant_key(tenant, email);

    let write = db.begin_write()?;
    {
        let mut email_idx = write.open_table(EMAIL_TO_USER)?;
        if email_idx.get(email_key.as_str())?.is_some() {
            anyhow::bail!("email already registered");
        }
        email_idx.insert(email_key.as_str(), user_key.as_str())?;
        let mut users = write.open_table(USERS)?;
        users.insert(user_key.as_str(), bytes.as_slice())?;
    }
    write.commit()?;
    Ok(User::from(&stored))
}

pub fn login(
    db: &Arc<Database>,
    tenant: &str,
    email: &str,
    password: &str,
) -> anyhow::Result<Option<User>> {
    require_safe_tenant(tenant)?;
    let email_key = tenant_key(tenant, email);

    let read = db.begin_read()?;
    let email_idx = read.open_table(EMAIL_TO_USER)?;
    let user_key = match email_idx.get(email_key.as_str())? {
        Some(v) => v.value().to_string(),
        None => return Ok(None),
    };
    let users = read.open_table(USERS)?;
    let bytes = match users.get(user_key.as_str())? {
        Some(v) => v.value().to_vec(),
        None => return Ok(None),
    };
    drop(users);
    drop(email_idx);
    drop(read);

    let mut stored: StoredUser = decode(&bytes)?;
    if !verify_password(password, &stored.password_hash) {
        return Ok(None);
    }

    let now = Utc::now().timestamp();
    stored.last_login_at = Some(now);
    let updated = encode_v1(&stored)?;
    let write = db.begin_write()?;
    {
        let mut users = write.open_table(USERS)?;
        users.insert(user_key.as_str(), updated.as_slice())?;
    }
    write.commit()?;

    Ok(Some(User::from(&stored)))
}

/// Enumerate every user in a tenant. Admin-only.
pub fn list_all_users(db: &Arc<Database>, tenant: &str) -> anyhow::Result<Vec<User>> {
    require_safe_tenant(tenant)?;
    let prefix = tenant_str_prefix(tenant);
    let read = db.begin_read()?;
    let users = read.open_table(USERS)?;
    let mut out = Vec::new();
    for entry in users.range(prefix.as_str()..)? {
        let entry = entry?;
        let key = entry.0.value();
        if !key.starts_with(&prefix) {
            break;
        }
        let stored: StoredUser = decode(entry.1.value())?;
        out.push(User::from(&stored));
    }
    out.sort_by(|a, b| b.created_at.cmp(&a.created_at));
    Ok(out)
}

pub fn get_user(
    db: &Arc<Database>,
    tenant: &str,
    user_id: &str,
) -> anyhow::Result<Option<User>> {
    require_safe_tenant(tenant)?;
    let user_key = tenant_key(tenant, user_id);
    let read = db.begin_read()?;
    let users = read.open_table(USERS)?;
    let bytes = match users.get(user_key.as_str())? {
        Some(v) => v.value().to_vec(),
        None => return Ok(None),
    };
    let stored: StoredUser = decode(&bytes)?;
    Ok(Some(User::from(&stored)))
}

/// Update a user's subscription tier. Validates the tier name is recognised
/// before writing. Returns the updated User if it existed, None if no such id.
///
/// This is the ONLY supported way to change a user's tier — registration
/// always writes "free" and there is no admin endpoint that bypasses this.
/// Future Stripe integration will call this from a webhook handler.
pub fn update_user_tier(
    db: &Arc<Database>,
    tenant: &str,
    user_id: &str,
    new_tier: &str,
) -> anyhow::Result<Option<User>> {
    require_safe_tenant(tenant)?;
    if !crate::auth::tiers::tier_exists(new_tier) {
        anyhow::bail!("unknown tier: {new_tier}");
    }

    let user_key = tenant_key(tenant, user_id);
    let read = db.begin_read()?;
    let users_r = read.open_table(USERS)?;
    let bytes = match users_r.get(user_key.as_str())? {
        Some(v) => v.value().to_vec(),
        None => return Ok(None),
    };
    drop(users_r);
    drop(read);

    let mut stored: StoredUser = decode(&bytes)?;
    stored.tier = new_tier.to_string();
    let updated = encode_v1(&stored)?;

    let write = db.begin_write()?;
    {
        let mut users_w = write.open_table(USERS)?;
        users_w.insert(user_key.as_str(), updated.as_slice())?;
    }
    write.commit()?;

    Ok(Some(User::from(&stored)))
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
    fn register_and_login() {
        let db = fresh_db();
        let user =
            register_user(&db, DEFAULT_TENANT, "test@example.com", "password123").unwrap();
        assert_eq!(user.email, "test@example.com");
        assert_eq!(user.tier, "free");
        assert_eq!(user.tenant_id, DEFAULT_TENANT);

        let logged_in =
            login(&db, DEFAULT_TENANT, "test@example.com", "password123").unwrap();
        assert!(logged_in.is_some());
        assert_eq!(logged_in.unwrap().id, user.id);
    }

    #[test]
    fn wrong_password_returns_none() {
        let db = fresh_db();
        register_user(&db, DEFAULT_TENANT, "x@x.com", "password123").unwrap();
        let result = login(&db, DEFAULT_TENANT, "x@x.com", "wrongpassword").unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn duplicate_email_fails_within_tenant() {
        let db = fresh_db();
        register_user(&db, DEFAULT_TENANT, "dup@x.com", "password123").unwrap();
        let err = register_user(&db, DEFAULT_TENANT, "dup@x.com", "password456");
        assert!(err.is_err());
    }

    #[test]
    fn same_email_allowed_across_tenants() {
        // Two tenants can both have a user named alice@example.com — that's
        // the foundational tenant isolation.
        let db = fresh_db();
        let a = register_user(&db, "tenant-a", "alice@example.com", "password123").unwrap();
        let b = register_user(&db, "tenant-b", "alice@example.com", "password123").unwrap();
        assert_ne!(a.id, b.id);
        assert_ne!(a.tenant_id, b.tenant_id);

        // Login is tenant-scoped — alice in tenant-a can't log in as tenant-b
        // if the password differs (here both happen to use the same password,
        // but the user IDs returned must differ).
        let la = login(&db, "tenant-a", "alice@example.com", "password123")
            .unwrap()
            .unwrap();
        let lb = login(&db, "tenant-b", "alice@example.com", "password123")
            .unwrap()
            .unwrap();
        assert_eq!(la.id, a.id);
        assert_eq!(lb.id, b.id);
    }

    #[test]
    fn list_all_users_is_tenant_scoped() {
        let db = fresh_db();
        register_user(&db, "tenant-a", "u1@x.com", "password123").unwrap();
        register_user(&db, "tenant-a", "u2@x.com", "password123").unwrap();
        register_user(&db, "tenant-b", "u3@x.com", "password123").unwrap();

        assert_eq!(list_all_users(&db, "tenant-a").unwrap().len(), 2);
        assert_eq!(list_all_users(&db, "tenant-b").unwrap().len(), 1);
    }

    #[test]
    fn short_password_rejected() {
        let db = fresh_db();
        let err = register_user(&db, DEFAULT_TENANT, "x@x.com", "short");
        assert!(err.is_err());
    }

    #[test]
    fn invalid_tenant_rejected() {
        let db = fresh_db();
        let err = register_user(&db, "../etc", "x@x.com", "password123");
        assert!(err.is_err());
    }
}
