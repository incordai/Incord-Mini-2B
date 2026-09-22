use std::sync::Arc;

use chrono::Utc;
use redb::{Database, ReadableTable};
use serde::{Deserialize, Serialize};

use crate::db::{decode, encode_v1, is_safe_tenant, tenant_key, GUEST_QUOTAS};

/// Guest quota record. Stored under `tenant::guest_id`.
///
/// Schema-versioned via [`crate::db::encode_v1`] / [`crate::db::decode`].
/// Field additions can bump the version byte without breaking older records.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct GuestQuota {
    pub guest_id: String,
    /// Tenant this guest belongs to.
    pub tenant_id: String,
    pub msg_count: u32,
    pub first_seen_at: i64,
    pub last_seen_at: i64,
    pub last_ip: Option<String>,
    pub last_user_agent: Option<String>,
    pub last_country: Option<String>,
    pub last_city: Option<String>,
    pub last_timezone: Option<String>,
}

fn require_safe_tenant(tenant: &str) -> anyhow::Result<()> {
    if !is_safe_tenant(tenant) {
        anyhow::bail!("invalid tenant id `{tenant}`");
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn check_and_increment(
    db: &Arc<Database>,
    tenant: &str,
    guest_id: &str,
    max_messages: u32,
    client_ip: Option<&str>,
    user_agent: Option<&str>,
    country: Option<&str>,
    city: Option<&str>,
    timezone: Option<&str>,
) -> anyhow::Result<Result<GuestQuota, GuestQuota>> {
    require_safe_tenant(tenant)?;
    let now = Utc::now().timestamp();
    let storage_key = tenant_key(tenant, guest_id);

    let write = db.begin_write()?;

    // On decode error treat as absent so the guest starts fresh; this keeps
    // local-dev tolerant if a record was written by an older binary.
    let current = {
        let table = write.open_table(GUEST_QUOTAS)?;
        match table.get(storage_key.as_str())? {
            Some(v) => decode::<GuestQuota>(v.value()).ok(),
            None => None,
        }
    };

    let mut quota = current.unwrap_or_else(|| GuestQuota {
        guest_id: guest_id.to_string(),
        tenant_id: tenant.to_string(),
        msg_count: 0,
        first_seen_at: now,
        last_seen_at: now,
        last_ip: None,
        last_user_agent: None,
        last_country: None,
        last_city: None,
        last_timezone: None,
    });

    if quota.msg_count >= max_messages {
        write.abort()?;
        return Ok(Err(quota));
    }

    quota.msg_count += 1;
    quota.last_seen_at = now;
    quota.last_ip = client_ip.map(|s| s.to_string()).or(quota.last_ip);
    quota.last_user_agent = user_agent.map(|s| s.to_string()).or(quota.last_user_agent);
    quota.last_country = country.map(|s| s.to_string()).or(quota.last_country);
    quota.last_city = city.map(|s| s.to_string()).or(quota.last_city);
    quota.last_timezone = timezone.map(|s| s.to_string()).or(quota.last_timezone);

    let bytes = encode_v1(&quota)?;
    {
        let mut table = write.open_table(GUEST_QUOTAS)?;
        table.insert(storage_key.as_str(), bytes.as_slice())?;
    }
    write.commit()?;
    Ok(Ok(quota))
}

pub fn get_quota(
    db: &Arc<Database>,
    tenant: &str,
    guest_id: &str,
) -> anyhow::Result<Option<GuestQuota>> {
    require_safe_tenant(tenant)?;
    let storage_key = tenant_key(tenant, guest_id);
    let read = db.begin_read()?;
    let table = read.open_table(GUEST_QUOTAS)?;
    Ok(match table.get(storage_key.as_str())? {
        Some(v) => decode::<GuestQuota>(v.value()).ok(),
        None => None,
    })
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
    fn quota_increments_and_enforces() {
        let db = fresh_db();
        for i in 1..=3 {
            let result = check_and_increment(
                &db, DEFAULT_TENANT, "guest-1", 3, None, None, None, None, None,
            )
            .unwrap();
            let quota = result.expect("should succeed under limit");
            assert_eq!(quota.msg_count, i);
            assert_eq!(quota.tenant_id, DEFAULT_TENANT);
        }
        let result = check_and_increment(
            &db, DEFAULT_TENANT, "guest-1", 3, None, None, None, None, None,
        )
        .unwrap();
        assert!(result.is_err(), "4th message should exceed quota");
    }

    #[test]
    fn separate_tenants_have_independent_quotas() {
        let db = fresh_db();
        // Same guest-id under two tenants — quotas don't share.
        check_and_increment(
            &db, "tenant-a", "guest-x", 1, None, None, None, None, None,
        )
        .unwrap()
        .unwrap();
        // tenant-a is now at max=1; tenant-b should still work for "the same" guest_id.
        let result = check_and_increment(
            &db, "tenant-b", "guest-x", 1, None, None, None, None, None,
        )
        .unwrap();
        assert!(result.is_ok());
    }

    #[test]
    fn separate_guests_independent() {
        let db = fresh_db();
        check_and_increment(
            &db, DEFAULT_TENANT, "guest-a", 1, None, None, None, None, None,
        )
        .unwrap()
        .unwrap();
        let result = check_and_increment(
            &db, DEFAULT_TENANT, "guest-b", 1, None, None, None, None, None,
        )
        .unwrap();
        assert!(result.is_ok());
    }

    #[test]
    fn client_info_persisted() {
        let db = fresh_db();
        let result = check_and_increment(
            &db,
            DEFAULT_TENANT,
            "guest-ip",
            10,
            Some("1.2.3.4"),
            Some("TestBrowser/1.0"),
            Some("US"),
            Some("New York"),
            Some("America/New_York"),
        )
        .unwrap()
        .unwrap();
        assert_eq!(result.last_ip.as_deref(), Some("1.2.3.4"));
        assert_eq!(result.last_user_agent.as_deref(), Some("TestBrowser/1.0"));
        assert_eq!(result.last_country.as_deref(), Some("US"));
        assert_eq!(result.last_city.as_deref(), Some("New York"));
        assert_eq!(result.last_timezone.as_deref(), Some("America/New_York"));

        let result2 = check_and_increment(
            &db,
            DEFAULT_TENANT,
            "guest-ip",
            10,
            Some("5.6.7.8"),
            None,
            Some("GB"),
            None,
            None,
        )
        .unwrap()
        .unwrap();
        assert_eq!(result2.last_ip.as_deref(), Some("5.6.7.8"));
        assert_eq!(result2.last_user_agent.as_deref(), Some("TestBrowser/1.0"));
        assert_eq!(result2.last_country.as_deref(), Some("GB"));
        assert_eq!(result2.last_city.as_deref(), Some("New York"));
        assert_eq!(result2.last_timezone.as_deref(), Some("America/New_York"));
    }
}
