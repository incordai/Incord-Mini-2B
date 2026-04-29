use std::sync::Arc;

use chrono::Utc;
use redb::{Database, ReadableTable};
use serde::{Deserialize, Serialize};

use crate::db::GUEST_QUOTAS;

const BINCODE_CFG: bincode::config::Configuration = bincode::config::standard();

/// Guest quota record.
///
/// # Backwards compatibility
/// Fields were added incrementally. Old bincode records without newer fields
/// will fail to deserialize — we treat that as "no existing record" so the
/// guest starts fresh. This is acceptable for local-dev data; a production
/// deployment would run a migration script instead.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct GuestQuota {
    pub guest_id: String,
    pub msg_count: u32,
    pub first_seen_at: i64,
    pub last_seen_at: i64,
    /// Most recent IP seen for this guest.
    #[serde(default)]
    pub last_ip: Option<String>,
    /// Most recent user-agent seen.
    #[serde(default)]
    pub last_user_agent: Option<String>,
    /// ISO 3166-1 alpha-2 country code from the most recent request.
    #[serde(default)]
    pub last_country: Option<String>,
    /// City name from the most recent request.
    #[serde(default)]
    pub last_city: Option<String>,
}

pub fn check_and_increment(
    db: &Arc<Database>,
    guest_id: &str,
    max_messages: u32,
    client_ip: Option<&str>,
    user_agent: Option<&str>,
    country: Option<&str>,
    city: Option<&str>,
) -> anyhow::Result<Result<GuestQuota, GuestQuota>> {
    let now = Utc::now().timestamp();
    let write = db.begin_write()?;
    // On decode error (e.g. old schema without last_ip/last_user_agent fields),
    // treat the record as absent — the guest simply starts fresh.
    let current = {
        let table = write.open_table(GUEST_QUOTAS)?;
        match table.get(guest_id)? {
            Some(v) => bincode::serde::decode_from_slice::<GuestQuota, _>(v.value(), BINCODE_CFG)
                .ok()
                .map(|(q, _)| q),
            None => None,
        }
    };

    let mut quota = current.unwrap_or_else(|| GuestQuota {
        guest_id: guest_id.to_string(),
        msg_count: 0,
        first_seen_at: now,
        last_seen_at: now,
        last_ip: None,
        last_user_agent: None,
        last_country: None,
        last_city: None,
    });

    if quota.msg_count >= max_messages {
        write.abort()?; // no change; quota exceeded
        return Ok(Err(quota));
    }

    quota.msg_count += 1;
    quota.last_seen_at = now;
    // Update fields with the most recent seen value (keep old value if new is absent).
    quota.last_ip = client_ip.map(|s| s.to_string()).or(quota.last_ip);
    quota.last_user_agent = user_agent.map(|s| s.to_string()).or(quota.last_user_agent);
    quota.last_country = country.map(|s| s.to_string()).or(quota.last_country);
    quota.last_city = city.map(|s| s.to_string()).or(quota.last_city);

    let bytes = bincode::serde::encode_to_vec(&quota, BINCODE_CFG)?;
    {
        let mut table = write.open_table(GUEST_QUOTAS)?;
        table.insert(guest_id, bytes.as_slice())?;
    }
    write.commit()?;
    Ok(Ok(quota))
}

pub fn get_quota(db: &Arc<Database>, guest_id: &str) -> anyhow::Result<Option<GuestQuota>> {
    let read = db.begin_read()?;
    let table = read.open_table(GUEST_QUOTAS)?;
    Ok(match table.get(guest_id)? {
        Some(v) => {
            // Same backwards-compat tolerance: on decode error return None.
            bincode::serde::decode_from_slice::<GuestQuota, _>(v.value(), BINCODE_CFG)
                .ok()
                .map(|(q, _)| q)
        }
        None => None,
    })
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
    fn quota_increments_and_enforces() {
        let db = fresh_db();
        // max = 3
        for i in 1..=3 {
            let result =
                check_and_increment(&db, "guest-1", 3, None, None, None, None).unwrap();
            let quota = result.expect("should succeed under limit");
            assert_eq!(quota.msg_count, i);
        }
        // 4th should fail
        let result = check_and_increment(&db, "guest-1", 3, None, None, None, None).unwrap();
        assert!(result.is_err(), "4th message should exceed quota");
    }

    #[test]
    fn separate_guests_independent() {
        let db = fresh_db();
        check_and_increment(&db, "guest-a", 1, None, None, None, None)
            .unwrap()
            .unwrap();
        // guest-a is now at max=1; guest-b should still work
        let result = check_and_increment(&db, "guest-b", 1, None, None, None, None).unwrap();
        assert!(result.is_ok());
    }

    #[test]
    fn client_info_persisted() {
        let db = fresh_db();
        let result = check_and_increment(
            &db,
            "guest-ip",
            10,
            Some("1.2.3.4"),
            Some("TestBrowser/1.0"),
            Some("US"),
            Some("New York"),
        )
        .unwrap()
        .unwrap();
        assert_eq!(result.last_ip.as_deref(), Some("1.2.3.4"));
        assert_eq!(result.last_user_agent.as_deref(), Some("TestBrowser/1.0"));
        assert_eq!(result.last_country.as_deref(), Some("US"));
        assert_eq!(result.last_city.as_deref(), Some("New York"));

        // Second call updates IP + country; UA and city kept if absent
        let result2 =
            check_and_increment(&db, "guest-ip", 10, Some("5.6.7.8"), None, Some("GB"), None)
                .unwrap()
                .unwrap();
        assert_eq!(result2.last_ip.as_deref(), Some("5.6.7.8"));
        // UA kept from first call when new one is absent
        assert_eq!(result2.last_user_agent.as_deref(), Some("TestBrowser/1.0"));
        assert_eq!(result2.last_country.as_deref(), Some("GB"));
        // City kept from first call
        assert_eq!(result2.last_city.as_deref(), Some("New York"));
    }
}
