use std::sync::Arc;

use chrono::Utc;
use redb::{Database, ReadableTable};
use serde::{Deserialize, Serialize};

use crate::db::GUEST_QUOTAS;

const BINCODE_CFG: bincode::config::Configuration = bincode::config::standard();

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct GuestQuota {
    pub guest_id: String,
    pub msg_count: u32,
    pub first_seen_at: i64,
    pub last_seen_at: i64,
}

pub fn check_and_increment(
    db: &Arc<Database>,
    guest_id: &str,
    max_messages: u32,
) -> anyhow::Result<Result<GuestQuota, GuestQuota>> {
    let now = Utc::now().timestamp();
    let write = db.begin_write()?;
    let current = {
        let table = write.open_table(GUEST_QUOTAS)?;
        match table.get(guest_id)? {
            Some(v) => {
                let (q, _): (GuestQuota, usize) =
                    bincode::serde::decode_from_slice(v.value(), BINCODE_CFG)?;
                Some(q)
            }
            None => None,
        }
    };

    let mut quota = current.unwrap_or_else(|| GuestQuota {
        guest_id: guest_id.to_string(),
        msg_count: 0,
        first_seen_at: now,
        last_seen_at: now,
    });

    if quota.msg_count >= max_messages {
        write.abort()?; // no change; quota exceeded
        return Ok(Err(quota));
    }

    quota.msg_count += 1;
    quota.last_seen_at = now;

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
            let (q, _): (GuestQuota, usize) =
                bincode::serde::decode_from_slice(v.value(), BINCODE_CFG)?;
            Some(q)
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
            let result = check_and_increment(&db, "guest-1", 3).unwrap();
            let quota = result.expect("should succeed under limit");
            assert_eq!(quota.msg_count, i);
        }
        // 4th should fail
        let result = check_and_increment(&db, "guest-1", 3).unwrap();
        assert!(result.is_err(), "4th message should exceed quota");
    }

    #[test]
    fn separate_guests_independent() {
        let db = fresh_db();
        check_and_increment(&db, "guest-a", 1).unwrap().unwrap();
        // guest-a is now at max=1; guest-b should still work
        let result = check_and_increment(&db, "guest-b", 1).unwrap();
        assert!(result.is_ok());
    }
}
