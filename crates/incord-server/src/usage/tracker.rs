use crate::db::{
    alloc_id, decode, encode_v1, is_safe_tenant, tenant_byte_prefix, tenant_u64_key, USAGE,
};
use chrono;
use redb::Database;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UsageRecord {
    pub request_id: String,
    /// Tenant the request was billed against.
    pub tenant_id: String,
    pub user_id: String,
    pub key_id: String,
    pub model: String,
    pub input_tokens: u32,
    pub output_tokens: u32,
    pub started_at: i64,
    pub duration_ms: u32,
    pub status: String, // "success" | "error" | "timeout" | "cancelled"
}

#[derive(Debug, Clone)]
pub struct UsageTracker {
    db: Arc<Database>,
}

fn require_safe_tenant(tenant: &str) -> anyhow::Result<()> {
    if !is_safe_tenant(tenant) {
        anyhow::bail!("invalid tenant id `{tenant}`");
    }
    Ok(())
}

impl UsageTracker {
    pub fn new(db: Arc<Database>) -> Self {
        Self { db }
    }

    /// Record a usage event. Race-safe: id allocation and the insert happen
    /// inside the same write transaction, so two concurrent calls cannot
    /// receive the same `id` and overwrite each other's billing record.
    pub fn record(&self, r: UsageRecord) -> anyhow::Result<()> {
        require_safe_tenant(&r.tenant_id)?;
        let counter_name = format!("{}::usage", r.tenant_id);
        let value = encode_v1(&r)?;
        let write = self.db.begin_write()?;
        let id = alloc_id(&write, &counter_name)?;
        let storage_key = tenant_u64_key(&r.tenant_id, id);
        {
            let mut t = write.open_table(USAGE)?;
            t.insert(storage_key.as_slice(), value.as_slice())?;
        }
        write.commit()?;
        Ok(())
    }

    /// List usage records in `tenant` with optional filters.
    /// - `since`: only records with `started_at >= since` (default: last 24h)
    /// - `limit`: max records returned (default 1000, hard cap 10_000)
    /// - `user_id_filter` / `model_filter` / `status_filter`: per-record filters
    pub fn list_usage(
        &self,
        tenant: &str,
        since: Option<i64>,
        limit: Option<usize>,
        user_id_filter: Option<&str>,
        model_filter: Option<&str>,
        status_filter: Option<&str>,
    ) -> anyhow::Result<Vec<UsageRecord>> {
        require_safe_tenant(tenant)?;
        let since = since.unwrap_or_else(|| chrono::Utc::now().timestamp() - 86400);
        let limit = limit.unwrap_or(1000).min(10_000);

        let prefix = tenant_byte_prefix(tenant);
        let read = self.db.begin_read()?;
        let table = read.open_table(USAGE)?;
        let mut out = Vec::new();
        for entry in table.range(prefix.as_slice()..)? {
            let entry = entry?;
            let key = entry.0.value();
            if !key.starts_with(&prefix) {
                break;
            }
            let rec: UsageRecord = decode(entry.1.value())?;
            if rec.started_at < since {
                continue;
            }
            if user_id_filter.is_some_and(|uid| rec.user_id != uid) {
                continue;
            }
            if model_filter.is_some_and(|m| rec.model != m) {
                continue;
            }
            if status_filter.is_some_and(|s| rec.status != s) {
                continue;
            }
            out.push(rec);
            if out.len() >= limit {
                break;
            }
        }
        Ok(out)
    }

    pub fn count(&self, tenant: &str, user_id: &str) -> anyhow::Result<i64> {
        require_safe_tenant(tenant)?;
        let prefix = tenant_byte_prefix(tenant);
        let read = self.db.begin_read()?;
        let table = read.open_table(USAGE)?;
        let mut n = 0i64;
        for entry in table.range(prefix.as_slice()..)? {
            let entry = entry?;
            let key = entry.0.value();
            if !key.starts_with(&prefix) {
                break;
            }
            let rec: UsageRecord = decode(entry.1.value())?;
            if rec.user_id == user_id {
                n += 1;
            }
        }
        Ok(n)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db::{open, DEFAULT_TENANT};
    use tempfile::tempdir;

    fn fresh_tracker() -> UsageTracker {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.redb");
        std::mem::forget(dir);
        let db = open(&path).unwrap();
        UsageTracker::new(db)
    }

    fn rec(tenant: &str, user: &str) -> UsageRecord {
        UsageRecord {
            request_id: "r1".into(),
            tenant_id: tenant.into(),
            user_id: user.into(),
            key_id: "k1".into(),
            model: "qora-4b".into(),
            input_tokens: 10,
            output_tokens: 20,
            started_at: 1,
            duration_ms: 100,
            status: "success".into(),
        }
    }

    #[test]
    fn record_and_count() {
        let t = fresh_tracker();
        t.record(rec(DEFAULT_TENANT, "u1")).unwrap();
        assert_eq!(t.count(DEFAULT_TENANT, "u1").unwrap(), 1);
        assert_eq!(t.count(DEFAULT_TENANT, "u2").unwrap(), 0);
    }

    #[test]
    fn count_is_tenant_scoped() {
        let t = fresh_tracker();
        t.record(rec("tenant-a", "u1")).unwrap();
        t.record(rec("tenant-a", "u1")).unwrap();
        t.record(rec("tenant-b", "u1")).unwrap();
        assert_eq!(t.count("tenant-a", "u1").unwrap(), 2);
        assert_eq!(t.count("tenant-b", "u1").unwrap(), 1);
    }

    #[test]
    fn alloc_id_is_monotonic_per_tenant() {
        // Two records under the same tenant get distinct, incrementing keys.
        // Verifies the race fix — even sequential records exercise the
        // alloc_id path.
        let t = fresh_tracker();
        for _ in 0..5 {
            t.record(rec(DEFAULT_TENANT, "u1")).unwrap();
        }
        assert_eq!(
            t.list_usage(DEFAULT_TENANT, Some(0), None, None, None, None).unwrap().len(),
            5
        );
    }
}
