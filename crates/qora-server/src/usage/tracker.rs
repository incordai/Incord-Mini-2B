use crate::db::{USAGE, next_usage_id};
use anyhow::Context;
use chrono;
use redb::{Database, ReadableTable};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UsageRecord {
    pub request_id: String,
    pub user_id: String,
    pub key_id: String,
    pub model: String,
    pub input_tokens: u32,
    pub output_tokens: u32,
    pub started_at: i64,
    pub duration_ms: u32,
    pub status: String, // "success" | "error" | "timeout" | "cancelled"
}

const BINCODE_CFG: bincode::config::Configuration = bincode::config::standard();

#[derive(Debug, Clone)]
pub struct UsageTracker {
    db: Arc<Database>,
}

impl UsageTracker {
    pub fn new(db: Arc<Database>) -> Self {
        Self { db }
    }

    pub fn record(&self, r: UsageRecord) -> anyhow::Result<()> {
        let id = next_usage_id(&self.db)?;
        let value =
            bincode::serde::encode_to_vec(&r, BINCODE_CFG).context("bincode UsageRecord")?;
        let write = self.db.begin_write()?;
        {
            let mut t = write.open_table(USAGE)?;
            t.insert(id, value.as_slice())?;
        }
        write.commit()?;
        Ok(())
    }

    /// List usage records with optional filters.
    /// - `since`: only records with started_at >= since (default: last 24 h)
    /// - `limit`: max records returned (default 1000, hard cap 10000)
    /// - `user_id`: optional filter
    /// - `model`: optional filter
    /// - `status`: optional filter ("success" | "error" | ...)
    pub fn list_usage(
        &self,
        since: Option<i64>,
        limit: Option<usize>,
        user_id_filter: Option<&str>,
        model_filter: Option<&str>,
        status_filter: Option<&str>,
    ) -> anyhow::Result<Vec<UsageRecord>> {
        let since = since.unwrap_or_else(|| chrono::Utc::now().timestamp() - 86400);
        let limit = limit.unwrap_or(1000).min(10000);

        let read = self.db.begin_read()?;
        let table = read.open_table(USAGE)?;
        let mut out = Vec::new();
        for entry in table.iter()? {
            let entry = entry?;
            let v = entry.1;
            let (rec, _): (UsageRecord, usize) =
                bincode::serde::decode_from_slice(v.value(), BINCODE_CFG)?;
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

    pub fn count(&self, user_id: &str) -> anyhow::Result<i64> {
        let read = self.db.begin_read()?;
        let table = read.open_table(USAGE)?;
        let mut n = 0i64;
        for entry in table.iter()? {
            let entry = entry?;
            let v = entry.1;
            let (rec, _): (UsageRecord, usize) =
                bincode::serde::decode_from_slice(v.value(), BINCODE_CFG)?;
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
    use crate::db::open;
    use tempfile::tempdir;

    #[test]
    fn record_and_count() {
        let dir = tempdir().unwrap();
        let db = open(&dir.path().join("test.redb")).unwrap();
        let t = UsageTracker::new(db);
        t.record(UsageRecord {
            request_id: "r1".into(),
            user_id: "u1".into(),
            key_id: "k1".into(),
            model: "qora-4b".into(),
            input_tokens: 10,
            output_tokens: 20,
            started_at: 1,
            duration_ms: 100,
            status: "success".into(),
        })
        .unwrap();
        assert_eq!(t.count("u1").unwrap(), 1);
        assert_eq!(t.count("u2").unwrap(), 0);
    }
}
