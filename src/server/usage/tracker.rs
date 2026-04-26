use std::sync::Arc;
use anyhow::Context;
use redb::{Database, ReadableTable};
use serde::{Deserialize, Serialize};
use crate::server::db::{USAGE, next_usage_id};

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
    pub status: String,    // "success" | "error" | "timeout" | "cancelled"
}

const BINCODE_CFG: bincode::config::Configuration = bincode::config::standard();

#[derive(Debug, Clone)]
pub struct UsageTracker {
    db: Arc<Database>,
}

impl UsageTracker {
    pub fn new(db: Arc<Database>) -> Self { Self { db } }

    pub fn record(&self, r: UsageRecord) -> anyhow::Result<()> {
        let id = next_usage_id(&self.db)?;
        let value = bincode::serde::encode_to_vec(&r, BINCODE_CFG).context("bincode UsageRecord")?;
        let write = self.db.begin_write()?;
        {
            let mut t = write.open_table(USAGE)?;
            t.insert(id, value.as_slice())?;
        }
        write.commit()?;
        Ok(())
    }

    pub fn count(&self, user_id: &str) -> anyhow::Result<i64> {
        let read = self.db.begin_read()?;
        let table = read.open_table(USAGE)?;
        let mut n = 0i64;
        for entry in table.iter()? {
            let (_, v) = entry?;
            let (rec, _): (UsageRecord, usize) =
                bincode::serde::decode_from_slice(v.value(), BINCODE_CFG)?;
            if rec.user_id == user_id { n += 1; }
        }
        Ok(n)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::server::db::open;
    use tempfile::tempdir;

    #[test]
    fn record_and_count() {
        let dir = tempdir().unwrap();
        let db = open(&dir.path().join("test.redb")).unwrap();
        let t = UsageTracker::new(db);
        t.record(UsageRecord {
            request_id: "r1".into(), user_id: "u1".into(), key_id: "k1".into(),
            model: "qora-4b".into(), input_tokens: 10, output_tokens: 20,
            started_at: 1, duration_ms: 100, status: "success".into(),
        }).unwrap();
        assert_eq!(t.count("u1").unwrap(), 1);
        assert_eq!(t.count("u2").unwrap(), 0);
    }
}
