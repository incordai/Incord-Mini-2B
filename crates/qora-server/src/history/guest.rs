//! Guest chat message storage (separate from authenticated user conversations).
//!
//! Free-tier anonymous chats are persisted so the admin can review usage,
//! moderate abuse, and build dataset insights. Each message is logged with
//! the guest_id, role, content, IP, user-agent, country, city, and timestamp.

use std::collections::HashMap;
use std::sync::Arc;

use chrono::Utc;
use redb::{Database, ReadableTable};
use serde::{Deserialize, Serialize};

use crate::auth::middleware::ClientInfo;
use crate::db::GUEST_MESSAGES;

const BINCODE_CFG: bincode::config::Configuration = bincode::config::standard();

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GuestMessage {
    pub idx: u64,
    pub guest_id: String,
    pub role: String,    // "user" | "assistant"
    pub content: String,
    pub ip: Option<String>,
    pub user_agent: Option<String>,
    /// ISO 3166-1 alpha-2 country code, e.g. "US". None when no geo data available.
    #[serde(default)]
    pub country: Option<String>,
    /// City name. None when no geo data available.
    #[serde(default)]
    pub city: Option<String>,
    pub timestamp: i64,
}

/// Append one message to the guest_messages table. Returns the assigned idx.
pub fn append(
    db: &Arc<Database>,
    guest_id: &str,
    role: &str,
    content: &str,
    client: &ClientInfo,
) -> anyhow::Result<u64> {
    let now = Utc::now().timestamp();
    let write = db.begin_write()?;
    let assigned_idx;
    {
        let mut table = write.open_table(GUEST_MESSAGES)?;
        // Find next idx — read max key + 1
        let last = table
            .iter()?
            .next_back()
            .transpose()?
            .map(|(k, _)| k.value())
            .unwrap_or(0);
        assigned_idx = last + 1;
        let msg = GuestMessage {
            idx: assigned_idx,
            guest_id: guest_id.to_string(),
            role: role.to_string(),
            content: content.to_string(),
            ip: client.ip.clone(),
            user_agent: client.user_agent.clone(),
            country: client.country.clone(),
            city: client.city.clone(),
            timestamp: now,
        };
        let bytes = bincode::serde::encode_to_vec(&msg, BINCODE_CFG)?;
        table.insert(assigned_idx, bytes.as_slice())?;
    }
    write.commit()?;
    Ok(assigned_idx)
}

/// List all messages for a single guest, in chronological order (up to `limit`).
pub fn list_for_guest(
    db: &Arc<Database>,
    guest_id: &str,
    limit: usize,
) -> anyhow::Result<Vec<GuestMessage>> {
    let read = db.begin_read()?;
    let table = read.open_table(GUEST_MESSAGES)?;
    let mut out = Vec::new();
    for entry in table.iter()? {
        let (_, v) = entry?;
        // Decode with graceful degradation — skip records that don't decode cleanly
        // (e.g. records written with a different schema version).
        let Ok((m, _)) =
            bincode::serde::decode_from_slice::<GuestMessage, _>(v.value(), BINCODE_CFG)
        else {
            continue;
        };
        if m.guest_id == guest_id {
            out.push(m);
            if out.len() >= limit {
                break;
            }
        }
    }
    Ok(out)
}

/// Aggregate stats per guest_id for the admin overview.
#[derive(Debug, Clone, Serialize)]
pub struct GuestSummary {
    pub guest_id: String,
    pub message_count: u32,
    pub first_at: i64,
    pub last_at: i64,
    pub last_ip: Option<String>,
    pub last_country: Option<String>,
    pub last_city: Option<String>,
    pub last_user_agent: Option<String>,
}

/// List all guest_ids with aggregate stats (admin overview).
/// Returns a HashMap keyed by guest_id; callers sort as needed.
pub fn list_all_guests(db: &Arc<Database>) -> anyhow::Result<HashMap<String, GuestSummary>> {
    let read = db.begin_read()?;
    let table = read.open_table(GUEST_MESSAGES)?;
    let mut acc: HashMap<String, GuestSummary> = HashMap::new();
    for entry in table.iter()? {
        let (_, v) = entry?;
        // Graceful degradation: skip records that fail to decode.
        let Ok((m, _)) =
            bincode::serde::decode_from_slice::<GuestMessage, _>(v.value(), BINCODE_CFG)
        else {
            continue;
        };
        let s = acc.entry(m.guest_id.clone()).or_insert(GuestSummary {
            guest_id: m.guest_id.clone(),
            message_count: 0,
            first_at: m.timestamp,
            last_at: m.timestamp,
            last_ip: m.ip.clone(),
            last_country: m.country.clone(),
            last_city: m.city.clone(),
            last_user_agent: m.user_agent.clone(),
        });
        s.message_count += 1;
        if m.timestamp < s.first_at {
            s.first_at = m.timestamp;
        }
        if m.timestamp > s.last_at {
            s.last_at = m.timestamp;
            s.last_ip = m.ip.clone();
            s.last_country = m.country.clone();
            s.last_city = m.city.clone();
            s.last_user_agent = m.user_agent.clone();
        }
    }
    Ok(acc)
}
