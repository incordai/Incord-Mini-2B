//! Guest chat message storage (separate from authenticated user conversations).
//!
//! Free-tier anonymous chats are persisted so the admin can review usage,
//! moderate abuse, and build dataset insights. Each message is logged with
//! the guest_id, role, content, IP, user-agent, country, city, and timestamp.
//!
//! Idx allocation goes through [`crate::db::alloc_id`], so concurrent writers
//! never collide on the same monotonic id even though multiple appends can
//! be in flight from different request handlers. The counter is tenant-scoped
//! (`{tenant}::guest_messages`) so each tenant has its own dense numbering.

use std::collections::HashMap;
use std::sync::Arc;

use chrono::Utc;
use redb::Database;
use serde::{Deserialize, Serialize};

use crate::auth::middleware::ClientInfo;
use crate::db::{
    alloc_id, decode, encode_v1, is_safe_tenant, tenant_byte_prefix, tenant_u64_key,
    GUEST_MESSAGES,
};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GuestMessage {
    pub idx: u64,
    /// Tenant this message belongs to.
    pub tenant_id: String,
    pub guest_id: String,
    pub role: String, // "user" | "assistant"
    pub content: String,
    pub ip: Option<String>,
    pub user_agent: Option<String>,
    #[serde(default)]
    pub country: Option<String>,
    #[serde(default)]
    pub city: Option<String>,
    #[serde(default)]
    pub timezone: Option<String>,
    pub timestamp: i64,
}

fn require_safe_tenant(tenant: &str) -> anyhow::Result<()> {
    if !is_safe_tenant(tenant) {
        anyhow::bail!("invalid tenant id `{tenant}`");
    }
    Ok(())
}

/// Append one message. Returns the assigned idx.
///
/// Race-safe: idx allocation reads + increments [`crate::db::COUNTERS`]
/// inside the same write transaction that inserts the message. Two concurrent
/// callers can never receive the same idx.
pub fn append(
    db: &Arc<Database>,
    tenant: &str,
    guest_id: &str,
    role: &str,
    content: &str,
    client: &ClientInfo,
) -> anyhow::Result<u64> {
    require_safe_tenant(tenant)?;
    let now = Utc::now().timestamp();
    let counter_name = format!("{tenant}::guest_messages");

    let write = db.begin_write()?;
    let assigned_idx = alloc_id(&write, &counter_name)?;
    let storage_key = tenant_u64_key(tenant, assigned_idx);
    let msg = GuestMessage {
        idx: assigned_idx,
        tenant_id: tenant.to_string(),
        guest_id: guest_id.to_string(),
        role: role.to_string(),
        content: content.to_string(),
        ip: client.ip.clone(),
        user_agent: client.user_agent.clone(),
        country: client.country.clone(),
        city: client.city.clone(),
        timezone: client.timezone.clone(),
        timestamp: now,
    };
    let bytes = encode_v1(&msg)?;
    {
        let mut table = write.open_table(GUEST_MESSAGES)?;
        table.insert(storage_key.as_slice(), bytes.as_slice())?;
    }
    write.commit()?;
    Ok(assigned_idx)
}

/// List all messages for a single guest within a tenant, in chronological
/// order (up to `limit`). Range-scoped to the tenant prefix so other tenants'
/// rows don't even get touched.
pub fn list_for_guest(
    db: &Arc<Database>,
    tenant: &str,
    guest_id: &str,
    limit: usize,
) -> anyhow::Result<Vec<GuestMessage>> {
    require_safe_tenant(tenant)?;
    let prefix = tenant_byte_prefix(tenant);
    let read = db.begin_read()?;
    let table = read.open_table(GUEST_MESSAGES)?;
    let mut out = Vec::new();
    for entry in table.range(prefix.as_slice()..)? {
        let (k, v) = entry?;
        if !k.value().starts_with(&prefix) {
            break;
        }
        let Ok(m) = decode::<GuestMessage>(v.value()) else {
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

/// Aggregate stats per guest_id within a tenant (admin overview).
#[derive(Debug, Clone, Serialize)]
pub struct GuestSummary {
    pub guest_id: String,
    pub tenant_id: String,
    pub message_count: u32,
    pub first_at: i64,
    pub last_at: i64,
    pub last_ip: Option<String>,
    pub last_country: Option<String>,
    pub last_city: Option<String>,
    pub last_timezone: Option<String>,
    pub last_user_agent: Option<String>,
}

/// List all guest_ids in a tenant with aggregate stats. Tenant-scoped scan.
pub fn list_all_guests(
    db: &Arc<Database>,
    tenant: &str,
) -> anyhow::Result<HashMap<String, GuestSummary>> {
    require_safe_tenant(tenant)?;
    let prefix = tenant_byte_prefix(tenant);
    let read = db.begin_read()?;
    let table = read.open_table(GUEST_MESSAGES)?;
    let mut acc: HashMap<String, GuestSummary> = HashMap::new();
    for entry in table.range(prefix.as_slice()..)? {
        let (k, v) = entry?;
        if !k.value().starts_with(&prefix) {
            break;
        }
        let Ok(m) = decode::<GuestMessage>(v.value()) else {
            continue;
        };
        let s = acc.entry(m.guest_id.clone()).or_insert(GuestSummary {
            guest_id: m.guest_id.clone(),
            tenant_id: m.tenant_id.clone(),
            message_count: 0,
            first_at: m.timestamp,
            last_at: m.timestamp,
            last_ip: m.ip.clone(),
            last_country: m.country.clone(),
            last_city: m.city.clone(),
            last_timezone: m.timezone.clone(),
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
            s.last_timezone = m.timezone.clone();
            s.last_user_agent = m.user_agent.clone();
        }
    }
    Ok(acc)
}
