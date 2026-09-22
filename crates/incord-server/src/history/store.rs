//! Conversation + message storage in redb.
//!
//! All keys are tenant-scoped; the multimap value stored under
//! `tenant::user_id` is `tenant::conv_id` so a tenant's user index can never
//! pull a different tenant's conversations.

use std::sync::Arc;

use chrono::Utc;
use redb::{Database, ReadableTable};
use uuid::Uuid;

use crate::db::{
    decode, encode_v1, is_safe_tenant, message_key, tenant_key, CONVERSATIONS, MESSAGES,
    USER_CONVERSATIONS,
};
use crate::history::{Conversation, StoredMessage};

fn require_safe_tenant(tenant: &str) -> anyhow::Result<()> {
    if !is_safe_tenant(tenant) {
        anyhow::bail!("invalid tenant id `{tenant}`");
    }
    Ok(())
}

/// Create a new conversation. Returns it with the assigned UUID.
pub fn create_conversation(
    db: &Arc<Database>,
    tenant: &str,
    user_id: &str,
    title: Option<String>,
) -> anyhow::Result<Conversation> {
    require_safe_tenant(tenant)?;
    let id = Uuid::new_v4().to_string();
    let now = Utc::now().timestamp();
    let conv = Conversation {
        id: id.clone(),
        tenant_id: tenant.to_string(),
        user_id: user_id.to_string(),
        title,
        created_at: now,
        updated_at: now,
        message_count: 0,
    };
    let bytes = encode_v1(&conv)?;
    let conv_storage_key = tenant_key(tenant, &id);
    let user_storage_key = tenant_key(tenant, user_id);

    let write = db.begin_write()?;
    {
        let mut convs = write.open_table(CONVERSATIONS)?;
        convs.insert(conv_storage_key.as_str(), bytes.as_slice())?;
        let mut idx = write.open_multimap_table(USER_CONVERSATIONS)?;
        idx.insert(user_storage_key.as_str(), conv_storage_key.as_str())?;
    }
    write.commit()?;
    Ok(conv)
}

/// Append a message to a conversation. Updates message_count + updated_at.
/// Returns the assigned idx.
pub fn append_message(
    db: &Arc<Database>,
    tenant: &str,
    conv_id: &str,
    user_id: &str,
    role: &str,
    content: &str,
    embedding: Option<Vec<f32>>,
) -> anyhow::Result<u32> {
    require_safe_tenant(tenant)?;
    let now = Utc::now().timestamp();
    let conv_storage_key = tenant_key(tenant, conv_id);

    let write = db.begin_write()?;
    let assigned_idx;
    {
        let mut convs = write.open_table(CONVERSATIONS)?;
        let conv_bytes = convs
            .get(conv_storage_key.as_str())?
            .ok_or_else(|| anyhow::anyhow!("conversation `{conv_id}` not found"))?
            .value()
            .to_vec();
        let mut conv: Conversation = decode(&conv_bytes)?;
        if conv.tenant_id != tenant {
            // This can't happen for a tenant-keyed lookup, but assert
            // defensively in case storage was corrupted.
            anyhow::bail!("conversation `{conv_id}` belongs to a different tenant");
        }
        if conv.user_id != user_id {
            anyhow::bail!(
                "conversation `{conv_id}` does not belong to user `{user_id}`"
            );
        }
        assigned_idx = conv.message_count;
        conv.message_count += 1;
        conv.updated_at = now;
        let updated_conv = encode_v1(&conv)?;
        convs.insert(conv_storage_key.as_str(), updated_conv.as_slice())?;

        let msg = StoredMessage {
            idx: assigned_idx,
            role: role.to_string(),
            content: content.to_string(),
            timestamp: now,
            embedding,
        };
        let msg_bytes = encode_v1(&msg)?;
        let key = message_key(tenant, conv_id, assigned_idx);
        let mut msgs = write.open_table(MESSAGES)?;
        msgs.insert(key.as_slice(), msg_bytes.as_slice())?;
    }
    write.commit()?;
    Ok(assigned_idx)
}

/// Delete a conversation and all its messages.
/// Returns true if the conversation existed and was deleted.
pub fn delete_conversation(
    db: &Arc<Database>,
    tenant: &str,
    conv_id: &str,
    user_id: &str,
) -> anyhow::Result<bool> {
    require_safe_tenant(tenant)?;
    let conv_storage_key = tenant_key(tenant, conv_id);
    let user_storage_key = tenant_key(tenant, user_id);

    // Phase 1: read to confirm existence + ownership.
    let conv_opt: Option<Conversation> = {
        let read = db.begin_read()?;
        let convs = read.open_table(CONVERSATIONS)?;
        match convs.get(conv_storage_key.as_str())? {
            None => None,
            Some(v) => Some(decode(v.value())?),
        }
    };

    let conv = match conv_opt {
        None => return Ok(false),
        Some(c) => c,
    };

    if conv.user_id != user_id {
        anyhow::bail!("conversation `{conv_id}` does not belong to user `{user_id}`");
    }

    // Phase 2: delete in a single write txn.
    let write = db.begin_write()?;
    {
        let mut msgs = write.open_table(MESSAGES)?;
        for i in 0..conv.message_count {
            let key = message_key(tenant, conv_id, i);
            msgs.remove(key.as_slice())?;
        }

        let mut idx = write.open_multimap_table(USER_CONVERSATIONS)?;
        idx.remove(user_storage_key.as_str(), conv_storage_key.as_str())?;

        let mut convs = write.open_table(CONVERSATIONS)?;
        convs.remove(conv_storage_key.as_str())?;
    }
    write.commit()?;
    Ok(true)
}
