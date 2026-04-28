//! Conversation + message storage in redb.

use std::sync::Arc;

use chrono::Utc;
use redb::{Database, ReadableTable};
use uuid::Uuid;

use crate::db::{CONVERSATIONS, MESSAGES, USER_CONVERSATIONS};
use crate::history::{Conversation, StoredMessage};

const BINCODE_CFG: bincode::config::Configuration = bincode::config::standard();

/// Build the composite key (conv_id bytes || u64 big-endian idx) for the messages table.
pub(crate) fn message_key(conv_id: &str, idx: u32) -> Vec<u8> {
    let mut buf = Vec::with_capacity(conv_id.len() + 8);
    buf.extend_from_slice(conv_id.as_bytes());
    buf.extend_from_slice(&(idx as u64).to_be_bytes());
    buf
}

/// Create a new conversation, returns it (with the assigned UUID).
pub fn create_conversation(
    db: &Arc<Database>,
    user_id: &str,
    title: Option<String>,
) -> anyhow::Result<Conversation> {
    let id = Uuid::new_v4().to_string();
    let now = Utc::now().timestamp();
    let conv = Conversation {
        id: id.clone(),
        user_id: user_id.to_string(),
        title,
        created_at: now,
        updated_at: now,
        message_count: 0,
    };
    let bytes = bincode::serde::encode_to_vec(&conv, BINCODE_CFG)?;

    let write = db.begin_write()?;
    {
        let mut convs = write.open_table(CONVERSATIONS)?;
        convs.insert(id.as_str(), bytes.as_slice())?;
        let mut idx = write.open_multimap_table(USER_CONVERSATIONS)?;
        idx.insert(user_id, id.as_str())?;
    }
    write.commit()?;
    Ok(conv)
}

/// Append a message to a conversation. Updates message_count + updated_at.
/// Returns the assigned idx.
pub fn append_message(
    db: &Arc<Database>,
    conv_id: &str,
    user_id: &str,
    role: &str,
    content: &str,
    embedding: Option<Vec<f32>>,
) -> anyhow::Result<u32> {
    let now = Utc::now().timestamp();

    let write = db.begin_write()?;
    let assigned_idx;
    {
        let mut convs = write.open_table(CONVERSATIONS)?;
        let conv_bytes = convs
            .get(conv_id)?
            .ok_or_else(|| anyhow::anyhow!("conversation `{conv_id}` not found"))?
            .value()
            .to_vec();
        let (mut conv, _): (Conversation, usize) =
            bincode::serde::decode_from_slice(&conv_bytes, BINCODE_CFG)?;
        if conv.user_id != user_id {
            anyhow::bail!(
                "conversation `{conv_id}` does not belong to user `{user_id}`"
            );
        }
        assigned_idx = conv.message_count;
        conv.message_count += 1;
        conv.updated_at = now;
        let updated_conv = bincode::serde::encode_to_vec(&conv, BINCODE_CFG)?;
        convs.insert(conv_id, updated_conv.as_slice())?;

        let msg = StoredMessage {
            idx: assigned_idx,
            role: role.to_string(),
            content: content.to_string(),
            timestamp: now,
            embedding,
        };
        let msg_bytes = bincode::serde::encode_to_vec(&msg, BINCODE_CFG)?;
        let key = message_key(conv_id, assigned_idx);
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
    conv_id: &str,
    user_id: &str,
) -> anyhow::Result<bool> {
    // Phase 1: read the conversation to check existence + ownership.
    let conv_opt: Option<Conversation> = {
        let read = db.begin_read()?;
        let convs = read.open_table(CONVERSATIONS)?;
        match convs.get(conv_id)? {
            None => None,
            Some(v) => {
                let bytes = v.value().to_vec();
                let (c, _): (Conversation, usize) =
                    bincode::serde::decode_from_slice(&bytes, BINCODE_CFG)?;
                Some(c)
            }
        }
    };

    let conv = match conv_opt {
        None => return Ok(false),
        Some(c) => c,
    };

    if conv.user_id != user_id {
        anyhow::bail!("conversation `{conv_id}` does not belong to user `{user_id}`");
    }

    // Phase 2: delete everything in a single write transaction.
    let write = db.begin_write()?;
    {
        // Delete all messages
        let mut msgs = write.open_table(MESSAGES)?;
        for i in 0..conv.message_count {
            let key = message_key(conv_id, i);
            msgs.remove(key.as_slice())?;
        }

        // Remove from user index
        let mut idx = write.open_multimap_table(USER_CONVERSATIONS)?;
        idx.remove(user_id, conv_id)?;

        // Remove conversation record
        let mut convs = write.open_table(CONVERSATIONS)?;
        convs.remove(conv_id)?;
    }
    write.commit()?;
    Ok(true)
}
