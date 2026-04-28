//! Read paths for conversations + messages.

use std::sync::Arc;

use redb::Database;

use crate::db::{CONVERSATIONS, MESSAGES, USER_CONVERSATIONS};
use crate::history::store::message_key;
use crate::history::{Conversation, StoredMessage};

const BINCODE_CFG: bincode::config::Configuration = bincode::config::standard();

/// Get a conversation by id, enforcing user_id ownership.
/// Returns None if it doesn't exist OR belongs to a different user (privacy: pretend it doesn't exist).
pub fn get_conversation(
    db: &Arc<Database>,
    conv_id: &str,
    user_id: &str,
) -> anyhow::Result<Option<Conversation>> {
    let read = db.begin_read()?;
    let convs = read.open_table(CONVERSATIONS)?;
    let bytes = match convs.get(conv_id)? {
        Some(v) => v.value().to_vec(),
        None => return Ok(None),
    };
    let (conv, _): (Conversation, usize) =
        bincode::serde::decode_from_slice(&bytes, BINCODE_CFG)?;
    // Privacy: pretend the conversation doesn't exist if it belongs to another user.
    if conv.user_id != user_id {
        return Ok(None);
    }
    Ok(Some(conv))
}

/// Get all messages in a conversation, in idx order.
pub fn get_messages(
    db: &Arc<Database>,
    conv_id: &str,
    user_id: &str,
) -> anyhow::Result<Vec<StoredMessage>> {
    let conv = get_conversation(db, conv_id, user_id)?
        .ok_or_else(|| anyhow::anyhow!("conversation not found or access denied"))?;

    let read = db.begin_read()?;
    let msgs = read.open_table(MESSAGES)?;
    let mut out = Vec::with_capacity(conv.message_count as usize);
    for i in 0..conv.message_count {
        let key = message_key(conv_id, i);
        if let Some(v) = msgs.get(key.as_slice())? {
            let (m, _): (StoredMessage, usize) =
                bincode::serde::decode_from_slice(v.value(), BINCODE_CFG)?;
            out.push(m);
        }
    }
    Ok(out)
}

/// List all conversations for a user, sorted by updated_at descending.
pub fn list_user_conversations(
    db: &Arc<Database>,
    user_id: &str,
) -> anyhow::Result<Vec<Conversation>> {
    let read = db.begin_read()?;
    let idx = read.open_multimap_table(USER_CONVERSATIONS)?;
    let mut conv_ids: Vec<String> = Vec::new();
    if let Ok(values) = idx.get(user_id) {
        for v in values {
            let v = v?;
            conv_ids.push(v.value().to_string());
        }
    }

    let convs_table = read.open_table(CONVERSATIONS)?;
    let mut conversations = Vec::with_capacity(conv_ids.len());
    for id in conv_ids {
        if let Some(v) = convs_table.get(id.as_str())? {
            let (c, _): (Conversation, usize) =
                bincode::serde::decode_from_slice(v.value(), BINCODE_CFG)?;
            conversations.push(c);
        }
    }
    // Sort by updated_at desc (most recently touched first)
    conversations.sort_by(|a, b| b.updated_at.cmp(&a.updated_at));
    Ok(conversations)
}

/// Find the top-K most relevant historical messages for a query, using cosine similarity
/// over stored embeddings. Messages without embeddings are skipped.
pub fn find_relevant_history(
    db: &Arc<Database>,
    conv_id: &str,
    user_id: &str,
    query_embedding: &[f32],
    top_k: usize,
) -> anyhow::Result<Vec<StoredMessage>> {
    let messages = get_messages(db, conv_id, user_id)?;
    let mut scored: Vec<(f32, StoredMessage)> = Vec::new();

    for m in messages {
        if let Some(emb) = &m.embedding {
            let score = cosine_similarity(query_embedding, emb);
            scored.push((score, m));
        }
    }

    // Sort descending by score
    scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    Ok(scored.into_iter().take(top_k).map(|(_, m)| m).collect())
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }
    dot / (norm_a * norm_b)
}
