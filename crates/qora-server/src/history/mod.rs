//! Chat history persistence with optional embeddings.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Conversation {
    pub id: String,       // UUID
    pub user_id: String,
    pub title: Option<String>,
    pub created_at: i64,  // unix seconds
    pub updated_at: i64,
    /// Total messages stored in this conversation.
    pub message_count: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StoredMessage {
    pub idx: u32,         // 0-indexed within the conversation
    pub role: String,     // "user" | "assistant" | "system"
    pub content: String,
    pub timestamp: i64,
    /// Embedding (if computed). None if no embedding model was available at write time.
    pub embedding: Option<Vec<f32>>,
}

pub mod guest;
pub mod retrieve;
pub mod store;
