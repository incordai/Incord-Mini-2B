use crate::gemv::HybridCache;
use crate::server::session::sampler::Sampler;
use uuid::Uuid;

pub struct Session {
    pub id: Uuid,
    pub key_id: String,
    pub user_id: String,
    pub model_name: String,
    pub cache: HybridCache,
    pub prompt_tokens: Vec<u32>,
    pub generated_tokens: Vec<u32>,
    pub max_new_tokens: usize,
    pub eos_token_id: u32,
    pub sampler: Sampler,
    pub started_at: i64,
}
