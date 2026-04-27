//! Model registry — runtime lookup table mapping model ids to engine instances.

use std::collections::HashMap;
use std::sync::Arc;

use crate::engine::chat::ChatEngine;
use crate::engine::embed::EmbeddingEngine;
use crate::engine::rerank::RerankEngine;
use crate::engine::stt::SttEngine;
use crate::engine::tts::TtsEngine;
use crate::engine::ModelEngine;

/// Holds all loaded engines. Populated at startup from TOML config; immutable thereafter.
/// Cheap to clone (Arcs only).
#[derive(Clone, Default)]
pub struct Registry {
    inner: Arc<RegistryInner>,
}

#[derive(Default)]
struct RegistryInner {
    by_id:  HashMap<String, Arc<dyn ModelEngine>>,
    chat:   HashMap<String, Arc<dyn ChatEngine>>,
    tts:    HashMap<String, Arc<dyn TtsEngine>>,
    stt:    HashMap<String, Arc<dyn SttEngine>>,
    embed:  HashMap<String, Arc<dyn EmbeddingEngine>>,
    rerank: HashMap<String, Arc<dyn RerankEngine>>,
}

impl Registry {
    pub fn builder() -> RegistryBuilder {
        RegistryBuilder::default()
    }

    pub fn contains(&self, id: &str) -> bool {
        self.inner.by_id.contains_key(id)
    }

    pub fn get(&self, id: &str) -> Option<Arc<dyn ModelEngine>> {
        self.inner.by_id.get(id).cloned()
    }

    pub fn get_chat(&self, id: &str) -> Option<Arc<dyn ChatEngine>> {
        self.inner.chat.get(id).cloned()
    }

    pub fn get_tts(&self, id: &str) -> Option<Arc<dyn TtsEngine>> {
        self.inner.tts.get(id).cloned()
    }

    pub fn get_stt(&self, id: &str) -> Option<Arc<dyn SttEngine>> {
        self.inner.stt.get(id).cloned()
    }

    pub fn get_embed(&self, id: &str) -> Option<Arc<dyn EmbeddingEngine>> {
        self.inner.embed.get(id).cloned()
    }

    pub fn get_rerank(&self, id: &str) -> Option<Arc<dyn RerankEngine>> {
        self.inner.rerank.get(id).cloned()
    }

    /// All registered model ids sorted — for stable output in /v1/models.
    pub fn ids(&self) -> Vec<&str> {
        let mut v: Vec<&str> = self.inner.by_id.keys().map(|s| s.as_str()).collect();
        v.sort_unstable();
        v
    }

    /// All engines (sorted by id) — useful for /v1/models.
    pub fn all(&self) -> Vec<Arc<dyn ModelEngine>> {
        let mut ids: Vec<&String> = self.inner.by_id.keys().collect();
        ids.sort_unstable();
        ids.into_iter().map(|id| self.inner.by_id[id].clone()).collect()
    }

    pub fn len(&self) -> usize { self.inner.by_id.len() }
    pub fn is_empty(&self) -> bool { self.inner.by_id.is_empty() }
}

#[derive(Default)]
pub struct RegistryBuilder {
    by_id:  HashMap<String, Arc<dyn ModelEngine>>,
    chat:   HashMap<String, Arc<dyn ChatEngine>>,
    tts:    HashMap<String, Arc<dyn TtsEngine>>,
    stt:    HashMap<String, Arc<dyn SttEngine>>,
    embed:  HashMap<String, Arc<dyn EmbeddingEngine>>,
    rerank: HashMap<String, Arc<dyn RerankEngine>>,
}

impl RegistryBuilder {
    pub fn add_chat(mut self, engine: Arc<dyn ChatEngine>) -> Self {
        let id = engine.id().to_string();
        self.by_id.insert(id.clone(), engine.clone() as Arc<dyn ModelEngine>);
        self.chat.insert(id, engine);
        self
    }

    pub fn add_tts(mut self, engine: Arc<dyn TtsEngine>) -> Self {
        let id = engine.id().to_string();
        self.by_id.insert(id.clone(), engine.clone() as Arc<dyn ModelEngine>);
        self.tts.insert(id, engine);
        self
    }

    pub fn add_stt(mut self, engine: Arc<dyn SttEngine>) -> Self {
        let id = engine.id().to_string();
        self.by_id.insert(id.clone(), engine.clone() as Arc<dyn ModelEngine>);
        self.stt.insert(id, engine);
        self
    }

    pub fn add_embed(mut self, engine: Arc<dyn EmbeddingEngine>) -> Self {
        let id = engine.id().to_string();
        self.by_id.insert(id.clone(), engine.clone() as Arc<dyn ModelEngine>);
        self.embed.insert(id, engine);
        self
    }

    pub fn add_rerank(mut self, engine: Arc<dyn RerankEngine>) -> Self {
        let id = engine.id().to_string();
        self.by_id.insert(id.clone(), engine.clone() as Arc<dyn ModelEngine>);
        self.rerank.insert(id, engine);
        self
    }

    pub fn build(self) -> Registry {
        Registry {
            inner: Arc::new(RegistryInner {
                by_id: self.by_id,
                chat:  self.chat,
                tts:   self.tts,
                stt:   self.stt,
                embed: self.embed,
                rerank: self.rerank,
            }),
        }
    }
}
