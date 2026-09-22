use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<Message>,

    #[serde(default)]
    pub stream: bool,

    #[serde(default)]
    pub max_tokens: Option<u32>,

    #[serde(default)]
    pub temperature: Option<f32>,

    #[serde(default)]
    pub top_p: Option<f32>,

    #[serde(default)]
    pub n: Option<u32>,

    #[serde(default)]
    pub stop: Option<StopValue>,

    #[serde(default)]
    pub presence_penalty: Option<f32>,

    #[serde(default)]
    pub frequency_penalty: Option<f32>,

    #[serde(default)]
    pub user: Option<String>,

    // QORA-specific extensions
    #[serde(default)]
    pub think_budget: Option<u32>,

    #[serde(default)]
    pub think: Option<bool>,

    #[serde(default)]
    pub show_think: Option<bool>,

    /// Reasoning effort: "low", "medium", or "high".
    /// Standard parameter for reasoning models (e.g. OpenAI).
    /// Maps to think_budget unless an explicit think_budget is provided.
    #[serde(default)]
    pub reasoning_effort: Option<ReasoningEffort>,

    /// If set, this chat is part of an existing conversation.
    /// New messages get appended; if use_history is true, prior context is injected.
    #[serde(default)]
    pub conversation_id: Option<String>,

    /// Inject relevant historical messages from this conversation as context.
    /// Requires conversation_id to be set.
    #[serde(default)]
    pub use_history: bool,

    /// If true, strip markdown formatting (`**bold**`, `# headings`, lists,
    /// links, code fences) from the response before returning it. Most useful
    /// for TTS / SMS pipelines that should not read asterisks aloud.
    /// Currently honoured for non-streaming requests only — streaming returns
    /// raw tokens as they're generated; clients can post-process those.
    #[serde(default)]
    pub strip_markdown: Option<bool>,
}

#[derive(Debug, Deserialize, Serialize, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum ReasoningEffort {
    Low,
    Medium,
    High,
}

impl ReasoningEffort {
    /// Maps the reasoning effort level to a default token budget:
    /// - Low: 512 tokens (fast, minimal latency)
    /// - Medium: 2048 tokens (balanced step-by-step reasoning)
    /// - High: 8192 tokens (deep reasoning, complex problems)
    pub fn default_budget(self) -> u32 {
        match self {
            ReasoningEffort::Low => 512,
            ReasoningEffort::Medium => 2048,
            ReasoningEffort::High => 8192,
        }
    }
}

impl std::fmt::Display for ReasoningEffort {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Low => write!(f, "low"),
            Self::Medium => write!(f, "medium"),
            Self::High => write!(f, "high"),
        }
    }
}

impl ChatCompletionRequest {
    /// Resolves effective thinking budget:
    /// If `think_budget` is explicitly set, it takes priority.
    /// Otherwise, if `reasoning_effort` is set, it maps to its preset budget.
    pub fn effective_think_budget(&self) -> Option<u32> {
        self.think_budget
            .or_else(|| self.reasoning_effort.map(|re| re.default_budget()))
    }
}

#[derive(Debug, Deserialize, Serialize, Clone)]
#[serde(untagged)]
pub enum StopValue {
    Single(String),
    Multiple(Vec<String>),
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct Message {
    pub role: String,
    #[serde(default)]
    pub content: MessageContent,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
#[serde(untagged)]
pub enum MessageContent {
    Text(String),
    Parts(Vec<ContentPart>),
}

impl Default for MessageContent {
    fn default() -> Self {
        Self::Text(String::new())
    }
}

#[derive(Debug, Deserialize, Serialize, Clone)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentPart {
    Text { text: String },
    ImageUrl { image_url: ImageUrl },
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct ImageUrl {
    pub url: String,
}
