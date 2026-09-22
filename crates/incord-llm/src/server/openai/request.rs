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

    // Incord-specific extensions
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_simple_text_message() {
        let json = r#"{
            "model": "incord-mini-2b",
            "messages": [
                {"role": "user", "content": "hi"}
            ]
        }"#;
        let req: ChatCompletionRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.model, "incord-mini-2b");
        assert_eq!(req.messages.len(), 1);
        assert!(matches!(req.messages[0].content, MessageContent::Text(_)));
    }

    #[test]
    fn parses_vision_content_parts() {
        let json = r#"{
            "model": "incord-mini-2b",
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "what's in this?"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}}
                ]
            }]
        }"#;
        let req: ChatCompletionRequest = serde_json::from_str(json).unwrap();
        if let MessageContent::Parts(parts) = &req.messages[0].content {
            assert_eq!(parts.len(), 2);
        } else {
            panic!("expected parts");
        }
    }

    #[test]
    fn missing_model_errors() {
        let json = r#"{"messages": []}"#;
        let r: Result<ChatCompletionRequest, _> = serde_json::from_str(json);
        assert!(r.is_err());
    }

    #[test]
    fn streaming_default_false() {
        let json = r#"{"model": "x", "messages": []}"#;
        let req: ChatCompletionRequest = serde_json::from_str(json).unwrap();
        assert!(!req.stream);
    }

    #[test]
    fn parses_reasoning_effort_levels() {
        let json_low = r#"{"model": "incord-mini-2b", "messages": [], "reasoning_effort": "low"}"#;
        let req_low: ChatCompletionRequest = serde_json::from_str(json_low).unwrap();
        assert_eq!(req_low.reasoning_effort, Some(ReasoningEffort::Low));
        assert_eq!(req_low.effective_think_budget(), Some(512));

        let json_med = r#"{"model": "incord-mini-2b", "messages": [], "reasoning_effort": "medium"}"#;
        let req_med: ChatCompletionRequest = serde_json::from_str(json_med).unwrap();
        assert_eq!(req_med.reasoning_effort, Some(ReasoningEffort::Medium));
        assert_eq!(req_med.effective_think_budget(), Some(2048));

        let json_high = r#"{"model": "incord-mini-2b", "messages": [], "reasoning_effort": "high"}"#;
        let req_high: ChatCompletionRequest = serde_json::from_str(json_high).unwrap();
        assert_eq!(req_high.reasoning_effort, Some(ReasoningEffort::High));
        assert_eq!(req_high.effective_think_budget(), Some(8192));

        // Explicit think_budget overrides reasoning_effort
        let json_override = r#"{"model": "incord-mini-2b", "messages": [], "reasoning_effort": "high", "think_budget": 1500}"#;
        let req_override: ChatCompletionRequest = serde_json::from_str(json_override).unwrap();
        assert_eq!(req_override.reasoning_effort, Some(ReasoningEffort::High));
        assert_eq!(req_override.effective_think_budget(), Some(1500));

        // Default with no effort or think_budget gives None
        let json_none = r#"{"model": "incord-mini-2b", "messages": []}"#;
        let req_none: ChatCompletionRequest = serde_json::from_str(json_none).unwrap();
        assert_eq!(req_none.reasoning_effort, None);
        assert_eq!(req_none.effective_think_budget(), None);
    }
}
