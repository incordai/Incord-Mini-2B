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
    fn default() -> Self { Self::Text(String::new()) }
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
            "model": "qora-4b",
            "messages": [
                {"role": "user", "content": "hi"}
            ]
        }"#;
        let req: ChatCompletionRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.model, "qora-4b");
        assert_eq!(req.messages.len(), 1);
        assert!(matches!(req.messages[0].content, MessageContent::Text(_)));
    }

    #[test]
    fn parses_vision_content_parts() {
        let json = r#"{
            "model": "qora-4b",
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
}
