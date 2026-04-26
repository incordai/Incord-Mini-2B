pub mod queue;
pub mod worker;

#[derive(Debug, Clone)]
pub enum Event {
    Token(u32, String), // token id + decoded string
    Finish(FinishReason),
    Error(String),
}

#[derive(Debug, Clone, Copy)]
pub enum FinishReason {
    Stop,
    Length,
    Cancelled,
    Timeout,
    Error,
}

impl FinishReason {
    pub fn as_openai(&self) -> &'static str {
        match self {
            Self::Stop | Self::Cancelled => "stop",
            Self::Length => "length",
            Self::Timeout => "stop",
            Self::Error => "stop",
        }
    }
}
