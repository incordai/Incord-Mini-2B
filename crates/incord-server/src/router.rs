//! Smart model routing.
//!
//! Given a chat request, decides which concrete chat model to dispatch to.
//! The "auto" model id (`qora-auto`) triggers this router; explicit model
//! ids (`qora-llm`, `qora-llm-mini`) bypass it.
//!
//! Strategy: heuristic on the user's latest message. Cheap to compute (no
//! ML in the router itself — the router IS the ML pre-filter), and good
//! enough to land 80%+ of greetings / chitchat on the small model and
//! anything substantial on the big one.
//!
//! Why heuristic rather than a tiny classifier: the router runs on every
//! request and must add zero perceivable latency. A regex+wordcount pass is
//! ~microseconds; loading a tiny ML classifier model would add tens of
//! milliseconds and a binary dependency. We can upgrade to a real classifier
//! later without changing the calling convention.

use crate::engine::chat::ChatMessage;

/// The well-known "auto-router" model id. When a request specifies this id,
/// the inference handler runs `route(...)` to pick a concrete model.
pub const AUTO_MODEL_ID: &str = "qora-auto";

/// Substrings that strongly suggest the user wants a thoughtful answer.
/// Lowercased — the matcher lowercases the message first.
const HEAVY_KEYWORDS: &[&str] = &[
    // Knowledge / definition requests (the missing class that sent
    // "tell me about bitcoin" to the small model).
    "tell me about", "tell me", "what is", "what are", "what's the",
    "who is", "who are", "who was", "where is", "where are", "where's",
    "when did", "when is", "when was", "describe", "introduce", "introduction to",
    "give me an overview", "overview of", "list", "name the",
    // Reasoning verbs
    "explain", "why does", "why is", "why are", "how does", "how do", "how can",
    "analyze", "analyse", "compare", "contrast", "evaluate", "summarize", "summarise",
    "reason", "think through", "step by step", "step-by-step", "break down",
    "deep dive", "deep-dive",
    // Coding / engineering
    "code", "function", "algorithm", "implement", "debug", "refactor",
    "optimize", "optimise", "trace", "profile", "compile", "stack trace",
    "architecture", "design pattern", "data structure", "complexity",
    // Math / logic
    "math", "calculate", "solve", "proof", "derivation", "equation",
    "formula", "theorem", "integral", "derivative",
    // Long-form / creative
    "write a", "draft a", "compose", "essay", "article",
    // Domain depth
    "research", "evidence", "citation", "study", "thesis",
];

/// Lightweight chitchat substrings — even if the message is long, these
/// generally don't need the big model. Lowercased.
const LIGHT_PHRASES: &[&str] = &[
    "hi", "hello", "hey", "yo", "sup", "good morning", "good evening",
    "good afternoon", "thanks", "thank you", "thx", "ty", "cheers",
    "ok", "okay", "got it", "cool", "nice", "great", "awesome",
    "bye", "goodbye", "see you", "see ya", "later", "ttyl",
    "how are you", "what's up", "whats up", "wassup",
];

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RoutedSize {
    Small,
    Big,
}

/// Inspect the latest user message and decide which size of model to use.
///
/// Policy (revised after live testing): **default to Big**. The small model
/// is reserved for *explicit* chitchat — greetings, one-word acks, farewells.
/// Anything that looks like a question, a knowledge request, or a real
/// instruction goes to the big model so we never give a half-answered
/// response. This trades a bit of latency on simple queries for never
/// missing a real reasoning task.
///
/// Heuristics, in order:
///   1. Empty → small (no work for big to do)
///   2. Contains any heavy keyword → big (covers "explain", "tell me", etc.)
///   3. Ends with `?` → big (questions need reasoning)
///   4. Word count > 30 OR char count > 200 → big
///   5. Short (≤ 4 words) AND matches a light-phrase prefix → small
///   6. Default → **big** (be safe; downgrade only when clearly chitchat)
pub fn route(messages: &[ChatMessage]) -> RoutedSize {
    let last_user = messages
        .iter()
        .rev()
        .find(|m| m.role == "user")
        .map(|m| m.content.as_str())
        .unwrap_or("");

    if last_user.trim().is_empty() {
        return RoutedSize::Small;
    }

    let lower = last_user.to_lowercase();
    let trimmed = lower.trim();
    let word_count = trimmed.split_whitespace().count();
    let char_count = last_user.chars().count();

    // Step 2: explicit heavy keyword → big
    if HEAVY_KEYWORDS.iter().any(|k| lower.contains(k)) {
        return RoutedSize::Big;
    }

    // Step 3: question mark → big (questions need real answers)
    if trimmed.ends_with('?') {
        return RoutedSize::Big;
    }

    // Step 4: long input → big
    if word_count > 30 || char_count > 200 {
        return RoutedSize::Big;
    }

    // Step 5: explicit short greeting → small
    if word_count <= 4 {
        let starts_light = LIGHT_PHRASES.iter().any(|p| {
            trimmed == *p || trimmed.starts_with(&format!("{p} ")) || trimmed.starts_with(&format!("{p}!"))
                || trimmed.starts_with(&format!("{p}?")) || trimmed.starts_with(&format!("{p},"))
                || trimmed.starts_with(&format!("{p}."))
        });
        if starts_light {
            return RoutedSize::Small;
        }
    }

    // Step 6: default big — safer for unknown intents.
    RoutedSize::Big
}

/// Resolve the auto-router decision into a concrete model id.
/// Falls back to whatever's actually loaded if the preferred id isn't present.
pub fn resolve_auto_model<'a>(
    decision: RoutedSize,
    available_chat_ids: &'a [String],
) -> Option<&'a str> {
    let prefer_for = |size: RoutedSize| match size {
        // The mini model has "mini" in its id; the big one is the bare "qora-llm" or contains a size like "4b"/"9b".
        RoutedSize::Small => available_chat_ids.iter().find(|id| {
            let l = id.to_lowercase();
            l.contains("mini") || l.contains("0.6") || l.contains("0.8") || l.contains("08b") || l.contains("06b")
        }),
        RoutedSize::Big => available_chat_ids.iter().find(|id| {
            let l = id.to_lowercase();
            l.contains("9b") || l.contains("4b") || l == "qora-llm"
        }),
    };

    prefer_for(decision)
        .or_else(|| prefer_for(if decision == RoutedSize::Small { RoutedSize::Big } else { RoutedSize::Small }))
        .map(|s| s.as_str())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn msg(role: &str, content: &str) -> ChatMessage {
        ChatMessage { role: role.to_string(), content: content.to_string() }
    }

    #[test]
    fn greeting_routes_small() {
        assert_eq!(route(&[msg("user", "hi")]), RoutedSize::Small);
        assert_eq!(route(&[msg("user", "hello there")]), RoutedSize::Small);
        assert_eq!(route(&[msg("user", "thanks!")]), RoutedSize::Small);
    }

    #[test]
    fn explain_routes_big() {
        assert_eq!(route(&[msg("user", "explain monads")]), RoutedSize::Big);
        assert_eq!(
            route(&[msg("user", "Why does the sky look blue at noon")]),
            RoutedSize::Big
        );
    }

    #[test]
    fn long_input_routes_big() {
        let long = "word ".repeat(40);
        assert_eq!(route(&[msg("user", &long)]), RoutedSize::Big);
    }

    #[test]
    fn code_keyword_routes_big() {
        assert_eq!(
            route(&[msg("user", "Can you debug this code?")]),
            RoutedSize::Big
        );
    }

    #[test]
    fn knowledge_request_routes_big() {
        // The original bug — "tell me about X" needs the big model.
        assert_eq!(
            route(&[msg("user", "tell me about bitcoin")]),
            RoutedSize::Big
        );
        assert_eq!(route(&[msg("user", "what is rust")]), RoutedSize::Big);
        assert_eq!(route(&[msg("user", "describe the moon")]), RoutedSize::Big);
    }

    #[test]
    fn question_mark_routes_big() {
        // Even without a heavy keyword, a question mark forces big.
        assert_eq!(
            route(&[msg("user", "is the weather nice today?")]),
            RoutedSize::Big
        );
    }

    #[test]
    fn unknown_intent_defaults_big() {
        // New default: anything that isn't a clear greeting goes big.
        assert_eq!(
            route(&[msg("user", "I had a long day at work")]),
            RoutedSize::Big
        );
    }

    #[test]
    fn resolve_falls_back_when_preferred_missing() {
        let only_big = vec!["qora-llm".to_string()];
        // Asks for small but only big is loaded → falls back to big.
        assert_eq!(resolve_auto_model(RoutedSize::Small, &only_big), Some("qora-llm"));
    }
}
