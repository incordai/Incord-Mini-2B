//! Markdown → plain-text stripper for chat completion responses.
//!
//! Why: API consumers piping LLM output to TTS / SMS / SQL / log files
//! generally want clean prose, not `**bold**` and `# heading` tokens. The
//! voice modal in particular CANNOT read markdown aloud — `**hi**` becomes
//! "asterisk asterisk hi asterisk asterisk" through the speech synth.
//!
//! Scope: handles the inline + block markdown the assistant typically emits
//! (bold, italic, code, headings, lists, links, blockquotes, hr). Does NOT
//! attempt full CommonMark parsing — we'd need pulldown-cmark for that and
//! the gain is negligible for chat output.

use once_cell::sync::Lazy;
use regex::Regex;

// Compiled once on first use. Each strip pass uses these regexes.
static FENCED_CODE: Lazy<Regex> = Lazy::new(|| Regex::new(r"```[a-zA-Z0-9_-]*\n([\s\S]*?)```").unwrap());
static INLINE_CODE: Lazy<Regex> = Lazy::new(|| Regex::new(r"`([^`]+)`").unwrap());
// Bold uses `[^\n]+?` so it can wrap italic markers (`**bold *italic* bold**`).
// Stripping bold first leaves `bold *italic* bold` for the italic pass.
static BOLD_AST: Lazy<Regex>    = Lazy::new(|| Regex::new(r"\*\*([^\n]+?)\*\*").unwrap());
static BOLD_UND: Lazy<Regex>    = Lazy::new(|| Regex::new(r"__([^\n]+?)__").unwrap());
static ITALIC_AST: Lazy<Regex>  = Lazy::new(|| Regex::new(r"(?:^|[^*])\*([^*\n][^*]*?)\*(?:[^*]|$)").unwrap());
static ITALIC_UND: Lazy<Regex>  = Lazy::new(|| Regex::new(r"(?:^|[^_])_([^_\n][^_]*?)_(?:[^_]|$)").unwrap());
static STRIKETHROUGH: Lazy<Regex> = Lazy::new(|| Regex::new(r"~~([^~]+)~~").unwrap());
// Important: leading space-class for block markers MUST be `[ \t]` (not `\s`)
// so it doesn't eat the preceding newline and collapse blank lines between
// paragraphs. Same for the trailing space — `[ \t]+` not `\s+`.
static HEADING: Lazy<Regex>     = Lazy::new(|| Regex::new(r"(?m)^#{1,6}[ \t]+").unwrap());
static IMAGE: Lazy<Regex>       = Lazy::new(|| Regex::new(r"!\[[^\]]*\]\([^)]+\)").unwrap());
static LINK: Lazy<Regex>        = Lazy::new(|| Regex::new(r"\[([^\]]+)\]\([^)]+\)").unwrap());
static UNORDERED_LIST: Lazy<Regex> = Lazy::new(|| Regex::new(r"(?m)^[ \t]*[-*+][ \t]+").unwrap());
static ORDERED_LIST: Lazy<Regex> = Lazy::new(|| Regex::new(r"(?m)^[ \t]*\d+\.[ \t]+").unwrap());
static BLOCKQUOTE: Lazy<Regex>  = Lazy::new(|| Regex::new(r"(?m)^>[ \t]*").unwrap());
static HR: Lazy<Regex>          = Lazy::new(|| Regex::new(r"(?m)^[\-*_]{3,}[ \t]*$").unwrap());
static MULTI_BLANK: Lazy<Regex> = Lazy::new(|| Regex::new(r"\n{3,}").unwrap());

/// Strip markdown formatting and return plain text suitable for TTS / logs.
///
/// Order matters: fenced code blocks first (so their content isn't mistaken
/// for inline syntax), then bold (which uses doubled markers and would
/// otherwise be eaten by the italic pass).
pub fn strip(text: &str) -> String {
    let mut s = text.to_string();

    // Fenced code blocks → their inner contents (drop the fence + lang).
    s = FENCED_CODE.replace_all(&s, "$1").to_string();
    // Inline code → the code text.
    s = INLINE_CODE.replace_all(&s, "$1").to_string();

    // Bold before italic — `**word**` must be eaten as bold first or italic
    // matches `*word*` inside it.
    s = BOLD_AST.replace_all(&s, "$1").to_string();
    s = BOLD_UND.replace_all(&s, "$1").to_string();

    // Italic — the lookbehind/ahead via captured non-marker chars preserves
    // surrounding context (we re-insert it during replace).
    s = ITALIC_AST.replace_all(&s, |c: &regex::Captures| {
        let full = c.get(0).unwrap().as_str();
        let inner = c.get(1).map_or("", |m| m.as_str());
        // First char is the leading non-* (or empty for start-of-string),
        // last char is the trailing non-* (or empty for end-of-string).
        let leading = if full.starts_with('*') { "" } else { &full[..1] };
        let trailing = if full.ends_with('*') { "" } else { &full[full.len() - 1..] };
        format!("{leading}{inner}{trailing}")
    }).to_string();
    s = ITALIC_UND.replace_all(&s, |c: &regex::Captures| {
        let full = c.get(0).unwrap().as_str();
        let inner = c.get(1).map_or("", |m| m.as_str());
        let leading = if full.starts_with('_') { "" } else { &full[..1] };
        let trailing = if full.ends_with('_') { "" } else { &full[full.len() - 1..] };
        format!("{leading}{inner}{trailing}")
    }).to_string();

    s = STRIKETHROUGH.replace_all(&s, "$1").to_string();

    // Block-level. Headings, lists, blockquote markers, hr.
    s = HEADING.replace_all(&s, "").to_string();
    s = UNORDERED_LIST.replace_all(&s, "").to_string();
    s = ORDERED_LIST.replace_all(&s, "").to_string();
    s = BLOCKQUOTE.replace_all(&s, "").to_string();
    s = HR.replace_all(&s, "").to_string();

    // Images first (they use the same `[...](...)` shape as links but with !).
    s = IMAGE.replace_all(&s, "").to_string();
    s = LINK.replace_all(&s, "$1").to_string();

    // Collapse 3+ consecutive blank lines.
    s = MULTI_BLANK.replace_all(&s, "\n\n").to_string();

    s.trim().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bold_and_italic() {
        assert_eq!(strip("**Hello** *world*"), "Hello world");
    }

    #[test]
    fn inline_and_block_code() {
        assert_eq!(strip("Run `cargo build` to compile."), "Run cargo build to compile.");
        assert_eq!(strip("```rust\nfn main() {}\n```"), "fn main() {}");
    }

    #[test]
    fn headings_and_lists() {
        let input = "# Title\n\n- one\n- two\n\n1. first\n2. second";
        let expected = "Title\n\none\ntwo\n\nfirst\nsecond";
        assert_eq!(strip(input), expected);
    }

    #[test]
    fn links_and_images() {
        assert_eq!(strip("See [the docs](https://example.com)."), "See the docs.");
        assert_eq!(strip("![alt](image.png) caption"), "caption");
    }

    #[test]
    fn blockquote_and_hr() {
        let input = "> wisdom\n\n---\n\nbody";
        let out = strip(input);
        assert!(out.contains("wisdom"));
        assert!(out.contains("body"));
        assert!(!out.contains("---"));
        assert!(!out.contains(">"));
    }

    #[test]
    fn empty_and_plain() {
        assert_eq!(strip(""), "");
        assert_eq!(strip("plain text with no formatting"), "plain text with no formatting");
    }

    #[test]
    fn nested_markup() {
        // Bold containing italic
        assert_eq!(strip("**bold *italic* bold**"), "bold italic bold");
    }
}
