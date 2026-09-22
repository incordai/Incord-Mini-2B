//! Normalise EU-style numeric formats in user queries to US format before
//! sending them to brain for retrieval. Brain stores corpus content in US
//! format (`$80,000.21`, `2.13%`); sparse BM25 retrieval is token-exact
//! and won't match `80.384,21` against a corpus that contains `80,384.21`.
//!
//! Used **only for the brain query string** — the user's actual message
//! reaches the LLM untouched.
//!
//! Handles the most common locale variants:
//!   - German / Italian : `80.384,21` → `80,384.21`
//!   - French / Polish  : `80 384,21` (regular space)  → `80,384.21`
//!                        `80\u{00A0}384,21` (NBSP)    → `80,384.21`
//!                        `80\u{202F}384,21` (NNBSP)   → `80,384.21`
//!   - German thousands without decimal, with currency context:
//!                        `$80.384` → `$80,384`  (only when the trailing
//!                                                group is exactly 3 digits
//!                                                AND preceded by `$` `€`
//!                                                `£` `¥` — otherwise
//!                                                `0.384` could legitimately
//!                                                be a US decimal)
//!   - Already-US strings (`80,384.21`) pass through unchanged.

use once_cell::sync::Lazy;
use regex::Regex;

// Full EU format: integer.thousands(.thousands)*,decimals -> US thousands+decimal.
// Example: 80.384,21 -> 80,384.21 ; 1.234.567,89 -> 1,234,567.89.
static RE_EU_FULL_DOT: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"(\d{1,3}(?:\.\d{3})+),(\d+)").expect("EU dot-thousands regex")
});

// Full EU format with space (regular, NBSP, NNBSP) as thousands separator.
static RE_EU_FULL_SPACE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"(\d{1,3}(?:[ \u{00A0}\u{202F}]\d{3})+),(\d+)")
        .expect("EU space-thousands regex")
});

// Currency-prefixed German thousands without decimal: $/€/£/¥ N.NNN where the
// post-dot group is exactly 3 digits and not followed by another digit.
// Example: $80.384 -> $80,384.
//
// `\b` enforces "not followed by another digit": `\d` is a word character,
// so `\d{3}\b` only matches when the next char is non-word or end-of-input.
// (Rust's `regex` crate has no lookaround — `\b` gets us the same outcome
// for digit/non-digit boundaries.)
static RE_CURRENCY_DOT_THOUSANDS: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"([\$€£¥])(\d{1,3})\.(\d{3})\b").expect("currency dot-thousands regex")
});

/// Normalise the query string. Applied in order:
///   1. EU full format with dot thousands (`80.384,21` → `80,384.21`)
///   2. EU full format with space thousands (`80 384,21` → `80,384.21`)
///   3. Currency-prefixed German thousands (`$80.384` → `$80,384`)
///
/// Step 1 must run first so that already-EU-full strings get fully
/// converted before the currency-thousands rule sees them; otherwise
/// `$80.384,21` would partially match step 3 first and produce
/// `$80,384,21` (broken).
pub fn normalize_numbers_to_us(s: &str) -> String {
    if !s.chars().any(|c| c == '.' || c == ',' || c.is_whitespace()) {
        return s.to_string();
    }

    let mut out = RE_EU_FULL_DOT
        .replace_all(s, |c: &regex::Captures| {
            let int_with_dots = &c[1];
            let decimals = &c[2];
            let int_us = int_with_dots.replace('.', ",");
            format!("{int_us}.{decimals}")
        })
        .into_owned();

    out = RE_EU_FULL_SPACE
        .replace_all(&out, |c: &regex::Captures| {
            let int_with_spaces = &c[1];
            let decimals = &c[2];
            let int_us: String = int_with_spaces
                .chars()
                .map(|ch| {
                    if ch == ' ' || ch == '\u{00A0}' || ch == '\u{202F}' {
                        ','
                    } else {
                        ch
                    }
                })
                .collect();
            format!("{int_us}.{decimals}")
        })
        .into_owned();

    out = RE_CURRENCY_DOT_THOUSANDS
        .replace_all(&out, |c: &regex::Captures| {
            let cur = &c[1];
            let int_part = &c[2];
            let group = &c[3];
            format!("{cur}{int_part},{group}")
        })
        .into_owned();

    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn currency_german_thousands_no_decimal() {
        assert_eq!(normalize_numbers_to_us("$80.384"), "$80,384");
        assert_eq!(normalize_numbers_to_us("€1.234"), "€1,234");
        assert_eq!(normalize_numbers_to_us("£999.500"), "£999,500");
    }

    #[test]
    fn full_eu_dot_format() {
        assert_eq!(normalize_numbers_to_us("80.384,21"), "80,384.21");
        assert_eq!(normalize_numbers_to_us("1.234.567,89"), "1,234,567.89");
        // Currency stays attached.
        assert_eq!(normalize_numbers_to_us("$80.384,21"), "$80,384.21");
    }

    #[test]
    fn full_eu_space_format() {
        assert_eq!(normalize_numbers_to_us("80 384,21"), "80,384.21");
        assert_eq!(normalize_numbers_to_us("1 234 567,89"), "1,234,567.89");
        // NBSP
        assert_eq!(normalize_numbers_to_us("80\u{00A0}384,21"), "80,384.21");
        // Narrow NBSP
        assert_eq!(normalize_numbers_to_us("80\u{202F}384,21"), "80,384.21");
    }

    #[test]
    fn already_us_format_unchanged() {
        assert_eq!(normalize_numbers_to_us("80,384.21"), "80,384.21");
        assert_eq!(normalize_numbers_to_us("$80,000"), "$80,000");
        assert_eq!(normalize_numbers_to_us("1,234,567.89"), "1,234,567.89");
    }

    #[test]
    fn us_decimals_without_currency_unchanged() {
        // "0.384" without a currency prefix should NOT be rewritten — it's
        // a legitimate US decimal (think coordinates, percentages, etc.).
        assert_eq!(normalize_numbers_to_us("RSI is 0.384"), "RSI is 0.384");
        assert_eq!(normalize_numbers_to_us("80.38"), "80.38");
    }

    #[test]
    fn percentage_strings_pass_through() {
        // EU percentages with comma decimal get normalised when paired with
        // a thousands group (rare); plain "2,13%" without the thousands dot
        // is intentionally NOT rewritten — could be US "2,13" comma-list
        // ambiguity. Keep this conservative.
        assert_eq!(normalize_numbers_to_us("2.13%"), "2.13%");
        assert_eq!(normalize_numbers_to_us("12.345,67%"), "12,345.67%");
    }

    #[test]
    fn full_btc_query_normalises() {
        let in_ = "What is happening with Bitcoin (BTC)? Current price: $80.384, +2.13% in 24h.";
        let out = normalize_numbers_to_us(in_);
        assert!(out.contains("$80,384"), "got: {out}");
        // The "2.13%" stays as-is (US decimal, no thousands ambiguity).
        assert!(out.contains("2.13%"), "got: {out}");
    }

    #[test]
    fn no_numbers_unchanged() {
        let s = "Hello world, what is going on today?";
        assert_eq!(normalize_numbers_to_us(s), s);
    }
}
