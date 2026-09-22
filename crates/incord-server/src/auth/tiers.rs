//! Subscription tiers and their per-tier limits.
//!
//! This is the single source of truth for what a user on each tier can do:
//! how fast (rpm/tpm), how many API keys they can mint, and whether there's a
//! daily message cap. Both the self-service `/v1/keys` endpoint and the
//! per-request rate-limit check read from here so the limits stay consistent.
//!
//! Pricing for these tiers lives in `PRICING_PLAN.md` and the frontend's
//! `TIERS` constant. Keep all three in sync when adjusting limits.

#[derive(Debug, Clone, Copy)]
pub struct TierLimits {
    /// Requests per minute (per API key).
    pub rpm: u32,
    /// Tokens per minute (per API key).
    pub tpm: u32,
    /// Maximum number of active API keys this user is allowed to hold.
    pub key_count_max: u32,
    /// Daily message cap. `None` = unlimited (subject to rpm/tpm only).
    pub daily_messages: Option<u32>,
    /// Maximum audio length per STT or TTS call, in seconds, for **API
    /// calls only** (i.e. when authenticated via API key). Platform users
    /// (web session, guest) are NOT subject to this — the limit is what
    /// programmatic-access customers get on their plan, not what we put
    /// in front of our own UX. None = unlimited.
    pub audio_max_seconds: Option<u32>,
}

/// Returns the limits for a tier name. Unknown tier strings fall back to
/// "free" — never elevate an unknown tier into a higher quota by accident.
pub fn limits_for_tier(tier: &str) -> TierLimits {
    match tier {
        // Free tier gets ZERO API keys — must upgrade to a paid plan to mint
        // any. Free users can still use the chat UI (web session), STT, TTS,
        // and the daily-100-message quota — they just can't create programmatic
        // API keys until they subscribe.
        "free" => TierLimits {
            rpm: 30,
            tpm: 30_000,
            key_count_max: 0,
            daily_messages: Some(100),
            audio_max_seconds: Some(5),
        },
        "starter" => TierLimits {
            rpm: 200,
            tpm: 500_000,
            key_count_max: 3,
            daily_messages: Some(1000),
            audio_max_seconds: Some(10),
        },
        "pro" => TierLimits {
            rpm: 1000,
            tpm: 5_000_000,
            key_count_max: 10,
            daily_messages: None,
            audio_max_seconds: Some(30),
        },
        "max" => TierLimits {
            rpm: 5000,
            tpm: 50_000_000,
            key_count_max: 50,
            daily_messages: None,
            audio_max_seconds: Some(60),
        },
        "team" => TierLimits {
            rpm: 10_000,
            tpm: 100_000_000,
            key_count_max: 100,
            daily_messages: None,
            audio_max_seconds: Some(60),
        },
        "enterprise" => TierLimits {
            rpm: u32::MAX,
            tpm: u32::MAX,
            key_count_max: u32::MAX,
            daily_messages: None,
            audio_max_seconds: None,
        },
        _ => TierLimits {
            rpm: 30,
            tpm: 30_000,
            key_count_max: 1,
            daily_messages: Some(100),
            audio_max_seconds: Some(5),
        },
    }
}

pub const TIER_NAMES: &[&str] = &["free", "starter", "pro", "max", "team", "enterprise"];

pub fn tier_exists(tier: &str) -> bool {
    TIER_NAMES.iter().any(|t| *t == tier)
}

/// Resolve the audio-duration cap (in seconds) for the calling identity.
///
/// Hard rule: **only API-key callers are limited.** Platform users (logged-in
/// session) and guests can use any audio length they want — the cap is a
/// programmatic-access feature so customers paying for API have predictable
/// per-call cost, not something we put in front of our own web UX.
///
/// Returns `None` when no cap should be applied (platform / guest / enterprise key).
pub fn audio_cap_for_identity(
    db: &std::sync::Arc<redb::Database>,
    identity: &crate::auth::middleware::AuthIdentity,
) -> Option<u32> {
    use crate::auth::middleware::AuthIdentity;
    match identity {
        AuthIdentity::ApiKey { key } => {
            // Look up the key holder's user record → tier → audio cap.
            // If no user record (ad-hoc admin-issued key for a label),
            // default to the most generous limit so admin-internal keys
            // aren't accidentally throttled.
            match crate::auth::users::get_user(db, &key.tenant_id, &key.user_id) {
                Ok(Some(user)) => limits_for_tier(&user.tier).audio_max_seconds,
                _ => None,
            }
        }
        // Platform / guest: unlimited audio length.
        AuthIdentity::User { .. } | AuthIdentity::Guest { .. } => None,
    }
}
