//! Per-tensor precision plans (roadmap Milestone 5: mixed precision).
//!
//! A plan gives every weight a storage format from its role and layer:
//!
//! ```text
//! head=q8,embed=q8,down@0-3=q8,attn=q8
//! ```
//!
//! Comma-separated `target[@layers]=format` rules over a default format; the
//! last matching rule wins. Targets are tensor classes (`embed`, `head`, `q`,
//! `k`, `v`, `o`, `gate`, `up`, `down`, DeltaNet `in_qkv`, `in_a`, `in_b`,
//! `in_z`, `out`) or groups (`attn`, `qkv`, `mlp`, `table`, `all`). Layers are
//! `N` or `A-B` (inclusive); rules with layers never match the embedding or
//! head. Formats: `f16`, `q8`, `q4`, `q3`, `q2`, `e8` (E8-lattice 2-bit).

use std::ops::RangeInclusive;

/// Weight storage format.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Fmt {
    F16,
    Q8,
    Q4,
    Q3,
    Q2,
    E8,
}

impl Fmt {
    /// Per-weight format byte used in `.incord` files.
    pub fn id(self) -> u8 {
        match self {
            Fmt::F16 => 0,
            Fmt::Q4 => 1,
            Fmt::Q8 => 2,
            other => other.low_bits().expect("sub-4-bit format").format_id(),
        }
    }

    /// The sub-4-bit layout, for Q3 / Q2 / E8.
    pub fn low_bits(self) -> Option<crate::quant::LowBits> {
        match self {
            Fmt::Q3 => Some(crate::quant::LowBits::Q3),
            Fmt::Q2 => Some(crate::quant::LowBits::Q2),
            Fmt::E8 => Some(crate::quant::LowBits::E8),
            _ => None,
        }
    }
}

impl std::str::FromStr for Fmt {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, String> {
        match s {
            "f16" => Ok(Fmt::F16),
            "q8" => Ok(Fmt::Q8),
            "q4" => Ok(Fmt::Q4),
            "q3" => Ok(Fmt::Q3),
            "q2" => Ok(Fmt::Q2),
            "e8" => Ok(Fmt::E8),
            _ => Err(format!("unknown format '{s}' (expected f16, q8, q4, q3, q2 or e8)")),
        }
    }
}

/// Role of a weight in the model.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Class {
    Embed,
    Head,
    Q,
    K,
    V,
    O,
    Gate,
    Up,
    Down,
    InQkv,
    InA,
    InB,
    InZ,
    Out,
}

impl Class {
    /// Class of a projection from its short field name (`q_proj`, ...).
    pub fn from_proj_name(name: &str) -> Option<Class> {
        Some(match name {
            "q_proj" => Class::Q,
            "k_proj" => Class::K,
            "v_proj" => Class::V,
            "o_proj" => Class::O,
            "gate_proj" => Class::Gate,
            "up_proj" => Class::Up,
            "down_proj" => Class::Down,
            "in_proj_qkv" => Class::InQkv,
            "in_proj_a" => Class::InA,
            "in_proj_b" => Class::InB,
            "in_proj_z" => Class::InZ,
            "out_proj" => Class::Out,
            _ => return None,
        })
    }

    /// Class and layer of a Hugging Face tensor name, e.g.
    /// `model.layers.7.mlp.down_proj.weight` → `(Down, Some(7))`.
    pub fn from_hf_name(name: &str) -> Option<(Class, Option<usize>)> {
        if name.ends_with("embed_tokens.weight") {
            return Some((Class::Embed, None));
        }
        if name.ends_with("lm_head.weight") {
            return Some((Class::Head, None));
        }
        let rest = name.split("layers.").nth(1)?;
        let layer: usize = rest.split('.').next()?.parse().ok()?;
        let proj = name.trim_end_matches(".weight").rsplit('.').next()?;
        Some((Class::from_proj_name(proj)?, Some(layer)))
    }

    fn in_group(self, target: &str) -> bool {
        use Class::*;
        match target {
            "all" => true,
            "table" => matches!(self, Embed | Head),
            "attn" => matches!(self, Q | K | V | O | InQkv | InA | InB | InZ | Out),
            "qkv" => matches!(self, Q | K | V | InQkv),
            "mlp" => matches!(self, Gate | Up | Down),
            _ => false,
        }
    }

    fn name(self) -> &'static str {
        use Class::*;
        match self {
            Embed => "embed",
            Head => "head",
            Q => "q",
            K => "k",
            V => "v",
            O => "o",
            Gate => "gate",
            Up => "up",
            Down => "down",
            InQkv => "in_qkv",
            InA => "in_a",
            InB => "in_b",
            InZ => "in_z",
            Out => "out",
        }
    }
}

const GROUPS: &[&str] = &["all", "table", "attn", "qkv", "mlp"];
const CLASSES: &[&str] = &[
    "embed", "head", "q", "k", "v", "o", "gate", "up", "down", "in_qkv", "in_a", "in_b", "in_z",
    "out",
];

#[derive(Clone, Debug, PartialEq)]
struct Rule {
    target: String,
    layers: Option<RangeInclusive<usize>>,
    fmt: Fmt,
}

/// A default format plus override rules.
#[derive(Clone, Debug, PartialEq)]
pub struct Plan {
    pub default: Fmt,
    rules: Vec<Rule>,
}

impl Plan {
    /// Every weight in `default`.
    pub fn uniform(default: Fmt) -> Self {
        Self {
            default,
            rules: Vec::new(),
        }
    }

    /// Parse `target[@layers]=fmt,...` over `default`. Empty spec = uniform.
    pub fn parse(default: Fmt, spec: &str) -> Result<Self, String> {
        let mut rules = Vec::new();
        for part in spec.split(',').map(str::trim).filter(|p| !p.is_empty()) {
            let (lhs, fmt) = part
                .split_once('=')
                .ok_or_else(|| format!("plan rule '{part}' needs '=format'"))?;
            let fmt: Fmt = fmt.trim().parse()?;
            let (target, layers) = match lhs.split_once('@') {
                Some((t, l)) => (t.trim(), Some(parse_layers(l.trim())?)),
                None => (lhs.trim(), None),
            };
            if !GROUPS.contains(&target) && !CLASSES.contains(&target) {
                return Err(format!("unknown plan target '{target}'"));
            }
            rules.push(Rule {
                target: target.to_string(),
                layers,
                fmt,
            });
        }
        Ok(Self { default, rules })
    }

    /// Format for a weight of `class` in `layer` (None for embed/head).
    pub fn format_for(&self, class: Class, layer: Option<usize>) -> Fmt {
        let mut fmt = self.default;
        for r in &self.rules {
            let class_ok = r.target == class.name() || class.in_group(&r.target);
            let layer_ok = match (&r.layers, layer) {
                (None, _) => true,
                (Some(range), Some(l)) => range.contains(&l),
                (Some(_), None) => false,
            };
            if class_ok && layer_ok {
                fmt = r.fmt;
            }
        }
        fmt
    }
}

fn parse_layers(s: &str) -> Result<RangeInclusive<usize>, String> {
    let bad = || format!("bad layer range '{s}' (expected N or A-B)");
    match s.split_once('-') {
        Some((a, b)) => {
            let (a, b): (usize, usize) = (a.parse().map_err(|_| bad())?, b.parse().map_err(|_| bad())?);
            if a > b {
                return Err(bad());
            }
            Ok(a..=b)
        }
        None => {
            let n: usize = s.parse().map_err(|_| bad())?;
            Ok(n..=n)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uniform_plan_uses_default() {
        let p = Plan::parse(Fmt::Q4, "").unwrap();
        assert_eq!(p, Plan::uniform(Fmt::Q4));
        assert_eq!(p.format_for(Class::Down, Some(3)), Fmt::Q4);
        assert_eq!(p.format_for(Class::Head, None), Fmt::Q4);
    }

    #[test]
    fn rules_groups_layers_and_last_match_wins() {
        let p = Plan::parse(Fmt::Q4, "table=q8, down@0-3=q8, attn=q8, q@5=f16").unwrap();
        assert_eq!(p.format_for(Class::Embed, None), Fmt::Q8);
        assert_eq!(p.format_for(Class::Head, None), Fmt::Q8);
        assert_eq!(p.format_for(Class::Down, Some(0)), Fmt::Q8);
        assert_eq!(p.format_for(Class::Down, Some(3)), Fmt::Q8);
        assert_eq!(p.format_for(Class::Down, Some(4)), Fmt::Q4);
        assert_eq!(p.format_for(Class::Gate, Some(0)), Fmt::Q4);
        assert_eq!(p.format_for(Class::K, Some(9)), Fmt::Q8);
        assert_eq!(p.format_for(Class::Q, Some(5)), Fmt::F16);
        assert_eq!(p.format_for(Class::Q, Some(6)), Fmt::Q8);
    }

    #[test]
    fn layer_rules_never_touch_tables() {
        let p = Plan::parse(Fmt::Q4, "all@0-41=q8").unwrap();
        assert_eq!(p.format_for(Class::Embed, None), Fmt::Q4);
        assert_eq!(p.format_for(Class::O, Some(41)), Fmt::Q8);
    }

    #[test]
    fn rejects_bad_specs() {
        for bad in ["down", "down=q5", "dwn=q8", "down@5-2=q8", "down@x=q8"] {
            assert!(Plan::parse(Fmt::Q4, bad).is_err(), "{bad} should be rejected");
        }
    }

    #[test]
    fn classes_from_hf_names() {
        assert_eq!(
            Class::from_hf_name("model.layers.7.mlp.down_proj.weight"),
            Some((Class::Down, Some(7)))
        );
        assert_eq!(
            Class::from_hf_name("model.layers.12.self_attn.q_proj.weight"),
            Some((Class::Q, Some(12)))
        );
        assert_eq!(Class::from_hf_name("model.embed_tokens.weight"), Some((Class::Embed, None)));
        assert_eq!(Class::from_hf_name("lm_head.weight"), Some((Class::Head, None)));
        assert_eq!(Class::from_hf_name("model.layers.3.input_layernorm.weight"), None);
    }
}
