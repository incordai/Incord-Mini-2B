//! Task benchmarks: does a compressed model still answer hard questions the
//! way the original does?
//!
//! Complements the logit-level metrics in `eval` (KL, top-1, perplexity) with
//! graded tasks. The data comes from `eval/fetch-tasks.sh` (Hugging Face rows
//! API pages, one directory per task); nothing here is committed data.
//!
//! Multiple-choice tasks are scored by log-likelihood, like
//! lm-evaluation-harness: every choice is a continuation of a context and the
//! highest-scoring continuation is the prediction. No sampling is involved, so
//! two models can be compared question by question.
//!
//! | task       | prompt                               | score                      |
//! |------------|--------------------------------------|----------------------------|
//! | arc        | question + lettered choices, "Answer:" | log p(" A".." E")        |
//! | mmlu       | subject header + lettered choices      | log p(" A".." D")        |
//! | hellaswag  | activity: context                      | log p(ending) / chars    |
//! | winogrande | sentence up to the blank, each option  | log p(rest of sentence)  |
//! | gsm8k      | chat prompt, greedy answer             | final number == gold     |

use std::path::Path;

pub const LETTERS: [&str; 5] = ["A", "B", "C", "D", "E"];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Task {
    Arc,
    Mmlu,
    Hellaswag,
    Winogrande,
    Gsm8k,
}

impl Task {
    pub const ALL: [Task; 5] = [Task::Arc, Task::Mmlu, Task::Hellaswag, Task::Winogrande, Task::Gsm8k];

    pub fn name(self) -> &'static str {
        match self {
            Task::Arc => "arc",
            Task::Mmlu => "mmlu",
            Task::Hellaswag => "hellaswag",
            Task::Winogrande => "winogrande",
            Task::Gsm8k => "gsm8k",
        }
    }

    pub fn parse(s: &str) -> Result<Task, String> {
        Task::ALL
            .into_iter()
            .find(|t| t.name() == s)
            .ok_or_else(|| format!("unknown task {s:?} (tasks: arc, mmlu, hellaswag, winogrande, gsm8k)"))
    }

    /// Multiple-choice score uses log-probability per character of the
    /// continuation (choices of different lengths), otherwise the raw sum.
    pub fn length_normalized(self) -> bool {
        self == Task::Hellaswag
    }
}

/// One multiple-choice question: choice `i` is `contexts[i]` (or the single
/// shared context) followed by `continuations[i]`.
#[derive(Clone, Debug, PartialEq)]
pub struct McItem {
    pub id: String,
    pub contexts: Vec<String>,
    pub continuations: Vec<String>,
    pub gold: usize,
}

impl McItem {
    pub fn choices(&self) -> usize {
        self.continuations.len()
    }

    pub fn context(&self, i: usize) -> &str {
        if self.contexts.len() == 1 { &self.contexts[0] } else { &self.contexts[i] }
    }
}

/// One free-answer question graded on its final number.
#[derive(Clone, Debug, PartialEq)]
pub struct GenItem {
    pub id: String,
    pub question: String,
    pub answer: f64,
}

pub enum Items {
    Mc(Vec<McItem>),
    Gen(Vec<GenItem>),
}

impl Items {
    pub fn len(&self) -> usize {
        match self {
            Items::Mc(v) => v.len(),
            Items::Gen(v) => v.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Keep `limit` items spread evenly over the whole set (deterministic, so
    /// every model sees the same questions; MMLU keeps all its subjects).
    pub fn subsample(self, limit: usize) -> Items {
        fn pick<T>(v: Vec<T>, limit: usize) -> Vec<T> {
            let n = v.len();
            if limit == 0 || limit >= n {
                return v;
            }
            let keep: Vec<usize> = (0..limit).map(|j| j * n / limit).collect();
            v.into_iter()
                .enumerate()
                .filter(|(i, _)| keep.binary_search(i).is_ok())
                .map(|(_, x)| x)
                .collect()
        }
        match self {
            Items::Mc(v) => Items::Mc(pick(v, limit)),
            Items::Gen(v) => Items::Gen(pick(v, limit)),
        }
    }
}

/// Load every row of `dir/<task>/page-*.json`, in file order.
fn load_rows(dir: &Path, task: Task) -> Result<Vec<serde_json::Value>, String> {
    let tdir = dir.join(task.name());
    let mut pages: Vec<_> = std::fs::read_dir(&tdir)
        .map_err(|e| format!("read {}: {e} (run eval/fetch-tasks.sh first)", tdir.display()))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "json"))
        .collect();
    pages.sort();
    let mut rows = Vec::new();
    for p in pages {
        let text = std::fs::read_to_string(&p).map_err(|e| format!("read {}: {e}", p.display()))?;
        let v: serde_json::Value =
            serde_json::from_str(&text).map_err(|e| format!("parse {}: {e}", p.display()))?;
        let page = v["rows"].as_array().ok_or_else(|| format!("{}: no rows", p.display()))?;
        rows.extend(page.iter().map(|r| r["row"].clone()));
    }
    if rows.is_empty() {
        return Err(format!("{}: no data", tdir.display()));
    }
    Ok(rows)
}

fn str_field<'a>(row: &'a serde_json::Value, key: &str) -> Result<&'a str, String> {
    row[key].as_str().ok_or_else(|| format!("row field {key:?} missing: {row}"))
}

fn str_list(v: &serde_json::Value) -> Result<Vec<String>, String> {
    v.as_array()
        .ok_or_else(|| format!("expected a list: {v}"))?
        .iter()
        .map(|s| s.as_str().map(str::to_string).ok_or_else(|| format!("expected a string: {s}")))
        .collect()
}

pub fn load(dir: &Path, task: Task) -> Result<Items, String> {
    let rows = load_rows(dir, task)?;
    let mut mc = Vec::new();
    let mut gen_items = Vec::new();
    for (i, row) in rows.iter().enumerate() {
        match task {
            Task::Arc => mc.push(arc_item(row)?),
            Task::Mmlu => mc.push(mmlu_item(row, i)?),
            Task::Hellaswag => mc.push(hellaswag_item(row, i)?),
            Task::Winogrande => mc.push(winogrande_item(row, i)?),
            Task::Gsm8k => gen_items.push(gsm8k_item(row, i)?),
        }
    }
    Ok(if task == Task::Gsm8k { Items::Gen(gen_items) } else { Items::Mc(mc) })
}

/// Question, lettered choices, then "Answer:"; continuations " A", " B", ...
fn lettered(header: &str, question: &str, choices: &[String]) -> Result<(String, Vec<String>), String> {
    if choices.len() < 2 || choices.len() > LETTERS.len() {
        return Err(format!("{} choices (supported 2..={})", choices.len(), LETTERS.len()));
    }
    let mut ctx = format!("{header}{}\n", question.trim());
    for (l, c) in LETTERS.iter().zip(choices) {
        ctx.push_str(&format!("{l}. {}\n", c.trim()));
    }
    ctx.push_str("Answer:");
    let conts = LETTERS[..choices.len()].iter().map(|l| format!(" {l}")).collect();
    Ok((ctx, conts))
}

fn arc_item(row: &serde_json::Value) -> Result<McItem, String> {
    let texts = str_list(&row["choices"]["text"])?;
    let labels = str_list(&row["choices"]["label"])?;
    let key = str_field(row, "answerKey")?;
    let gold = labels
        .iter()
        .position(|l| l == key)
        .ok_or_else(|| format!("arc: answer {key:?} not among labels {labels:?}"))?;
    let (ctx, conts) = lettered("", &format!("Question: {}", str_field(row, "question")?), &texts)?;
    Ok(McItem { id: str_field(row, "id")?.to_string(), contexts: vec![ctx], continuations: conts, gold })
}

fn mmlu_item(row: &serde_json::Value, i: usize) -> Result<McItem, String> {
    let subject = str_field(row, "subject")?;
    let header = format!(
        "The following is a multiple choice question about {}.\n\n",
        subject.replace('_', " ")
    );
    let choices = str_list(&row["choices"])?;
    let gold = row["answer"].as_u64().ok_or("mmlu: answer missing")? as usize;
    let (ctx, conts) = lettered(&header, str_field(row, "question")?, &choices)?;
    if gold >= conts.len() {
        return Err(format!("mmlu: answer {gold} out of range"));
    }
    Ok(McItem { id: format!("{subject}-{i}"), contexts: vec![ctx], continuations: conts, gold })
}

/// HellaSwag text cleanup, as in lm-evaluation-harness: WikiHow "[title]"
/// markers become sentence breaks, other bracketed tags are dropped (plus a
/// final trim, so a dropped leading tag leaves no stray space).
pub fn hellaswag_clean(text: &str) -> String {
    let text = text.trim().replace(" [title]", ". ");
    let mut out = String::with_capacity(text.len());
    let mut depth = 0usize;
    for ch in text.chars() {
        match ch {
            '[' => depth += 1,
            ']' if depth > 0 => depth -= 1,
            _ if depth == 0 => out.push(ch),
            _ => {}
        }
    }
    out.replace("  ", " ").trim().to_string()
}

fn capitalize(s: &str) -> String {
    let mut c = s.chars();
    match c.next() {
        Some(f) => f.to_uppercase().chain(c).collect(),
        None => String::new(),
    }
}

fn hellaswag_item(row: &serde_json::Value, i: usize) -> Result<McItem, String> {
    let ctx = format!("{} {}", str_field(row, "ctx_a")?, capitalize(str_field(row, "ctx_b")?));
    let query = hellaswag_clean(&format!("{}: {ctx}", str_field(row, "activity_label")?));
    let conts = str_list(&row["endings"])?
        .iter()
        .map(|e| format!(" {}", hellaswag_clean(e)))
        .collect::<Vec<_>>();
    let gold: usize = str_field(row, "label")?
        .parse()
        .map_err(|_| format!("hellaswag: bad label in row {i}"))?;
    if gold >= conts.len() {
        return Err(format!("hellaswag: label {gold} out of range"));
    }
    Ok(McItem { id: format!("hs-{i}"), contexts: vec![query], continuations: conts, gold })
}

/// Partial scoring: each option fills the blank, and the model scores the
/// (identical) rest of the sentence after it.
fn winogrande_item(row: &serde_json::Value, i: usize) -> Result<McItem, String> {
    let sentence = str_field(row, "sentence")?;
    let blank = sentence.find('_').ok_or_else(|| format!("winogrande: no blank in row {i}"))?;
    let (before, after) = (&sentence[..blank], sentence[blank + 1..].trim());
    let contexts = ["option1", "option2"]
        .iter()
        .map(|k| Ok(format!("{before}{}", str_field(row, k)?)))
        .collect::<Result<Vec<_>, String>>()?;
    let gold = match str_field(row, "answer")? {
        "1" => 0,
        "2" => 1,
        a => return Err(format!("winogrande: bad answer {a:?}")),
    };
    let cont = format!(" {after}");
    Ok(McItem { id: format!("wg-{i}"), contexts, continuations: vec![cont.clone(), cont], gold })
}

fn gsm8k_item(row: &serde_json::Value, i: usize) -> Result<GenItem, String> {
    let answer = str_field(row, "answer")?;
    let gold = answer
        .rsplit("####")
        .next()
        .and_then(|s| parse_number(s.trim()))
        .ok_or_else(|| format!("gsm8k: no final answer in row {i}"))?;
    Ok(GenItem { id: format!("gsm-{i}"), question: str_field(row, "question")?.to_string(), answer: gold })
}

/// The chat message sent to the model for a GSM8K question.
pub fn gsm8k_prompt(question: &str) -> String {
    format!(
        "{}\n\nSolve the problem step by step. End with a final line of the form \
         \"The answer is N\" where N is a number.",
        question.trim()
    )
}

/// Parse a number like "1,234", "-5", "$18", "3.50".
pub fn parse_number(s: &str) -> Option<f64> {
    let t: String = s.chars().filter(|c| !matches!(c, ',' | '$' | ' ')).collect();
    t.trim_end_matches('.').parse().ok()
}

/// Every number in `text`, in order (commas inside digits allowed).
fn numbers(text: &str) -> Vec<f64> {
    let b = text.as_bytes();
    let mut out = Vec::new();
    let mut i = 0;
    while i < b.len() {
        let neg = b[i] == b'-' && i + 1 < b.len() && b[i + 1].is_ascii_digit();
        if b[i].is_ascii_digit() || neg {
            let start = i;
            i += 1;
            while i < b.len()
                && (b[i].is_ascii_digit()
                    || ((b[i] == b',' || b[i] == b'.') && i + 1 < b.len() && b[i + 1].is_ascii_digit()))
            {
                i += 1;
            }
            if let Some(v) = parse_number(&text[start..i]) {
                out.push(v);
            }
        } else {
            i += 1;
        }
    }
    out
}

/// The model's final answer: the first number after the last "answer is",
/// otherwise the last number in the text.
pub fn extract_answer(text: &str) -> Option<f64> {
    let lower = text.to_lowercase();
    if let Some(p) = lower.rfind("answer is") {
        if let Some(&v) = numbers(&text[p..]).first() {
            return Some(v);
        }
    }
    numbers(text).last().copied()
}

pub fn same_number(a: f64, b: f64) -> bool {
    (a - b).abs() <= 1e-6 * b.abs().max(1.0)
}

/// Index of the best choice. Ties go to the lower index.
pub fn argmax(scores: &[f64]) -> usize {
    let mut best = 0;
    for (i, &s) in scores.iter().enumerate() {
        if s > scores[best] {
            best = i;
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn arc_letters_and_gold() {
        let row = json!({"id": "x1", "question": "Which is a gas?",
            "choices": {"text": ["rock", "steam", "ice"], "label": ["1", "2", "3"]}, "answerKey": "2"});
        let it = arc_item(&row).unwrap();
        assert_eq!(it.gold, 1);
        assert_eq!(it.continuations, vec![" A", " B", " C"]);
        assert_eq!(it.context(2), "Question: Which is a gas?\nA. rock\nB. steam\nC. ice\nAnswer:");
    }

    #[test]
    fn mmlu_header_and_answer() {
        let row = json!({"question": "2+2?", "subject": "elementary_mathematics",
            "choices": ["3", "4", "5", "6"], "answer": 1});
        let it = mmlu_item(&row, 7).unwrap();
        assert_eq!(it.gold, 1);
        assert!(it.context(0).starts_with("The following is a multiple choice question about elementary mathematics.\n\n2+2?\nA. 3\n"));
        assert_eq!(it.id, "elementary_mathematics-7");
    }

    #[test]
    fn hellaswag_cleanup() {
        assert_eq!(hellaswag_clean(" Wash hands [title] Rinse [step] well "), "Wash hands. Rinse well");
        let row = json!({"activity_label": "Cooking", "ctx_a": "A man cuts onions.", "ctx_b": "he",
            "endings": ["cries.", "[header] sings."], "label": "0"});
        let it = hellaswag_item(&row, 3).unwrap();
        assert_eq!(it.context(0), "Cooking: A man cuts onions. He");
        assert_eq!(it.continuations, vec![" cries.", " sings."]);
    }

    #[test]
    fn winogrande_partial_scoring() {
        let row = json!({"sentence": "Sarah was better than Maria so _ got easy cases.",
            "option1": "Sarah", "option2": "Maria", "answer": "2"});
        let it = winogrande_item(&row, 0).unwrap();
        assert_eq!(it.gold, 1);
        assert_eq!(it.context(0), "Sarah was better than Maria so Sarah");
        assert_eq!(it.context(1), "Sarah was better than Maria so Maria");
        assert_eq!(it.continuations[0], " got easy cases.");
        assert_eq!(it.continuations[0], it.continuations[1]);
    }

    #[test]
    fn gsm8k_gold_and_extraction() {
        let row = json!({"question": "q", "answer": "work\n#### 1,234"});
        assert_eq!(gsm8k_item(&row, 0).unwrap().answer, 1234.0);
        assert_eq!(extract_answer("so 3 + 4 = 7. The answer is $18."), Some(18.0));
        assert_eq!(extract_answer("The answer is: 1,050 dollars"), Some(1050.0));
        assert_eq!(extract_answer("we get -5 then 12"), Some(12.0));
        assert_eq!(extract_answer("It costs 3.50 total."), Some(3.5));
        assert_eq!(extract_answer("no digits"), None);
        assert!(same_number(18.0, 18.0000000001));
        assert!(!same_number(18.0, 18.5));
    }

    #[test]
    fn subsample_is_even_and_deterministic() {
        let v: Vec<GenItem> = (0..10)
            .map(|i| GenItem { id: i.to_string(), question: String::new(), answer: 0.0 })
            .collect();
        let Items::Gen(s) = Items::Gen(v).subsample(4) else { unreachable!() };
        let ids: Vec<_> = s.iter().map(|g| g.id.as_str()).collect();
        assert_eq!(ids, vec!["0", "2", "5", "7"]);
    }

    #[test]
    fn argmax_ties_go_low() {
        assert_eq!(argmax(&[1.0, 3.0, 3.0]), 1);
        assert_eq!(argmax(&[-2.0, -1.0]), 1);
    }
}
