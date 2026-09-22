//! Concrete TtsEngine wrapping the incord-tts crate.
//!
//! Integration approach: lib linkage (not subprocess).
//! The TTS engine crate (separate repo) exposes a `[lib]` target used here.
//! The incord-tts library exposes all modules publicly via lib.rs.
//!
//! Voice model: This 1.7B model has an EMPTY spk_id map — it does NOT use built-in
//! speaker IDs.  Instead it requires a reference WAV for voice cloning.
//! Voices are stored in `model_dir/voices/{name}.wav` (25 voices ship with the model).
//! On synthesize, we:
//!   1. Resolve voices/{voice_name}.wav
//!   2. Extract mel-spectrogram + speaker embedding via the speaker encoder
//!   3. Pass voice_embedding to generate_speech()
//!
//! Audio output: generate_speech returns Vec<f32> (mono, 24 kHz, [-1.0, 1.0]).
//! We convert to 16-bit PCM and encode as WAV in-memory.

use async_trait::async_trait;
use std::path::PathBuf;
use std::sync::Arc;

use incord_tts::audio_features::MelConfig;
use incord_tts::code_predictor::CodePredictorWeights;
use incord_tts::decoder::SpeechDecoderWeights;
use incord_tts::generate_new::{TTSParams, generate_speech};
use incord_tts::speaker_encoder::{SpeakerEncoderWeights, extract_speaker_embedding};
use incord_tts::talker::TalkerWeights;
use incord_tts::tokenizer::TTSTokenizer;

use crate::engine::tts::{TtsEngine, TtsOutput, TtsRequest};
use crate::engine::{ModelEngine, Task};

/// Incord-TTS sample rate (fixed by the model architecture).
const SAMPLE_RATE: u32 = 24_000;

/// Default voice name (must exist in voices/ directory).
const DEFAULT_VOICE: &str = "luna";
const DEFAULT_LANGUAGE: &str = "english";

// ---------------------------------------------------------------------------
// Model handle
// ---------------------------------------------------------------------------

/// Opaque handle to the loaded TTS model weights + tokenizer + voice data.
/// All fields are plain heap data, so Send + Sync holds.
pub struct TtsModelHandle {
    pub talker: TalkerWeights,
    pub predictor: CodePredictorWeights,
    pub decoder: SpeechDecoderWeights,
    /// Speaker encoder for extracting voice embeddings from reference WAV files.
    /// `None` for the customvoice variant — that model has voices baked into the
    /// talker, so synthesis skips the embedding step.
    pub speaker_encoder: Option<SpeakerEncoderWeights>,
    pub tokenizer: TTSTokenizer,
    /// language-name → u32 id map, from config.json.
    pub language_map: std::collections::HashMap<String, u32>,
    /// speaker-name → u32 id map, from config.json. Empty for models that rely on
    /// reference WAVs (cloning) instead of built-in speakers.
    pub speaker_map: std::collections::HashMap<String, u32>,
    /// Directory containing voices/{name}.wav reference files.
    pub voices_dir: PathBuf,
}

// Safety: all fields are Vec<f32>/Vec<half::f16>/etc — plain heap data with no
// thread-local state. TTSTokenizer wraps a HuggingFace tokenizer (Arc-backed, Send+Sync).
unsafe impl Send for TtsModelHandle {}
unsafe impl Sync for TtsModelHandle {}

// ---------------------------------------------------------------------------
// Engine struct
// ---------------------------------------------------------------------------

/// Wrapper around the incord-tts crate's synthesis API.
///
/// On creation (`load`), reads model weights and tokenizer from disk.
/// On `synthesize`, runs inference on a Tokio blocking thread (TTS is heavy CPU).
pub struct IncordTtsEngine {
    pub id: String,
    pub description: String,
    pub model: Arc<TtsModelHandle>,
}

impl IncordTtsEngine {
    /// Load the TTS model from `model_dir`.
    ///
    /// Expected layout (mirrors the Incord-TTS project):
    /// ```text
    /// model_dir/
    ///   model.incord-tts      -- binary weights (talker + predictor + decoder + speaker_encoder)
    ///   tokenizer.json      -- HuggingFace tokenizer
    ///   config.json         -- language id maps (spk_id is empty for this model)
    ///   voices/             -- reference WAV files: luna.wav, ryan.wav, etc.
    /// ```
    pub fn load(
        id: impl Into<String>,
        model_dir: PathBuf,
        description: impl Into<String>,
    ) -> anyhow::Result<Self> {
        let id = id.into();
        let description = description.into();

        let model_path = model_dir.join("model.incord-tts");
        if !model_path.exists() {
            anyhow::bail!("TTS model binary not found at {}", model_path.display());
        }

        tracing::info!(path = ?model_path, "loading incord-tts weights...");
        let (talker, predictor, decoder, speaker_encoder_opt) =
            incord_tts::save::load_model(&model_path)
                .map_err(|e| anyhow::anyhow!("load_model failed: {e}"))?;

        if speaker_encoder_opt.is_none() {
            tracing::warn!(
                path = ?model_path,
                "no speaker encoder in model — voice cloning disabled, will use built-in voices"
            );
        }
        let speaker_encoder = speaker_encoder_opt;

        let tokenizer_path = model_dir.join("tokenizer.json");
        if !tokenizer_path.exists() {
            anyhow::bail!("TTS tokenizer not found at {}", tokenizer_path.display());
        }
        let tokenizer = TTSTokenizer::from_file(&tokenizer_path)
            .map_err(|e| anyhow::anyhow!("loading tokenizer: {e}"))?;

        // Load config for language + speaker id maps.
        let config_path = model_dir.join("config.json");
        let (language_map, speaker_map) = if config_path.exists() {
            let cfg = incord_tts::config::IncordTTSConfig::from_file(&config_path)
                .map_err(|e| anyhow::anyhow!("loading config.json: {e}"))?;
            (cfg.talker_config.codec_language_id, cfg.talker_config.spk_id)
        } else {
            tracing::warn!(
                "config.json not found at {}; language and speaker maps will be empty",
                config_path.display()
            );
            (Default::default(), Default::default())
        };

        // Resolve voices directory. Standard layout: `model_dir/voices/*.wav`.
        // Custom-voice variant (1.7B-customvoice): WAVs sit at the model dir root
        // (angry.wav, baseline.wav, whisper.wav). Fall back to model_dir itself.
        let voices_subdir = model_dir.join("voices");
        let voices_dir = if voices_subdir.exists() {
            voices_subdir
        } else {
            tracing::info!(
                "no voices/ subdir at {} — using model_dir root for voice WAVs (custom_voice layout)",
                model_dir.display()
            );
            model_dir.clone()
        };

        tracing::info!(
            model_id = %id,
            languages = ?language_map.keys().collect::<Vec<_>>(),
            speakers = ?speaker_map.keys().collect::<Vec<_>>(),
            voices_dir = ?voices_dir,
            "TTS model loaded"
        );

        Ok(Self {
            id,
            description,
            model: Arc::new(TtsModelHandle {
                talker,
                predictor,
                decoder,
                speaker_encoder,
                tokenizer,
                language_map,
                speaker_map,
                voices_dir,
            }),
        })
    }
}

// ---------------------------------------------------------------------------
// Trait impls
// ---------------------------------------------------------------------------

impl ModelEngine for IncordTtsEngine {
    fn id(&self) -> &str {
        &self.id
    }

    fn task(&self) -> Task {
        Task::Tts
    }

    fn description(&self) -> &str {
        &self.description
    }
}

#[async_trait]
impl TtsEngine for IncordTtsEngine {
    async fn synthesize(&self, req: TtsRequest) -> anyhow::Result<TtsOutput> {
        let model = self.model.clone();
        let voice = req.voice.unwrap_or_else(|| DEFAULT_VOICE.to_string());
        let language = req.language.unwrap_or_else(|| DEFAULT_LANGUAGE.to_string());
        let text = req.input;
        let instruct = req.instruct.and_then(|s| {
            let trimmed = s.trim();
            if trimmed.is_empty() { None } else { Some(trimmed.to_string()) }
        });

        let result = tokio::task::spawn_blocking(move || -> anyhow::Result<TtsOutput> {
            // Resolve language id.
            let language_id = *model.language_map.get(&language).ok_or_else(|| {
                anyhow::anyhow!(
                    "unknown language '{}'. Available: {:?}",
                    language,
                    model.language_map.keys().collect::<Vec<_>>()
                )
            })?;

            // Resolve speaker_id from voice name if the model has a built-in
            // speaker map. Lookup is case-insensitive (frontend can send "Vivian"
            // or "vivian"). Falls back to 0 when the map is empty (cloning mode).
            let speaker_id: u32 = if !model.speaker_map.is_empty() {
                let key = voice.to_lowercase();
                *model.speaker_map.get(&key).ok_or_else(|| {
                    let mut available: Vec<&String> = model.speaker_map.keys().collect();
                    available.sort();
                    anyhow::anyhow!(
                        "unknown voice '{}'. Available speakers: {:?}",
                        voice,
                        available
                    )
                })?
            } else {
                0
            };

            // Compute voice embedding only if the model has a speaker encoder.
            // Customvoice variant has no encoder → falls through with None and the
            // talker uses its built-in speaker_id from the lookup above.
            let voice_embedding: Option<Vec<f32>> = if let Some(encoder) = &model.speaker_encoder {
                // Resolve voice reference WAV.
                // Accept: bare name ("luna") → voices/luna.wav, or an absolute path.
                let voice_path = if std::path::Path::new(&voice).is_absolute() {
                    PathBuf::from(&voice)
                } else {
                    model.voices_dir.join(format!("{voice}.wav"))
                };

                if !voice_path.exists() {
                    anyhow::bail!(
                        "voice WAV not found at {}. Available voices are the .wav files in {}",
                        voice_path.display(),
                        model.voices_dir.display()
                    );
                }

                let (ref_audio, _sr) = incord_tts::wav::read_wav(&voice_path).map_err(|e| {
                    anyhow::anyhow!("reading voice WAV {}: {e}", voice_path.display())
                })?;

                let mel_config = MelConfig::speaker_encoder();
                let mel_spec =
                    incord_tts::audio_features::extract_mel_spectrogram(&ref_audio, &mel_config);

                Some(extract_speaker_embedding(&mel_spec, encoder, 128))
            } else {
                tracing::debug!(
                    voice = %voice,
                    "no speaker encoder — using model's built-in voices (voice name passed through to talker)"
                );
                None
            };

            // Run synthesis. Returns Vec<f32> samples at 24 kHz, range [-1.0, 1.0].
            let samples: Vec<f32> = generate_speech(
                &model.talker,
                &model.predictor,
                &model.decoder,
                &model.tokenizer,
                &text,
                speaker_id, // ignored when voice_embedding is Some
                language_id,
                None, // voice_codes: no .codes file for HTTP path
                voice_embedding.as_deref(),
                instruct.as_deref(),
                &TTSParams::default(),
                None, // seed: non-deterministic by default
            );

            // Encode f32 PCM → 16-bit WAV bytes (in-memory).
            let wav_bytes = encode_wav_f32(&samples, SAMPLE_RATE);

            Ok(TtsOutput {
                audio: wav_bytes,
                format: "wav".to_string(),
                sample_rate: SAMPLE_RATE,
            })
        })
        .await??;

        Ok(result)
    }
}

// ---------------------------------------------------------------------------
// WAV encoder (in-memory)
// ---------------------------------------------------------------------------

/// Encode f32 samples ([-1.0, 1.0], mono, `sample_rate` Hz) to 16-bit PCM WAV bytes.
///
/// Mirrors `incord_tts::wav::write_wav` but writes into a Vec<u8> rather than a file.
fn encode_wav_f32(samples: &[f32], sample_rate: u32) -> Vec<u8> {
    let num_samples = samples.len() as u32;
    let num_channels: u16 = 1;
    let bytes_per_sample: u32 = 2; // 16-bit
    let data_size = num_samples * bytes_per_sample;
    let chunk_size = 36 + data_size;
    let byte_rate = sample_rate * num_channels as u32 * bytes_per_sample;
    let block_align = num_channels * bytes_per_sample as u16;
    let bits_per_sample: u16 = 16;

    let mut buf: Vec<u8> = Vec::with_capacity(44 + data_size as usize);

    // RIFF header
    buf.extend_from_slice(b"RIFF");
    buf.extend_from_slice(&chunk_size.to_le_bytes());
    buf.extend_from_slice(b"WAVE");

    // fmt chunk
    buf.extend_from_slice(b"fmt ");
    buf.extend_from_slice(&16u32.to_le_bytes());
    buf.extend_from_slice(&1u16.to_le_bytes());
    buf.extend_from_slice(&num_channels.to_le_bytes());
    buf.extend_from_slice(&sample_rate.to_le_bytes());
    buf.extend_from_slice(&byte_rate.to_le_bytes());
    buf.extend_from_slice(&block_align.to_le_bytes());
    buf.extend_from_slice(&bits_per_sample.to_le_bytes());

    // data chunk
    buf.extend_from_slice(b"data");
    buf.extend_from_slice(&data_size.to_le_bytes());

    for &s in samples {
        let clamped = s.clamp(-1.0, 1.0);
        let i16_val = (clamped * 32767.0) as i16;
        buf.extend_from_slice(&i16_val.to_le_bytes());
    }

    buf
}
