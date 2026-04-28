//! Concrete TtsEngine wrapping the qora-tts crate.
//!
//! Integration approach: lib linkage (not subprocess).
//! We added `[lib]` to D:/Burn-AI/QORA-TTS/Cargo.toml and add it as a path dep.
//! The qora-tts library exposes all modules publicly via lib.rs.
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

use std::path::PathBuf;
use std::sync::Arc;
use async_trait::async_trait;

use qora_tts::audio_features::MelConfig;
use qora_tts::code_predictor::CodePredictorWeights;
use qora_tts::decoder::SpeechDecoderWeights;
use qora_tts::generate_new::{generate_speech, TTSParams};
use qora_tts::speaker_encoder::{extract_speaker_embedding, SpeakerEncoderWeights};
use qora_tts::talker::TalkerWeights;
use qora_tts::tokenizer::TTSTokenizer;

use crate::engine::tts::{TtsEngine, TtsOutput, TtsRequest};
use crate::engine::{ModelEngine, Task};

/// QORA-TTS sample rate (fixed by the model architecture).
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
    pub speaker_encoder: SpeakerEncoderWeights,
    pub tokenizer: TTSTokenizer,
    /// language-name → u32 id map, from config.json.
    pub language_map: std::collections::HashMap<String, u32>,
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

/// Wrapper around the qora-tts crate's synthesis API.
///
/// On creation (`load`), reads model weights and tokenizer from disk.
/// On `synthesize`, runs inference on a Tokio blocking thread (TTS is heavy CPU).
pub struct QoraTtsEngine {
    pub id: String,
    pub description: String,
    pub model: Arc<TtsModelHandle>,
}

impl QoraTtsEngine {
    /// Load the TTS model from `model_dir`.
    ///
    /// Expected layout (mirrors the QORA-TTS project):
    /// ```text
    /// model_dir/
    ///   model.qora-tts      -- binary weights (talker + predictor + decoder + speaker_encoder)
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

        let model_path = model_dir.join("model.qora-tts");
        if !model_path.exists() {
            anyhow::bail!(
                "TTS model binary not found at {}",
                model_path.display()
            );
        }

        tracing::info!(path = ?model_path, "loading qora-tts weights...");
        let (talker, predictor, decoder, speaker_encoder_opt) =
            qora_tts::save::load_model(&model_path)
                .map_err(|e| anyhow::anyhow!("load_model failed: {e}"))?;

        let speaker_encoder = speaker_encoder_opt.ok_or_else(|| {
            anyhow::anyhow!(
                "Model at {} does not contain a speaker encoder. \
                 Voice cloning requires a speaker encoder in the .qora-tts binary.",
                model_path.display()
            )
        })?;

        let tokenizer_path = model_dir.join("tokenizer.json");
        if !tokenizer_path.exists() {
            anyhow::bail!(
                "TTS tokenizer not found at {}",
                tokenizer_path.display()
            );
        }
        let tokenizer = TTSTokenizer::from_file(&tokenizer_path)
            .map_err(|e| anyhow::anyhow!("loading tokenizer: {e}"))?;

        // Load config for language id map.
        let config_path = model_dir.join("config.json");
        let language_map = if config_path.exists() {
            let cfg = qora_tts::config::QoraTTSConfig::from_file(&config_path)
                .map_err(|e| anyhow::anyhow!("loading config.json: {e}"))?;
            cfg.talker_config.codec_language_id
        } else {
            tracing::warn!(
                "config.json not found at {}; language map will be empty",
                config_path.display()
            );
            Default::default()
        };

        let voices_dir = model_dir.join("voices");
        if !voices_dir.exists() {
            tracing::warn!(
                "voices directory not found at {}; voice names will be resolved as absolute paths",
                voices_dir.display()
            );
        }

        tracing::info!(
            model_id = %id,
            languages = ?language_map.keys().collect::<Vec<_>>(),
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
                voices_dir,
            }),
        })
    }
}

// ---------------------------------------------------------------------------
// Trait impls
// ---------------------------------------------------------------------------

impl ModelEngine for QoraTtsEngine {
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
impl TtsEngine for QoraTtsEngine {
    async fn synthesize(&self, req: TtsRequest) -> anyhow::Result<TtsOutput> {
        let model = self.model.clone();
        let voice = req.voice.unwrap_or_else(|| DEFAULT_VOICE.to_string());
        let language = req.language.unwrap_or_else(|| DEFAULT_LANGUAGE.to_string());
        let text = req.input;

        let result = tokio::task::spawn_blocking(move || -> anyhow::Result<TtsOutput> {
            // Resolve language id.
            let language_id = *model.language_map.get(&language).ok_or_else(|| {
                anyhow::anyhow!(
                    "unknown language '{}'. Available: {:?}",
                    language,
                    model.language_map.keys().collect::<Vec<_>>()
                )
            })?;

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

            // Load reference audio and extract speaker embedding.
            let (ref_audio, _sr) = qora_tts::wav::read_wav(&voice_path)
                .map_err(|e| anyhow::anyhow!("reading voice WAV {}: {e}", voice_path.display()))?;

            let mel_config = MelConfig::speaker_encoder();
            let mel_spec = qora_tts::audio_features::extract_mel_spectrogram(&ref_audio, &mel_config);

            let voice_embedding = extract_speaker_embedding(&mel_spec, &model.speaker_encoder, 128);

            // Run synthesis. Returns Vec<f32> samples at 24 kHz, range [-1.0, 1.0].
            // speaker_id is 0 (unused when voice_embedding is supplied).
            let samples: Vec<f32> = generate_speech(
                &model.talker,
                &model.predictor,
                &model.decoder,
                &model.tokenizer,
                &text,
                0,              // speaker_id: not used when voice_embedding is Some
                language_id,
                None,           // voice_codes: no .codes file for HTTP path
                Some(&voice_embedding),
                &TTSParams::default(),
                None,           // seed: non-deterministic by default
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
/// Mirrors `qora_tts::wav::write_wav` but writes into a Vec<u8> rather than a file.
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
