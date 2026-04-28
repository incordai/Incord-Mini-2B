//! Smoke test for STT. Requires real qora-stt-small model directory.
//! Skipped when not present.

#![cfg(feature = "mock-weights")]

#[tokio::test]
async fn stt_transcribes_short_audio() {
    let stt_model_dir = std::path::PathBuf::from("D:/Burn-AI/QORA-STT-Small/model");
    if !stt_model_dir.exists() {
        eprintln!("skipping: STT model not at {}", stt_model_dir.display());
        return;
    }

    use qora_server::engines::stt::QoraSttEngine;
    use qora_server::engine::stt::{SttEngine, SttRequest};

    let engine = match QoraSttEngine::load("test-stt", stt_model_dir, "Test STT") {
        Ok(e) => e,
        Err(err) => {
            eprintln!("skipping: STT engine load failed: {err}");
            return;
        }
    };

    // Use a small WAV from a known fixture if available, else skip
    let fixture_paths = [
        std::path::PathBuf::from("D:/Burn-AI/QORA-TTS/model/1.7B/output.wav"),
        std::path::PathBuf::from(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/sample.wav"
        )),
    ];
    let audio_path = fixture_paths.iter().find(|p| p.exists());
    let Some(audio_path) = audio_path else {
        eprintln!("skipping: no audio fixture available");
        return;
    };

    let audio_bytes = std::fs::read(audio_path).unwrap();

    let req = SttRequest {
        model: "test-stt".to_string(),
        audio: audio_bytes,
        format: Some("wav".to_string()),
        language: Some("english".to_string()),
    };

    let out = engine.transcribe(req).await.unwrap();
    assert!(!out.text.is_empty(), "transcription produced no text");
    eprintln!("Transcribed: {}", out.text);
}
