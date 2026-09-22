//! Smoke test for STT. Requires a real STT model directory, given by
//! `INCORD_STT_MODEL_DIR`. Skipped when unset or not present.

#![cfg(feature = "mock-weights")]

#[tokio::test]
async fn stt_transcribes_short_audio() {
    let Ok(dir) = std::env::var("INCORD_STT_MODEL_DIR") else {
        eprintln!("skipping: INCORD_STT_MODEL_DIR not set");
        return;
    };
    let stt_model_dir = std::path::PathBuf::from(dir);
    if !stt_model_dir.exists() {
        eprintln!("skipping: STT model not at {}", stt_model_dir.display());
        return;
    }

    use incord_server::engine::stt::{SttEngine, SttRequest};
    use incord_server::engines::stt::IncordSttEngine;

    let engine = match IncordSttEngine::load("test-stt", stt_model_dir, "Test STT") {
        Ok(e) => e,
        Err(err) => {
            eprintln!("skipping: STT engine load failed: {err}");
            return;
        }
    };

    // Use a small WAV from INCORD_STT_TEST_WAV or the repo fixture, else skip
    let mut fixture_paths = vec![std::path::PathBuf::from(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/sample.wav"
    ))];
    if let Ok(wav) = std::env::var("INCORD_STT_TEST_WAV") {
        fixture_paths.insert(0, std::path::PathBuf::from(wav));
    }
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
