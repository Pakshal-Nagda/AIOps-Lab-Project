import os
import sys

# Ensure root directory is always on python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
import numpy as np
from fastapi.testclient import TestClient

from src.aligner import (
    split_sutra_lines, 
    space_agnostic_align, 
    SushrotaVerifier, 
    ContinuousSutraAligner
)
from src.vad_checker import StreamingVAD, detect_long_pauses
from backend.main import app

client = TestClient(app)

# -------------------------------------------------------------
# 1. Text Segmentation Unit Tests
# -------------------------------------------------------------
def test_split_sutra_lines_multiline():
    raw_verse = """
    णमो अरिहंताणं
    णमो सिद्धाणं
    णमो आयरियाणं
    """
    lines = split_sutra_lines(raw_verse)
    assert len(lines) == 3
    assert lines[0] == "णमो अरिहंताणं"
    assert lines[1] == "णमो सिद्धाणं"
    assert lines[2] == "णमो आयरियाणं"

def test_split_sutra_lines_danda():
    raw_verse = "उवसग्गहरं पासं। पासं वंदामि कम्मघणमुक्कं। विसहर विस निन्नासं। मंगल कल्लाण आवासं॥"
    lines = split_sutra_lines(raw_verse)
    assert len(lines) == 4
    assert lines[0] == "उवसग्गहरं पासं"
    assert lines[3] == "मंगल कल्लाण आवासं"

def test_split_sutra_empty():
    assert split_sutra_lines("") == []
    assert split_sutra_lines("   \n\n  ") == []

# -------------------------------------------------------------
# 2. Space-Agnostic Phonetic Alignment Unit Tests
# -------------------------------------------------------------
def test_space_agnostic_alignment_spaces_ignored():
    ref = "णमो अरिहंताणं णमो सिद्धाणं"
    # Transcribed has no space between words
    heard = "णमोअरिहंताणं णमोसिद्धाणं"
    tokens, correct, total = space_agnostic_align(ref, heard)
    assert correct == total
    assert all(t["type"] in ("correct", "preserve") for t in tokens)
    # Check that original spaces are preserved in output tokens
    space_tokens = [t for t in tokens if t["type"] == "preserve" and t["text"] == " "]
    assert len(space_tokens) == 3

def test_space_agnostic_pronunciation_mistake():
    ref = "णमो अरिहंताणं"
    # User omitted 'ि'
    heard = "णमोअरहंताणं"
    tokens, correct, total = space_agnostic_align(ref, heard)
    assert correct < total
    missing_tok = [t for t in tokens if t["type"] == "missing"]
    assert len(missing_tok) == 1
    assert missing_tok[0]["expected"] == "ि"

def test_space_agnostic_alignment_empty():
    tokens, correct, total = space_agnostic_align("", "")
    assert tokens == []
    assert correct == 0
    assert total == 0

# -------------------------------------------------------------
# 3. Streaming VAD Unit Tests
# -------------------------------------------------------------
def test_streaming_vad_speech_and_boundary():
    vad = StreamingVAD(
        sample_rate=16000,
        energy_threshold=0.01,
        min_speech_duration_s=0.2,
        line_pause_s=0.5,
        max_allowed_pause_s=1.5
    )

    # 0.4s speech
    speech_chunk = np.random.uniform(-0.15, 0.15, int(16000 * 0.4)).astype(np.float32)
    ev1 = vad.feed_chunk(speech_chunk)
    assert any(e["event"] == "speech_start" for e in ev1)
    assert vad.is_speaking is True

    # 0.6s silence to trigger line boundary
    silence_chunk = np.zeros(int(16000 * 0.6), dtype=np.float32)
    ev2 = vad.feed_chunk(silence_chunk)
    boundary_ev = [e for e in ev2 if e["event"] in ("utterance_boundary", "line_boundary")]
    assert len(boundary_ev) == 1
    assert boundary_ev[0]["pause_duration"] >= 0.5
    assert vad.is_speaking is False

def test_streaming_vad_hesitation_warning():
    vad = StreamingVAD(
        sample_rate=16000,
        energy_threshold=0.01,
        line_pause_s=0.5,
        max_allowed_pause_s=1.0
    )

    # Feed 1.2s silence
    silence = np.zeros(int(16000 * 1.2), dtype=np.float32)
    events = vad.feed_chunk(silence)
    pause_warnings = [e for e in events if e["event"] == "long_pause_warning"]
    assert len(pause_warnings) == 1
    assert pause_warnings[0]["pause_duration"] >= 1.0

def test_streaming_vad_short_blip_ignored():
    vad = StreamingVAD(
        sample_rate=16000,
        energy_threshold=0.01,
        min_speech_duration_s=0.2,
        line_pause_s=0.5
    )
    # 50ms short click/noise (below min_speech_duration_s=0.2)
    short_click = np.random.uniform(-0.1, 0.1, int(16000 * 0.05)).astype(np.float32)
    vad.feed_chunk(short_click)
    # Follow with 0.6s silence to trigger line boundary check
    silence = np.zeros(int(16000 * 0.6), dtype=np.float32)
    events = vad.feed_chunk(silence)
    # line_boundary should NOT be emitted because speech was < 200ms
    assert not any(e["event"] == "line_boundary" for e in events)
    assert vad.is_speaking is False

# -------------------------------------------------------------
# 4. ContinuousSutraAligner Progression & Inline Pause Tests
# -------------------------------------------------------------
def test_continuous_sutra_aligner_multiline_and_pauses():
    verifier = SushrotaVerifier()
    text = "णमो अरिहंताणं\nणमो सिद्धाणं"
    aligner = ContinuousSutraAligner(text, verifier, max_pause_s=1.5)

    assert aligner.cursor_clean_idx == 0
    assert aligner.cursor_orig_idx == 0

    dummy_audio = np.zeros(16000, dtype=np.float32)

    # Process first utterance with normal cadence pause (0.7s)
    chunk1 = aligner.process_utterance(dummy_audio, pause_duration=0.7)
    assert chunk1 is not None
    assert chunk1["is_long_pause"] is False
    assert chunk1["pause_duration"] == 0.7

    # Process second utterance with hesitation pause (2.1s)
    chunk2 = aligner.process_utterance(dummy_audio, pause_duration=2.1)
    assert chunk2 is not None
    assert chunk2["is_long_pause"] is True
    assert chunk2["total_long_pauses"] == 1

    summary = aligner.get_session_summary()
    assert summary["total_long_pauses"] == 1
    assert summary["total_chunks"] == 2

# -------------------------------------------------------------
# 5. FastAPI Endpoints & WebSocket Integration Tests
# -------------------------------------------------------------
def test_api_health():
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"
    assert data["vocab_size"] == 257
    assert "mlflow_serving" in data
    assert "url" in data["mlflow_serving"]

def test_mlflow_model_serving_invocations():
    import requests
    try:
        ping = requests.get("http://localhost:5001/ping", timeout=1.0)
        if ping.status_code == 200:
            payload = {
                "dataframe_records": [
                    {"audio_path": "data/processed_audio/clip_1794387514520_Sweta_p23.wav"}
                ]
            }
            res = requests.post("http://localhost:5001/invocations", json=payload, timeout=3.0)
            assert res.status_code == 200
            preds = res.json().get("predictions", [])
            assert len(preds) == 1
            assert "transcription" in preds[0]
            assert preds[0]["status"] == "success"
    except Exception:
        pass

def test_api_metrics():
    res = client.get("/metrics")
    assert res.status_code == 200
    assert "api_requests_total" in res.text

def test_websocket_recite_flow():
    with client.websocket_connect("/ws/recite") as ws:
        # 1. Start handshake
        ws.send_json({
            "action": "start",
            "reference_text": "णमो अरिहंताणं\nणमो सिद्धाणं",
            "energy_threshold": 0.01,
            "max_pause_s": 1.5
        })
        resp = ws.receive_json()
        assert resp["event"] == "session_started"
        assert resp["total_phonetic_chars"] > 0

        # 2. Stream speech chunk (16kHz int16 PCM)
        speech = (np.sin(np.linspace(0, 50, 8000)) * 8000).astype(np.int16).tobytes()
        ws.send_bytes(speech)

        # 3. Stream silence chunk (1.0s) to trigger utterance completion
        silence = np.zeros(16000, dtype=np.int16).tobytes()
        ws.send_bytes(silence)

        # Expect speech_started then chunk_annotated
        ev1 = ws.receive_json()
        assert ev1["event"] == "speech_started"

        ev2 = ws.receive_json()
        assert ev2["event"] == "chunk_annotated"
        assert "tokens" in ev2
        assert "overall_accuracy" in ev2
        assert "pause_duration" in ev2

        # 4. Stop session
        ws.send_json({"action": "stop"})
        ev3 = ws.receive_json()
        assert ev3["event"] == "session_completed"
        assert "summary" in ev3

# -------------------------------------------------------------
# 5. Model Evaluation & Error Metric Unit Tests
# -------------------------------------------------------------
try:
    from tests.evaluate import compute_wer, compute_cer, evaluate_model
except ModuleNotFoundError:
    from evaluate import compute_wer, compute_cer, evaluate_model

def test_evaluation_error_metrics():
    ref = "णमो अरिहंताणं णमो सिद्धाणं"
    hyp = "णमो अरिहंताणं णमो सिद्धाणं"
    w_edits, w_total = compute_wer(ref, hyp)
    assert w_edits == 0
    assert w_total == 4

    c_edits, c_total = compute_cer(ref, hyp)
    assert c_edits == 0
    assert c_total > 0

    # With 1 substitution
    hyp_err = "णमो अरिहंताणं णमो बुद्धाणं"
    w_edits, _ = compute_wer(ref, hyp_err)
    assert w_edits == 1

def test_evaluate_model_integration():
    metrics = evaluate_model(
        model_dir="models",
        val_manifest="data/val_manifest.json",
        metrics_out="metrics.json"
    )
    assert "val_wer" in metrics
    assert "val_cer" in metrics
    assert "val_char_accuracy" in metrics
    assert metrics["val_char_accuracy"] > 50.0
    assert metrics["eval_samples"] == 11

# -------------------------------------------------------------
# 6. REST API & Data Collector Endpoint Tests
# -------------------------------------------------------------
def test_collector_prompts_api():
    res = client.get("/api/v1/collector/prompts")
    assert res.status_code == 200
    data = res.json()
    assert "prompts" in data
    assert len(data["prompts"]) > 0

def test_collector_update_prompts_api():
    payload = {"raw_text": "णमो अरिहंताणं\nणमो सिद्धाणं\nणमो आयरियाणं"}
    res = client.post("/api/v1/collector/prompts", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "success"
    assert data["total_prompts"] == 3

def test_collector_stats_api():
    res = client.get("/api/v1/collector/stats")
    assert res.status_code == 200
    data = res.json()
    assert "total_clips" in data

def test_collector_audio_upload_flow():
    import io
    import soundfile as sf
    bio = io.BytesIO()
    # Generate 1 sec dummy 16kHz sine wave audio
    dummy = (np.sin(np.linspace(0, 10, 16000)) * 10000).astype(np.int16)
    sf.write(bio, dummy, 16000, format="WAV", subtype="PCM_16")
    bio.seek(0)

    files = {"file": ("test_clip.wav", bio.getvalue(), "audio/wav")}
    data = {
        "text": "णमो अरिहंताणं",
        "speaker": "PyTestUser",
        "prompt_id": 1
    }
    res = client.post("/api/v1/collector/upload", files=files, data=data)
    assert res.status_code == 200
    resp_data = res.json()
    assert resp_data["status"] == "success"
    assert "filename" in resp_data

def test_analyze_recitation_api():
    sample_wav = "data/processed_audio/clip_1794387514520_Sweta_p23.wav"
    with open(sample_wav, "rb") as f:
        files = {"file": ("clip.wav", f.read(), "audio/wav")}
    data = {"reference_text": "खामेमि सव्व जीवे सव्वे जीवा खमंतु मे"}
    res = client.post("/api/v1/analyze", files=files, data=data)
    assert res.status_code == 200
    resp = res.json()
    assert resp["status"] == "success"
    assert "transcription" in resp
    assert "similarity_score" in resp
    assert "discrepancies" in resp
    assert "diff_tokens" in resp

def test_analyze_recitation_missing_reference():
    sample_wav = "data/processed_audio/clip_1794387514520_Sweta_p23.wav"
    with open(sample_wav, "rb") as f:
        files = {"file": ("clip.wav", f.read(), "audio/wav")}
    res = client.post("/api/v1/analyze", files=files)
    assert res.status_code == 422

# -------------------------------------------------------------
# 7. Ray Data Cleaning Logic Unit Tests
# -------------------------------------------------------------
from scripts.prepare_dataset import clean_devanagari_text

def test_ray_devanagari_text_cleaning():
    raw_sutra = "  उवसग्गहरं पासं।  पासं वंदामि -- कम्मघणमुक्कं॥  "
    cleaned, is_valid, reason = clean_devanagari_text(raw_sutra)
    assert is_valid is True
    assert "।" not in cleaned
    assert "॥" not in cleaned
    assert "--" not in cleaned
    assert cleaned == "उवसग्गहरं पासं पासं वंदामि कम्मघणमुक्कं"

def test_ray_devanagari_empty_or_non_indic():
    _, is_valid, reason = clean_devanagari_text("   \n\t   ")
    assert is_valid is False
    assert reason == "empty_text"

    _, is_valid, reason = clean_devanagari_text("Hello World 12345")
    assert is_valid is False
    assert reason == "no_devanagari_characters"
