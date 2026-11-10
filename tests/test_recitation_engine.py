import pytest
import numpy as np
from fastapi.testclient import TestClient

from src.aligner import (
    split_sutra_lines, 
    space_agnostic_align, 
    SushrotaVerifier, 
    ContinuousSutraAligner, 
    LiveSutraSession
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
    raw_verse = "ॐ त्र्यम्बकं यजामहे। सुगन्धिं पुष्टिवर्धनम्। उर्वारुकमिव बन्धनान्। मृत्योर्मुक्षीय मामृतात्॥"
    lines = split_sutra_lines(raw_verse)
    assert len(lines) == 4
    assert lines[0] == "ॐ त्र्यम्बकं यजामहे"
    assert lines[3] == "मृत्योर्मुक्षीय मामृतात्"

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
