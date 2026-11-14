# Test Plan Document

**Project Name:** Project Medhā: Sanskrit & Prakrit Live Recitation Verifier  
**Course:** DA3408 - AIOps Lab  

---

## 1. Test Strategy & Objectives

The primary objective of the Medhā verification suite is to ensure that the recitation engine provides accurate, low-latency, and resilient pronunciation verification for classical Sanskrit and Prakrit oral recitations.

The testing strategy spans four testing tiers:
1. **Unit Testing:** Validates mathematical correctness of space-agnostic phonetic alignment, verse line segmentation, streaming VAD state transitions, and distributed Unicode data cleaning.
2. **Integration Testing:** Tests interactions between the FastAPI gateway, WebSocket streaming pipelines, audio file decoders, and MLflow serving endpoints.
3. **Model Quality & Performance Benchmarking:** Measures empirical speech recognition metrics (Character Error Rate, Word Error Rate, Real-Time Factor, and inference latency) against defined acceptance thresholds.
4. **Data & Pipeline Testing:** Verifies DVC continuous integration DAG execution and Ray distributed task scheduling.

---

## 2. Test Environment Specification

* **Operating System:** Linux x86_64
* **Python Runtime:** Python 3.14.4 virtual environment (`./venv/bin/python`)
* **Test Framework:** `pytest` (v9.2.0)
* **HTTP & WebSocket Client:** `fastapi.testclient.TestClient` (Starlette TestClient with `httpx` and `websockets`)
* **Acoustic Inference Engine:** MLflow Model Server on port `5001` with fallback to `onnxruntime` INT8
* **Distributed Compute Engine:** Ray Core 2.44.1

---

## 3. Performance & Quality Acceptance Thresholds

| Metric | Target Acceptance Criteria | Minimum Tolerable |
| :--- | :--- | :--- |
| **Character Accuracy (Validation)** | $\ge 70.0\%$ | $65.0\%$ |
| **Character Error Rate (CER)** | $\le 25.0\%$ | $30.0\%$ |
| **Word Error Rate (WER)** | $\le 80.0\%$ (Sandhi-sensitive) | $85.0\%$ |
| **Average Inference Latency** | $\le 400\text{ ms / clip}$ | $600\text{ ms / clip}$ |
| **Real-Time Factor (RTF)** | $\le 0.08\times$ | $0.15\times$ |
| **VAD Line Pause Threshold** | $0.5\text{s} - 1.0\text{s}$ | $0.5\text{s} - 1.2\text{s}$ |
| **VAD Hesitation Alert Threshold** | $> 1.5\text{s}$ | $> 1.5\text{s}$ |
| **Automated Test Pass Rate** | $100\%$ (24 / 24) | $100\%$ |

---

## 4. Master Test Cases Matrix (TC-01 to TC-24)

All 24 test cases below are implemented in `tests/test_recitation_engine.py` and execute automatically via `pytest`.

### Tier 1: Text Segmentation & Manuscript Parsing Unit Tests

| Test ID | Test Name | Target Component | Input Specification | Expected Outcome |
| :--- | :--- | :--- | :--- | :--- |
| **TC-01** | `test_split_sutra_lines_multiline` | `split_sutra_lines` | Multi-line verse string with leading/trailing whitespaces | Parses into exactly 3 clean trimmed line strings. |
| **TC-02** | `test_split_sutra_lines_danda` | `split_sutra_lines` | Verse separated by single danda (`।`) and double danda (`॥`) | Splits into 4 distinct verse lines correctly without danda markers. |
| **TC-03** | `test_split_sutra_empty` | `split_sutra_lines` | Empty string `""` and whitespace-only string `"   \n\n  "` | Returns an empty list `[]` without raising exceptions. |

### Tier 2: Space-Agnostic Phonetic Alignment Unit Tests

| Test ID | Test Name | Target Component | Input Specification | Expected Outcome |
| :--- | :--- | :--- | :--- | :--- |
| **TC-04** | `test_space_agnostic_alignment_spaces_ignored` | `space_agnostic_align` | Reference: `"णमो अरिहंताणं णमो सिद्धाणं"`, Heard: `"णमोअरिहंताणं णमोसिद्धाणं"` (no spaces) | `correct == total`, all tokens marked as `correct` or `preserve`; original spaces preserved. |
| **TC-05** | `test_space_agnostic_pronunciation_mistake` | `space_agnostic_align` | Reference: `"णमो अरिहंताणं"`, Heard: `"णमोअरहंताणं"` (omitted vowel matra `'ि'`) | `correct < total`, missing vowel token flagged with `type="missing"` and `expected="ि"`. |
| **TC-06** | `test_space_agnostic_alignment_empty` | `space_agnostic_align` | Reference: `""`, Heard: `""` | Returns empty tokens `[]`, `correct=0`, `total=0`. |

### Tier 3: Streaming Voice Activity Detection (VAD) Unit Tests

| Test ID | Test Name | Target Component | Input Specification | Expected Outcome |
| :--- | :--- | :--- | :--- | :--- |
| **TC-07** | `test_streaming_vad_speech_and_boundary` | `StreamingVAD` | 0.4s active speech chunk followed by 0.6s silence chunk | Emits `speech_start`, tracks `is_speaking=True`, then emits `line_boundary` with `pause_duration >= 0.5s`. |
| **TC-08** | `test_streaming_vad_hesitation_warning` | `StreamingVAD` | Continuous silence chunk lasting 1.2s when `max_allowed_pause_s=1.0` | Emits `long_pause_warning` event with `pause_duration >= 1.0s`. |
| **TC-09** | `test_streaming_vad_short_blip_ignored` | `StreamingVAD` | 50ms noise click (below `min_speech_duration_s=0.2s`) followed by 0.6s silence | Discards transient noise; does NOT emit `line_boundary` and resets `is_speaking=False`. |

### Tier 4: Continuous Aligner State Machine Unit Tests

| Test ID | Test Name | Target Component | Input Specification | Expected Outcome |
| :--- | :--- | :--- | :--- | :--- |
| **TC-10** | `test_continuous_sutra_aligner_multiline_and_pauses` | `ContinuousSutraAligner` | Two-line reference manuscript with 0.7s normal pause on chunk 1 and 2.1s pause on chunk 2 | Chunk 1 flags `is_long_pause=False`; Chunk 2 flags `is_long_pause=True`; session summary tracks 1 hesitation pause. |

### Tier 5: Service Health & Observability Integration Tests

| Test ID | Test Name | Target Component | Input Specification | Expected Outcome |
| :--- | :--- | :--- | :--- | :--- |
| **TC-11** | `test_api_health` | `GET /health` | HTTP GET request | HTTP 200, `"status": "healthy"`, `"vocab_size": 257`, MLflow status object present. |
| **TC-12** | `test_mlflow_model_serving_invocations` | `POST /invocations` | HTTP POST with sample audio path to MLflow serving port 5001 | HTTP 200, returns prediction object containing transcription and status. |
| **TC-13** | `test_api_metrics` | `GET /metrics` | HTTP GET request | HTTP 200, response text contains Prometheus metric `api_requests_total`. |

### Tier 6: WebSocket Streaming Recitation Integration Tests

| Test ID | Test Name | Target Component | Input Specification | Expected Outcome |
| :--- | :--- | :--- | :--- | :--- |
| **TC-14** | `test_websocket_recite_flow` | `WS /ws/recite` | WebSocket handshake `start`, binary 16kHz speech chunk, 1.0s silence chunk, then `stop` | Receives `session_started`, `speech_started`, `chunk_annotated` (with tokens), and `session_completed` summary. |

### Tier 7: Empirical Model Evaluation & Error Metrics Tests

| Test ID | Test Name | Target Component | Input Specification | Expected Outcome |
| :--- | :--- | :--- | :--- | :--- |
| **TC-15** | `test_evaluation_error_metrics` | `compute_wer`, `compute_cer` | Identical strings and strings with 1 substitution error | Identical strings return 0 edits; substitution string returns exactly 1 word edit. |
| **TC-16** | `test_evaluate_model_integration` | `evaluate_model` | Validation manifest `data/val_manifest.json` and quantized models | Generates `metrics.json` with `val_char_accuracy > 50.0%` over 11 evaluation samples. |

### Tier 8: Data Collector REST API Endpoints Tests

| Test ID | Test Name | Target Component | Input Specification | Expected Outcome |
| :--- | :--- | :--- | :--- | :--- |
| **TC-17** | `test_collector_prompts_api` | `GET /api/v1/collector/prompts` | HTTP GET request | HTTP 200, returns array of prompts with `id` and `text`. |
| **TC-18** | `test_collector_update_prompts_api` | `POST /api/v1/collector/prompts` | HTTP POST with 3-line raw text string | HTTP 200, returns `status="success"` and updates prompt count to 3. |
| **TC-19** | `test_collector_stats_api` | `GET /api/v1/collector/stats` | HTTP GET request | HTTP 200, returns metadata summary with `total_clips`. |
| **TC-20** | `test_collector_audio_upload_flow` | `POST /api/v1/collector/upload` | Multipart form with 1.0s synthetic sine wave WAV, prompt ID, and speaker name | HTTP 200, saves audio to `data/raw/` and returns success status with generated filename. |

### Tier 9: Audio File Verification API Tests

| Test ID | Test Name | Target Component | Input Specification | Expected Outcome |
| :--- | :--- | :--- | :--- | :--- |
| **TC-21** | `test_analyze_recitation_api` | `POST /api/v1/analyze` | Multipart form with processed speech WAV clip and matching reference text | HTTP 200, returns transcription, similarity score, discrepancy list, diff tokens, and latency. |
| **TC-22** | `test_analyze_recitation_missing_reference` | `POST /api/v1/analyze` | Multipart form with audio clip but missing required `reference_text` parameter | HTTP 422 Unprocessable Entity. |

### Tier 10: Ray Distributed Data Cleaning Unit Tests

| Test ID | Test Name | Target Component | Input Specification | Expected Outcome |
| :--- | :--- | :--- | :--- | :--- |
| **TC-23** | `test_ray_devanagari_text_cleaning` | `clean_devanagari_text` | Devanagari text with dashes and danda markers (`।`, `॥`) | Strips punctuation, normalizes Unicode NFC, returns clean text and `is_valid=True`. |
| **TC-24** | `test_ray_devanagari_empty_or_non_indic` | `clean_devanagari_text` | Whitespace-only string and non-Indic ASCII English text | Flags invalid with reasons `"empty_text"` and `"no_devanagari_characters"`. |

---

## 5. Execution Instructions

To execute the complete test suite:

```bash
# Activate environment and run pytest
./venv/bin/pytest tests/ -v

# Run with coverage inspection
./venv/bin/pytest tests/ --durations=10

# Validate DVC pipeline status
./venv/bin/dvc status
```
