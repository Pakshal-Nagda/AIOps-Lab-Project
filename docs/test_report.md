# Test Execution & Evaluation Report

**Project Name:** Project Medhā: Sanskrit & Prakrit Live Recitation Verifier  
**Course:** DA3408 - AIOps Lab  
**Report Date:** November 14, 2026  
**Test Suite:** `tests/test_recitation_engine.py` & `tests/evaluate.py`  
**Execution Environment:** Linux x86_64, Python 3.14.4, Pytest 9.2.0  
**Overall Result:** **PASSED (100% Pass Rate)**  

---

## 1. Executive Summary

All planned automated test cases spanning unit logic, phonetic alignment, streaming VAD, API endpoints, WebSocket live streaming, Ray data cleaning, and empirical model evaluation were executed against the Project Medhā test suite. 

* **Total Test Cases Executed:** 27  
* **Passed:** 27  
* **Failed:** 0  
* **Pass Rate:** **100%**  
* **Total Pytest Execution Duration:** 8.22 seconds  
* **Acoustic Character Accuracy:** **75.31%** (Exceeds 70.0% acceptance threshold)  
* **Character Error Rate (CER):** **24.69%** (Meets $\le 25.0\%$ target threshold)  
* **Real-Time Factor (RTF):** **0.0399x** (~25x faster than real-time audio playback)  
* **Average Inference Latency:** **342.4 ms** (Well below 500 ms threshold)  

---

## 2. Master Test Execution Results

| Test ID | Test Category | Test Case Method | Status | Duration |
| :--- | :--- | :--- | :--- | :--- |
| **TC-01** | Text Segmentation | `test_split_sutra_lines_multiline` | **PASS** | 0.001s |
| **TC-02** | Text Segmentation | `test_split_sutra_lines_danda` | **PASS** | 0.001s |
| **TC-03** | Text Segmentation | `test_split_sutra_empty` | **PASS** | 0.001s |
| **TC-04** | Phonetic Alignment | `test_space_agnostic_alignment_spaces_ignored` | **PASS** | 0.002s |
| **TC-05** | Phonetic Alignment | `test_space_agnostic_pronunciation_mistake` | **PASS** | 0.002s |
| **TC-06** | Phonetic Alignment | `test_space_agnostic_alignment_empty` | **PASS** | 0.001s |
| **TC-07** | Streaming VAD | `test_streaming_vad_speech_and_boundary` | **PASS** | 0.003s |
| **TC-08** | Streaming VAD | `test_streaming_vad_hesitation_warning` | **PASS** | 0.002s |
| **TC-09** | Streaming VAD | `test_streaming_vad_short_blip_ignored` | **PASS** | 0.002s |
| **TC-10** | Continuous Aligner | `test_continuous_sutra_aligner_multiline_and_pauses` | **PASS** | 0.045s |
| **TC-11** | Gateway API | `test_api_health` | **PASS** | 0.012s |
| **TC-12** | Model Serving | `test_mlflow_model_serving_invocations` | **PASS** | 0.028s |
| **TC-13** | Observability | `test_api_metrics` | **PASS** | 0.014s |
| **TC-14** | Live Streaming | `test_websocket_recite_flow` | **PASS** | 0.065s |
| **TC-15** | Model Evaluation | `test_evaluation_error_metrics` | **PASS** | 0.002s |
| **TC-16** | Model Evaluation | `test_evaluate_model_integration` | **PASS** | 4.820s |
| **TC-17** | Data Collector | `test_collector_prompts_api` | **PASS** | 0.008s |
| **TC-18** | Data Collector | `test_collector_update_prompts_api` | **PASS** | 0.009s |
| **TC-19** | Data Collector | `test_collector_stats_api` | **PASS** | 0.008s |
| **TC-20** | Data Collector | `test_collector_audio_upload_flow` | **PASS** | 0.035s |
| **TC-21** | Audio Analysis | `test_analyze_recitation_api` | **PASS** | 0.410s |
| **TC-22** | Audio Analysis | `test_analyze_recitation_missing_reference` | **PASS** | 0.007s |
| **TC-23** | Audio Analysis | `test_analyze_recitation_non_devanagari_text` | **PASS** | 0.008s |
| **TC-24** | Text Cleaning | `test_clean_devanagari_input_valid` | **PASS** | 0.001s |
| **TC-25** | Text Cleaning | `test_clean_devanagari_input_non_indic_treated_empty` | **PASS** | 0.001s |
| **TC-26** | Ray Data Prep | `test_ray_devanagari_text_cleaning` | **PASS** | 0.001s |
| **TC-27** | Ray Data Prep | `test_ray_devanagari_empty_or_non_indic` | **PASS** | 0.001s |

---

## 3. Empirical Model Performance & Acoustic Quality Metrics

Evaluated on the held-out validation set (`data/val_manifest.json`) comprising 11 multi-speaker clips totaling 94.5 seconds of recorded recitation:

```json
{
  "val_wer": 0.7836,
  "val_cer": 0.2469,
  "val_char_accuracy": 75.31,
  "eval_samples": 11,
  "total_audio_seconds": 94.5,
  "avg_latency_ms": 342.4,
  "real_time_factor": 0.0399,
  "total_infer_time_seconds": 3.77
}
```

### Analysis:
* **Character Accuracy (75.31%):** Achieved without GPU acceleration via CPU transfer learning on Conformer-CTC. This provides accurate phonetic error identification when compared against canonical reference texts.
* **Character Error Rate (24.69%):** Outperforms the project target threshold of $\le 25.0\%$.
* **Word Error Rate (78.36%):** As noted in the High-Level Design, WER is naturally high for Sanskrit/Prakrit due to arbitrary whitespace insertions across continuous *sandhi* phrases. The space-agnostic aligner bypasses this limitation.
* **Inference Speed & Real-Time Factor (0.0399x):** The quantized INT8 ONNX acoustic model transcribes an 8.6-second verse line in approximately **342.4 ms**, enabling near-instantaneous real-time visual feedback on live recitation screens.

---

## 4. Distributed Data Cleaning Performance (Ray Core)

Measured during `scripts/prepare_dataset.py` execution across 4 parallel Ray worker tasks:

```json
{
  "pipeline": "ray_speech_data_cleaning",
  "raw_samples": 78,
  "cleaned_samples": 78,
  "rejected_samples": 0,
  "total_audio_duration_seconds": 633.62,
  "total_audio_duration_minutes": 10.56,
  "train_samples": 67,
  "val_samples": 11,
  "processing_time_seconds": 5.68,
  "throughput_samples_per_second": 13.74,
  "ray_cpus_utilized": 4
}
```

* **Throughput:** 13.74 audio clips/second.
* **Integrity:** Zero corrupted clips; 100% of collected recordings successfully resampled to 16kHz mono, trimmed of trailing silence, and normalized to 0.95 peak amplitude.

---

## 5. Acceptance Criteria Verification Matrix

| Evaluation Criteria | Target Metric | Measured Value | Compliance Status |
| :--- | :--- | :--- | :--- |
| **Pytest Pass Rate** | $100\%$ | $100\%$ (27/27) | **MET** |
| **Character Accuracy** | $\ge 70.0\%$ | **$75.31\%$** | **MET** |
| **Character Error Rate** | $\le 25.0\%$ | **$24.69\%$** | **MET** |
| **Average Latency** | $\le 500\text{ ms}$ | **$342.4\text{ ms}$** | **MET** |
| **Real-Time Factor** | $\le 0.10\times$ | **$0.0399\times$** | **MET** |
| **DVC CI Pipeline** | Clean DAG | Verified (`dvc.yaml`) | **MET** |
| **MLflow Model Serving** | Model Registry Serving | Active (`sushrota-asr@champion`) | **MET** |
| **Hesitation Detection** | Alert on pause $>1.5$s | Verified | **MET** |

---

## 6. Conclusion

The Project Medhā recitation engine successfully satisfies all architectural, quality, latency, and test requirements specified in the project guidelines. The system is validated for live recitation verification and edge deployment.
