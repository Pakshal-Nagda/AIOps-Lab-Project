#!/usr/bin/env python3
"""
Project Medhā - Model Evaluation & Quality Metrics
=================================================
Evaluates the fine-tuned quantized Sanskrit/Prakrit ASR model on validation audio:
1. Loads quantized ONNX model, preprocessor, and vocabulary.
2. Runs inference over all validation recordings in val_manifest.json.
3. Computes standard speech recognition metrics:
   - Word Error Rate (WER)
   - Character Error Rate (CER)
   - Phonetic Character Accuracy (%)
   - Real-Time Factor (RTF)
   - Average Inference Latency (ms)
4. Emits `metrics.json` for DVC Continuous Integration tracking.
5. Can be executed as a DVC stage or imported and executed by pytest.

Usage:
    python tests/evaluate.py \
        --model-dir models \
        --val-manifest data/val_manifest.json \
        --metrics-out metrics.json
"""

import os
import sys
import json
import time
import argparse
import logging
from typing import List, Dict, Tuple, Optional

import numpy as np

# Ensure root workspace is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.aligner import SushrotaVerifier, NON_PHONETIC_CHARS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("evaluate_model")


def levenshtein_distance(ref: List[str], hyp: List[str]) -> int:
    """Computes Levenshtein edit distance between two sequences."""
    n, m = len(ref), len(hyp)
    dp = [[0] * (m + 1) for _ in range(n + 1)]

    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref[i - 1] == hyp[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])

    return dp[n][m]


def compute_wer(ref_text: str, hyp_text: str) -> Tuple[int, int]:
    """Computes word edit distance and total reference words."""
    ref_words = [w for w in ref_text.strip().split() if w]
    hyp_words = [w for w in hyp_text.strip().split() if w]
    if not ref_words:
        return len(hyp_words), 0
    return levenshtein_distance(ref_words, hyp_words), len(ref_words)


def compute_cer(ref_text: str, hyp_text: str) -> Tuple[int, int]:
    """Computes character edit distance and total reference characters (ignoring typographic punctuation)."""
    clean_ref = [c for c in ref_text if c not in NON_PHONETIC_CHARS]
    clean_hyp = [c for c in hyp_text if c not in NON_PHONETIC_CHARS]
    if not clean_ref:
        return len(clean_hyp), 0
    return levenshtein_distance(clean_ref, clean_hyp), len(clean_ref)


def evaluate_model(
    model_dir: str = "models",
    val_manifest: str = "data/val_manifest.json",
    metrics_out: Optional[str] = "metrics.json",
    log_mlflow: bool = False,
    mlflow_uri: str = "sqlite:///mlflow.db"
) -> Dict[str, float]:
    """
    Evaluates the model on val_manifest and returns the metrics dict.
    Optionally writes to metrics_out (for DVC) and logs to MLflow.
    """
    if not os.path.exists(val_manifest):
        raise FileNotFoundError(f"Validation manifest '{val_manifest}' not found!")

    val_samples = []
    with open(val_manifest, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                val_samples.append(json.loads(line.strip()))

    if not val_samples:
        raise ValueError("Validation manifest is empty!")

    logger.info(f"Loaded {len(val_samples)} validation samples. Initializing SushrotaVerifier from '{model_dir}'...")
    verifier = SushrotaVerifier(model_dir=model_dir)

    total_word_edits = 0
    total_words = 0
    total_char_edits = 0
    total_chars = 0
    total_audio_duration = 0.0
    latencies = []

    logger.info("Running evaluation across validation dataset...")
    for idx, sample in enumerate(val_samples):
        audio_path = sample["audio_filepath"]
        ref_text = sample["text"]
        duration = sample.get("duration", 0.0)
        total_audio_duration += duration

        t0 = time.time()
        hyp_text = verifier.transcribe(audio_path)
        infer_time = time.time() - t0
        latencies.append(infer_time)

        # Word Error metrics
        w_edits, w_count = compute_wer(ref_text, hyp_text)
        total_word_edits += w_edits
        total_words += w_count

        # Character Error metrics
        c_edits, c_count = compute_cer(ref_text, hyp_text)
        total_char_edits += c_edits
        total_chars += c_count

    overall_wer = (total_word_edits / max(1, total_words))
    overall_cer = (total_char_edits / max(1, total_chars))
    char_accuracy = max(0.0, 1.0 - overall_cer) * 100.0
    avg_latency_ms = (sum(latencies) / len(latencies)) * 1000.0
    total_inference_time = sum(latencies)
    rtf = total_inference_time / max(0.001, total_audio_duration)

    metrics = {
        "val_wer": round(overall_wer, 4),
        "val_cer": round(overall_cer, 4),
        "val_char_accuracy": round(char_accuracy, 2),
        "eval_samples": len(val_samples),
        "total_audio_seconds": round(total_audio_duration, 2),
        "avg_latency_ms": round(avg_latency_ms, 2),
        "real_time_factor": round(rtf, 4),
        "total_infer_time_seconds": round(total_inference_time, 2)
    }

    if metrics_out:
        with open(metrics_out, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

    logger.info("=" * 60)
    logger.info("VALIDATION EVALUATION COMPLETE!")
    logger.info("=" * 60)
    logger.info(f"Evaluated Samples:      {len(val_samples)} ({round(total_audio_duration, 2)}s audio)")
    logger.info(f"Word Error Rate (WER):  {round(overall_wer * 100, 2)}%")
    logger.info(f"Char Error Rate (CER):  {round(overall_cer * 100, 2)}%")
    logger.info(f"Character Accuracy:     {round(char_accuracy, 2)}%")
    logger.info(f"Average Latency:        {round(avg_latency_ms, 2)} ms / clip")
    logger.info(f"Real-Time Factor (RTF): {round(rtf, 4)}x")
    if metrics_out:
        logger.info(f"Metrics Output:         {metrics_out}")
    logger.info("=" * 60)

    if log_mlflow:
        try:
            import mlflow
            mlflow.set_tracking_uri(mlflow_uri)
            with mlflow.start_run(run_name="onnx-quantized-eval"):
                mlflow.log_metrics(metrics)
            logger.info("Logged evaluation metrics to MLflow!")
        except Exception as e:
            logger.warning(f"Could not log to MLflow: {e}")

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate quantized ASR model on validation manifest")
    parser.add_argument("--model-dir", type=str, default="models", help="Directory containing ONNX model artifacts")
    parser.add_argument("--val-manifest", type=str, default="data/val_manifest.json", help="Path to val_manifest.json")
    parser.add_argument("--metrics-out", type=str, default="metrics.json", help="Output path for metrics.json")
    parser.add_argument("--mlflow-uri", type=str, default="sqlite:///mlflow.db", help="MLflow tracking URI")
    parser.add_argument("--log-mlflow", action="store_true", help="Log evaluation metrics to MLflow")
    args = parser.parse_args()

    evaluate_model(
        model_dir=args.model_dir,
        val_manifest=args.val_manifest,
        metrics_out=args.metrics_out,
        log_mlflow=args.log_mlflow,
        mlflow_uri=args.mlflow_uri
    )


if __name__ == "__main__":
    main()
