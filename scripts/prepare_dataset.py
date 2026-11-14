#!/usr/bin/env python3
"""
Project Medhā - Distributed Speech Dataset Cleaning & Preparation with Ray
==========================================================================
Scalable data cleaning and manifest generation pipeline for Sanskrit/Prakrit ASR:
1. Distributed execution across CPU cores using Ray for high-throughput scaling.
2. Robust Audio Cleaning:
   - Corrupted file detection and validation.
   - Multi-channel to mono conversion.
   - Resampling to 16,000 Hz.
   - Leading/trailing silence trimming.
   - Outlier duration filtering (min_duration to max_duration).
   - Energy check (RMS filtering for silent or noisy empty recordings).
   - Peak amplitude normalization (0.95 / -0.5 dBFS) for speaker microphone parity.
3. Robust Text Cleaning:
   - Unicode Devanagari normalization (NFC).
   - Punctuation stripping (dandas, quotes, special chars).
   - Multi-space and whitespace collapsing.
   - Devanagari character verification.
4. Stratified/Reproducible Train-Validation Split and NeMo Manifest Output.
5. Emits pipeline cleaning metrics (JSON) for AIOps monitoring and DVC tracking.

Usage:
    python scripts/prepare_dataset.py \
        --input-dir data/raw \
        --output-dir data \
        --val-split 0.15 \
        --seed 42 \
        --num-cpus 4
"""

import os
import sys
import json
import csv
import glob
import time
import random
import unicodedata
import argparse
import logging
from typing import List, Dict, Any, Tuple

import numpy as np
import soundfile as sf
import librosa
import ray

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("ray_prepare_dataset")

def clean_devanagari_text(text: str) -> Tuple[str, bool, str]:
    """
    Cleans and normalizes Sanskrit/Prakrit Devanagari text.
    Returns (cleaned_text, is_valid, reject_reason).
    """
    if not text or not text.strip():
        return "", False, "empty_text"

    # 1. Unicode NFC Normalization
    text = unicodedata.normalize("NFC", text.strip())

    # 2. Remove unwanted punctuation (dandas, quotes, brackets, etc.)
    unwanted_chars = "।॥.,?!;:'\"-–—()[]{}/*_~`«»#@+=<>"
    for ch in unwanted_chars:
        text = text.replace(ch, " ")

    # 3. Collapse multiple whitespace
    text = " ".join(text.split())

    if not text:
        return "", False, "empty_after_punctuation_removal"

    # 4. Verify presence of Devanagari Unicode characters (0x0900 to 0x097F)
    has_devanagari = any("\u0900" <= ch <= "\u097F" for ch in text)
    if not has_devanagari:
        return text, False, "no_devanagari_characters"

    return text, True, ""

@ray.remote
def clean_single_sample_remote(
    raw_item: Dict[str, Any],
    output_audio_dir: str,
    target_sr: int = 16000,
    min_duration: float = 0.5,
    max_duration: float = 20.0,
    top_db_trim: float = 30.0,
    peak_norm_val: float = 0.95
) -> Dict[str, Any]:
    """
    Ray remote task that cleans audio and text for a single recording.
    Runs concurrently across Ray worker processes.
    """
    audio_path = raw_item["audio_path"]
    raw_text = raw_item["raw_text"]
    speaker = raw_item.get("speaker", "unknown")
    base_name = raw_item["base_name"]

    # 1. Validate Audio File Existence
    if not os.path.exists(audio_path):
        return {"status": "rejected", "reason": "audio_file_not_found", "file": audio_path}

    # 2. Text Normalization & Quality Check
    cleaned_text, is_text_valid, text_reason = clean_devanagari_text(raw_text)
    if not is_text_valid:
        return {"status": "rejected", "reason": text_reason, "file": audio_path}

    # 3. Audio Loading & Format Conversion
    try:
        try:
            audio_data, sr = sf.read(audio_path, dtype="float32")
        except Exception:
            # Fallback for m4a/aac/mp3/webm
            audio_data, sr = librosa.load(audio_path, sr=None, mono=False)
    except Exception as e:
        return {"status": "rejected", "reason": f"audio_corrupt_read_error: {str(e)}", "file": audio_path}

    if audio_data is None or len(audio_data) == 0:
        return {"status": "rejected", "reason": "empty_audio_buffer", "file": audio_path}

    # 4. Channel Averaging (Stereo -> Mono)
    if audio_data.ndim > 1:
        audio_data = audio_data.mean(axis=0 if audio_data.shape[0] < audio_data.shape[1] else 1)

    # 5. Resampling to target_sr (16kHz)
    if sr != target_sr:
        try:
            audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=target_sr)
            sr = target_sr
        except Exception as e:
            return {"status": "rejected", "reason": f"resampling_failed: {str(e)}", "file": audio_path}

    # 6. Silence Trimming (Leading & Trailing dead air)
    try:
        trimmed_data, _ = librosa.effects.trim(audio_data, top_db=top_db_trim)
        if len(trimmed_data) > int(target_sr * 0.2):
            audio_data = trimmed_data
    except Exception:
        pass  # keep untrimmed if trimming encounters an issue

    # 7. Energy Check (Detect Silent / Inaudible Audio)
    rms_energy = np.sqrt(np.mean(audio_data ** 2))
    if rms_energy < 1e-4:
        return {"status": "rejected", "reason": "audio_inaudible_silence", "file": audio_path}

    # 8. Duration Outlier Filtering
    duration = len(audio_data) / float(target_sr)
    if duration < min_duration:
        return {
            "status": "rejected",
            "reason": f"duration_too_short ({round(duration, 2)}s < {min_duration}s)",
            "file": audio_path
        }
    if duration > max_duration:
        return {
            "status": "rejected",
            "reason": f"duration_too_long ({round(duration, 2)}s > {max_duration}s)",
            "file": audio_path
        }

    # 9. Peak Amplitude Normalization
    max_amp = np.max(np.abs(audio_data))
    if max_amp > 1e-4:
        audio_data = (audio_data / max_amp) * peak_norm_val

    # 10. Save Processed 16kHz Mono 16-bit PCM WAV
    out_filename = f"{base_name}.wav"
    out_filepath = os.path.join(output_audio_dir, out_filename)
    try:
        sf.write(out_filepath, audio_data, target_sr, subtype="PCM_16")
    except Exception as e:
        return {"status": "rejected", "reason": f"audio_write_failed: {str(e)}", "file": audio_path}

    return {
        "status": "success",
        "audio_filepath": os.path.abspath(out_filepath),
        "duration": round(duration, 3),
        "text": cleaned_text,
        "speaker": speaker,
        "rms_energy": float(round(rms_energy, 5))
    }

def discover_raw_samples(input_dir: str) -> List[Dict[str, Any]]:
    """Discovers audio-text pairs from metadata.csv or paired files in input_dir."""
    samples = []
    csv_path = os.path.join(input_dir, "metadata.csv")
    if os.path.exists(csv_path):
        logger.info(f"Ingesting raw metadata from: {csv_path}")
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            
            # Identify columns
            col_map = {"filename": 0, "text": 1, "speaker": 2}
            if header and any(k in header[0].lower() for k in ["file", "audio", "path"]):
                header_lower = [h.strip().lower() for h in header]
                for idx, h in enumerate(header_lower):
                    if "file" in h or "audio" in h or "path" in h:
                        col_map["filename"] = idx
                    elif "text" in h or "transcript" in h or "verse" in h:
                        col_map["text"] = idx
                    elif "speaker" in h:
                        col_map["speaker"] = idx
            else:
                if header:
                    samples.append({
                        "audio_path": os.path.join(input_dir, header[0].strip()),
                        "raw_text": header[1].strip() if len(header) > 1 else "",
                        "speaker": header[2].strip() if len(header) > 2 else "unknown",
                        "base_name": os.path.splitext(header[0].strip())[0]
                    })

            for row in reader:
                if len(row) < 2:
                    continue
                audio_file = row[col_map["filename"]].strip()
                text = row[col_map["text"]].strip()
                speaker = row[col_map["speaker"]].strip() if len(row) > col_map.get("speaker", 2) else "unknown"

                audio_full = os.path.join(input_dir, audio_file)
                if not os.path.exists(audio_full):
                    matches = glob.glob(os.path.join(input_dir, f"{os.path.splitext(audio_file)[0]}.*"))
                    if matches:
                        audio_full = matches[0]

                samples.append({
                    "audio_path": audio_full,
                    "raw_text": text,
                    "speaker": speaker,
                    "base_name": os.path.splitext(os.path.basename(audio_full))[0]
                })

        return samples

    # Alternative: Paired .txt and audio files
    logger.info("No metadata.csv found. Scanning for paired audio and .txt files...")
    audio_exts = [".wav", ".mp3", ".m4a", ".ogg", ".flac", ".webm"]
    txt_files = glob.glob(os.path.join(input_dir, "*.txt"))

    for txt_file in txt_files:
        base_name = os.path.splitext(os.path.basename(txt_file))[0]
        audio_match = None
        for ext in audio_exts:
            candidate = os.path.join(input_dir, f"{base_name}{ext}")
            if os.path.exists(candidate):
                audio_match = candidate
                break

        if not audio_match:
            continue

        with open(txt_file, "r", encoding="utf-8") as f:
            text = f.read().strip()

        samples.append({
            "audio_path": audio_match,
            "raw_text": text,
            "speaker": "unknown",
            "base_name": base_name
        })

    return samples

def main():
    parser = argparse.ArgumentParser(description="Distributed Speech Dataset Cleaning with Ray")
    parser.add_argument("--input-dir", type=str, default="data/raw", help="Directory with raw audio & metadata")
    parser.add_argument("--output-dir", type=str, default="data", help="Output directory for manifests & processed audio")
    parser.add_argument("--val-split", type=float, default=0.15, help="Validation fraction (e.g. 0.15 for 15 percent)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible split")
    parser.add_argument("--num-cpus", type=int, default=None, help="Number of Ray worker CPU cores (default: auto)")
    parser.add_argument("--min-duration", type=float, default=0.5, help="Minimum clip duration in seconds")
    parser.add_argument("--max-duration", type=float, default=20.0, help="Maximum clip duration in seconds")
    parser.add_argument("--top-db-trim", type=float, default=30.0, help="Top dB threshold for silence trimming")
    args = parser.parse_args()

    if not os.path.exists(args.input_dir):
        logger.error(f"Input directory '{args.input_dir}' not found!")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)
    processed_audio_dir = os.path.join(args.output_dir, "processed_audio")
    os.makedirs(processed_audio_dir, exist_ok=True)

    # 1. Discover raw input items
    raw_samples = discover_raw_samples(args.input_dir)
    if not raw_samples:
        logger.error(f"No audio samples found in '{args.input_dir}'!")
        sys.exit(1)

    logger.info(f"Discovered {len(raw_samples)} raw speech recordings. Initializing Ray distributed cluster...")

    # 2. Initialize Ray
    t_start = time.time()
    ray.init(
        ignore_reinit_error=True,
        num_cpus=args.num_cpus,
        include_dashboard=False,
        logging_level=logging.WARNING
    )
    cluster_cpus = ray.cluster_resources().get("CPU", 1.0)
    logger.info(f"Ray cluster ready! Active CPU workers: {cluster_cpus}")

    # 3. Dispatch parallel cleaning tasks
    logger.info("Executing parallel audio/text cleaning across Ray workers...")
    futures = [
        clean_single_sample_remote.remote(
            sample,
            output_audio_dir=processed_audio_dir,
            target_sr=16000,
            min_duration=args.min_duration,
            max_duration=args.max_duration,
            top_db_trim=args.top_db_trim,
            peak_norm_val=0.95
        )
        for sample in raw_samples
    ]

    results = ray.get(futures)
    ray.shutdown()
    elapsed_time = time.time() - t_start

    # 4. Filter and partition results
    clean_records = []
    rejected_records = []
    rejection_reasons = {}

    for res in results:
        if res.get("status") == "success":
            clean_records.append({
                "audio_filepath": res["audio_filepath"],
                "duration": res["duration"],
                "text": res["text"]
            })
        else:
            rejected_records.append(res)
            reason = res.get("reason", "unknown")
            rejection_reasons[reason] = rejection_reasons.get(reason, 0) + 1

    if not clean_records:
        logger.error("All samples were rejected during cleaning! Check rejection reasons below:")
        for r, cnt in rejection_reasons.items():
            logger.error(f"  {r}: {cnt}")
        sys.exit(1)

    # 5. Reproducible Train / Validation Split
    random.seed(args.seed)
    random.shuffle(clean_records)

    val_count = max(1, int(len(clean_records) * args.val_split))
    val_records = clean_records[:val_count]
    train_records = clean_records[val_count:]

    train_manifest_path = os.path.join(args.output_dir, "train_manifest.json")
    val_manifest_path = os.path.join(args.output_dir, "val_manifest.json")

    with open(train_manifest_path, "w", encoding="utf-8") as f:
        for r in train_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    with open(val_manifest_path, "w", encoding="utf-8") as f:
        for r in val_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    total_duration = sum(r["duration"] for r in clean_records)
    train_duration = sum(r["duration"] for r in train_records)
    val_duration = sum(r["duration"] for r in val_records)
    throughput = len(raw_samples) / max(0.001, elapsed_time)

    # 6. Save Data Engineering Stats for Monitoring & AIOps Audit
    stats = {
        "pipeline": "ray_speech_data_cleaning",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "raw_samples": len(raw_samples),
        "cleaned_samples": len(clean_records),
        "rejected_samples": len(rejected_records),
        "rejection_reasons": rejection_reasons,
        "total_audio_duration_seconds": round(total_duration, 2),
        "total_audio_duration_minutes": round(total_duration / 60.0, 2),
        "total_audio_duration_hours": round(total_duration / 3600.0, 4),
        "train_samples": len(train_records),
        "train_duration_seconds": round(train_duration, 2),
        "val_samples": len(val_records),
        "val_duration_seconds": round(val_duration, 2),
        "processing_time_seconds": round(elapsed_time, 2),
        "throughput_samples_per_second": round(throughput, 2),
        "ray_cpus_utilized": int(cluster_cpus)
    }

    stats_path = os.path.join(args.output_dir, "cleaning_stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    logger.info("=" * 65)
    logger.info("RAY DISTRIBUTED DATA CLEANING & PREPARATION COMPLETE!")
    logger.info("=" * 65)
    logger.info(f"Raw Input Audio Clips:        {len(raw_samples)}")
    logger.info(f"Successfully Cleaned:         {len(clean_records)} ({round(total_duration / 60, 2)} min)")
    logger.info(f"Filtered / Outliers:          {len(rejected_records)}")
    if rejection_reasons:
        for r, cnt in rejection_reasons.items():
            logger.info(f"   - {r}: {cnt}")
    logger.info(f"Training Split:               {len(train_records)} clips ({round(train_duration / 60, 2)} min)")
    logger.info(f"Validation Split:             {len(val_records)} clips ({round(val_duration / 60, 2)} min)")
    logger.info(f"Ray Processing Time:          {round(elapsed_time, 2)}s ({round(throughput, 1)} clips/sec)")
    logger.info(f"Train Manifest:               {train_manifest_path}")
    logger.info(f"Validation Manifest:          {val_manifest_path}")
    logger.info(f"Data Cleaning Audit Stats:    {stats_path}")
    logger.info("=" * 65)

if __name__ == "__main__":
    main()
