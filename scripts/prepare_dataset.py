#!/usr/bin/env python3
"""
Project Medhā - Speech Dataset Preparation & Manifest Generator
===============================================================
Prepares user-recorded Sanskrit/Prakrit speech audio for NeMo Conformer-CTC fine-tuning:
1. Normalizes audio to 16kHz Mono 16-bit PCM WAV.
2. Generates standard NeMo manifest JSON lines containing:
   {"audio_filepath": "/path/to/clip.wav", "duration": 4.25, "text": "णमो अरिहंताणं"}
3. Splits data into train and validation sets (default: 85% train, 15% val).
4. Prints comprehensive dataset statistics (total duration, vocabulary coverage, audio lengths).

Supported Input Formats:
- Option A (Directory with metadata.csv):
    data/raw/
      audio_001.wav
      audio_002.m4a
      metadata.csv (Format: filename,text)
- Option B (Matching .wav and .txt pairs):
    data/raw/
      verse_01.wav
      verse_01.txt (containing ground-truth Devanagari verse)

Usage:
    python scripts/prepare_dataset.py --input-dir data/raw --output-dir data/ --val-split 0.15
"""

import os
import sys
import json
import csv
import glob
import random
import argparse
import logging
from typing import List, Dict, Tuple
import soundfile as sf
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("prepare_dataset")


def convert_audio_to_16k_mono(input_path: str, output_path: str) -> float:
    """
    Reads an audio file (WAV, MP3, M4A, FLAC) and writes 16kHz mono 16-bit WAV.
    Returns the exact duration in seconds.
    """
    try:
        data, sr = sf.read(input_path, dtype="float32")
    except Exception:
        # Fallback to librosa if soundfile fails on m4a/mp3
        import librosa
        data, sr = librosa.load(input_path, sr=None, mono=False)

    if data.ndim > 1:
        data = data.mean(axis=0 if data.shape[0] < data.shape[1] else 1)

    if sr != 16000:
        import librosa
        data = librosa.resample(data, orig_sr=sr, target_sr=16000)

    # Normalize amplitude slightly if audio is too quiet
    max_val = np.max(np.abs(data))
    if max_val > 0.01:
        data = data / max_val * 0.95

    sf.write(output_path, data, 16000, subtype="PCM_16")
    duration = len(data) / 16000.0
    return round(duration, 3)


def parse_dataset(input_dir: str, processed_dir: str) -> List[Dict]:
    """Scans input_dir for metadata.csv or paired .txt files and prepares normalized WAVs."""
    os.makedirs(processed_dir, exist_ok=True)
    records = []

    # Check for metadata.csv
    csv_path = os.path.join(input_dir, "metadata.csv")
    if os.path.exists(csv_path):
        logger.info(f"Found metadata file at {csv_path}")
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            
            # Check if header exists
            first_row = header
            has_header = False
            if header and len(header) >= 2 and any(k in header[0].lower() for k in ["file", "audio", "path"]):
                has_header = True

            rows = list(reader)
            if not has_header and first_row:
                rows.insert(0, first_row)

            for row in rows:
                if len(row) < 2:
                    continue
                audio_name, text = row[0].strip(), row[1].strip()
                audio_file = os.path.join(input_dir, audio_name)
                if not os.path.exists(audio_file):
                    # Try matching with extensions
                    matches = glob.glob(os.path.join(input_dir, f"{os.path.splitext(audio_name)[0]}.*"))
                    if matches:
                        audio_file = matches[0]
                    else:
                        logger.warning(f"Audio file not found: {audio_name}")
                        continue

                out_wav = os.path.join(processed_dir, f"{os.path.splitext(os.path.basename(audio_file))[0]}.wav")
                dur = convert_audio_to_16k_mono(audio_file, out_wav)
                records.append({
                    "audio_filepath": os.path.abspath(out_wav),
                    "duration": dur,
                    "text": text
                })
        return records

    # Alternatively: paired audio and .txt files
    logger.info("Scanning for paired audio and .txt files...")
    audio_exts = [".wav", ".mp3", ".m4a", ".ogg", ".flac"]
    txt_files = glob.glob(os.path.join(input_dir, "*.txt"))

    for txt_file in txt_files:
        base_name = os.path.splitext(os.path.basename(txt_file))[0]
        # Find matching audio
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

        if not text:
            continue

        out_wav = os.path.join(processed_dir, f"{base_name}.wav")
        dur = convert_audio_to_16k_mono(audio_match, out_wav)
        records.append({
            "audio_filepath": os.path.abspath(out_wav),
            "duration": dur,
            "text": text
        })

    return records


def main():
    parser = argparse.ArgumentParser(description="Prepare dataset and manifest for NeMo ASR fine-tuning")
    parser.add_argument("--input-dir", type=str, default="data/raw", help="Directory with raw audio and text/csv")
    parser.add_argument("--output-dir", type=str, default="data", help="Output directory for manifests and processed audio")
    parser.add_argument("--val-split", type=float, default=0.15, help="Fraction of samples to hold out for validation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for repeatable split")
    args = parser.parse_args()

    if not os.path.exists(args.input_dir):
        logger.error(f"Input directory '{args.input_dir}' does not exist! Please create it and add your recordings.")
        sys.exit(1)

    processed_dir = os.path.join(args.output_dir, "processed_audio")
    records = parse_dataset(args.input_dir, processed_dir)

    if not records:
        logger.error(f"No valid audio + text pairs found in '{args.input_dir}'.")
        logger.info("Please provide either:")
        logger.info("  1. metadata.csv with columns: audio_filename,ground_truth_text")
        logger.info("  2. Matching pairs: e.g. verse_01.wav and verse_01.txt")
        sys.exit(1)

    random.seed(args.seed)
    random.shuffle(records)

    val_count = max(1, int(len(records) * args.val_split))
    val_records = records[:val_count]
    train_records = records[val_count:]

    train_manifest = os.path.join(args.output_dir, "train_manifest.json")
    val_manifest = os.path.join(args.output_dir, "val_manifest.json")

    with open(train_manifest, "w", encoding="utf-8") as f:
        for r in train_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    with open(val_manifest, "w", encoding="utf-8") as f:
        for r in val_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    total_dur = sum(r["duration"] for r in records)
    train_dur = sum(r["duration"] for r in train_records)
    val_dur = sum(r["duration"] for r in val_records)

    logger.info("=" * 60)
    logger.info("DATASET PREPARATION COMPLETED SUCCESSFULLY!")
    logger.info("=" * 60)
    logger.info(f"Total audio clips:       {len(records)} ({round(total_dur / 60, 2)} minutes)")
    logger.info(f"Training split:          {len(train_records)} clips ({round(train_dur / 60, 2)} minutes)")
    logger.info(f"Validation split:        {len(val_records)} clips ({round(val_dur / 60, 2)} minutes)")
    logger.info(f"Train manifest:          {train_manifest}")
    logger.info(f"Validation manifest:     {val_manifest}")
    logger.info("=" * 60)
    logger.info("You can now run fine-tuning with:")
    logger.info(f"    python scripts/train.py --train-manifest {train_manifest} --val-manifest {val_manifest}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
