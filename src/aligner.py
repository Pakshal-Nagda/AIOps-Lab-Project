import os
import re
import json
import difflib
from typing import Union, List, Dict, Optional, Tuple
import numpy as np
import soundfile as sf
import onnxruntime as ort

# Punctuation and non-phonetic characters to ignore during Sanskrit/Prakrit phonetic comparison
NON_PHONETIC_CHARS = set(" \t\r\n।॥-–—.,?!;:()[]{}'\"“”‘’*")

def clean_devanagari_input(text: str) -> str:
    """
    Cleans non-Devanagari characters from reference text,
    preserving newlines, spaces, and Devanagari characters (U+0900 to U+097F).
    If no Devanagari characters exist in the input, returns an empty string "".
    """
    if not text or not re.search(r'[\u0900-\u097F]', text):
        return ""

    # Keep only Devanagari characters and whitespace (spaces, tabs, newlines)
    cleaned = re.sub(r'[^\u0900-\u097F\s]', '', text)

    # Normalize lines: collapse inline whitespace while keeping newlines
    lines = []
    for line in cleaned.splitlines():
        line_clean = re.sub(r'[^\S\r\n]+', ' ', line).strip()
        if line_clean and re.search(r'[\u0900-\u097F]', line_clean):
            lines.append(line_clean)

    if not lines:
        return ""

    return "\n".join(lines)

def space_agnostic_align(ref_text: str, trans_text: str) -> Tuple[List[Dict], int, int]:
    """
    Phonetically aligns reference text and transcribed text ignoring all spaces and punctuation.
    Spaces and punctuation in Sanskrit/Prakrit are purely typographic conventions and have
    no phonetic existence. They are preserved in output for reading formatting, but never
    cause pronunciation mismatches, omissions, or additions.
    
    Returns:
        (tokens, correct_count, total_phonetic_count)
    """
    clean_ref = []
    clean_to_orig = []
    for i, c in enumerate(ref_text):
        if c not in NON_PHONETIC_CHARS:
            clean_ref.append(c)
            clean_to_orig.append(i)
    clean_ref_str = "".join(clean_ref)

    clean_trans = [c for c in trans_text if c not in NON_PHONETIC_CHARS]
    clean_trans_str = "".join(clean_trans)

    matcher = difflib.SequenceMatcher(None, clean_ref_str, clean_trans_str)
    char_status = {}
    insertions = {}

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            for ci in range(i1, i2):
                char_status[ci] = {
                    "status": "correct",
                    "expected": clean_ref[ci],
                    "heard": clean_trans_str[ci - i1 + j1]
                }
        elif tag == "replace":
            for ci in range(i1, i2):
                h = clean_trans_str[j1:j2] if ci == i1 else ""
                char_status[ci] = {
                    "status": "mispronounced",
                    "expected": clean_ref[ci],
                    "heard": h
                }
        elif tag == "delete":
            for ci in range(i1, i2):
                char_status[ci] = {
                    "status": "missing",
                    "expected": clean_ref[ci],
                    "heard": ""
                }
        elif tag == "insert":
            insertions[i1] = clean_trans_str[j1:j2]

    tokens = []
    clean_ptr = 0
    correct_cnt = 0
    total_cnt = len(clean_ref)

    for orig_idx, ch in enumerate(ref_text):
        # Insert any unexpected extra sounds spoken by user at this phonetic juncture
        if clean_ptr in insertions:
            tokens.append({
                "type": "extra",
                "expected": "",
                "heard": insertions.pop(clean_ptr)
            })

        if clean_ptr < len(clean_to_orig) and orig_idx == clean_to_orig[clean_ptr]:
            info = char_status.get(clean_ptr, {"status": "correct", "expected": ch, "heard": ch})
            if info["status"] == "correct":
                correct_cnt += 1
            tokens.append({
                "type": info["status"],
                "text": ch,
                "expected": info.get("expected", ch),
                "heard": info.get("heard", "")
            })
            clean_ptr += 1
        else:
            # Non-phonetic character (space, newline, danda, hyphen): preserve exactly as formatted!
            tokens.append({
                "type": "preserve",
                "text": ch
            })

    if clean_ptr in insertions:
        tokens.append({
            "type": "extra",
            "expected": "",
            "heard": insertions.pop(clean_ptr)
        })

    return tokens, correct_cnt, total_cnt


def split_sutra_lines(text: str) -> List[str]:
    """
    Intelligently splits a multi-line or danda-separated Sanskrit/Prakrit sutra into clean lines.
    Preserves line breaks if present; otherwise splits on danda (।, ॥) or punctuation.
    """
    if not text or not text.strip():
        return []
    
    # Check if text contains explicit newlines
    raw_lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
    if len(raw_lines) > 1:
        lines = []
        for line in raw_lines:
            cleaned = re.sub(r'[।॥]+$', '', line).strip()
            if cleaned:
                lines.append(cleaned)
        return lines

    # Single block of text: split on danda (।, ॥) or semicolon
    parts = re.split(r'[।॥;\n]+', text)
    cleaned = [p.strip() for p in parts if p.strip()]
    return cleaned if cleaned else [text.strip()]


class SushrotaVerifier:
    """
    Production-ready Sanskrit Recitation Verifier using Quantized ONNX Runtime.
    Eliminates all heavy PyTorch and NeMo dependencies for fast, lightweight serving.
    """
    def __init__(self, model_dir: str = "models"):
        self.model_dir = model_dir

        prep_path = os.path.join(model_dir, "preprocessor.onnx")
        model_asr_path = os.path.join(model_dir, "model_asr.onnx")
        legacy_int8_path = os.path.join(model_dir, "sushrota_sanskrit_ctc_int8.onnx")
        vocab_path = os.path.join(model_dir, "sanskrit_vocab.json")

        asr_path = model_asr_path if os.path.exists(model_asr_path) else legacy_int8_path

        if not os.path.exists(prep_path):
            raise FileNotFoundError(f"Missing preprocessor ONNX model at '{prep_path}'.")
        if not os.path.exists(asr_path):
            raise FileNotFoundError(f"Missing ASR ONNX model at '{asr_path}'. Run scripts/convert_nemo_to_onnx.py first.")
        if not os.path.exists(vocab_path):
            raise FileNotFoundError(f"Missing vocabulary JSON at '{vocab_path}'.")

        print(f"Loading ONNX Preprocessor from {prep_path}...")
        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 2
        opts.intra_op_num_threads = 4
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        self.prep_session = ort.InferenceSession(prep_path, sess_options=opts, providers=['CPUExecutionProvider'])

        print(f"Loading Quantized ASR Model from {asr_path}...")
        self.asr_session = ort.InferenceSession(asr_path, sess_options=opts, providers=['CPUExecutionProvider'])

        print(f"Loading Sanskrit Vocabulary from {vocab_path}...")
        with open(vocab_path, "r", encoding="utf-8") as f:
            self.vocab = json.load(f)

        import requests
        self.session_http = requests.Session()
        self.mlflow_serving_url = os.getenv("MLFLOW_SERVING_URL", "http://localhost:5001/invocations")
        print(f"SushrotaVerifier initialized successfully! Vocabulary size: {len(self.vocab)} (MLflow Serving URL: {self.mlflow_serving_url})")

    def transcribe(self, audio_input: Union[str, np.ndarray], sample_rate: int = 16000) -> str:
        """
        Loads audio from filepath or accepts raw numpy array (16kHz float32),
        first attempting inference via MLflow Model Serving (/invocations).
        If MLflow serving is unreachable, falls back to direct ONNX Runtime.
        """
        # 1. Attempt inference via MLflow Model Serving
        if self.mlflow_serving_url:
            try:
                import base64
                import io

                payload = None
                if isinstance(audio_input, str) and os.path.exists(audio_input):
                    payload = {"dataframe_records": [{"audio_path": os.path.abspath(audio_input)}]}
                elif isinstance(audio_input, np.ndarray):
                    wav_data = audio_input.copy()
                    if wav_data.ndim > 1:
                        wav_data = wav_data.mean(axis=1)
                    if wav_data.dtype == np.int16:
                        wav_data = wav_data.astype(np.float32) / 32768.0
                    bio = io.BytesIO()
                    sf.write(bio, wav_data, sample_rate, format="WAV", subtype="PCM_16")
                    b64_str = base64.b64encode(bio.getvalue()).decode("ascii")
                    payload = {"dataframe_records": [{"audio_base64": b64_str}]}

                if payload:
                    resp = self.session_http.post(self.mlflow_serving_url, json=payload, timeout=1.5)
                    if resp.status_code == 200:
                        preds = resp.json().get("predictions", [])
                        if preds and "transcription" in preds[0] and preds[0].get("status") == "success":
                            return preds[0]["transcription"]
            except Exception:
                # Graceful fallback to direct local ONNX execution
                pass

        # 2. Local ONNX Runtime Execution (Fallback & Offline)
        if isinstance(audio_input, str):
            wav, sr = sf.read(audio_input, dtype="float32")
        elif isinstance(audio_input, np.ndarray):
            wav = audio_input.copy()
            sr = sample_rate
            if wav.dtype == np.int16:
                wav = wav.astype(np.float32) / 32768.0
            else:
                wav = wav.astype(np.float32)
        else:
            raise TypeError(f"audio_input must be a file path (str) or numpy array, got {type(audio_input)}")

        # Convert stereo to mono if necessary
        if wav.ndim > 1:
            wav = wav.mean(axis=1)

        # Resample to 16kHz if necessary
        if sr != 16000:
            import librosa
            wav = librosa.resample(wav, orig_sr=sr, target_sr=16000)

        wav = np.ascontiguousarray(wav, dtype=np.float32)

        if len(wav) < 160:  # Minimum 10ms of audio required
            return ""

        audio_signal = wav[np.newaxis, :]
        audio_len = np.array([len(wav)], dtype=np.int64)

        # 1. Feature extraction (16kHz audio -> 80-mel log spectrogram)
        feats, flen = self.prep_session.run(
            None,
            {"audio_signal": audio_signal, "length": audio_len}
        )

        # 2. Acoustic model inference (80-mel log spectrogram -> Sanskrit CTC logits)
        logits = self.asr_session.run(
            None,
            {"audio_signal": feats, "length": flen}
        )[0]

        # 3. Greedy CTC decoding: argmax -> collapse repeats -> remove blank (idx 0)
        pred_ids = np.argmax(logits[0], axis=-1)
        out, prev = [], -1
        for i in pred_ids:
            i = int(i)
            if i != prev and i != 0:
                if i < len(self.vocab):
                    out.append(self.vocab[i])
            prev = i

        # Replace SentencePiece delimiter with space
        transcription = "".join(out).replace(" ", " ").replace("▁", " ").strip()
        return transcription

    def verify_recitation(self, transcribed_text: str, reference_text: str) -> dict:
        """
        Compares the transcribed text against reference stotra text with space-agnostic alignment.
        Returns match accuracy percentage, phonetic discrepancies, and token-level diff.
        """
        tokens, correct_cnt, total_cnt = space_agnostic_align(reference_text, transcribed_text)
        
        discrepancies = []
        for t in tokens:
            if t["type"] in ("mispronounced", "missing", "extra"):
                discrepancies.append({
                    "type": t["type"],
                    "expected": t.get("expected", ""),
                    "heard": t.get("heard", ""),
                    "text": t.get("text", "")
                })

        score = round((correct_cnt / total_cnt * 100), 2) if total_cnt > 0 else 100.0

        return {
            "transcription": transcribed_text,
            "match_accuracy_percentage": score,
            "discrepancies": discrepancies,
            "diff_tokens": tokens
        }


class ContinuousSutraAligner:
    """
    Production-grade Continuous Recitation Aligner for arbitrary-length sutras (from 1 to 100+ lines).
    Features:
    - Decoupled streaming: user recites freely at their natural pace without waiting for annotations.
    - Continuous text flow: seamless Devanagari manuscript view without rigid per-line cards.
    - Intra-line and inter-line pause detection: flags unnatural hesitation pauses (> max_pause_s).
    - Unbroken recitation across line boundaries without penalty.
    - Strictly space-agnostic alignment: spaces and linebreaks preserved for rendering, but ignored in scoring.
    """
    def __init__(
        self,
        reference_text: str,
        verifier: SushrotaVerifier,
        max_pause_s: float = 1.5
    ):
        self.verifier = verifier
        self.max_pause_s = max_pause_s
        self.orig_text = clean_devanagari_input(reference_text)
        ref_lines = [l for l in self.orig_text.splitlines() if l.strip()]
        self.total_lines = len(ref_lines) if ref_lines else (1 if self.orig_text else 0)

        # Build clean phonetic representation and index map
        self.clean_ref: List[str] = []
        self.clean_to_orig: List[int] = []
        for idx, ch in enumerate(self.orig_text):
            if ch not in NON_PHONETIC_CHARS:
                self.clean_ref.append(ch)
                self.clean_to_orig.append(idx)
        self.clean_ref_str = "".join(self.clean_ref)

        self.cursor_clean_idx = 0
        self.cursor_orig_idx = 0
        self.total_phonetic_chars = len(self.clean_ref)

        self.total_correct_chars = 0
        self.total_processed_chars = 0
        self.all_pauses: List[float] = []
        self.long_pauses: List[float] = []
        self.annotated_chunks: List[Dict] = []

    def process_utterance(self, audio_data: np.ndarray, pause_duration: float = 0.0) -> Optional[Dict]:
        """
        Processes an in-flight utterance segment:
        1. Transcribes audio to Devanagari text.
        2. Aligns phonetic tokens to the current window in reference text.
        3. Appends inline pause annotations if pause > max_pause_s.
        4. Advances cursor and updates overall recitation metrics.
        """
        transcription = self.verifier.transcribe(audio_data)
        clean_trans = [c for c in transcription if c not in NON_PHONETIC_CHARS]
        clean_trans_str = "".join(clean_trans)

        if not clean_trans_str:
            return None

        # Determine phonetic span in reference text
        start_clean = self.cursor_clean_idx
        trans_len = len(clean_trans_str)

        # Search window for optimal alignment span
        min_len = max(1, trans_len - 5)
        max_len = min(len(self.clean_ref) - start_clean, trans_len + 15)

        best_score = -1.0
        best_clean_end = min(len(self.clean_ref), start_clean + trans_len)

        if max_len >= min_len:
            for l in range(min_len, max_len + 1):
                candidate = self.clean_ref_str[start_clean : start_clean + l]
                ratio = difflib.SequenceMatcher(None, candidate, clean_trans_str).ratio()
                if ratio > best_score:
                    best_score = ratio
                    best_clean_end = start_clean + l

        # Map clean span back to original text slice (including whitespace & newlines)
        orig_start = self.cursor_orig_idx
        if best_clean_end < len(self.clean_to_orig):
            # Include formatting up to the start of the next clean character
            orig_end = self.clean_to_orig[best_clean_end]
        else:
            orig_end = len(self.orig_text)

        orig_slice = self.orig_text[orig_start : orig_end]

        # Perform space-agnostic character alignment on this slice
        tokens, correct_cnt, slice_total = space_agnostic_align(orig_slice, clean_trans_str)

        # Update cumulative counts
        self.total_correct_chars += correct_cnt
        self.total_processed_chars += slice_total
        self.cursor_clean_idx = best_clean_end
        self.cursor_orig_idx = orig_end

        # Pause duration analysis
        is_long_pause = pause_duration > self.max_pause_s
        if pause_duration > 0:
            self.all_pauses.append(pause_duration)
        if is_long_pause:
            self.long_pauses.append(pause_duration)
            pause_token = {
                "type": "pause_badge",
                "text": f"[⏸️ {pause_duration:.1f}s]",
                "duration_s": round(pause_duration, 1)
            }
            tokens.insert(0, pause_token)

        overall_acc = round(
            (self.total_correct_chars / self.total_processed_chars * 100) if self.total_processed_chars > 0 else 100.0,
            1
        )
        progress = round(
            (self.cursor_clean_idx / self.total_phonetic_chars * 100) if self.total_phonetic_chars > 0 else 100.0,
            1
        )

        lines_done = min(self.total_lines, len(self.annotated_chunks) + 1)
        chunk_result = {
            "transcription": transcription,
            "tokens": tokens,
            "orig_text_segment": orig_slice,
            "pause_duration": round(pause_duration, 2),
            "is_long_pause": is_long_pause,
            "overall_accuracy": overall_acc,
            "accuracy_percentage": overall_acc,
            "progress_percentage": progress,
            "line_index": lines_done,
            "lines_completed": lines_done,
            "total_lines": self.total_lines,
            "total_long_pauses": len(self.long_pauses),
            "total_hesitations": len(self.long_pauses),
            "hesitation_count": len(self.long_pauses),
            "is_completed": self.cursor_clean_idx >= self.total_phonetic_chars
        }
        self.annotated_chunks.append(chunk_result)
        return chunk_result

    def get_session_summary(self) -> Dict:
        """Calculates final comprehensive session evaluation."""
        overall_acc = round(
            (self.total_correct_chars / self.total_processed_chars * 100) if self.total_processed_chars > 0 else 100.0,
            1
        )
        avg_pause = round(sum(self.all_pauses) / len(self.all_pauses), 2) if self.all_pauses else 0.0
        lines_done = min(self.total_lines, len(self.annotated_chunks))

        return {
            "overall_accuracy": overall_acc,
            "accuracy_percentage": overall_acc,
            "total_phonetic_chars": self.total_phonetic_chars,
            "processed_chars": self.total_processed_chars,
            "correct_chars": self.total_correct_chars,
            "total_chunks": len(self.annotated_chunks),
            "lines_completed": lines_done,
            "total_lines": self.total_lines,
            "total_long_pauses": len(self.long_pauses),
            "total_hesitations": len(self.long_pauses),
            "hesitation_count": len(self.long_pauses),
            "average_pause_seconds": avg_pause,
            "completed_all": self.cursor_clean_idx >= self.total_phonetic_chars
        }
