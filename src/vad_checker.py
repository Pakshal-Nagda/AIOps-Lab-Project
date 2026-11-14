"""
Project Medhā - Voice Activity Detection & Pause Analysis
=========================================================
Provides both static audio file pause analysis and real-time streaming VAD
for live recitation boundary and hesitation detection.
"""

import numpy as np
import librosa
from typing import List, Dict, Tuple, Optional


def detect_long_pauses(audio_path: str, max_allowed_pause: float = 1.5) -> List[Dict]:
    """
    Loads static audio and flags segments where silence/pauses exceed max_allowed_pause seconds.
    Used for batch analysis and endpoint validation.
    """
    y, sr = librosa.load(audio_path, sr=16000)

    # Use librosa's non-silent interval detector to find speech chunks
    intervals = librosa.effects.split(y, top_db=30)

    pauses = []
    for i in range(len(intervals) - 1):
        end_current_speech = intervals[i][1] / sr
        start_next_speech = intervals[i + 1][0] / sr
        pause_duration = start_next_speech - end_current_speech

        if pause_duration > max_allowed_pause:
            pauses.append({
                "after_second": round(end_current_speech, 2),
                "pause_duration_seconds": round(pause_duration, 2)
            })

    return pauses


class StreamingVAD:
    """
    Real-Time Streaming Voice Activity Detector (VAD).
    Analyzes incoming 16kHz PCM audio chunks to:
    1. Delimit recitation line boundaries based on natural cadence pauses (e.g. 0.7s - 1.0s).
    2. Flag hesitation / unusually long pauses (> max_allowed_pause, e.g. 1.5s).
    3. Accumulate active speech audio buffers for incremental ASR transcription.
    """
    def __init__(
        self,
        sample_rate: int = 16000,
        energy_threshold: float = 0.012,
        min_speech_duration_s: float = 0.35,
        line_pause_s: float = 0.75,
        max_allowed_pause_s: float = 1.5
    ):
        self.sample_rate = sample_rate
        self.energy_threshold = energy_threshold
        self.min_speech_samples = int(sample_rate * min_speech_duration_s)
        self.line_pause_samples = int(sample_rate * line_pause_s)
        self.max_allowed_pause_samples = int(sample_rate * max_allowed_pause_s)

        # State tracking
        self.speech_buffer: List[np.ndarray] = []
        self.total_speech_samples: int = 0
        self.silence_samples: int = 0
        self.inter_line_silence_samples: int = 0
        self.is_speaking: bool = False
        self.long_pause_emitted: bool = False
        self.session_time_samples: int = 0

    def feed_chunk(self, chunk: np.ndarray) -> List[Dict]:
        """
        Feeds a chunk of 16kHz mono float32 audio.
        Returns a list of detected events, such as:
        - {"event": "speech_start", "lead_pause_duration": float, "is_long_pause": bool}
        - {"event": "line_boundary", "audio": np.ndarray, "pause_duration": float, "is_long_pause": bool}
        - {"event": "long_pause_warning", "pause_duration": float}
        """
        if chunk.ndim > 1:
            chunk = chunk.mean(axis=-1)
        chunk = chunk.astype(np.float32)

        chunk_len = len(chunk)
        self.session_time_samples += chunk_len

        rms = float(np.sqrt(np.mean(chunk ** 2))) if chunk_len > 0 else 0.0
        is_speech = rms > self.energy_threshold

        events = []

        if is_speech:
            if not self.is_speaking:
                self.is_speaking = True
                lead_pause_s = self.inter_line_silence_samples / self.sample_rate
                is_long_lead = lead_pause_s >= (self.max_allowed_pause_samples / self.sample_rate)
                events.append({
                    "event": "speech_start",
                    "time": round(self.session_time_samples / self.sample_rate, 2),
                    "lead_pause_duration": round(lead_pause_s, 2),
                    "is_long_pause": is_long_lead
                })

            self.speech_buffer.append(chunk)
            self.total_speech_samples += chunk_len
            self.silence_samples = 0
            self.inter_line_silence_samples = 0
            self.long_pause_emitted = False

            # If continuous recitation without pause exceeds 10s, yield utterance chunk to prevent UI lag
            if self.total_speech_samples >= (self.sample_rate * 10):
                accumulated_audio = np.concatenate(self.speech_buffer)
                events.append({
                    "event": "utterance_boundary",
                    "audio": accumulated_audio,
                    "pause_duration": 0.0,
                    "is_long_pause": False
                })
                self.speech_buffer = []
                self.total_speech_samples = 0

        else:
            self.silence_samples += chunk_len
            self.inter_line_silence_samples += chunk_len

            if self.is_speaking:
                # In speech, now encountering trailing silence
                self.speech_buffer.append(chunk)

                # Check if silence reached cadence pause duration (0.7s)
                if self.silence_samples >= self.line_pause_samples:
                    if self.total_speech_samples >= self.min_speech_samples:
                        # Full utterance captured. Trim excessive trailing silence from speech buffer.
                        accumulated_audio = np.concatenate(self.speech_buffer)
                        pause_duration = self.silence_samples / self.sample_rate
                        is_long = pause_duration >= (self.max_allowed_pause_samples / self.sample_rate)

                        events.append({
                            "event": "utterance_boundary",
                            "audio": accumulated_audio,
                            "pause_duration": round(pause_duration, 2),
                            "is_long_pause": is_long
                        })

                    # Reset buffer for next utterance (regardless of whether discarded as noise blip)
                    self.speech_buffer = []
                    self.total_speech_samples = 0
                    self.is_speaking = False
            else:
                # Silence between lines or before recitation starts
                if not self.long_pause_emitted and self.inter_line_silence_samples >= self.max_allowed_pause_samples:
                    self.long_pause_emitted = True
                    pause_duration = self.inter_line_silence_samples / self.sample_rate
                    events.append({
                        "event": "long_pause_warning",
                        "pause_duration": round(pause_duration, 2)
                    })

        return events

    def flush(self) -> Optional[np.ndarray]:
        """Flushes any remaining speech buffer at session close."""
        if self.speech_buffer and self.total_speech_samples >= self.min_speech_samples:
            accumulated = np.concatenate(self.speech_buffer)
            self.speech_buffer = []
            self.total_speech_samples = 0
            self.is_speaking = False
            return accumulated
        return None
