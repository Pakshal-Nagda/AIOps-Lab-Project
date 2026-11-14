from fastapi import FastAPI, UploadFile, File, Form, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, JSONResponse
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
import shutil
import os
import re
import time
import json
import requests
import numpy as np

from src.vad_checker import detect_long_pauses, StreamingVAD
from src.aligner import SushrotaVerifier, ContinuousSutraAligner

app = FastAPI(
    title="Project Medhā - Sanskrit Recitation & Verification API",
    description="Production-grade AIOps model serving for Sanskrit & Prakrit live recitation analysis, VAD pause detection, and phonetic alignment.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================
# Prometheus Metrics Instrumentation
# ==========================================
REQUEST_COUNT = Counter('api_requests_total', 'Total HTTP API requests', ['endpoint'])
INFERENCE_TIME = Histogram('inference_duration_seconds', 'Time spent running ASR inference and VAD')
LIVE_SESSIONS_COUNT = Counter('live_recitation_sessions_total', 'Total live WebSocket recitation sessions started')
LIVE_LINES_COUNT = Counter('live_lines_verified_total', 'Total individual sutra lines verified live')
ACTIVE_WEBSOCKETS = Gauge('active_websocket_connections', 'Current active live recitation WebSocket sessions')

# Load the model into memory once when server boots
print("Initializing Sushrota ASR Engine...")
verifier = SushrotaVerifier()

@app.get("/health")
def health_check():
    """Health check endpoint for container orchestrators and monitoring probes."""
    mlflow_status = "offline"
    mlflow_url = verifier.mlflow_serving_url
    if mlflow_url:
        try:
            ping_url = mlflow_url.replace("/invocations", "/ping")
            r = requests.get(ping_url, timeout=1.0)
            if r.status_code == 200:
                mlflow_status = "active"
        except Exception:
            pass

    return {
        "status": "healthy",
        "service": "Project Medhā Recitation Engine",
        "model_dir": verifier.model_dir,
        "vocab_size": len(verifier.vocab),
        "mlflow_serving": {
            "url": mlflow_url,
            "status": mlflow_status
        }
    }

@app.get("/metrics")
def metrics():
    """Prometheus exporter endpoint scraping inference metrics and request counts."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.post("/api/v1/analyze")
async def analyze_recitation(
    file: UploadFile = File(...), 
    reference_text: str = Form(...)
):
    """
    Standard REST endpoint for batch audio file recitation verification.
    Accepts audio file (WAV/MP3/OGG) and expected Sanskrit/Prakrit reference text.
    """
    REQUEST_COUNT.labels(endpoint='/api/v1/analyze').inc()
    start_time = time.time()
    
    temp_file = f"temp_{int(time.time()*1000)}_{file.filename}"
    with open(temp_file, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    try:
        # 1. Run VAD to detect pauses > 1.5s
        pauses = detect_long_pauses(temp_file, max_allowed_pause=1.5)
        
        # 2. Run Sushrota ASR Model
        transcribed = verifier.transcribe(temp_file)
        
        # 3. Akshara & Phonetic Comparison
        comparison = verifier.verify_recitation(transcribed, reference_text)
        
        duration = time.time() - start_time
        INFERENCE_TIME.observe(duration)
        
        return {
            "status": "success",
            "reference_text": reference_text,
            "transcription": transcribed,
            "similarity_score": comparison["match_accuracy_percentage"],
            "discrepancies": comparison["discrepancies"],
            "diff_tokens": comparison["diff_tokens"],
            "long_pauses": pauses,
            "latency_seconds": round(duration, 3)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
        
    finally:
        if os.path.exists(temp_file):
            os.remove(temp_file)

# ==========================================================
# Collaborative Speech Data Collection Endpoints
# ==========================================================
DATA_DIR = os.getenv("DATA_DIR", "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
METADATA_CSV = os.path.join(RAW_DATA_DIR, "metadata.csv")
PROMPTS_JSON = os.path.join(DATA_DIR, "prompts.json")

import csv
import librosa

@app.get("/api/v1/collector/prompts")
def get_prompts():
    """Returns the list of sutra verse prompts for speakers to record."""
    if not os.path.exists(PROMPTS_JSON):
        default_prompts = [
            {"id": 1, "text": "णमो अरिहंताणं"},
            {"id": 2, "text": "णमो सिद्धाणं"},
            {"id": 3, "text": "णमो आयरियाणं"},
            {"id": 4, "text": "णमो उवज्झायाणं"},
            {"id": 5, "text": "णमो लोए सव्वसाहूणं"}
        ]
        os.makedirs(DATA_DIR, exist_ok=True)
        with open(PROMPTS_JSON, "w", encoding="utf-8") as f:
            json.dump(default_prompts, f, ensure_ascii=False, indent=2)
        return {"prompts": default_prompts}

    with open(PROMPTS_JSON, "r", encoding="utf-8") as f:
        prompts = json.load(f)
    return {"prompts": prompts}

@app.post("/api/v1/collector/prompts")
def update_prompts(data: dict):
    """Allows uploading new verse prompts (e.g. from pasted multi-line text)."""
    os.makedirs(DATA_DIR, exist_ok=True)
    if "raw_text" in data:
        lines = [l.strip() for l in data["raw_text"].strip().splitlines() if l.strip()]
        prompts = [{"id": i + 1, "text": line} for i, line in enumerate(lines)]
    elif "prompts" in data:
        prompts = data["prompts"]
    else:
        raise HTTPException(status_code=400, detail="Missing 'raw_text' or 'prompts' field")

    with open(PROMPTS_JSON, "w", encoding="utf-8") as f:
        json.dump(prompts, f, ensure_ascii=False, indent=2)
    return {"status": "success", "total_prompts": len(prompts)}

@app.post("/api/v1/collector/upload")
async def upload_collector_audio(
    file: UploadFile = File(...),
    prompt_id: int = Form(...),
    text: str = Form(...),
    speaker: str = Form("Anonymous")
):
    """
    Receives recorded audio from a speaker, normalizes it to 16kHz mono WAV,
    saves it to data/raw/, and appends an entry to data/raw/metadata.csv.
    """
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    clean_speaker = re.sub(r'[^a-zA-Z0-9_-]', '_', speaker.strip()) or "Anonymous"
    timestamp_ms = int(time.time() * 1000)
    out_filename = f"clip_{timestamp_ms}_{clean_speaker}_p{prompt_id}.wav"
    out_filepath = os.path.join(RAW_DATA_DIR, out_filename)

    temp_path = os.path.join(RAW_DATA_DIR, f"temp_{timestamp_ms}_{file.filename}")
    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # =====================================
        # UNIVERSAL AUDIO DECODING
        # =====================================

        import soundfile as sf
        import subprocess

        converted_path = temp_path + ".wav"

        result = subprocess.run(
            [
                "ffmpeg",
                "-nostdin",
                "-y",
                "-i", temp_path,
                "-vn",
                "-ac", "1",
                "-ar", "16000",
                "-c:a", "pcm_s16le",
                converted_path
            ],
            capture_output=True,
            text=True,
            timeout=120
        )

        if result.returncode != 0:
            raise ValueError(
                "Audio decoding failed: " + result.stderr[-1500:]
            )

        wav, sr = sf.read(
            converted_path,
            dtype="float32"
        )

        os.remove(converted_path)

        # Slight normalization
        max_val = np.max(np.abs(wav))
        if max_val > 0.01:
            wav = wav / max_val * 0.95

        sf.write(out_filepath, wav, 16000, subtype="PCM_16")
        duration = round(len(wav) / 16000.0, 2)

        # Append to metadata.csv
        file_exists = os.path.exists(METADATA_CSV)
        with open(METADATA_CSV, "a", encoding="utf-8", newline="") as csvfile:
            writer = csv.writer(csvfile)
            if not file_exists:
                writer.writerow(["filename", "text", "speaker", "duration_seconds", "timestamp"])
            writer.writerow([out_filename, text, clean_speaker, duration, time.strftime("%Y-%m-%d %H:%M:%S")])

        # Compute running stats
        total_clips = 0
        total_seconds = 0.0
        with open(METADATA_CSV, "r", encoding="utf-8") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                total_clips += 1
                try:
                    total_seconds += float(row.get("duration_seconds", 0.0))
                except ValueError:
                    pass

        return {
            "status": "success",
            "filename": out_filename,
            "duration": duration,
            "prompt_id": prompt_id,
            "total_clips": total_clips,
            "total_duration_minutes": round(total_seconds / 60.0, 2)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

        if os.path.exists(converted_path):
            os.remove(converted_path)

@app.get("/api/v1/collector/stats")
def get_collector_stats():
    """Returns summary statistics for recorded dataset."""
    if not os.path.exists(METADATA_CSV):
        return {
            "total_clips": 0,
            "total_minutes": 0.0,
            "speakers": {},
            "recent_recordings": []
        }

    total_clips = 0
    total_seconds = 0.0
    speakers = {}

    with open(METADATA_CSV, "r", encoding="utf-8") as csvfile:
        reader = list(csv.DictReader(csvfile))
        total_clips = len(reader)
        for row in reader:
            spk = row.get("speaker", "Unknown")
            speakers[spk] = speakers.get(spk, 0) + 1
            try:
                total_seconds += float(row.get("duration_seconds", 0.0))
            except ValueError:
                pass
        recent = reader[-5:] if reader else []

    return {
        "total_clips": total_clips,
        "total_minutes": round(total_seconds / 60.0, 2),
        "speakers": speakers,
        "recent_recordings": recent
    }

@app.websocket("/ws/recite")
async def websocket_recite_endpoint(websocket: WebSocket):
    """
    Real-Time Streaming WebSocket Endpoint for Continuous Decoupled Recitation.
    
    Protocol:
    1. Client initiates connection and sends initial handshake JSON:
       {"action": "start", "reference_text": "...", "energy_threshold": 0.012, "max_pause_s": 1.5}
    2. Server sends {"event": "session_started", "total_phonetic_chars": N}
    3. Client streams 16kHz 16-bit PCM binary audio chunks continuously without blocking recitation.
    4. VAD tracks active speech and pauses (both intra-line and inter-line).
       - When an utterance boundary is detected: server runs ASR and space-agnostic phonetic alignment,
         immediately pushing {"event": "chunk_annotated", "tokens": [...], ...}.
       - Unusually long pauses (> 1.5s) are embedded as inline pause annotations.
       - If pause > 1.5s is ongoing: emits {"event": "long_pause_warning", "pause_duration": ...}
    5. Client can send {"action": "stop"} to finalize session and receive full evaluation summary.
    """
    await websocket.accept()
    ACTIVE_WEBSOCKETS.inc()
    LIVE_SESSIONS_COUNT.inc()

    session: ContinuousSutraAligner = None
    vad: StreamingVAD = None

    try:
        while True:
            message = await websocket.receive()
            
            # 1. Handling JSON control messages (Handshake / Stop)
            if "text" in message:
                data = json.loads(message["text"])
                action = data.get("action", "")

                if action == "start":
                    ref_text = data.get("reference_text", "")
                    max_pause_s = float(data.get("max_pause_s", 1.5))
                    energy_threshold = float(data.get("energy_threshold", 0.012))
                    line_pause_s = float(data.get("line_pause_s", 0.70))

                    session = ContinuousSutraAligner(
                        reference_text=ref_text,
                        verifier=verifier,
                        max_pause_s=max_pause_s
                    )
                    vad = StreamingVAD(
                        sample_rate=16000,
                        energy_threshold=energy_threshold,
                        min_speech_duration_s=0.30,
                        line_pause_s=line_pause_s,
                        max_allowed_pause_s=max_pause_s
                    )

                    await websocket.send_json({
                        "event": "session_started",
                        "total_phonetic_chars": session.total_phonetic_chars
                    })

                elif action == "stop":
                    # Client requested session stop; flush any remaining speech in buffer
                    if vad and session:
                        residual_audio = vad.flush()
                        if residual_audio is not None and len(residual_audio) > 3200:
                            t0 = time.time()
                            chunk_res = session.process_utterance(residual_audio, pause_duration=0.0)
                            if chunk_res:
                                INFERENCE_TIME.observe(time.time() - t0)
                                LIVE_LINES_COUNT.inc()
                                await websocket.send_json({
                                    "event": "chunk_annotated",
                                    **chunk_res
                                })

                        await websocket.send_json({
                            "event": "session_completed",
                            "summary": session.get_session_summary()
                        })
                    break

            # 2. Handling binary 16kHz PCM audio chunks
            elif "bytes" in message:
                if not vad or not session:
                    continue

                raw_bytes = message["bytes"]
                if not raw_bytes:
                    continue

                # Unpack 16-bit signed PCM to float32 (-1.0 to 1.0)
                pcm_int16 = np.frombuffer(raw_bytes, dtype=np.int16)
                float_chunk = pcm_int16.astype(np.float32) / 32768.0

                events = vad.feed_chunk(float_chunk)

                for ev in events:
                    if ev["event"] == "speech_start":
                        await websocket.send_json({
                            "event": "speech_started",
                            "lead_pause_duration": ev.get("lead_pause_duration", 0.0),
                            "is_long_pause": ev.get("is_long_pause", False)
                        })

                    elif ev["event"] in ("utterance_boundary", "line_boundary"):
                        audio_segment = ev["audio"]
                        pause_duration = ev["pause_duration"]

                        # Run ASR inference and pronunciation verification
                        t0 = time.time()
                        chunk_res = session.process_utterance(audio_segment, pause_duration=pause_duration)
                        if chunk_res:
                            INFERENCE_TIME.observe(time.time() - t0)
                            LIVE_LINES_COUNT.inc()

                            await websocket.send_json({
                                "event": "chunk_annotated",
                                **chunk_res
                            })

                            if chunk_res["is_completed"]:
                                await websocket.send_json({
                                    "event": "session_completed",
                                    "summary": session.get_session_summary()
                                })

                    elif ev["event"] == "long_pause_warning":
                        await websocket.send_json({
                            "event": "long_pause_warning",
                            "pause_duration": ev["pause_duration"]
                        })

    except WebSocketDisconnect:
        print("WebSocket client disconnected normally.")
    except Exception as e:
        print(f"Error in WebSocket recitation stream: {e}")
    finally:
        ACTIVE_WEBSOCKETS.dec()

