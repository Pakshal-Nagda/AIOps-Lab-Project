from fastapi import FastAPI, UploadFile, File, Form, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, JSONResponse
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
import shutil
import os
import time
import json
import numpy as np

from src.vad_checker import detect_long_pauses, StreamingVAD
from src.aligner import SushrotaVerifier, ContinuousSutraAligner, LiveSutraSession, split_sutra_lines

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
    return {
        "status": "healthy",
        "service": "Project Medhā Recitation Engine",
        "model_dir": verifier.model_dir,
        "vocab_size": len(verifier.vocab)
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

