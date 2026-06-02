"""
Vocal Agent Web Service
=======================
FastAPI backend providing standardized RESTful APIs and WebSocket
for real-time voice AI assistant capabilities.

API Endpoints (v1):
    GET  /api/v1/health         - Service health check
    POST /api/v1/stt            - Speech-to-Text (audio file → text)
    POST /api/v1/chat           - Chat with AI agent (text → text)
    POST /api/v1/tts            - Text-to-Speech (text → audio WAV)
    POST /api/v1/conversation   - Full pipeline (audio → text + response + audio)
    WS   /ws/v1/voice           - Real-time voice conversation
"""

import os
import io
import re
import time
import asyncio
import logging
import tempfile
from datetime import datetime
from typing import Optional

import numpy as np
import soundfile as sf
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from agent_client import knowledge_agent_client
from kokoro_onnx import Kokoro
from misaki import en, espeak, zh

load_dotenv()

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("vocal-agent")

# ---------------------------------------------------------------------------
# Pydantic Schemas (standardized request / response models)
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    """Service health status."""
    status: str = "ok"
    version: str = "1.0.0"
    models_loaded: bool = False
    timestamp: str


class ChatRequest(BaseModel):
    """Chat request payload."""
    message: str = Field(..., min_length=1, description="User message text")


class ChatResponse(BaseModel):
    """Chat response payload."""
    response: str
    timestamp: str


class STTResponse(BaseModel):
    """Speech-to-Text result."""
    text: str
    language: str = "en"
    duration_seconds: Optional[float] = None
    timestamp: str


class TTSRequest(BaseModel):
    """Text-to-Speech request payload."""
    text: str = Field(..., min_length=1, description="Text to synthesize")
    voice: Optional[str] = Field(default=None, description="Voice profile name")


class ConversationResponse(BaseModel):
    """Full-pipeline conversation result."""
    user_text: str
    agent_response: str
    timestamp: str


class ErrorResponse(BaseModel):
    """Standardized error response."""
    error: str
    detail: Optional[str] = None
    timestamp: str


# ---------------------------------------------------------------------------
# Server Configuration (from .env)
# ---------------------------------------------------------------------------

SERVER_HOST = os.getenv("SERVER_HOST", "0.0.0.0")
SERVER_PORT = int(os.getenv("SERVER_PORT", "8000"))


# ---------------------------------------------------------------------------
# FastAPI Application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Vocal Agent API",
    description="Real-time voice AI assistant with STT, LLM, and TTS capabilities.",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
)

# ---------------------------------------------------------------------------
# CORS Configuration
# ---------------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 建议在生产环境中将 "*" 替换为实际的前端域名白名单
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Global Model References
# ---------------------------------------------------------------------------

g2p_en = None
g2p_zh = None
kokoro_model = None
whisper_model = None
models_loaded = False


@app.on_event("startup")
async def load_models():
    """Load all ML models on server startup."""
    global g2p_en, g2p_zh, kokoro_model, whisper_model, models_loaded

    logger.info("Loading TTS models (Kokoro + G2P en/zh) ...")
    fallback = espeak.EspeakFallback(british=False)
    g2p_en = en.G2P(trf=False, british=False, fallback=fallback)
    g2p_zh = zh.ZHG2P()
    kokoro_model = Kokoro("kokoro-v1.0.onnx", "voices-v1.0.bin")
    logger.info("✅ TTS models loaded.")

    logger.info("Loading STT model (faster-whisper base) ...")
    from faster_whisper import WhisperModel
    whisper_model = WhisperModel("base", device="cpu", compute_type="int8")
    logger.info("✅ STT model loaded.")

    models_loaded = True
    logger.info("🚀 All models ready — server is accepting requests.")


def _ensure_models():
    """Raise 503 if models are still loading."""
    if not models_loaded:
        raise HTTPException(
            status_code=503,
            detail="Models are still loading, please try again shortly.",
        )


# ---------------------------------------------------------------------------
# Helper: synthesize text → WAV bytes
# ---------------------------------------------------------------------------

def _synthesize_wav(text: str, voice: Optional[str] = None) -> bytes:
    """Convert a text string to WAV audio bytes via Kokoro TTS."""
    # Split text into sentences to prevent exceeding the 510 phoneme limit
    raw_chunks = re.split(r'(?<=[.!?。！？])|\n+', text.strip())
    chunks = [c.strip() for c in raw_chunks if c.strip()]
    total_chunks = len(chunks)
    
    logger.info(f"🎙️ Starting TTS synthesis. Text length: {len(text)}, split into {total_chunks} chunks.")
    start_time = time.time()
    
    all_samples = []
    sample_rate = 24000  # Default sample rate
    
    for i, chunk in enumerate(chunks, 1):
        # Dynamically route Chinese and English
        is_chinese = bool(re.search(r'[\u4e00-\u9fff]', chunk))
        lang_tag = "ZH" if is_chinese else "EN"
        
        if is_chinese:
            phonemes, _ = g2p_zh(chunk)
            current_voice = voice if voice else "zf_xiaoxiao"
        else:
            phonemes, _ = g2p_en(chunk)
            current_voice = voice if voice else "af_heart"
            
        if not phonemes:
            logger.warning(f"  [{i}/{total_chunks}] ⚠️ No phonemes generated for chunk.")
            continue
            
        try:
            chunk_start = time.time()
            samples, sr = kokoro_model.create(phonemes, current_voice, is_phonemes=True)
            all_samples.append(samples)
            sample_rate = sr
            elapsed = time.time() - chunk_start
            preview = (chunk[:20] + "...") if len(chunk) > 20 else chunk
            logger.info(f"  [{i}/{total_chunks}] ✅ [{lang_tag}] Synth in {elapsed:.2f}s | Voice: {current_voice} | Text: {preview}")
        except Exception as e:
            logger.warning(f"  [{i}/{total_chunks}] ❌ Skipping TTS for chunk due to error: {e}")
            continue

    if not all_samples:
        raise ValueError("Could not synthesize any audio from the provided text.")

    final_samples = np.concatenate(all_samples)
    buf = io.BytesIO()
    sf.write(buf, final_samples.astype(np.float32), sample_rate, format="WAV")
    buf.seek(0)
    
    total_elapsed = time.time() - start_time
    total_audio_len = len(final_samples) / sample_rate
    logger.info(f"🏁 TTS synthesis completed in {total_elapsed:.2f}s. Generated {total_audio_len:.2f}s of audio.")
    
    return buf.read()


def _transcribe_audio(audio_bytes: bytes) -> tuple[str, float]:
    """Transcribe audio bytes via faster-whisper. Returns (text, duration)."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    try:
        segments, info = whisper_model.transcribe(tmp_path, beam_size=1)
        text = " ".join(seg.text for seg in segments).strip()
        duration = getattr(info, "duration", 0.0)
    finally:
        os.unlink(tmp_path)

    return text, duration


# ---------------------------------------------------------------------------
# REST API v1
# ---------------------------------------------------------------------------

@app.get("/api/v1/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Check service health and model readiness."""
    return HealthResponse(
        status="ok" if models_loaded else "loading",
        models_loaded=models_loaded,
        timestamp=datetime.now().isoformat(),
    )


@app.get("/api/v1/config", tags=["System"])
async def get_config():
    """
    Return server configuration for frontend clients.

    Frontend can call this on startup to discover the correct
    API base URL and WebSocket URL dynamically.
    """
    return {
        "api_base_url": os.getenv("API_BASE_URL", ""),
        "ws_base_url": os.getenv("WS_BASE_URL", ""),
        "version": "1.0.0",
    }


@app.post("/api/v1/stt", response_model=STTResponse, tags=["Speech"])
async def speech_to_text(file: UploadFile = File(..., description="Audio file (WAV/WebM)")):
    """
    Convert uploaded audio to text.

    Accepts WAV or WebM audio files recorded from browser.
    """
    _ensure_models()

    audio_bytes = await file.read()
    text, duration = await asyncio.to_thread(_transcribe_audio, audio_bytes)

    if not text:
        raise HTTPException(status_code=400, detail="Could not transcribe any speech from the audio.")

    return STTResponse(
        text=text,
        duration_seconds=round(duration, 2),
        timestamp=datetime.now().isoformat(),
    )


@app.post("/api/v1/chat", response_model=ChatResponse, tags=["Agent"])
async def chat(request: ChatRequest):
    """
    Send a text message to the AI knowledge agent and receive a text response.

    The agent may use tools (Google Search, Wikipedia, Arxiv) as needed.
    """
    _ensure_models()

    response = await asyncio.to_thread(knowledge_agent_client, request.message)
    if not response:
        raise HTTPException(status_code=500, detail="Agent returned no valid response.")

    return ChatResponse(
        response=response,
        timestamp=datetime.now().isoformat(),
    )


@app.post("/api/v1/tts", tags=["Speech"])
async def text_to_speech(request: TTSRequest):
    """
    Convert text to speech audio.

    Returns a WAV audio stream.
    """
    _ensure_models()

    try:
        wav_bytes = await asyncio.to_thread(_synthesize_wav, request.text, request.voice)
        return StreamingResponse(io.BytesIO(wav_bytes), media_type="audio/wav")
    except Exception as e:
        logger.error(f"TTS synthesis failed: {e}")
        raise HTTPException(status_code=500, detail=f"TTS synthesis error: {str(e)}")


@app.post("/api/v1/conversation", response_model=ConversationResponse, tags=["Pipeline"])
async def conversation(file: UploadFile = File(..., description="Audio file (WAV/WebM)")):
    """
    Full conversation pipeline: audio → STT → Agent → response.

    Upload an audio file; the server transcribes it, queries the AI agent,
    and returns both the transcription and the agent's text response.
    Use `/api/v1/tts` separately to synthesize the response into audio.
    """
    _ensure_models()

    audio_bytes = await file.read()
    user_text, _ = await asyncio.to_thread(_transcribe_audio, audio_bytes)

    if not user_text:
        raise HTTPException(status_code=400, detail="Could not transcribe any speech.")

    agent_response = await asyncio.to_thread(knowledge_agent_client, user_text)
    if not agent_response:
        raise HTTPException(status_code=500, detail="Agent returned no valid response.")

    return ConversationResponse(
        user_text=user_text,
        agent_response=agent_response,
        timestamp=datetime.now().isoformat(),
    )


# ---------------------------------------------------------------------------
# WebSocket v1 — Real-time TTS Stream
# ---------------------------------------------------------------------------

@app.websocket("/ws/v1/tts")
async def websocket_tts(ws: WebSocket):
    """
    Real-time text-to-speech over WebSocket with Abort support.
    
    Protocol:
        Client → Server: JSON {"text": "...", "voice": "..." (optional)}
        Client → Server: JSON {"action": "abort"} (Cancels ongoing TTS)
        Server → Client: binary (WAV audio chunk for each sentence)
        Server → Client: JSON {"type": "tts_end"}
    """
    await ws.accept()
    logger.info("🔌 WebSocket TTS client connected")
    
    current_task = None
    
    async def process_text_task(text_to_process: str, voice_override: Optional[str]):
        """Background task to synthesize and stream chunks."""
        raw_chunks = re.split(r'(?<=[.!?。！？])|\n+', text_to_process.strip())
        chunks = [c.strip() for c in raw_chunks if c.strip()]
        
        try:
            for chunk in chunks:
                if not chunk:
                    continue
                
                # Check for cancellation before expensive operations
                if asyncio.current_task().cancelled():
                    break
                    
                wav_bytes = await asyncio.to_thread(_synthesize_wav, chunk, voice_override)
                
                # Check again after yielding to thread
                if asyncio.current_task().cancelled():
                    break
                    
                await ws.send_bytes(wav_bytes)
                
            # Signal end if completed naturally
            if not asyncio.current_task().cancelled():
                await ws.send_json({"type": "tts_end"})
        except asyncio.CancelledError:
            logger.info("🛑 TTS background task was aborted cleanly.")
        except Exception as e:
            logger.error(f"TTS WebSocket background task error: {e}")

    try:
        while True:
            try:
                data = await ws.receive_json()
            except Exception:
                # Catch invalid JSON or disconnect during receive
                break
                
            action = data.get("action", "speak")
            
            if action == "abort":
                if current_task and not current_task.done():
                    current_task.cancel()
                    logger.info("🛑 Received abort signal, cancelled ongoing TTS task.")
                continue
                
            text = data.get("text", "")
            voice = data.get("voice", None)
            
            if text:
                if current_task and not current_task.done():
                    current_task.cancel()
                    logger.info("⚠️ New text arrived, implicitly cancelled previous TTS task.")
                
                current_task = asyncio.create_task(process_text_task(text, voice))
                
    except WebSocketDisconnect:
        logger.info("🔌 WebSocket TTS client disconnected")
    except Exception as e:
        logger.error(f"WebSocket TTS error: {e}")
    finally:
        if current_task and not current_task.done():
            current_task.cancel()


# ---------------------------------------------------------------------------
# WebSocket v1 — Real-time voice conversation
# ---------------------------------------------------------------------------

@app.websocket("/ws/v1/voice")
async def websocket_voice(ws: WebSocket):
    """
    Real-time voice conversation over WebSocket.

    Protocol:
        Client → Server:  binary (audio WAV/WebM bytes)
        Server → Client:  JSON   {"type": "transcription", "text": "..."}
        Server → Client:  JSON   {"type": "response", "text": "..."}
        Server → Client:  binary (WAV audio chunk for each sentence)
        Server → Client:  JSON   {"type": "audio_end"}
    """
    await ws.accept()
    logger.info("🔌 WebSocket client connected")

    try:
        while True:
            # 1. Receive audio from browser
            audio_data = await ws.receive_bytes()
            logger.info(f"📥 Received audio: {len(audio_data)} bytes")

            # 2. STT
            try:
                user_text, _ = await asyncio.to_thread(_transcribe_audio, audio_data)
            except Exception as e:
                await ws.send_json({"type": "error", "message": f"STT failed: {e}"})
                continue

            if not user_text:
                await ws.send_json({"type": "error", "message": "Could not transcribe audio"})
                continue

            await ws.send_json({"type": "transcription", "text": user_text})

            # 3. Agent
            try:
                agent_response = await asyncio.to_thread(knowledge_agent_client, user_text)
            except Exception as e:
                await ws.send_json({"type": "error", "message": f"Agent error: {e}"})
                continue

            if not agent_response:
                await ws.send_json({"type": "error", "message": "No agent response"})
                continue

            await ws.send_json({"type": "response", "text": agent_response})

            # 4. TTS — stream sentence by sentence
            raw_sentences = re.split(r'(?<=[.!?。！？])|\n+', agent_response.strip())
            for sentence in raw_sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue
                try:
                    wav_bytes = await asyncio.to_thread(_synthesize_wav, sentence)
                    await ws.send_bytes(wav_bytes)
                except Exception as e:
                    logger.error(f"TTS error for sentence: {e}")

            await ws.send_json({"type": "audio_end"})

    except WebSocketDisconnect:
        logger.info("🔌 WebSocket client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")


# ---------------------------------------------------------------------------
# Static files (frontend) — must be mounted LAST
# ---------------------------------------------------------------------------

app.mount("/", StaticFiles(directory="static", html=True), name="static")


# ---------------------------------------------------------------------------
# Entry point: python server.py
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    logger.info(f"Starting server on {SERVER_HOST}:{SERVER_PORT}")
    uvicorn.run("server:app", host=SERVER_HOST, port=SERVER_PORT, reload=True)
