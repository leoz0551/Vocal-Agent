import asyncio
import io
import re
import time
import json
import logging
import base64
import torch
import numpy as np
import threading
from typing import Optional, Callable, Any
from fastapi import WebSocket, WebSocketDisconnect

from agent_client import create_streaming_agent, knowledge_agent_client_stream

logger = logging.getLogger("vocal-agent-v2")

# Global VAD model cache
_vad_model = None
_vad_utils = None

def get_vad_model():
    global _vad_model, _vad_utils
    if _vad_model is None:
        logger.info("Loading Silero VAD model for V2...")
        try:
            import os
            torch_hub_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "torch_hub")
            torch.hub.set_dir(torch_hub_dir)
            _vad_model, _vad_utils = torch.hub.load(repo_or_dir="snakers4/silero-vad", model="silero_vad", trust_repo=True)
            logger.info("✅ Silero VAD model loaded.")
        except Exception as e:
            logger.error(f"Failed to load VAD model: {e}")
            raise
    return _vad_model, _vad_utils

class RealtimeVoiceSession:
    def __init__(self, ws: WebSocket, synthesize_fn: Callable, whisper_model: Any, gpu_lock: asyncio.Lock):
        self.ws = ws
        self.synthesize_fn = synthesize_fn
        self.whisper_model = whisper_model
        self.gpu_lock = gpu_lock
        
        self.audio_queue = asyncio.Queue()
        self.is_connected = False
        self.is_initialized = False
        self.system_prompt = "You are a helpful assistant."
        self.voice_id = "af_heart"
        self.language = None
        self.agent = None
        
        self.receive_task: Optional[asyncio.Task] = None
        self.vad_task: Optional[asyncio.Task] = None
        self.pipeline_task: Optional[asyncio.Task] = None
        
        # Load VAD
        model, _ = get_vad_model()
        self.vad_model = model

    async def start(self):
        self.is_connected = True
        logger.info("🔌 V2 WebSocket client connected. Waiting for init...")
        
        try:
            # 1. Wait for init message
            init_data = await self.ws.receive_json()
            if init_data.get("action") == "init":
                self.system_prompt = init_data.get("system_prompt", self.system_prompt)
                self.voice_id = init_data.get("voice_id", self.voice_id)
                lang_val = init_data.get("language")
                self.language = lang_val if lang_val and lang_val != "auto" else None
                self.agent = create_streaming_agent(self.system_prompt)
                self.is_initialized = True
                await self.ws.send_json({"type": "init_success", "message": "Session initialized."})
                logger.info(f"Session initialized with voice: {self.voice_id}")
            else:
                await self.ws.send_json({"type": "error", "message": "Expected init action."})
                return

            # 2. Start concurrent loops
            self.receive_task = asyncio.create_task(self._receive_loop())
            self.vad_task = asyncio.create_task(self._vad_loop())
            
            # Keep alive: wait until EITHER task finishes (e.g. receive_task breaks on disconnect)
            await asyncio.wait(
                [self.receive_task, self.vad_task],
                return_when=asyncio.FIRST_COMPLETED
            )
            
        except WebSocketDisconnect:
            logger.info("🔌 V2 WebSocket client disconnected")
        except Exception as e:
            logger.error(f"V2 Session Error: {e}")
        finally:
            await self.close()

    async def _receive_loop(self):
        """Continuously receive WebSocket data."""
        logger.debug("[V2] Starting receive loop...")
        try:
            while self.is_connected:
                message = await self.ws.receive()
                
                if message["type"] == "websocket.disconnect":
                    logger.info("🔌 [V2] Client disconnected naturally.")
                    self.is_connected = False
                    break
                elif message.get("bytes"):
                    # Received raw PCM chunk
                    await self.audio_queue.put(message.get("bytes"))
                elif message.get("text"):
                    try:
                        data = json.loads(message.get("text"))
                        logger.debug(f"[V2] Received WS Action: {data}")
                        if data.get("action") == "abort":
                            await self.handle_interruption()
                    except json.JSONDecodeError:
                        pass
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"[V2] Receive loop error: {e}")
            self.is_connected = False

    async def _vad_loop(self):
        """Process raw PCM chunks via Silero VAD to detect speech bounds."""
        logger.debug("[V2] Starting VAD loop...")
        vad_chunk_size = 512
        pcm_buffer = bytearray()
        
        is_speaking = False
        speech_buffer = []
        pre_speech_buffer = [] # Rolling buffer to keep history before VAD triggers
        
        silence_chunks = 0
        speech_chunks = 0
        MAX_SILENCE_CHUNKS = int((16000 / vad_chunk_size) * 1.7) # 1.7s of silence triggers STT
        MIN_SPEECH_CHUNKS = 4 # ~120ms of continuous speech to trigger interruption
        PRE_SPEECH_HISTORY = 15 # ~480ms history to avoid cutting off start of words
        
        try:
            while self.is_connected:
                chunk = await self.audio_queue.get()
                pcm_buffer.extend(chunk)
                
                # process in 512-frame chunks (1024 bytes for int16)
                while len(pcm_buffer) >= vad_chunk_size * 2:
                    frame_bytes = pcm_buffer[:vad_chunk_size * 2]
                    pcm_buffer = pcm_buffer[vad_chunk_size * 2:]
                    
                    # Convert Int16 PCM to Float32 Tensor for Silero
                    pcm_data = np.frombuffer(frame_bytes, dtype=np.int16)
                    float_data = pcm_data.astype(np.float32) / 32768.0
                    tensor_chunk = torch.from_numpy(float_data)
                    
                    if not is_speaking:
                        # Keep a rolling history of recent chunks
                        pre_speech_buffer.append(float_data)
                        if len(pre_speech_buffer) > PRE_SPEECH_HISTORY:
                            pre_speech_buffer.pop(0)
                    
                    # Get Speech Probability
                    speech_prob = self.vad_model(tensor_chunk, 16000).item()
                    
                    if speech_prob > 0.7: # Higher threshold to avoid noise
                        silence_chunks = 0
                        speech_chunks += 1
                        if not is_speaking and speech_chunks >= MIN_SPEECH_CHUNKS:
                            is_speaking = True
                            logger.info("🗣️ [VAD] User started speaking! (Debounced)")
                            # Interrupt AI if it is speaking
                            if self.pipeline_task and not self.pipeline_task.done():
                                await self.handle_interruption()
                            
                            # Prepend the history buffer so we don't clip the first word
                            speech_buffer = list(pre_speech_buffer)
                            pre_speech_buffer = []
                    else:
                        speech_chunks = 0
                    
                    if is_speaking:
                        speech_buffer.append(float_data)
                        if speech_prob < 0.5: # Lower threshold for silence
                            silence_chunks += 1
                            
                        if silence_chunks > MAX_SILENCE_CHUNKS:
                            # User stopped speaking
                            is_speaking = False
                            logger.info(f"🔇 [VAD] User stopped speaking (collected {len(speech_buffer)} chunks).")
                            
                            if speech_buffer:
                                full_speech = np.concatenate(speech_buffer)
                                # Only process if utterance is > 0.5 seconds
                                if len(full_speech) > 16000 * 0.5:
                                    logger.debug(f"[V2] Triggering pipeline with {len(full_speech)} frames.")
                                    self.pipeline_task = asyncio.create_task(self.run_pipeline(full_speech))
                                else:
                                    logger.info("[V2] Speech too short, ignoring.")
                            speech_buffer = []
                            pre_speech_buffer = []
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"[V2] VAD loop error: {e}")


    async def handle_interruption(self):
        """Cancel ongoing LLM/TTS tasks and notify frontend to stop playback."""
        logger.info("🛑 Interruption triggered!")
        if hasattr(self, 'cancel_event'):
            self.cancel_event.set()
        if self.pipeline_task and not self.pipeline_task.done():
            self.pipeline_task.cancel()
            
        # Send clear signal to frontend to stop audio playback
        try:
            await self.ws.send_json({"type": "clear_audio"})
        except Exception:
            pass

    async def _tts_worker(self, sentence_queue: asyncio.Queue, cancel_event: threading.Event):
        """Background worker to synthesize text while LLM is still streaming."""
        try:
            while True:
                sentence = await sentence_queue.get()
                if sentence is None: # Sentinel to exit
                    break
                if cancel_event.is_set():
                    break
                
                # Synthesize
                logger.debug(f"[V2] TTS worker synthesizing: {sentence}")
                async with self.gpu_lock:
                    wav_bytes = await asyncio.to_thread(self.synthesize_fn, sentence, self.voice_id, cancel_event)
                
                if wav_bytes and not asyncio.current_task().cancelled():
                    await self.ws.send_bytes(wav_bytes)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"[V2] TTS worker error: {e}")

    async def run_pipeline(self, audio_data: np.ndarray):
        """STT -> Stream LLM -> TTS sentences -> send to WS."""
        try:
            logger.info(f"🎙️ [V2] Running STT...")
            await self.ws.send_json({"type": "thinking", "message": "Transcribing..."})
            
            # 1. STT (faster-whisper accepts float32 numpy arrays)
            def transcribe_sync():
                kwargs = {"beam_size": 1, "vad_filter": True}
                if self.language:
                    kwargs["language"] = self.language
                segments, info = self.whisper_model.transcribe(audio_data, **kwargs)
                return " ".join(seg.text for seg in segments).strip()

            async with self.gpu_lock:
                user_text = await asyncio.to_thread(transcribe_sync)
            
            # Filter hallucinations and extremely short audio
            cleaned_text = re.sub(r'[^\w\s]', '', user_text).strip()
            filler_words = {"呃", "啊", "嗯", "哦", "嗯嗯", "呃呃"}
            is_only_filler = all(char in filler_words for char in cleaned_text)
            
            if not user_text or not user_text.strip() or len(cleaned_text) < 2 or is_only_filler:
                logger.info(f"[V2] STT resulted in empty or garbage text: '{user_text}', ignoring.")
                await self.ws.send_json({"type": "audio_end"})
                return
                
            logger.info(f"📝 [V2] STT: {user_text}")
            await self.ws.send_json({"type": "transcription", "text": user_text})
            await self.ws.send_json({"type": "thinking", "message": "Thinking..."})

            # 2. LLM Streaming
            self.cancel_event = threading.Event()
            full_response = ""
            
            sentence_queue = asyncio.Queue()
            tts_task = asyncio.create_task(self._tts_worker(sentence_queue, self.cancel_event))
            
            from sentence_splitter import SentenceSplitter
            splitter = SentenceSplitter(first_min=4, merge_min=16)
            
            llm_stream = knowledge_agent_client_stream(self.agent, user_text)
            
            logger.debug("[V2] Waiting for LLM stream...")
            for chunk in llm_stream:
                if asyncio.current_task().cancelled():
                    break
                    
                full_response += chunk
                
                # Send text chunk to frontend for typewriter effect
                await self.ws.send_json({"type": "response_chunk", "text": chunk})
                
                sentences = splitter.feed(chunk)
                for sentence_to_synth in sentences:
                    await sentence_queue.put(sentence_to_synth)

            # Flush any remaining text
            sentences = splitter.flush()
            for sentence_to_synth in sentences:
                await sentence_queue.put(sentence_to_synth)
                
            # Sentinel for TTS worker
            await sentence_queue.put(None)
            
            # Wait for all queued sentences to be synthesized and sent
            await tts_task

            await self.ws.send_json({"type": "audio_end"})
            logger.info("✅ Pipeline complete.")
            
        except asyncio.CancelledError:
            logger.info("🛑 Pipeline cancelled mid-execution.")
            # Cancel the TTS worker if the pipeline was cancelled
            if 'tts_task' in locals() and not tts_task.done():
                tts_task.cancel()
        except Exception as e:
            logger.error(f"Pipeline error: {e}")
            try:
                await self.ws.send_json({"type": "error", "message": str(e)})
            except:
                pass

    async def close(self):
        self.is_connected = False
        if self.receive_task: self.receive_task.cancel()
        if self.vad_task: self.vad_task.cancel()
        if self.pipeline_task: self.pipeline_task.cancel()
        try:
            await self.ws.close()
        except:
            pass
