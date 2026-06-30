# Audio_Streaming_Parakeet.py
#########################################
# Real-time Streaming Transcription with Parakeet Models
# This module provides WebSocket-based real-time transcription using Parakeet models
# with support for all variants (standard, ONNX, MLX) and chunked processing.
#
####################
# Function List
#
# 1. ParakeetStreamingTranscriber - Main class for streaming transcription
# 2. create_websocket_server() - Create WebSocket server for real-time transcription
# 3. process_audio_stream() - Process streaming audio chunks
#
####################

import asyncio
import base64
import json
import sys
import time
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np
from loguru import logger


def is_transcription_error_message(text: str) -> bool:
    """Lazily resolve STT sentinel detection to avoid heavy imports at module load."""
    lib_mod = sys.modules.get(
        "tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Lib"
    )
    if lib_mod is None:
        return False
    _impl = getattr(lib_mod, "is_transcription_error_message", None)
    if not callable(_impl):
        return False
    try:
        return bool(_impl(text))
    except Exception:
        return False

# Import transcription functions
from .Audio_Transcription_Nemo import (
    transcribe_with_parakeet,
)
from .Audio_Transcription_Parakeet_MLX import (
    transcribe_with_parakeet_mlx,
)

logger = logger


@dataclass
class StreamingConfig:
    """Configuration for streaming transcription."""
    model_variant: str = 'standard'  # 'standard', 'onnx', 'mlx'
    sample_rate: int = 16000
    chunk_duration: float = 2.0  # Seconds of audio to accumulate before transcribing
    overlap_duration: float = 0.5  # Overlap between chunks
    max_buffer_duration: float = 30.0  # Maximum buffer size in seconds
    enable_partial: bool = True  # Send partial results
    partial_interval: float = 0.5  # How often to send partials
    language: Optional[str] = None


@dataclass
class AudioBuffer:
    """Manages audio buffering for streaming."""
    sample_rate: int
    max_duration: float
    data: list = field(default_factory=list)

    def add(self, audio_chunk: np.ndarray):
        """Add audio chunk to buffer."""
        self.data.append(audio_chunk)

        # Trim if buffer exceeds max duration by keeping the most recent portion
        total_samples = sum(len(chunk) for chunk in self.data)
        max_samples = int(self.sample_rate * self.max_duration)

        if total_samples > max_samples:
            combined = np.concatenate(self.data)
            self.data = [combined[-max_samples:]]

    def get_duration(self) -> float:
        """Get current buffer duration in seconds."""
        if not self.data:
            return 0.0
        total_samples = sum(len(chunk) for chunk in self.data)
        return total_samples / self.sample_rate

    def get_audio(self, duration: Optional[float] = None) -> Optional[np.ndarray]:
        """Get audio from buffer."""
        if not self.data:
            return None

        combined = np.concatenate(self.data)

        if duration is not None:
            samples_needed = int(duration * self.sample_rate)
            if len(combined) >= samples_needed:
                return combined[:samples_needed]
            return None

        return combined

    def consume(self, duration: float, overlap: float = 0.0):
        """Consume duration from buffer, keeping overlap."""
        if not self.data:
            return

        combined = np.concatenate(self.data)
        samples_to_consume = int((duration - overlap) * self.sample_rate)

        if samples_to_consume > 0 and len(combined) > samples_to_consume:
            # Keep the remaining audio
            remaining = combined[samples_to_consume:]
            self.data = [remaining]
        else:
            self.data.clear()

    def clear(self):
        """Clear the buffer."""
        self.data.clear()


class ParakeetStreamingTranscriber:
    """
    Real-time streaming transcriber using Parakeet models.

    Supports WebSocket-based streaming with chunked processing
    and partial results for all Parakeet variants.
    """

    def __init__(self, config: Optional[StreamingConfig] = None):
        """Initialize the streaming transcriber."""
        self.config = config or StreamingConfig()
        self.buffer = AudioBuffer(
            sample_rate=self.config.sample_rate,
            max_duration=self.config.max_buffer_duration
        )
        self.model = None
        self.is_running = False
        self.last_partial_time = 0
        self.transcription_history = []

    def initialize(self):
        """Load the model based on configuration.
        Returns an awaitable so callers may `await initialize()` or call synchronously.
        """
        if self.config.model_variant == 'mlx':
            # MLX model is loaded on-demand in transcribe function
            logger.info("Using Parakeet MLX variant")
        else:
            # Load standard or ONNX variant (import inside to respect test patches)
            from .Audio_Transcription_Nemo import load_parakeet_model as _load_parakeet_model
            self.model = _load_parakeet_model(self.config.model_variant)
            if self.model is None:
                logger.warning(f"Could not load Parakeet {self.config.model_variant} model; proceeding without preloaded model")
            else:
                logger.info(f"Loaded Parakeet {self.config.model_variant} model")

        async def _noop():
            return None

        return _noop()

    async def process_audio_chunk(self, audio_data: bytes) -> Optional[dict[str, Any]]:
        """
        Process a chunk of audio data.

        Args:
            audio_data: Base64-encoded audio data

        Returns:
            Transcription result or None if not ready
        """
        try:
            # Accept both base64-encoded strings and raw float32 bytes
            if isinstance(audio_data, (str, bytes)):
                if isinstance(audio_data, str):
                    audio_bytes = base64.b64decode(audio_data)
                else:
                    # Heuristic: if bytes look like base64 text (ASCII), decode; otherwise treat as raw
                    try:
                        ascii_sample = audio_data[:16]
                        if all(32 <= b <= 126 or b in (10, 13) for b in ascii_sample):
                            audio_bytes = base64.b64decode(audio_data)
                        else:
                            audio_bytes = audio_data
                    except Exception:
                        audio_bytes = audio_data
            else:
                # Fallback: attempt to coerce to bytes via base64
                audio_bytes = base64.b64decode(audio_data)

            # Convert to numpy array (assuming float32 samples)
            # Ensure buffer aligns to float32 size
            remainder = len(audio_bytes) % 4
            if remainder:
                audio_bytes = audio_bytes[:-remainder]
            audio_array = np.frombuffer(audio_bytes, dtype=np.float32)

            # Add to buffer
            self.buffer.add(audio_array)

            result = None

            # Check if we have enough audio for transcription
            if self.buffer.get_duration() >= self.config.chunk_duration:
                # Get audio chunk
                audio_chunk = self.buffer.get_audio(self.config.chunk_duration)

                if audio_chunk is not None:
                    # Transcribe
                    text = await self._transcribe_chunk(audio_chunk)

                    if isinstance(text, str) and is_transcription_error_message(text):
                        logger.error(f"ParakeetStreamingTranscriber STT error sentinel: {text}")
                    elif text:
                        # Consume buffer with overlap
                        self.buffer.consume(
                            self.config.chunk_duration,
                            self.config.overlap_duration
                        )

                        # Add to history
                        self.transcription_history.append(text)

                        result = {
                            "type": "transcription",
                            "text": text,
                            "timestamp": time.time(),
                            "is_final": True
                        }

            # Send partial if enabled and enough time has passed
            elif self.config.enable_partial:
                now = time.time()
                if now - self.last_partial_time >= self.config.partial_interval:
                    partial_audio = self.buffer.get_audio()
                    if partial_audio is not None and len(partial_audio) > 0:
                        partial_text = await self._transcribe_chunk(partial_audio)
                        if isinstance(partial_text, str) and is_transcription_error_message(partial_text):
                            logger.error(f"ParakeetStreamingTranscriber STT error sentinel (partial): {partial_text}")
                        elif partial_text:
                            self.last_partial_time = now
                            result = {
                                "type": "partial",
                                "text": partial_text,
                                "timestamp": now,
                                "is_final": False
                            }

            return result

        except Exception as e:
            logger.exception(f"Error processing audio chunk: {e}")
            return {"type": "error", "message": "Audio chunk processing failed"}

    async def _transcribe_chunk(self, audio_chunk: np.ndarray) -> Optional[str]:
        """Transcribe an audio chunk using the appropriate variant.
        If a model has been injected (e.g., tests), prefer it regardless of variant.
        """
        try:
            # Prefer injected model for testability
            if self.model is not None and hasattr(self.model, 'transcribe'):
                result = await asyncio.to_thread(self.model.transcribe, audio_chunk)
                if isinstance(result, list):
                    result = result[0] if result else ""
                if hasattr(result, 'text'):
                    return result.text
                return str(result) if result is not None else None

            if self.config.model_variant == 'mlx':
                result = await asyncio.to_thread(
                    transcribe_with_parakeet_mlx,
                    audio_chunk,
                    sample_rate=self.config.sample_rate,
                    language=self.config.language
                )
                return result
            else:
                result = await asyncio.to_thread(
                    transcribe_with_parakeet,
                    audio_chunk,
                    sample_rate=self.config.sample_rate,
                    variant=self.config.model_variant
                )
                return result
        except Exception as e:
            logger.exception(f"Transcription error: {e}")
            return None

    async def flush(self) -> Optional[dict[str, Any]]:
        """Process any remaining audio in the buffer."""
        if self.buffer.get_duration() > 0:
            audio = self.buffer.get_audio()
            if audio is not None:
                text = await self._transcribe_chunk(audio)
                self.buffer.clear()

                if isinstance(text, str) and is_transcription_error_message(text):
                    logger.error(f"ParakeetStreamingTranscriber STT error sentinel on flush: {text}")
                elif text:
                    self.transcription_history.append(text)
                    return {
                        "type": "final",
                        "text": text,
                        "timestamp": time.time(),
                        "is_final": True
                    }
        return None

    def get_full_transcript(self) -> str:
        """Get the complete transcript so far."""
        return " ".join(self.transcription_history)

    def reset(self):
        """Reset the transcriber state."""
        self.buffer.clear()
        self.transcription_history.clear()
        self.last_partial_time = 0


async def handle_websocket_transcription(
    websocket,
    config: Optional[StreamingConfig] = None
):
    """
    Compatibility wrapper that routes legacy Parakeet websocket traffic through the unified streaming handler.

    Args:
        websocket: WebSocket connection (legacy interface with recv/send is supported)
        config: Streaming configuration (mapped to UnifiedStreamingConfig with model='parakeet')
    """
    try:
        # Defer imports to avoid circular dependencies during module import
        from .Audio_Streaming_Unified import UnifiedStreamingConfig, handle_unified_websocket
    except Exception as import_err:
        logger.exception(f"Failed to import unified streaming handler: {import_err}")
        raise

    # Map legacy StreamingConfig to UnifiedStreamingConfig
    base_cfg = config or StreamingConfig()
    unified_cfg = UnifiedStreamingConfig(
        model='parakeet',
        model_variant=base_cfg.model_variant,
        sample_rate=base_cfg.sample_rate,
        chunk_duration=base_cfg.chunk_duration,
        overlap_duration=base_cfg.overlap_duration,
        max_buffer_duration=base_cfg.max_buffer_duration,
        enable_partial=base_cfg.enable_partial,
        partial_interval=base_cfg.partial_interval,
        language=getattr(base_cfg, "language", None)
    )

    class _LegacyWebSocketAdapter:
        """Adapts a legacy recv/send websocket to the unified receive_text/send_json API."""

        def __init__(self, ws):
            self._ws = ws
            # Preserve common attributes for downstream checks
            self.closed = getattr(ws, "closed", False)

        async def receive_text(self):
            raw = await self._ws.recv()
            try:
                payload = json.loads(raw)
            except Exception:
                return raw

            # Translate legacy "start" message into unified "config"
            if payload.get("type") == "start":
                cfg = payload.get("config") or {}
                payload = {
                    "type": "config",
                    **cfg
                }
            return json.dumps(payload)

        async def send_json(self, payload):
            if hasattr(self._ws, "send_json"):
                return await self._ws.send_json(payload)
            return await self._ws.send(json.dumps(payload))

        async def close(self, code: int | None = None, reason: str | None = None):
            if hasattr(self._ws, "close"):
                try:
                    return await self._ws.close(code, reason)
                except TypeError:
                    # websockets' close signature differs
                    return await self._ws.close()

    # Wrap legacy websocket interface when needed
    if not hasattr(websocket, "receive_text") and hasattr(websocket, "recv"):
        websocket = _LegacyWebSocketAdapter(websocket)

    await handle_unified_websocket(websocket, config=unified_cfg)


def create_streaming_generator(
    audio_source: Callable[[], bytes],
    config: Optional[StreamingConfig] = None
) -> AsyncGenerator[str, None]:
    """
    Create an async generator for streaming transcription.

    Args:
        audio_source: Callable that returns audio bytes
        config: Streaming configuration

    Yields:
        Transcribed text segments
    """
    transcriber = ParakeetStreamingTranscriber(config)
    transcriber.initialize()

    async def generate():
        try:
            while True:
                # Get audio from source
                audio_bytes = await asyncio.to_thread(audio_source)

                if audio_bytes is None:
                    # End of stream
                    final = await transcriber.flush()
                    if final:
                        yield final["text"]
                    break

                # Convert to base64 for processing
                audio_b64 = base64.b64encode(audio_bytes).decode('utf-8')

                # Process chunk
                result = await transcriber.process_audio_chunk(audio_b64)

                if result and result.get("is_final"):
                    yield result["text"]

        except Exception as e:
            logger.exception(f"Streaming error: {e}")
            yield "[Error: Streaming transcription failed]"

    return generate()


#######################################################################################################################
# End of Audio_Streaming_Parakeet.py
#######################################################################################################################
