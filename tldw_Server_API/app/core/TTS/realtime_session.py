# realtime_session.py
# Description: Realtime TTS session primitives and fallback session implementation.
#
from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from typing import Any

from loguru import logger

from tldw_Server_API.app.api.v1.schemas.audio_schemas import OpenAISpeechRequest


@dataclass
class RealtimeSessionConfig:
    """Configuration for realtime TTS sessions."""
    model: str
    voice: str
    response_format: str
    speed: float = 1.0
    lang_code: str | None = None
    extra_params: dict[str, Any] | None = None
    provider: str | None = None


@dataclass
class RealtimeSessionHandle:
    """Handle returned by TTSServiceV2.open_realtime_session."""
    session: RealtimeTTSSession
    provider: str | None = None
    warning: str | None = None


class RealtimeTTSSession:
    """Base interface for realtime TTS sessions."""

    async def push_text(self, delta: str) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    async def commit(self) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    async def finish(self) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    async def audio_stream(self) -> AsyncGenerator[bytes, None]:  # pragma: no cover - interface
        raise NotImplementedError

    @property
    def error(self) -> Exception | None:
        return None


class BufferedRealtimeSession(RealtimeTTSSession):
    """Fallback realtime session that buffers text and uses generate_speech on commit."""

    def __init__(
        self,
        *,
        tts_service: Any,
        config: RealtimeSessionConfig,
        provider_hint: str | None = None,
        route: str = "audio.stream.tts.realtime",
        user_id: int | None = None,
    ) -> None:
        self._tts_service = tts_service
        self._config = config
        self._provider_hint = provider_hint
        self._route = route
        self._user_id = user_id
        self._buffer = ""
        self._text_queue: asyncio.Queue[str | None] = asyncio.Queue()
        self._audio_queue: asyncio.Queue[bytes | None] = asyncio.Queue()
        self._closed = False
        self._error: Exception | None = None
        self._worker_task = asyncio.create_task(self._worker())

    @property
    def error(self) -> Exception | None:
        return self._error

    async def push_text(self, delta: str) -> None:
        if self._closed:
            return
        if not isinstance(delta, str):
            return
        self._buffer += delta

    async def commit(self) -> None:
        if self._closed:
            return
        if self._buffer.strip():
            await self._text_queue.put(self._buffer)
            self._buffer = ""

    async def finish(self) -> None:
        if self._closed:
            return
        if self._buffer.strip():
            await self._text_queue.put(self._buffer)
            self._buffer = ""
        await self._text_queue.put(None)
        self._closed = True

    async def audio_stream(self) -> AsyncGenerator[bytes, None]:
        while True:
            chunk = await self._audio_queue.get()
            if chunk is None:
                break
            if chunk:
                yield chunk

    async def _worker(self) -> None:
        try:
            while True:
                text = await self._text_queue.get()
                if text is None:
                    break
                if not text.strip():
                    continue
                request = OpenAISpeechRequest(
                    model=self._config.model,
                    input=text,
                    voice=self._config.voice,
                    response_format=self._config.response_format,
                    speed=self._config.speed,
                    stream=True,
                    lang_code=self._config.lang_code,
                    extra_params=self._config.extra_params,
                )
                async for chunk in self._tts_service.generate_speech(
                    request,
                    provider=self._provider_hint,
                    fallback=True,
                    voice_to_voice_route=self._route,
                    user_id=self._user_id,
                ):
                    if chunk:
                        await self._audio_queue.put(chunk)
        except Exception as exc:
            self._error = exc
            logger.error(
                "Buffered realtime TTS session failed ({})",
                type(exc).__name__,
            )
        finally:
            await self._audio_queue.put(None)
