"""Default realtime STT -> LLM -> TTS pipeline adapter."""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import os
import tempfile
import wave
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any, Literal

from tldw_Server_API.app.api.v1.schemas.audio_schemas import OpenAISpeechRequest
from tldw_Server_API.app.core.Audio.Realtime.models import RealtimeSessionConfig
from tldw_Server_API.app.core.Audio.Realtime.pipeline import (
    RealtimePipelineAudioDelta,
    RealtimePipelineAudioDone,
    RealtimePipelineEvent,
    RealtimePipelineTextDelta,
    RealtimePipelineTextDone,
    RealtimePipelineTranscriptDelta,
    RealtimePipelineTranscriptDone,
    RealtimePipelineTurnDone,
)

PipelineStage = Literal["stt", "llm", "tts"]

REALTIME_TTS_ROUTE = "audio.stream.tts.realtime"
DEFAULT_REALTIME_OUTPUT_SAMPLE_RATE_HZ = 24000


class RealtimePipelineError(RuntimeError):
    """Raised when one stage of the realtime pipeline fails."""

    def __init__(self, *, stage: PipelineStage, message: str, cause: BaseException | None = None) -> None:
        super().__init__(message)
        self.stage = stage
        self.__cause__ = cause


class DefaultRealtimePipeline:
    """Adapter from realtime turns to existing STT, chat, and TTS services."""

    def __init__(
        self,
        *,
        stt_transcribe_pcm16: Callable[..., Awaitable[str]],
        chat_call: Callable[..., Awaitable[Any]],
        tts_service_factory: Callable[[], Any],
        default_model: str,
        default_voice: str,
        provider_hint: str | None,
        user_id: int | None,
    ) -> None:
        self._stt_transcribe_pcm16 = stt_transcribe_pcm16
        self._chat_call = chat_call
        self._tts_service_factory = tts_service_factory
        self._default_model = default_model
        self._default_voice = default_voice
        self._provider_hint = provider_hint
        self._user_id = user_id

    async def transcribe_pcm16(self, audio: bytes, *, sample_rate_hz: int, language: str | None) -> str:
        """Transcribe one committed PCM16 audio turn."""

        try:
            transcript = await self._stt_transcribe_pcm16(
                audio,
                sample_rate_hz=sample_rate_hz,
                language=language,
            )
        except Exception as exc:
            raise RealtimePipelineError(stage="stt", message="Realtime transcription failed", cause=exc) from exc
        return transcript

    async def stream_turn(
        self,
        transcript: str,
        *,
        config: RealtimeSessionConfig,
    ) -> AsyncIterator[RealtimePipelineEvent]:
        """Stream one assistant response turn as text, spoken transcript, and PCM audio."""

        try:
            tts_session = await self._open_tts_session(config)
        except Exception as exc:
            raise RealtimePipelineError(stage="tts", message="Realtime TTS session failed", cause=exc) from exc

        audio_events: asyncio.Queue[RealtimePipelineAudioDelta | RealtimePipelineAudioDone | BaseException] = (
            asyncio.Queue()
        )
        audio_task = asyncio.create_task(_drain_tts_audio(tts_session, audio_events))
        tts_finished = False
        stream_completed = False

        try:
            try:
                chat_result = await self._chat_call(**self._chat_kwargs(transcript, config))
                async for delta in _iter_text_deltas(chat_result):
                    if not delta:
                        continue
                    try:
                        await tts_session.push_text(delta)
                    except Exception as exc:
                        raise RealtimePipelineError(
                            stage="tts",
                            message="Realtime TTS text push failed",
                            cause=exc,
                        ) from exc
                    yield RealtimePipelineTextDelta(delta)
                    yield RealtimePipelineTranscriptDelta(delta)
            except RealtimePipelineError:
                raise
            except Exception as exc:
                raise RealtimePipelineError(stage="llm", message="Realtime LLM streaming failed", cause=exc) from exc

            try:
                await tts_session.commit()
                await tts_session.finish()
                tts_finished = True
            except Exception as exc:
                raise RealtimePipelineError(stage="tts", message="Realtime TTS commit failed", cause=exc) from exc

            yield RealtimePipelineTextDone()
            yield RealtimePipelineTranscriptDone()

            async for audio_event in _drain_remaining_audio_events(audio_events, audio_task):
                yield audio_event

            error = getattr(tts_session, "error", None)
            if error is not None:
                raise RealtimePipelineError(stage="tts", message="Realtime TTS audio failed", cause=error) from error
            stream_completed = True
            yield RealtimePipelineTurnDone()
        finally:
            if not stream_completed:
                await _cleanup_tts_session(tts_session, audio_task, tts_finished=tts_finished)

    def _chat_kwargs(self, transcript: str, config: RealtimeSessionConfig) -> dict[str, Any]:
        messages = [{"role": "user", "content": transcript}]
        return {
            "api_endpoint": self._provider_hint,
            "messages_payload": messages,
            "message": transcript,
            "system_message": config.instructions,
            "model": config.model or self._default_model,
            "api_model": config.model or self._default_model,
            "stream": True,
            "streaming": True,
            "user": str(self._user_id) if self._user_id is not None else None,
        }

    async def _open_tts_session(self, config: RealtimeSessionConfig) -> Any:
        service = self._tts_service_factory()
        if inspect.isawaitable(service):
            service = await service

        request = _build_speech_request(config, default_model=self._default_model, default_voice=self._default_voice)
        open_realtime_session = getattr(service, "open_realtime_session", None)
        if callable(open_realtime_session):
            handle = await _call_open_realtime_session(
                open_realtime_session,
                request=request,
                provider_hint=self._provider_hint,
                route=REALTIME_TTS_ROUTE,
                user_id=self._user_id,
            )
            return handle.session

        from tldw_Server_API.app.core.TTS.realtime_session import BufferedRealtimeSession

        return BufferedRealtimeSession(
            tts_service=service,
            config=_tts_config_from_request(request, self._provider_hint),
            provider_hint=self._provider_hint,
            route=REALTIME_TTS_ROUTE,
            user_id=self._user_id,
        )


async def _cleanup_tts_session(session: Any, audio_task: asyncio.Task[None], *, tts_finished: bool) -> None:
    try:
        if not tts_finished:
            await _close_tts_session_safely(session)
    finally:
        if not audio_task.done():
            audio_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await audio_task


async def _close_tts_session_safely(session: Any) -> None:
    for method_name in ("close", "aclose", "abort", "cancel"):
        method = getattr(session, method_name, None)
        if not callable(method):
            continue
        cleanup_failed = False
        try:
            maybe_result = method()
            if inspect.isawaitable(maybe_result):
                await maybe_result
        except Exception:
            cleanup_failed = True
        if not cleanup_failed:
            return
    await _finish_tts_session_safely(session)


async def _finish_tts_session_safely(session: Any) -> None:
    finish = getattr(session, "finish", None)
    if not callable(finish):
        return
    try:
        maybe_result = finish()
        if inspect.isawaitable(maybe_result):
            await maybe_result
    except Exception:
        return


async def default_stt_transcribe_pcm16(audio: bytes, *, sample_rate_hz: int, language: str | None) -> str:
    """Transcribe raw mono PCM16 audio through the configured STT adapter."""

    temp_path = _write_pcm16_wav(audio, sample_rate_hz=sample_rate_hz)
    try:
        from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.stt_provider_adapter import (
            get_stt_provider_registry,
            resolve_default_transcription_model,
        )

        registry = get_stt_provider_registry()
        model_name = resolve_default_transcription_model("whisper-1")
        provider, model, _variant = registry.resolve_provider_for_model(model_name)
        adapter = registry.get_adapter(provider)
        artifact = await asyncio.to_thread(
            adapter.transcribe_batch,
            temp_path,
            model=model,
            language=language,
        )
    finally:
        with contextlib.suppress(OSError):
            os.unlink(temp_path)

    if isinstance(artifact, dict):
        text = artifact.get("text")
        return text if isinstance(text, str) else ""
    return ""


def build_default_realtime_pipeline(principal: Any | None = None, user_id: int | None = None) -> DefaultRealtimePipeline:
    """Build the production realtime pipeline without importing providers at module import time."""

    from tldw_Server_API.app.core.Chat.chat_service import perform_chat_api_call_async
    from tldw_Server_API.app.core.TTS.tts_service_v2 import get_tts_service_v2

    resolved_user_id = _resolve_user_id(principal, user_id)
    return DefaultRealtimePipeline(
        stt_transcribe_pcm16=default_stt_transcribe_pcm16,
        chat_call=perform_chat_api_call_async,
        tts_service_factory=lambda: get_tts_service_v2(),
        default_model=_default_realtime_model(),
        default_voice=_default_realtime_voice(),
        provider_hint=_default_provider_hint(),
        user_id=resolved_user_id,
    )


def _build_speech_request(
    config: RealtimeSessionConfig,
    *,
    default_model: str,
    default_voice: str,
) -> OpenAISpeechRequest:
    tts_model = config.metadata.get("tts_model") if isinstance(config.metadata, dict) else None
    output_sample_rate = config.output_sample_rate_hz or DEFAULT_REALTIME_OUTPUT_SAMPLE_RATE_HZ
    return OpenAISpeechRequest(
        model=str(tts_model or _default_tts_model(default_model)),
        input="",
        voice=config.voice or default_voice,
        response_format="pcm",
        stream=True,
        target_sample_rate=output_sample_rate,
    )


async def _call_open_realtime_session(
    open_realtime_session: Callable[..., Any],
    *,
    request: OpenAISpeechRequest,
    provider_hint: str | None,
    route: str,
    user_id: int | None,
) -> Any:
    signature = inspect.signature(open_realtime_session)
    if "request" in signature.parameters:
        maybe_handle = open_realtime_session(
            **_filter_callable_kwargs(
                signature,
                {
                    "request": request,
                    "provider_hint": provider_hint,
                    "route": route,
                    "user_id": user_id,
                },
            )
        )
    else:
        maybe_handle = open_realtime_session(
            **_filter_callable_kwargs(
                signature,
                {
                    "config": _tts_config_from_request(request, provider_hint),
                    "provider_hint": provider_hint,
                    "route": route,
                    "user_id": user_id,
                },
            )
        )
    return await maybe_handle if inspect.isawaitable(maybe_handle) else maybe_handle


def _filter_callable_kwargs(signature: inspect.Signature, kwargs: dict[str, Any]) -> dict[str, Any]:
    if any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in signature.parameters.values()):
        return kwargs
    return {key: value for key, value in kwargs.items() if key in signature.parameters}


def _tts_config_from_request(request: OpenAISpeechRequest, provider_hint: str | None) -> Any:
    from tldw_Server_API.app.core.TTS.realtime_session import RealtimeSessionConfig as TTSRealtimeSessionConfig

    return TTSRealtimeSessionConfig(
        model=request.model,
        voice=request.voice,
        response_format=request.response_format,
        speed=request.speed,
        lang_code=request.lang_code,
        extra_params={"target_sample_rate": request.target_sample_rate},
        provider=provider_hint,
    )


async def _drain_tts_audio(session: Any, queue: asyncio.Queue[Any]) -> None:
    try:
        async for chunk in session.audio_stream():
            if chunk:
                await queue.put(RealtimePipelineAudioDelta(bytes(chunk)))
        await queue.put(RealtimePipelineAudioDone())
    except Exception as exc:
        await queue.put(exc)


async def _drain_remaining_audio_events(
    queue: asyncio.Queue[RealtimePipelineAudioDelta | RealtimePipelineAudioDone | BaseException],
    audio_task: asyncio.Task[None],
) -> AsyncIterator[RealtimePipelineAudioDelta | RealtimePipelineAudioDone]:
    while True:
        event = await queue.get()
        if isinstance(event, BaseException):
            raise RealtimePipelineError(stage="tts", message="Realtime TTS audio failed", cause=event) from event
        yield event
        if isinstance(event, RealtimePipelineAudioDone):
            break
    await audio_task


async def _iter_text_deltas(result: Any) -> AsyncIterator[str]:
    if hasattr(result, "__aiter__"):
        async for chunk in result:
            delta = _extract_text_delta(chunk, streaming=True)
            if delta:
                yield delta
        return

    delta = _extract_text_delta(result, streaming=False)
    if delta:
        yield delta


def _extract_text_delta(chunk: Any, *, streaming: bool) -> str:
    if chunk is None:
        return ""
    if isinstance(chunk, bytes | bytearray):
        chunk = chunk.decode("utf-8", errors="replace")
    if isinstance(chunk, str):
        return _extract_text_from_string_chunk(chunk)
    if isinstance(chunk, dict):
        return _extract_text_from_mapping(chunk, streaming=streaming)
    return _extract_text_from_object(chunk, streaming=streaming)


def _extract_text_from_string_chunk(chunk: str) -> str:
    raw = chunk.strip("\r\n")
    if not raw or raw.strip().lower() == "data: [done]":
        return ""
    if raw.startswith("data:"):
        import json

        payload = raw[5:].strip()
        try:
            parsed = json.loads(payload)
        except json.JSONDecodeError:
            return ""
        return _extract_text_from_mapping(parsed, streaming=True)
    return chunk


def _extract_text_from_mapping(chunk: dict[str, Any], *, streaming: bool) -> str:
    choices = chunk.get("choices")
    if isinstance(choices, list) and choices:
        first = choices[0]
        if isinstance(first, dict):
            if streaming:
                delta = first.get("delta")
                if isinstance(delta, dict) and isinstance(delta.get("content"), str):
                    return delta["content"]
            message = first.get("message")
            if isinstance(message, dict) and isinstance(message.get("content"), str):
                return message["content"]
            text = first.get("text")
            if isinstance(text, str):
                return text
    for key in ("content", "text", "response"):
        value = chunk.get(key)
        if isinstance(value, str):
            return value
    return ""


def _extract_text_from_object(chunk: Any, *, streaming: bool) -> str:
    choices = getattr(chunk, "choices", None)
    if isinstance(choices, list) and choices:
        first = choices[0]
        if streaming:
            delta = getattr(first, "delta", None)
            content = getattr(delta, "content", None)
            if isinstance(content, str):
                return content
        message = getattr(first, "message", None)
        content = getattr(message, "content", None)
        if isinstance(content, str):
            return content
        text = getattr(first, "text", None)
        if isinstance(text, str):
            return text
    return ""


def _write_pcm16_wav(audio: bytes, *, sample_rate_hz: int) -> str:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
        temp_path = temp_file.name
    try:
        with wave.open(temp_path, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate_hz)
            wav_file.writeframes(audio)
    except Exception:
        with contextlib.suppress(OSError):
            os.unlink(temp_path)
        raise
    return temp_path


def _resolve_user_id(principal: Any | None, explicit_user_id: int | None) -> int | None:
    if explicit_user_id is not None:
        return explicit_user_id
    for attr in ("id_int", "user_id", "id"):
        value = getattr(principal, attr, None)
        if value is None:
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return None


def _default_realtime_model() -> str:
    return os.getenv("REALTIME_CHAT_MODEL", "gpt-4o-mini")


def _default_realtime_voice() -> str:
    return os.getenv("REALTIME_TTS_VOICE", "alloy")


def _default_tts_model(_chat_model: str) -> str:
    return os.getenv("REALTIME_TTS_MODEL", "tts-1")


def _default_provider_hint() -> str | None:
    return os.getenv("REALTIME_PROVIDER_HINT", "openai")
