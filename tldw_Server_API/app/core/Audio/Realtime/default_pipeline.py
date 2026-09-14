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
from tldw_Server_API.app.core.Audio.Realtime.constants import REALTIME_MAX_OUTPUT_CHUNK_BYTES
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
        provider_hint: str | None = None,
        chat_provider_hint: str | None = None,
        tts_provider_hint: str | None = None,
        user_id: int | None = None,
        tts_credential_scope: Callable[..., Any] | None = None,
    ) -> None:
        self._stt_transcribe_pcm16 = stt_transcribe_pcm16
        self._chat_call = chat_call
        self._tts_service_factory = tts_service_factory
        self._default_model = default_model
        self._default_voice = default_voice
        self._chat_provider_hint = chat_provider_hint if chat_provider_hint is not None else provider_hint
        self._tts_provider_hint = tts_provider_hint if tts_provider_hint is not None else provider_hint
        self._user_id = user_id
        self._tts_credential_scope = tts_credential_scope
        self._history: list[dict[str, str]] = []

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

        scope_stack = contextlib.AsyncExitStack()
        try:
            tts_session = await self._open_tts_session(config, scope_stack=scope_stack)
        except asyncio.CancelledError:
            await scope_stack.aclose()
            raise
        except Exception as exc:
            await scope_stack.aclose()
            raise RealtimePipelineError(stage="tts", message="Realtime TTS session failed", cause=exc) from exc

        events: asyncio.Queue[Any] = asyncio.Queue(maxsize=16)
        tts_finished = False
        stream_completed = False
        assistant_parts: list[str] = []

        async def produce_text() -> None:
            """Feed TTS while publishing text through the same bounded queue as audio."""
            nonlocal tts_finished
            chat_result = None
            try:
                chat_result = await self._chat_call(**self._chat_kwargs(transcript, config))
                async for delta in _iter_text_deltas(chat_result):
                    if not delta:
                        continue
                    assistant_parts.append(delta)
                    try:
                        await tts_session.push_text(delta)
                    except Exception as exc:
                        raise RealtimePipelineError(
                            stage="tts",
                            message="Realtime TTS text push failed",
                            cause=exc,
                        ) from exc
                    await events.put(RealtimePipelineTextDelta(delta))
                    await events.put(RealtimePipelineTranscriptDelta(delta))
                try:
                    await tts_session.commit()
                    await tts_session.finish()
                    tts_finished = True
                except Exception as exc:
                    raise RealtimePipelineError(stage="tts", message="Realtime TTS commit failed", cause=exc) from exc
                await events.put(RealtimePipelineTextDone())
                await events.put(RealtimePipelineTranscriptDone())
                await events.put(None)
            except RealtimePipelineError as exc:
                await events.put(exc)
            except Exception as exc:  # noqa: BLE001 - normalize arbitrary provider failures at the stream boundary
                await events.put(RealtimePipelineError(stage="llm", message="Realtime LLM streaming failed", cause=exc))
            finally:
                close = getattr(chat_result, "aclose", None)
                if callable(close):
                    await close()

        audio_task = asyncio.create_task(_drain_tts_audio(tts_session, events))
        text_task = asyncio.create_task(produce_text())
        try:
            finished_producers = 0
            while finished_producers < 2:
                event = await events.get()
                if event is None:
                    finished_producers += 1
                elif isinstance(event, BaseException):
                    raise event
                else:
                    yield event
            await text_task
            await audio_task
            self._history.extend(
                [
                    {"role": "user", "content": transcript},
                    {"role": "assistant", "content": "".join(assistant_parts)},
                ]
            )
            self._history = self._history[-20:]
            stream_completed = True
            yield RealtimePipelineTurnDone()
        finally:
            for task in (text_task, audio_task):
                if not task.done():
                    task.cancel()
            try:
                if not stream_completed:
                    await _cleanup_tts_session(tts_session, audio_task, tts_finished=tts_finished)
            finally:
                try:
                    await asyncio.gather(text_task, audio_task, return_exceptions=True)
                finally:
                    await scope_stack.aclose()

    def _chat_kwargs(self, transcript: str, config: RealtimeSessionConfig) -> dict[str, Any]:
        messages = [*self._history, {"role": "user", "content": transcript}]
        return {
            "api_endpoint": self._chat_provider_hint,
            "messages_payload": messages,
            "message": transcript,
            "system_message": config.instructions,
            "model": config.model or self._default_model,
            "api_model": config.model or self._default_model,
            "stream": True,
            "streaming": True,
            "user": str(self._user_id) if self._user_id is not None else None,
        }

    async def _open_tts_session(self, config: RealtimeSessionConfig, *, scope_stack: contextlib.AsyncExitStack) -> Any:
        service = self._tts_service_factory()
        if inspect.isawaitable(service):
            service = await service

        request = _build_speech_request(config, default_model=self._default_model, default_voice=self._default_voice)
        overrides = None
        if self._tts_credential_scope is not None:
            _user_id, overrides, _runtime, _credentials = await scope_stack.enter_async_context(
                self._tts_credential_scope(provider=self._tts_provider_hint, model=request.model)
            )
        open_realtime_session = getattr(service, "open_realtime_session", None)
        if callable(open_realtime_session):
            handle = await _call_open_realtime_session(
                open_realtime_session,
                request=request,
                provider_hint=self._tts_provider_hint,
                route=REALTIME_TTS_ROUTE,
                user_id=self._user_id,
                provider_overrides=overrides,
            )
            if self._tts_credential_scope is not None:
                from tldw_Server_API.app.core.Chat.bounded_daemon import await_owned_worker

                try:
                    await await_owned_worker(_runtime.mark_used(_credentials))
                except BaseException:  # noqa: BLE001 - the acquired provider session must close even on cancellation
                    await _close_tts_session_safely(handle.session)
                    raise
            return handle.session

        from tldw_Server_API.app.core.TTS.realtime_session import BufferedRealtimeSession

        return BufferedRealtimeSession(
            tts_service=service,
            config=_tts_config_from_request(request, self._tts_provider_hint),
            provider_hint=self._tts_provider_hint,
            provider_overrides=overrides,
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
        except Exception:  # noqa: BLE001 - try the next provider cleanup method if one is unavailable
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
    except Exception:  # noqa: BLE001 - best-effort cleanup for injected provider sessions
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


def build_default_realtime_pipeline(
    principal: Any | None = None,
    user_id: int | None = None,
    request: Any | None = None,
) -> DefaultRealtimePipeline:
    """Build the production realtime pipeline without importing providers at module import time."""

    from functools import partial
    from types import SimpleNamespace

    from tldw_Server_API.app.core.Audio.tts_service import tts_provider_credential_scope
    from tldw_Server_API.app.core.AuthNZ.byok_helpers import derive_trusted_credential_scope
    from tldw_Server_API.app.core.AuthNZ.llm_provider_overrides import capture_provider_override_call_snapshot
    from tldw_Server_API.app.core.AuthNZ.provider_credential_runtime import (
        PROVIDER_CALL_CREDENTIALS_CONTEXT_KEY,
        ProviderCredentialRuntime,
    )
    from tldw_Server_API.app.core.Chat.bounded_daemon import await_owned_worker
    from tldw_Server_API.app.core.Chat.chat_service import perform_chat_api_call_async
    from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.stt_policy import (
        apply_transcript_text_policy,
        resolve_effective_stt_policy,
    )
    from tldw_Server_API.app.core.TTS.tts_service_v2 import get_tts_service_v2
    from tldw_Server_API.app.core.Usage.audio_quota import consume_daily_minutes

    resolved_user_id = _resolve_user_id(principal, user_id)
    current_user = principal or SimpleNamespace(id=resolved_user_id)

    async def transcribe(audio: bytes, *, sample_rate_hz: int, language: str | None) -> str:
        """Enforce daily usage and redact before any transcript reaches the session or LLM."""
        allowed, _remaining = await consume_daily_minutes(
            resolved_user_id,
            len(audio) / (2 * sample_rate_hz * 60),
        )
        if not allowed:
            raise RealtimePipelineError(stage="stt", message="Realtime transcription quota exceeded")
        policy = await resolve_effective_stt_policy(principal=principal, user_id=resolved_user_id, db=None)
        text = await default_stt_transcribe_pcm16(audio, sample_rate_hz=sample_rate_hz, language=language)
        return apply_transcript_text_policy(text, policy=policy, is_partial=False)

    async def chat(**kwargs: Any) -> AsyncIterator[Any]:
        """Retain one authoritative provider credential snapshot until stream cleanup."""

        async def stream() -> AsyncIterator[Any]:
            runtime_user, teams, orgs, trusted_url = derive_trusted_credential_scope(request, current_user)
            runtime = ProviderCredentialRuntime(
                user_id=runtime_user,
                team_ids=teams,
                org_ids=orgs,
                trusted_base_url_override=trusted_url,
                override_snapshot_resolver=capture_provider_override_call_snapshot,
            )
            result = None
            try:
                credentials = await await_owned_worker(runtime.resolve(kwargs["api_endpoint"], model=kwargs["model"]))
                kwargs[PROVIDER_CALL_CREDENTIALS_CONTEXT_KEY] = credentials
                result = await perform_chat_api_call_async(**kwargs)
                await await_owned_worker(runtime.mark_used(credentials))
                if hasattr(result, "__aiter__"):
                    async for chunk in result:
                        yield chunk
                else:
                    yield result
            finally:
                try:
                    close = getattr(result, "aclose", None)
                    if callable(close):
                        await close()
                finally:
                    await await_owned_worker(runtime.close())

        return stream()

    return DefaultRealtimePipeline(
        stt_transcribe_pcm16=transcribe,
        chat_call=chat,
        tts_service_factory=lambda: get_tts_service_v2(),
        default_model=_default_realtime_model(),
        default_voice=_default_realtime_voice(),
        chat_provider_hint=_default_chat_provider_hint(),
        tts_provider_hint=_default_tts_provider_hint(),
        user_id=resolved_user_id,
        tts_credential_scope=partial(tts_provider_credential_scope, request=request, current_user=current_user),
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
    provider_overrides: dict[str, Any] | None = None,
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
                    "provider_overrides": provider_overrides,
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
                    "provider_overrides": provider_overrides,
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
    """Publish audio with backpressure and report failures before completion."""
    try:
        async for chunk in session.audio_stream():
            if chunk:
                for audio_chunk in _split_output_audio_chunk(bytes(chunk)):
                    await queue.put(RealtimePipelineAudioDelta(audio_chunk))
        error = getattr(session, "error", None)
        if error is not None:
            raise error
        await queue.put(RealtimePipelineAudioDone())
        await queue.put(None)
    except Exception as exc:  # noqa: BLE001 - provider errors are returned through the bounded stream
        await queue.put(RealtimePipelineError(stage="tts", message="Realtime TTS audio failed", cause=exc))


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


def _default_chat_provider_hint() -> str | None:
    return os.getenv("REALTIME_CHAT_PROVIDER_HINT") or _default_provider_hint()


def _default_tts_provider_hint() -> str | None:
    return os.getenv("REALTIME_TTS_PROVIDER_HINT") or _default_provider_hint()


def _split_output_audio_chunk(chunk: bytes) -> list[bytes]:
    if len(chunk) <= REALTIME_MAX_OUTPUT_CHUNK_BYTES:
        return [chunk]
    return [
        chunk[index : index + REALTIME_MAX_OUTPUT_CHUNK_BYTES]
        for index in range(0, len(chunk), REALTIME_MAX_OUTPUT_CHUNK_BYTES)
    ]
