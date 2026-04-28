"""
speech_chat_service.py

Non-streaming Speech-to-Speech chat orchestration (STT → LLM → TTS).

This module wires together:
  - STT: Audio_Transcription_Lib.transcribe_audio
  - LLM: adapter registry (async) with legacy fallback
  - TTS: TTSServiceV2.generate_speech
  - Session storage: ChaChaNotes conversations/messages

It is intentionally lean and avoids advanced features (queues, moderation,
tool calling) to keep the v1 speech chat path simple and testable.
"""
from __future__ import annotations

import asyncio
import base64
import io
import json
import os
import time
import uuid
from typing import Any

import numpy as np
import soundfile as sf
from fastapi import HTTPException, status
from loguru import logger

from tldw_Server_API.app.api.v1.schemas.audio_schemas import (
    OpenAISpeechRequest,
    SpeechChatRequest,
    SpeechChatResponse,
    SpeechChatTiming,
    SpeechChatTokenUsage,
)
from tldw_Server_API.app.api.v1.schemas.chat_request_schemas import (
    DEFAULT_LLM_PROVIDER,
    get_api_keys,
)
from tldw_Server_API.app.core.AuthNZ.byok_runtime import (
    record_byok_missing_credentials,
    resolve_byok_credentials,
)
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User
from tldw_Server_API.app.core.Chat.chat_helpers import (
    extract_response_content,
    get_or_create_character_context,
    get_or_create_conversation,
    load_conversation_history,
)
from tldw_Server_API.app.core.Chat.chat_service import (
    perform_chat_api_call_async as chat_api_call_async,
)
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Lib import (
    is_transcription_error_message,
    transcribe_audio,
)
from tldw_Server_API.app.core.LLM_Calls.adapter_registry import get_registry
from tldw_Server_API.app.core.LLM_Calls.adapter_utils import (
    ensure_app_config,
    normalize_provider,
    resolve_provider_api_key_from_config,
    resolve_provider_model,
    split_system_message,
)
from tldw_Server_API.app.core.LLM_Calls.provider_metadata import provider_requires_api_key
from tldw_Server_API.app.core.MCP_unified.modules.registry import get_module_registry
from tldw_Server_API.app.core.MCP_unified.protocol import RequestContext
from tldw_Server_API.app.core.Metrics.metrics_manager import get_metrics_registry
from tldw_Server_API.app.core.testing import is_truthy
from tldw_Server_API.app.core.TTS.tts_exceptions import (
    TTSAuthenticationError,
    TTSError,
    TTSInvalidVoiceReferenceError,
    TTSProviderNotConfiguredError,
    TTSQuotaExceededError,
    TTSRateLimitError,
    TTSValidationError,
)
from tldw_Server_API.app.core.TTS.tts_request_resolution import (
    resolve_tts_request_defaults,
)
from tldw_Server_API.app.core.TTS.tts_service_v2 import TTSServiceV2

_ALLOWED_AUDIO_FORMATS = {"wav", "mp3", "ogg", "opus", "aac", "flac", "webm", "m4a"}


def _normalize_audio_format(input_format: str) -> str:
    """Normalize common audio format strings (extensions or MIME types) to bare extensions."""
    fmt = input_format.lower().strip()
    # Strip any MIME subtype and parameters (e.g., audio/wav;codec=... -> wav)
    if "/" in fmt:
        fmt = fmt.split("/", 1)[1]
    if ";" in fmt:
        fmt = fmt.split(";", 1)[0]
    # Drop leading "x-" if present (audio/x-wav)
    if fmt.startswith("x-"):
        fmt = fmt[2:]
    return fmt


def _decode_base64_audio(data: str) -> bytes:
    """Decode base64 audio, raising HTTP 400 on failure."""
    try:
        return base64.b64decode(data, validate=True)
    except Exception as e:  # noqa: BLE001
        logger.warning("Failed to decode base64 audio")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid base64 encoding for input_audio",
        ) from e


def _load_audio_to_mono_np(audio_bytes: bytes) -> tuple[np.ndarray, int]:
    """
    Load audio bytes into a mono float32 numpy array and return (audio, sample_rate).

    Uses soundfile to support common formats (wav, mp3, ogg, etc.). Raises HTTP 400 on failure.
    """
    try:
        with io.BytesIO(audio_bytes) as buf:
            audio, sample_rate = sf.read(buf, dtype="float32", always_2d=False)
    except Exception as e:  # noqa: BLE001
        logger.warning("Failed to read audio bytes for speech chat")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Unsupported or corrupt audio format in input_audio",
        ) from e

    if audio.size == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Empty audio input",
        )

    # Ensure mono
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    return audio, int(sample_rate or 16000)


def _validate_audio_constraints(
    *,
    audio_bytes: bytes,
    duration_sec: float,
    input_format: str,
) -> None:
    """Validate input audio size/duration/format for speech chat."""
    if not input_format or not isinstance(input_format, str):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="input_audio_format is required",
        )
    fmt = _normalize_audio_format(input_format)
    if fmt not in _ALLOWED_AUDIO_FORMATS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported input_audio_format '{input_format}'",
        )
    # Size limit (bytes): allow env override but fall back safely on parse errors.
    _raw_max_bytes = os.getenv("AUDIO_CHAT_MAX_BYTES")
    if _raw_max_bytes is not None:
        try:
            max_bytes = int(_raw_max_bytes)
        except (ValueError, TypeError):
            logger.debug("AUDIO_CHAT_MAX_BYTES parse failed; using default 20MB")
            max_bytes = 20 * 1024 * 1024
    else:
        max_bytes = 20 * 1024 * 1024

    if audio_bytes and len(audio_bytes) > max_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="input_audio exceeds size limit for speech chat",
        )
    # Duration limit (seconds): allow env override but fall back safely on parse errors.
    _raw_max_duration = os.getenv("AUDIO_CHAT_MAX_DURATION_SEC")
    if _raw_max_duration is not None:
        try:
            max_duration = float(_raw_max_duration)
        except (ValueError, TypeError):
            logger.debug("AUDIO_CHAT_MAX_DURATION_SEC parse failed; using default 120s")
            max_duration = 120.0
    else:
        max_duration = 120.0

    if duration_sec > max_duration:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="input_audio duration exceeds allowed limit for speech chat",
        )


def _actions_enabled() -> bool:
    """Return True when action execution is enabled for audio chat."""
    return is_truthy(os.getenv("AUDIO_CHAT_ENABLE_ACTIONS"))


async def _execute_action(action_name: str, transcript: str, current_user: User) -> dict[str, Any]:
    """
    Execute a tool/workflow via MCP modules when available; fail soft with status.
    """
    allow_env = os.getenv("AUDIO_CHAT_ALLOWED_ACTIONS", "")
    if allow_env:
        allowed = {a.strip() for a in allow_env.split(",") if a.strip()}
        if allowed and action_name not in allowed:
            return {
                "action": action_name,
                "status": "not_allowed",
                "message": "Action not allowed",
                "user_id": getattr(current_user, "id", None),
            }
    user_id = getattr(current_user, "id", None)
    ctx = RequestContext(
        request_id=str(uuid.uuid4()),
        user_id=str(user_id) if user_id is not None else None,
        client_id=None,
        metadata={"source": "audio.chat"},
    )
    registry = get_module_registry()
    try:
        module = await registry.find_module_for_tool(action_name)
    except Exception:  # noqa: BLE001  # defensive: action failures must not break speech chat
        logger.warning("Action lookup failed during speech chat")
        return {
            "action": action_name,
            "status": "error",
            "message": "Action lookup failed; see server logs for details.",
            "user_id": user_id,
        }

    if module is None:
        return {
            "action": action_name,
            "status": "not_found",
            "message": "No module registered for this action",
            "user_id": user_id,
        }

    try:
        result = await module.execute_tool(action_name, {"input": transcript}, context=ctx)
        return {
            "action": action_name,
            "status": "ok",
            "result": result,
            "user_id": user_id,
        }
    except Exception:  # noqa: BLE001  # defensive: action failures must not break speech chat
        logger.warning("Action execution failed during speech chat")
        return {
            "action": action_name,
            "status": "error",
            "message": "Action execution failed; see server logs for details.",
            "user_id": user_id,
        }


async def _maybe_execute_action(
    *,
    transcript: str,
    request_data: SpeechChatRequest,
    current_user: User,
) -> dict[str, Any] | None:
    """
    Execute an action/workflow when enabled and requested.

    The action hint can come from request metadata or llm_config.extra_params["action"].
    """
    if not _actions_enabled():
        return None
    action_hint = None
    if request_data.metadata and isinstance(request_data.metadata, dict):
        action_hint = request_data.metadata.get("action")
    if not action_hint and request_data.llm_config and request_data.llm_config.extra_params:
        action_hint = request_data.llm_config.extra_params.get("action")
    if not action_hint:
        return None
    return await _execute_action(str(action_hint), transcript, current_user)


def _is_transcription_error(msg: str) -> bool:
    """Delegate to the shared transcription error sentinel helper."""
    return is_transcription_error_message(msg)


def _strip_whisper_metadata_header_from_text(text: str) -> str:
    """
    Remove the Whisper metadata header from a plain-text transcript, if present.

    The faster-whisper pipeline may prepend a header like:
        "This text was transcribed using whisper model: ...\\n"
        "Detected language: ...\\n\\n"
    For LLM input we want only the user content, so we trim this header.
    """
    if not isinstance(text, str):
        return text

    header_prefix = "This text was transcribed using whisper model:"
    if not text.startswith(header_prefix):
        return text

    # Preferred: split on the blank line separating header from content
    parts = text.split("\n\n", 1)
    if len(parts) == 2:
        return parts[1]

    # Fallback: drop the first two lines (model + language) if present
    lines = text.splitlines()
    if len(lines) >= 3:
        return "\n".join(lines[2:])
    return text


def _map_tts_exception(exc: Exception) -> HTTPException:
    """Map TTS exceptions to HTTPException consistent with /audio/speech."""
    if isinstance(exc, TTSInvalidVoiceReferenceError):
        logger.warning("TTS voice reference error in speech chat")
        return HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc))
    if isinstance(exc, TTSValidationError):
        logger.warning("TTS validation error in speech chat")
        return HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    if isinstance(exc, TTSProviderNotConfiguredError):
        logger.error("TTS provider not configured in speech chat")
        return HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="TTS service unavailable",
        )
    if isinstance(exc, TTSAuthenticationError):
        logger.error("TTS authentication error in speech chat")
        return HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="TTS provider authentication failed",
        )
    if isinstance(exc, TTSRateLimitError):
        logger.warning("TTS rate limit exceeded in speech chat")
        return HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="TTS provider rate limit exceeded. Please try again later.",
        )
    if isinstance(exc, TTSQuotaExceededError):
        logger.warning("TTS quota exceeded in speech chat")
        return HTTPException(
            status_code=status.HTTP_402_PAYMENT_REQUIRED,
            detail="TTS quota exceeded. Please review your plan or quota.",
        )

    # Fallback for other TTSError subclasses and unexpected errors
    if isinstance(exc, TTSError):
        logger.error("TTS provider error in speech chat")
        return HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="TTS provider error while generating speech",
        )

    logger.error("Unexpected TTS error in speech chat")
    return HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail="Unexpected error during TTS generation",
    )


async def run_speech_chat_turn(
    *,
    request_data: SpeechChatRequest,
    request: Any | None = None,
    current_user: User,
    chat_db: CharactersRAGDB,
    tts_service: TTSServiceV2,
) -> SpeechChatResponse:
    """
    Execute a single non-streaming speech chat turn.

    Steps:
      1. Decode and normalize input audio to mono.
      2. Run STT to obtain a user transcript.
      3. Resolve character + conversation and load recent history.
      4. Call LLM via adapter registry (async) with legacy fallback.
      5. Persist user/assistant messages into ChaChaNotes.
      6. Run TTS to synthesize assistant reply and base64-encode it.
    """
    # --- Decode audio ---
    req_start = time.time()
    raw_audio_bytes = _decode_base64_audio(request_data.input_audio)
    audio_np, sample_rate = _load_audio_to_mono_np(raw_audio_bytes)
    duration_sec = float(len(audio_np) / float(sample_rate or 16000))
    _validate_audio_constraints(
        audio_bytes=raw_audio_bytes,
        duration_sec=duration_sec,
        input_format=request_data.input_audio_format,
    )

    # --- STT ---
    stt_start = time.time()
    stt_provider = None
    stt_language = None
    if request_data.stt_config is not None:
        stt_provider = request_data.stt_config.provider
        stt_language = request_data.stt_config.language
    try:
        transcript = await asyncio.to_thread(
            transcribe_audio,
            audio_data=audio_np,
            transcription_provider=stt_provider,
            sample_rate=sample_rate,
            speaker_lang=stt_language,
        )
    except HTTPException:
        raise
    except Exception as e:  # noqa: BLE001
        logger.error("Speech chat STT failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Transcription failed for speech chat",
        ) from e

    if not isinstance(transcript, str):
        logger.error(f"Speech chat STT returned non-string transcript: type={type(transcript)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Transcription failed for speech chat. Please verify STT configuration in config.txt.",
        )

    if _is_transcription_error(transcript):
        logger.error("Speech chat STT returned error sentinel")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Transcription failed for speech chat. Please try again or verify STT configuration in config.txt.",
        )

    # Remove known Whisper metadata header lines so the LLM receives only
    # user content in the prompt.
    transcript = _strip_whisper_metadata_header_from_text(transcript)
    transcript = transcript.strip()
    if not transcript:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Transcription produced empty text from input_audio. Please verify STT configuration in config.txt.",
        )
    stt_ms = (time.time() - stt_start) * 1000.0

    # --- Conversation & character context ---
    loop = None
    try:
        loop = asyncio.get_running_loop()
    except Exception:
        loop = None

    if loop is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Async event loop unavailable for speech chat",
        )

    character_card, character_db_id = await get_or_create_character_context(
        chat_db,
        character_id=None,
        loop=loop,
    )
    if not character_db_id:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unable to resolve character context for speech chat",
        )

    client_id = getattr(chat_db, "client_id", None) or str(
        getattr(current_user, "id", "speech_chat_client")
    )
    character_name = (character_card or {}).get("name") or "Assistant"
    conversation_id, _was_created = await get_or_create_conversation(
        db=chat_db,
        conversation_id=request_data.session_id,
        character_id=character_db_id,
        character_name=character_name,
        client_id=client_id,
        loop=loop,
    )

    # Load prior history for context (simple fixed limit)
    try:
        history_messages = await load_conversation_history(
            chat_db,
            conversation_id,
            character_card,
            limit=20,
            loop=loop,
        )
    except Exception:  # noqa: BLE001
        logger.error("Failed to load conversation history for speech chat")
        history_messages = []

    # --- LLM call ---
    if request_data.llm_config is None or not request_data.llm_config.model:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="llm_config.model is required for speech chat v1",
        )

    llm_provider = (
        (request_data.llm_config.api_provider or "").strip().lower()
        or DEFAULT_LLM_PROVIDER
    )
    llm_model = request_data.llm_config.model

    messages_payload = list(history_messages)
    messages_payload.append({"role": "user", "content": transcript})

    def _fallback_resolver(name: str) -> str | None:
        try:
            return get_api_keys().get(name)
        except Exception:
            return None

    user_id_int = getattr(current_user, "id_int", None)
    if user_id_int is None:
        try:
            user_id_int = int(getattr(current_user, "id", None))
        except Exception:
            user_id_int = None
    byok_resolution = await resolve_byok_credentials(
        llm_provider,
        user_id=user_id_int,
        request=request,
        fallback_resolver=_fallback_resolver,
    )
    provider_api_key = byok_resolution.api_key

    llm_start = time.time()
    try:
        adapter = get_registry().get_adapter(normalize_provider(llm_provider))
        system_message = (character_card or {}).get("system_prompt")
        request_messages = messages_payload
        if system_message is None:
            system_message, request_messages = split_system_message(messages_payload)

        app_config = ensure_app_config(byok_resolution.app_config)
        api_key = provider_api_key or resolve_provider_api_key_from_config(llm_provider, app_config)
        provider_key = (llm_provider or "").strip().lower()
        if provider_requires_api_key(provider_key) and not api_key:
            record_byok_missing_credentials(provider_key, operation="speech_chat")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail={
                    "error_code": "missing_provider_credentials",
                    "message": f"Provider '{llm_provider}' requires an API key.",
                },
            )

        if adapter is not None:
            resolved_model = llm_model or resolve_provider_model(llm_provider, app_config)
            if not resolved_model:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="llm_config.model is required for speech chat v1",
                )
            request_payload = {
                "messages": request_messages,
                "system_message": system_message,
                "model": resolved_model,
                "api_key": api_key,
                "temperature": request_data.llm_config.temperature,
                "max_tokens": request_data.llm_config.max_tokens,
                "response_format": {"type": "text"},
                "user": str(getattr(current_user, "id", client_id)),
                "app_config": app_config,
            }
            try:
                llm_response = await adapter.achat(request_payload)
            except NotImplementedError:
                llm_response = await asyncio.to_thread(adapter.chat, request_payload)
        else:
            llm_response = await chat_api_call_async(
                api_endpoint=llm_provider,
                messages_payload=request_messages,
                api_key=api_key,
                temp=request_data.llm_config.temperature,
                maxp=None,
                model=llm_model,
                topk=None,
                topp=None,
                max_tokens=request_data.llm_config.max_tokens,
                response_format={"type": "text"},
                streaming=False,
                user_identifier=str(getattr(current_user, "id", client_id)),
                system_message=system_message,
                app_config=app_config,
            )
    except HTTPException:
        raise
    except Exception as e:  # noqa: BLE001
        logger.error("Speech chat LLM call failed")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="LLM provider error during speech chat",
        ) from e

    llm_ms = (time.time() - llm_start) * 1000.0

    await byok_resolution.touch_last_used()

    assistant_text = extract_response_content(llm_response) or ""
    assistant_text = assistant_text.strip()
    if not assistant_text:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="LLM returned empty completion for speech chat",
        )

    # Extract token usage if available
    token_usage: SpeechChatTokenUsage | None = None
    if isinstance(llm_response, dict):
        usage = llm_response.get("usage") or {}
        if isinstance(usage, dict):
            token_usage = SpeechChatTokenUsage(
                prompt_tokens=usage.get("prompt_tokens"),
                completion_tokens=usage.get("completion_tokens"),
                total_tokens=usage.get("total_tokens"),
            )

    # --- Optional action/workflow execution ---
    action_result: dict[str, Any] | None = await _maybe_execute_action(
        transcript=transcript,
        request_data=request_data,
        current_user=current_user,
    )

    # --- Persist messages into ChaChaNotes ---
    try:
        chat_db.add_message(
            {
                "conversation_id": conversation_id,
                "sender": "user",
                "content": transcript,
                "client_id": client_id,
            }
        )
        chat_db.add_message(
            {
                "conversation_id": conversation_id,
                "sender": "assistant",
                "content": assistant_text,
                "client_id": client_id,
            }
        )
        if action_result is not None:
            try:
                tool_content = json.dumps(action_result)
            except TypeError:
                logger.warning("Failed to serialize action_result for chat history")
            else:
                chat_db.add_message(
                    {
                        "conversation_id": conversation_id,
                        "sender": "tool",
                        "content": tool_content,
                        "client_id": client_id,
                    }
                )
    except Exception:  # noqa: BLE001
        logger.error("Failed to persist speech chat messages")
        # Do not fail the user-facing request solely due to DB persistence issues

    # --- TTS ---
    tts_start = time.time()
    tts_config = request_data.tts_config
    response_format = (tts_config.response_format if tts_config and tts_config.response_format else "mp3")
    resolved_tts = resolve_tts_request_defaults(
        provider=(tts_config.provider if tts_config else None),
        model=(tts_config.model if tts_config else None),
        voice=(tts_config.voice if tts_config else None),
    )

    tts_request = OpenAISpeechRequest(
        model=resolved_tts.model,
        input=assistant_text,
        voice=resolved_tts.voice,
        response_format=response_format,
        speed=(tts_config.speed if tts_config and tts_config.speed is not None else 1.0),
        extra_params=(tts_config.extra_params if tts_config else None),
        stream=False,
    )

    try:
        audio_chunks = []
        async for chunk in tts_service.generate_speech(
            tts_request,
            provider=resolved_tts.provider,
            fallback=True,
        ):
            if chunk:
                audio_chunks.append(chunk)
        audio_bytes = b"".join(audio_chunks)
    except Exception as e:  # noqa: BLE001
        raise _map_tts_exception(e) from e

    if not audio_bytes:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="TTS produced empty audio for speech chat",
        )

    tts_ms = (time.time() - tts_start) * 1000.0

    # --- Build response ---
    mime_map: dict[str, str] = {
        "mp3": "audio/mpeg",
        "opus": "audio/opus",
        "aac": "audio/aac",
        "flac": "audio/flac",
        "wav": "audio/wav",
        "pcm": "audio/L16; rate=24000; channels=1",
    }
    mime_type = mime_map.get(response_format, "audio/mpeg")

    output_b64 = base64.b64encode(audio_bytes).decode("ascii")

    timing = SpeechChatTiming(
        stt_ms=stt_ms,
        llm_ms=llm_ms,
        tts_ms=tts_ms,
    )

    # End-to-end latency metric (defensive: metrics must not break the request)
    try:
        total_latency = max(0.0, time.time() - req_start)
        reg = get_metrics_registry()
        reg.observe(
            "audio_chat_latency_seconds",
            total_latency,
            labels={
                "stt_provider": stt_provider or "default",
                "llm_provider": llm_provider,
                "tts_provider": resolved_tts.provider,
            },
        )
    except Exception:  # noqa: BLE001
        logger.debug("Failed to record audio_chat_latency_seconds metric")

    return SpeechChatResponse(
        session_id=conversation_id,
        user_transcript=transcript,
        assistant_text=assistant_text,
        output_audio=output_b64,
        output_audio_mime_type=mime_type,
        timing=timing,
        token_usage=token_usage,
        metadata=request_data.metadata,
        action_result=action_result,
    )
