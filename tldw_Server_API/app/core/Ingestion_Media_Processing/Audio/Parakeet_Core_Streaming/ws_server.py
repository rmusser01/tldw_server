"""
Minimal WebSocket handler for Parakeet Core Streaming.

Protocol (JSON frames):
- Config frame (fields are optional unless noted):
  {
    "type": "config",                            # required
    "model": "parakeet",                        # default: parakeet
    "model_variant": "standard|onnx|mlx",       # default: standard
    "sample_rate": 16000,                        # default: 16000
    "chunk_duration": 2.0,                       # seconds per final segment
    "overlap_duration": 0.5,                     # seconds kept as context
    "language": "en",                           # optional language hint
    "enable_partial": true,                      # emit partial results
    "insights": { ... },                         # optional live insights config
    "diarization": true                          # or "diarize": true; enable speaker diarization
  }
- Client sends: {"type": "audio", "data": "<base64 float32 mono>"}
- Server emits partials: {"type": "partial", "text": "...", "is_final": false, ...segment metadata}
- Server emits finals:   {"type": "final",   "text": "...", "is_final": true,  ...segment metadata}
- Client can send: {"type": "commit"} to flush remaining audio and receive {"type": "full_transcript"}
- Client can send: {"type": "reset"} to clear state; server replies {"type": "status", "state": "reset"}
- Client can send: {"type": "stop"} to end the session
- Client can send: {"type": "ping"}; server replies {"type": "pong"}

This module does not enforce auth/quotas; it is a drop-in core for external servers.
"""

from __future__ import annotations

import contextlib
import json

from fastapi import APIRouter, WebSocket
from loguru import logger

from ..model_utils import normalize_model_and_variant
from .config import StreamingConfig
from .transcriber import DecodeFn, ParakeetCoreTranscriber

_PARAKEET_WS_NONCRITICAL_EXCEPTIONS = (
    AssertionError,
    AttributeError,
    ConnectionError,
    FileNotFoundError,
    ImportError,
    IndexError,
    KeyError,
    LookupError,
    OSError,
    PermissionError,
    RuntimeError,
    TimeoutError,
    TypeError,
    ValueError,
    UnicodeDecodeError,
    json.JSONDecodeError,
)

# Optional enhancements: live insights and diarization
try:  # pragma: no cover - optional, heavy dependency surface
    from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Streaming_Insights import (
        LiveInsightSettings,
        LiveMeetingInsights,
    )
except _PARAKEET_WS_NONCRITICAL_EXCEPTIONS:  # pragma: no cover - tests may not load insights deps
    LiveInsightSettings = None  # type: ignore
    LiveMeetingInsights = None  # type: ignore

try:  # pragma: no cover - optional
    # Reuse streaming diarizer from unified path
    from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Streaming_Unified import (
        StreamingDiarizer,
    )
except _PARAKEET_WS_NONCRITICAL_EXCEPTIONS:  # pragma: no cover
    StreamingDiarizer = None  # type: ignore


router = APIRouter()


@router.websocket("/core/parakeet/stream")
async def websocket_parakeet_core(
    websocket: WebSocket,
    decode_fn: DecodeFn | None = None,
):
    """
    Handle a long-lived WebSocket session for Parakeet Core streaming: accept configuration, receive audio chunks, stream partial and final transcript frames, and manage session lifecycle.

    This endpoint implements a minimal Parakeet Core streaming protocol over a WebSocket. It accepts a JSON "config" message to initialize or reconfigure the transcriber (including model and variant changes), processes "audio" messages containing audio payloads into transcript frames, supports "commit" to flush and emit a final transcript and optional diarization summary, and responds to "reset", "stop", and "ping" control messages. When available and enabled, it integrates optional live insights and streaming diarization engines; it also returns structured error and status messages and performs cleanup of optional engines and the WebSocket on exit.

    Parameters:
        websocket (WebSocket): WebSocket connection to receive messages from and send responses to.
        decode_fn (Optional[DecodeFn]): Optional decoding callback used by the ParakeetCoreTranscriber; if not provided or unavailable for the selected model variant, model-unavailable errors will be sent.
    """
    await websocket.accept()

    config = StreamingConfig()
    transcriber = ParakeetCoreTranscriber(config=config, decode_fn=decode_fn)
    insights_settings = None
    insights_engine = None
    diarizer = None

    try:
        while True:
            msg = await websocket.receive_text()
            try:
                data = json.loads(msg)
            except _PARAKEET_WS_NONCRITICAL_EXCEPTIONS:
                await websocket.send_json({"type": "error", "message": "Invalid JSON"})
                continue

            mtype = str(data.get("type") or "").lower()

            if mtype == "config":
                # update config
                cfg = data
                # detect if model selection changes require a new transcriber
                old_model = config.model
                old_variant = config.model_variant
                if "sample_rate" in cfg:
                    config.sample_rate = int(cfg.get("sample_rate") or config.sample_rate)
                if "chunk_duration" in cfg:
                    config.chunk_duration = float(cfg.get("chunk_duration") or config.chunk_duration)
                if "overlap_duration" in cfg:
                    config.overlap_duration = float(cfg.get("overlap_duration") or config.overlap_duration)
                if "enable_partial" in cfg:
                    config.enable_partial = bool(cfg.get("enable_partial"))
                if "partial_interval" in cfg:
                    config.partial_interval = float(cfg.get("partial_interval") or config.partial_interval)
                if "language" in cfg:
                    # store language hint; Parakeet core does not enforce it but downstream tools may use it
                    try:
                        config.language = str(cfg.get("language") or "") or None
                    except _PARAKEET_WS_NONCRITICAL_EXCEPTIONS:
                        config.language = None
                # Normalize model + variant, supporting forms like
                # "parakeet-tdt-0.6b-v3-onnx" and "parakeet-onnx".
                variant_override = cfg.get("variant") or cfg.get("model_variant")
                if ("model" in cfg) or (variant_override is not None):
                    raw_model = str(cfg.get("model") or config.model)
                    new_model, new_variant = normalize_model_and_variant(
                        raw_model,
                        current_model=config.model,
                        current_variant=config.model_variant,
                        variant_override=variant_override,
                    )
                    config.model = new_model
                    config.model_variant = new_variant

                if (config.model != old_model) or (config.model_variant != old_variant):
                    # reset state and rebuild transcriber for new model selection
                    with contextlib.suppress(_PARAKEET_WS_NONCRITICAL_EXCEPTIONS):
                        transcriber.close()
                    transcriber = ParakeetCoreTranscriber(config=config, decode_fn=decode_fn)
                    if getattr(transcriber, 'decode_fn', None) is None:
                        await websocket.send_json({
                            "type": "error",
                            "error_type": "model_unavailable",
                            "message": "Selected model variant is not available",
                            "details": {
                                "model": config.model,
                                "variant": config.model_variant,
                                "suggestion": "Install required dependencies or choose another variant",
                            },
                        })
                        # In this minimal core, Whisper fallback is not engaged automatically.
                        await websocket.send_json({
                            "type": "warning",
                            "state": "fallback_not_applied",
                            "message": "Whisper fallback not enabled in core route",
                        })
                # Optional live insights configuration
                try:
                    if LiveInsightSettings is not None and LiveMeetingInsights is not None:
                        insights_payload = cfg.get("insights") or cfg.get("meeting_insights")
                        if insights_payload is not None:
                            try:
                                insights_settings = LiveInsightSettings.from_client_payload(insights_payload)  # type: ignore[attr-defined]
                            except _PARAKEET_WS_NONCRITICAL_EXCEPTIONS as insight_err:
                                logger.error(f"Failed to parse live insights config: {insight_err}")
                                insights_settings = LiveInsightSettings(enabled=False)  # type: ignore[call-arg]
                        elif cfg.get("insights_enabled") is True and insights_settings is None:
                            insights_settings = LiveInsightSettings(enabled=True)  # type: ignore[call-arg]
                        elif cfg.get("insights_enabled") is False:
                            insights_settings = LiveInsightSettings(enabled=False)  # type: ignore[call-arg]

                        # Initialize engine if enabled
                        if insights_engine is None and insights_settings and insights_settings.enabled:
                            try:
                                insights_engine = LiveMeetingInsights(websocket, insights_settings)  # type: ignore[call-arg]
                                await websocket.send_json({
                                    "type": "status",
                                    "state": "insights_enabled",
                                    "insights": insights_engine.describe(),  # type: ignore[union-attr]
                                })
                            except _PARAKEET_WS_NONCRITICAL_EXCEPTIONS as insight_err:
                                logger.error(f"Live insights init failed: {insight_err}")
                                insights_engine = None
                                await websocket.send_json({
                                    "type": "status",
                                    "state": "insights_unavailable",
                                    "message": "Live insights disabled: initialization failed",
                                })
                        elif insights_settings and not insights_settings.enabled:
                            await websocket.send_json({"type": "status", "state": "insights_disabled"})
                except _PARAKEET_WS_NONCRITICAL_EXCEPTIONS as _insights_cfg_err:
                    logger.debug(f"Insights config handling skipped: {_insights_cfg_err}")

                # Optional diarization configuration
                try:
                    diarize_flag = bool(cfg.get("diarize") or cfg.get("diarization"))
                    if StreamingDiarizer is not None and diarize_flag:
                        diarizer = StreamingDiarizer(
                            sample_rate=config.sample_rate,
                            store_audio=bool(cfg.get("store_audio", False)),
                            storage_dir=cfg.get("diarization_storage_dir"),
                            num_speakers=cfg.get("num_speakers"),
                        )
                        try:
                            ok = await diarizer.ensure_ready()
                            if ok:
                                await websocket.send_json({"type": "status", "state": "diarization_enabled"})
                            else:
                                await websocket.send_json({"type": "status", "state": "diarization_unavailable"})
                        except _PARAKEET_WS_NONCRITICAL_EXCEPTIONS:
                            await websocket.send_json({"type": "status", "state": "diarization_unavailable"})
                except _PARAKEET_WS_NONCRITICAL_EXCEPTIONS as _diar_cfg_err:
                    logger.debug(f"Diarization config handling skipped: {_diar_cfg_err}")

                await websocket.send_json({"type": "status", "state": "configured"})

            elif mtype == "audio":
                payload = data.get("data")
                if payload is None:
                    await websocket.send_json({"type": "error", "message": "Missing audio payload"})
                    continue
                if getattr(transcriber, 'decode_fn', None) is None:
                    await websocket.send_json({
                        "type": "error",
                        "error_type": "model_unavailable",
                        "message": "Selected model variant is not available",
                        "details": {
                            "model": config.model,
                            "variant": config.model_variant,
                        },
                    })
                    continue
                frame = await transcriber.process_audio_chunk(payload)
                if frame:
                    # Diarization: label segment if final and diarizer available
                    try:
                        if diarizer and frame.get("is_final"):
                            audio_np = frame.get("_audio_chunk")
                            if audio_np is not None:
                                # inject text into meta for alignment
                                seg_meta = dict(frame)
                                seg_meta["text"] = frame.get("text", "")
                                speaker_info = await diarizer.label_segment(audio_np, seg_meta)
                                if speaker_info:
                                    frame["speaker_id"] = speaker_info.get("speaker_id")
                                    frame["speaker_label"] = speaker_info.get("speaker_label")
                    except _PARAKEET_WS_NONCRITICAL_EXCEPTIONS as _diar_err:
                        logger.debug(f"Diarization label_segment failed: {_diar_err}")
                    # Live insights: send segment to engine
                    try:
                        if insights_engine and frame.get("is_final"):
                            await insights_engine.on_transcript(frame)
                    except _PARAKEET_WS_NONCRITICAL_EXCEPTIONS as _insight_err:
                        logger.debug(f"Insights on_transcript failed: {_insight_err}")
                    # strip internal fields if present
                    frame.pop("_audio_chunk", None)
                    await websocket.send_json(frame)

            elif mtype == "commit":
                frame = await transcriber.flush()
                if frame:
                    # Final flush may include last audio; diarize and insights
                    try:
                        if diarizer and frame.get("_audio_chunk") is not None:
                            seg_meta = dict(frame)
                            seg_meta["text"] = frame.get("text", "")
                            speaker_info = await diarizer.label_segment(frame.get("_audio_chunk"), seg_meta)
                            if speaker_info:
                                frame["speaker_id"] = speaker_info.get("speaker_id")
                                frame["speaker_label"] = speaker_info.get("speaker_label")
                    except _PARAKEET_WS_NONCRITICAL_EXCEPTIONS as _diar_err:
                        logger.debug(f"Diarization label_segment (flush) failed: {_diar_err}")
                    try:
                        if insights_engine:
                            await insights_engine.on_transcript(frame)
                    except _PARAKEET_WS_NONCRITICAL_EXCEPTIONS as _insight_err:
                        logger.debug(f"Insights on_transcript (flush) failed: {_insight_err}")
                    frame.pop("_audio_chunk", None)
                    await websocket.send_json(frame)
                await websocket.send_json({
                    "type": "full_transcript",
                    "text": transcriber.get_full_transcript(),
                })
                # Finalize insights and diarization summaries
                try:
                    if insights_engine:
                        await insights_engine.on_commit(transcriber.get_full_transcript())
                except _PARAKEET_WS_NONCRITICAL_EXCEPTIONS as _insight_err:
                    logger.debug(f"Insights on_commit failed: {_insight_err}")
                try:
                    if diarizer:
                        mapping, audio_path, speakers = await diarizer.finalize()
                        if mapping or audio_path or speakers:
                            speaker_map = [
                                {
                                    "segment_id": seg_id,
                                    "speaker_id": info.get("speaker_id"),
                                    "speaker_label": info.get("speaker_label"),
                                }
                                for seg_id, info in sorted(mapping.items())
                            ]
                            await websocket.send_json({
                                "type": "diarization_summary",
                                "speaker_map": speaker_map,
                                "audio_path": audio_path,
                                "speakers": speakers,
                            })
                        # Emit a status frame reflecting persistence state
                        try:
                            if getattr(diarizer, "store_audio", False):
                                method = getattr(diarizer, "persistence_method", None)
                                if audio_path and method and method != "soundfile":
                                    await websocket.send_json({
                                        "type": "status",
                                        "state": "diarization_persist_degraded",
                                        "persistence_method": method,
                                    })
                                elif (not audio_path) or (method is None):
                                    await websocket.send_json({
                                        "type": "status",
                                        "state": "diarization_persist_disabled",
                                        "persistence_method": method,
                                    })
                        except _PARAKEET_WS_NONCRITICAL_EXCEPTIONS:
                            pass
                except _PARAKEET_WS_NONCRITICAL_EXCEPTIONS as _diar_err:
                    logger.debug(f"Diarization finalize failed: {_diar_err}")
                transcriber.reset()

            elif mtype == "reset":
                transcriber.reset()
                try:
                    if insights_engine:
                        await insights_engine.reset()
                except _PARAKEET_WS_NONCRITICAL_EXCEPTIONS as _insight_err:
                    logger.debug(f"Insights reset failed: {_insight_err}")
                try:
                    if diarizer:
                        await diarizer.reset()
                except _PARAKEET_WS_NONCRITICAL_EXCEPTIONS as _diar_err:
                    logger.debug(f"Diarizer reset failed: {_diar_err}")
                await websocket.send_json({"type": "status", "state": "reset"})

            elif mtype == "stop":
                break

            elif mtype == "ping":
                await websocket.send_json({"type": "pong"})

            else:
                await websocket.send_json({"type": "error", "message": "Unknown message type"})

    except _PARAKEET_WS_NONCRITICAL_EXCEPTIONS as e:
        logger.error(f"Parakeet core WS error: {e}")
        with contextlib.suppress(_PARAKEET_WS_NONCRITICAL_EXCEPTIONS):
            await websocket.send_json({"type": "error", "message": "Parakeet core websocket failed"})
    finally:
        try:
            with contextlib.suppress(_PARAKEET_WS_NONCRITICAL_EXCEPTIONS):
                await transcriber.flush()
            with contextlib.suppress(_PARAKEET_WS_NONCRITICAL_EXCEPTIONS):
                transcriber.close()
            if insights_engine:
                with contextlib.suppress(_PARAKEET_WS_NONCRITICAL_EXCEPTIONS):
                    await insights_engine.close()
            if diarizer:
                with contextlib.suppress(_PARAKEET_WS_NONCRITICAL_EXCEPTIONS):
                    await diarizer.close()
            await websocket.close()
        except _PARAKEET_WS_NONCRITICAL_EXCEPTIONS:
            pass


__all__ = ["router", "websocket_parakeet_core"]
