"""Speaker diarization adapter.

This module includes the diarization adapter for speaker identification.
"""

from __future__ import annotations

import asyncio
import os
from typing import Any

from loguru import logger

from tldw_Server_API.app.core.Workflows.adapters._common import (
    resolve_workflow_file_path,
    resolve_workflow_file_uri,
)
from tldw_Server_API.app.core.Workflows.adapters._registry import registry
from tldw_Server_API.app.core.Workflows.adapters.audio._config import AudioDiarizeConfig


@registry.register(
    "audio_diarize",
    category="audio",
    description="Speaker diarization",
    parallelizable=False,
    config_model=AudioDiarizeConfig,
    tags=["audio"],
)
async def run_audio_diarize_adapter(config: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    """Perform speaker diarization on an audio file.

    Config:
      - audio_path: str - Path to audio file
      - audio_uri: str - Alternative: file:// URI
      - num_speakers: int - Expected number of speakers (optional, auto-detect if not set)
      - min_speakers: int - Minimum speakers for auto-detect (default: 1)
      - max_speakers: int - Maximum speakers for auto-detect (default: 10)
      - model: str - Diarization model to use (default: "pyannote")
    Output:
      - segments: list[dict] - Speaker segments with timestamps
      - speakers: list[str] - Unique speaker labels
      - total_duration: float - Total audio duration in seconds
    """
    from tldw_Server_API.app.core.Chat.prompt_template_manager import apply_template_to_string as _tmpl

    if callable(context.get("is_cancelled")) and context["is_cancelled"]():
        return {"__status__": "cancelled"}

    audio_path = config.get("audio_path")
    audio_uri = config.get("audio_uri") or config.get("file_uri")

    if audio_uri:
        try:
            audio_path = resolve_workflow_file_uri(audio_uri, context, config)
        except Exception:
            return {"error": "invalid_audio_uri", "segments": [], "speakers": []}
    elif audio_path:
        if isinstance(audio_path, str):
            audio_path = _tmpl(audio_path, context) or audio_path
        try:
            audio_path = resolve_workflow_file_path(audio_path, context, config)
        except Exception:
            return {"error": "audio_access_denied", "segments": [], "speakers": []}
    else:
        # Try to get from previous step
        prev = context.get("prev") or context.get("last") or {}
        if isinstance(prev, dict):
            audio_path = prev.get("audio_path") or prev.get("file_path")
        if not audio_path:
            return {"error": "missing_audio_path", "segments": [], "speakers": []}

    num_speakers = config.get("num_speakers")
    min_speakers = int(config.get("min_speakers", 1))
    max_speakers = int(config.get("max_speakers", 10))

    try:
        # Try to use pyannote for diarization
        try:
            import torch
            from pyannote.audio import Pipeline

            # Load diarization pipeline
            pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=os.getenv("HF_TOKEN"),
            )

            # Move to GPU if available
            if torch.cuda.is_available():
                pipeline.to(torch.device("cuda"))

            # Run diarization
            if num_speakers:
                diarization = pipeline(str(audio_path), num_speakers=num_speakers)
            else:
                diarization = pipeline(
                    str(audio_path),
                    min_speakers=min_speakers,
                    max_speakers=max_speakers,
                )

            # Extract segments
            segments = []
            speakers = set()
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                segments.append(
                    {
                        "start": turn.start,
                        "end": turn.end,
                        "speaker": speaker,
                        "duration": turn.end - turn.start,
                    }
                )
                speakers.add(speaker)

            total_duration = max((s["end"] for s in segments), default=0)

            return {
                "segments": segments,
                "speakers": sorted(speakers),
                "total_duration": total_duration,
            }

        except ImportError:
            logger.warning("Pyannote not available, using simplified diarization")

            # Fallback: use whisper with speaker detection if available
            try:
                from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Lib import (
                    transcribe_audio_with_whisper,
                )

                result = await asyncio.to_thread(
                    transcribe_audio_with_whisper,
                    str(audio_path),
                    diarize=True,
                )

                if isinstance(result, dict) and result.get("segments"):
                    segments = []
                    speakers = set()
                    for seg in result["segments"]:
                        speaker = seg.get("speaker", "SPEAKER_0")
                        segments.append(
                            {
                                "start": seg.get("start", 0),
                                "end": seg.get("end", 0),
                                "speaker": speaker,
                                "text": seg.get("text", ""),
                            }
                        )
                        speakers.add(speaker)

                    return {
                        "segments": segments,
                        "speakers": sorted(speakers),
                        "total_duration": result.get("duration", 0),
                    }

            except Exception:
                logger.debug("Whisper diarization fallback failed")

            return {
                "error": "diarization_unavailable",
                "segments": [],
                "speakers": [],
                "message": "Install pyannote-audio for speaker diarization",
            }

    except Exception:
        logger.exception("Audio diarize adapter error")
        return {"error": "diarization_error", "segments": [], "speakers": []}
