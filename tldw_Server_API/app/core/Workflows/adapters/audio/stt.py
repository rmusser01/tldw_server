"""Speech-to-text adapter.

This module includes the STT adapter for audio transcription.
"""

from __future__ import annotations

from typing import Any

from tldw_Server_API.app.core.exceptions import AdapterError
from tldw_Server_API.app.core.Workflows.adapters._common import resolve_workflow_file_uri
from tldw_Server_API.app.core.Workflows.adapters._registry import registry
from tldw_Server_API.app.core.Workflows.adapters.audio._config import STTConfig


@registry.register(
    "stt_transcribe",
    category="audio",
    description="Speech-to-text transcription",
    parallelizable=False,
    config_model=STTConfig,
    tags=["audio", "speech"],
)
async def run_stt_transcribe_adapter(config: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    """Transcribe audio file locally with optional diarization.

    Config:
      - file_uri: str - file:// path to audio/video file (required)
      - model: str = "large-v3" - Whisper model name
      - language: str (optional) - Source language code
      - hotwords: list[str] | str (optional) - Hotwords for improved recognition
      - diarize: bool = False - Enable speaker diarization
      - word_timestamps: bool = False - Include word-level timestamps
    Output:
      - {"text": str, "segments": [...], "language": str}
    """
    file_uri = str(config.get("file_uri") or "").strip()
    if not (file_uri and file_uri.startswith("file://")):
        return {"error": "missing_or_invalid_file_uri"}
    try:
        resolved_path = resolve_workflow_file_uri(file_uri, context, config)
    except AdapterError:
        return {"error": "file_access_denied"}
    model = str(config.get("model") or "large-v3")
    language = config.get("language") or None
    hotwords = config.get("hotwords") or None
    diarize = bool(config.get("diarize", False))
    word_ts = bool(config.get("word_timestamps", False))
    try:
        from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Lib import speech_to_text

        # When language is None, allow the STT backend to auto-detect.
        segs_or_pair = speech_to_text(
            str(resolved_path),
            whisper_model=model,
            selected_source_lang=language,
            vad_filter=False,
            diarize=diarize,
            word_timestamps=word_ts,
            return_language=True,
            hotwords=hotwords,
        )
        if isinstance(segs_or_pair, tuple) and len(segs_or_pair) == 2:
            segments, lang = segs_or_pair
        else:
            segments, lang = segs_or_pair, None
        text = " ".join([s.get("Text", "").strip() for s in (segments or []) if isinstance(s, dict)])
        return {"text": text, "segments": segments, "language": lang}
    except Exception:
        return {"error": "stt_error"}
