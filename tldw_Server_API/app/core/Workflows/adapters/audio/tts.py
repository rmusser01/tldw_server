"""Text-to-speech adapter.

This module includes the TTS adapter for speech synthesis.
"""

from __future__ import annotations

import asyncio
import contextlib
import uuid
from typing import Any

try:
    from tldw_Server_API.app.api.v1.schemas.audio_schemas import (  # type: ignore
        NormalizationOptions,
        OpenAISpeechRequest,
    )
except ImportError:
    NormalizationOptions = None  # type: ignore[assignment]
    OpenAISpeechRequest = None  # type: ignore[assignment]

try:
    from tldw_Server_API.app.core.TTS.tts_service_v2 import get_tts_service_v2  # type: ignore
except ImportError:
    get_tts_service_v2 = None  # type: ignore[assignment]

try:
    from tldw_Server_API.app.core.TTS.utils import clean_text_for_tts
except ImportError:
    def clean_text_for_tts(text: str | None) -> str:  # type: ignore[no-redef]
        return str(text or "")

from tldw_Server_API.app.core.Workflows.adapters._common import (
    AsyncFileWriter,
    resolve_artifact_filename,
    resolve_artifacts_dir,
    watchlist_artifact_metadata,
)
from tldw_Server_API.app.core.Workflows.adapters._registry import registry
from tldw_Server_API.app.core.Workflows.adapters.audio._config import TTSConfig

_WORKFLOW_TTS_NONCRITICAL_EXCEPTIONS = (
    AttributeError,
    ConnectionError,
    KeyError,
    OSError,
    RuntimeError,
    TimeoutError,
    TypeError,
    UnicodeDecodeError,
    ValueError,
)


@registry.register(
    "tts",
    category="audio",
    description="Text-to-speech synthesis",
    parallelizable=True,
    config_model=TTSConfig,
    tags=["audio", "speech"],
)
async def run_tts_adapter(config: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    """Synthesize speech from text using the internal TTS service.

    Config:
      - input: str (templated) - Text to synthesize; defaults to last.text
      - model: str = "kokoro" - TTS model (kokoro, tts-1, etc.)
      - voice: str = "af_heart" - Voice to use
      - response_format: Literal["mp3", "wav", "opus", "flac", "aac", "pcm"] = "mp3"
      - speed: float = 1.0 - Speech speed multiplier
      - provider: str (optional) - Provider hint
    Output:
      - {"audio_uri": "file://...", "format": str, "model": str,
         "voice": str, "size_bytes": int}
    """
    if OpenAISpeechRequest is None or NormalizationOptions is None or get_tts_service_v2 is None:
        return {"error": "tts_unavailable"}

    # Resolve input text
    from tldw_Server_API.app.core.Chat.prompt_template_manager import apply_template_to_string as _tmpl

    text_t = str(config.get("input") or "").strip()
    if text_t:
        text = _tmpl(text_t, context) or text_t
    else:
        text = None
        try:
            # Prefer last.text, then inputs.summary, then inputs.text
            last = context.get("prev") or context.get("last") or {}
            text = str(last.get("text")) if isinstance(last, dict) and last.get("text") else None
        except _WORKFLOW_TTS_NONCRITICAL_EXCEPTIONS:
            text = None
        if not text and isinstance(context.get("inputs"), dict):
            text = str(context["inputs"].get("summary") or context["inputs"].get("text") or "")
    text = clean_text_for_tts(text or "")
    if not text.strip():
        return {"error": "missing_input_text"}

    model = str(config.get("model") or "kokoro")
    voice = str(config.get("voice") or "af_heart")
    fmt = str(config.get("response_format") or "mp3").lower()
    try:
        speed = float(config.get("speed", 1.0))
    except (TypeError, ValueError):
        speed = 1.0
    provider = str(config.get("provider") or "").strip() or None

    # Optional advanced fields
    lang_code = str(config.get("lang_code") or "").strip() or None
    normalization = None
    try:
        norm_cfg = config.get("normalization_options") or config.get("normalization")
        if isinstance(norm_cfg, dict):
            normalization = NormalizationOptions(**norm_cfg)
    except _WORKFLOW_TTS_NONCRITICAL_EXCEPTIONS:
        normalization = None
    voice_reference = str(config.get("voice_reference") or "").strip() or None
    reference_duration_min = None
    try:
        if config.get("reference_duration_min") is not None:
            reference_duration_min = float(config.get("reference_duration_min"))
    except (TypeError, ValueError):
        reference_duration_min = None
    # Merge provider-specific options into extra_params
    extra_params = config.get("extra_params") if isinstance(config.get("extra_params"), dict) else {}
    provider_opts = config.get("provider_options") if isinstance(config.get("provider_options"), dict) else {}
    try:
        if provider_opts:
            extra_params = {**(extra_params or {}), **provider_opts}
    except (TypeError, ValueError):
        pass

    req = OpenAISpeechRequest(
        model=model,
        input=text,
        voice=voice,
        response_format=fmt,
        speed=speed,
        stream=True,
        lang_code=lang_code,
        normalization_options=normalization,
        voice_reference=voice_reference,
        reference_duration_min=reference_duration_min,
        extra_params=extra_params,
    )

    # Prepare output path under Databases/artifacts/<step_run_id or ts>/speech.ext
    import time as _time

    step_run_id = str(context.get("step_run_id") or f"tts_{int(_time.time() * 1000)}")
    out_dir = resolve_artifacts_dir(step_run_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    ext = "mp3" if fmt not in {"wav", "opus", "flac", "aac", "pcm"} else fmt
    # Optional file naming template
    try:
        tmpl = str(config.get("output_filename_template") or "").strip()
    except (TypeError, ValueError):
        tmpl = ""
    if tmpl:
        try:
            # Expose common fields in template context
            tctx = {
                **context,
                "voice": voice,
                "model": model,
                "ext": ext,
                "run_id": str(context.get("run_id") or ""),
                "step_id": str(context.get("step_run_id") or ""),
                "timestamp": str(int(_time.time())),
            }
            fname = (_tmpl(tmpl, tctx) or tmpl).strip()
            if not fname:
                fname = f"speech.{ext}"
            if not fname.lower().endswith(f".{ext}"):
                fname = f"{fname}.{ext}"
        except _WORKFLOW_TTS_NONCRITICAL_EXCEPTIONS:
            fname = f"speech.{ext}"
    else:
        fname = f"speech.{ext}"
    fname = resolve_artifact_filename(fname, ext, default_stem="speech")
    out_path = out_dir / fname

    size_bytes = 0
    try:
        service = await get_tts_service_v2()
        async with AsyncFileWriter(out_path) as writer:
            async for chunk in service.generate_speech(req, provider=provider):
                # Cooperative cancel during streaming
                try:
                    if callable(context.get("is_cancelled")) and context["is_cancelled"]():
                        return {"__status__": "cancelled"}
                except _WORKFLOW_TTS_NONCRITICAL_EXCEPTIONS:
                    pass
                if isinstance(chunk, (bytes, bytearray)):
                    await writer.write(chunk)
                    size_bytes += len(chunk)
                else:
                    # Some providers may stream text errors when stream_errors_as_audio is enabled
                    data = bytes(chunk)
                    await writer.write(data)
                    size_bytes += len(data)
    except _WORKFLOW_TTS_NONCRITICAL_EXCEPTIONS:
        return {"error": "tts_unavailable"}
    except Exception:
        return {"error": "tts_unavailable"}

    # Optional post-process normalization via ffmpeg (best-effort)
    pp = config.get("post_process") or {}
    normalized = False
    normalized_path = out_path
    try:
        if isinstance(pp, dict) and pp.get("normalize"):
            import shutil

            ffmpeg_path = shutil.which("ffmpeg")
            if ffmpeg_path:
                # Use EBU R128 loudness normalization as a sane default
                target_lufs = float(pp.get("target_lufs", -16.0))
                true_peak = float(pp.get("true_peak_dbfs", -1.5))
                lra = float(pp.get("lra", 11.0))
                norm_out = out_dir / f"normalized.{ext}"
                cmd = [
                    ffmpeg_path,
                    "-y",
                    "-nostdin",
                    "-i",
                    str(out_path),
                    "-af",
                    f"loudnorm=I={target_lufs}:TP={true_peak}:LRA={lra}",
                    str(norm_out),
                ]
                try:
                    proc = await asyncio.create_subprocess_exec(
                        *cmd,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE,
                        cwd=str(out_dir),
                    )
                    try:
                        await asyncio.wait_for(proc.communicate(), timeout=120)
                    except asyncio.TimeoutError:
                        proc.kill()
                        with contextlib.suppress(_WORKFLOW_TTS_NONCRITICAL_EXCEPTIONS):
                            await proc.communicate()
                    else:
                        if proc.returncode == 0:
                            normalized = True
                            normalized_path = norm_out
                        else:
                            normalized = False
                except _WORKFLOW_TTS_NONCRITICAL_EXCEPTIONS:
                    normalized = False
            else:
                normalized = False
    except _WORKFLOW_TTS_NONCRITICAL_EXCEPTIONS:
        normalized = False

    # Prepare outputs and optional artifacts
    outputs: dict[str, Any] = {
        "audio_uri": f"file://{normalized_path}",
        "format": ext,
        "model": model,
        "voice": voice,
        "size_bytes": size_bytes,
        "normalized": normalized,
    }

    # Create audio artifact and attach a download link if requested
    attach_download = bool(config.get("attach_download_link"))
    save_transcript = bool(config.get("save_transcript"))
    audio_artifact_id = None
    try:
        if callable(context.get("add_artifact")):
            import mimetypes

            mime, _ = mimetypes.guess_type(str(out_path))
            audio_artifact_id = f"tts_{uuid.uuid4()}"
            artifact_metadata = config.get("artifact_metadata")
            if not isinstance(artifact_metadata, dict):
                artifact_metadata = {}
            context["add_artifact"](
                type="tts_audio",
                uri=f"file://{normalized_path}",
                size_bytes=size_bytes,
                mime_type=mime or "application/octet-stream",
                metadata={
                    "model": model,
                    "voice": voice,
                    "format": ext,
                    **watchlist_artifact_metadata(context),
                    **artifact_metadata,
                },
                artifact_id=audio_artifact_id,
            )
    except _WORKFLOW_TTS_NONCRITICAL_EXCEPTIONS:
        audio_artifact_id = None

    if attach_download and audio_artifact_id:
        outputs["download_url"] = f"/api/v1/workflows/artifacts/{audio_artifact_id}/download"

    # Optional transcript artifact
    if save_transcript and text:
        try:
            tx = out_dir / "transcript.txt"
            tx.write_text(text or "", encoding="utf-8")
            if callable(context.get("add_artifact")):
                context["add_artifact"](
                    type="tts_transcript",
                    uri=f"file://{tx}",
                    size_bytes=len(text.encode("utf-8")),
                    mime_type="text/plain",
                    metadata={"model": model, "voice": voice},
                )
            outputs["transcript"] = text
        except _WORKFLOW_TTS_NONCRITICAL_EXCEPTIONS:
            pass

    return outputs
