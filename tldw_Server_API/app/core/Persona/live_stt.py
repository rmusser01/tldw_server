"""Public Persona speech selection, PCM and transcript boundaries."""

from typing import Any


def normalize_persona_live_stt_model(raw_model: Any) -> tuple[str, str, str | None]:
    """Resolve a supported speech model identifier.

    Args:
        raw_model: Configured model name or None for the existing default.

    Returns:
        Backend, variant and optional Whisper model size.

    Raises:
        ValueError: The requested speech identifier is unsupported.
    """
    from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.model_utils import (
        normalize_model_and_variant,
    )

    normalized_raw = str(raw_model or "").strip().lower() or None
    whisper_sizes = {
        "tiny",
        "tiny.en",
        "base",
        "base.en",
        "small",
        "small.en",
        "medium",
        "medium.en",
        "large",
        "large-v1",
        "large-v2",
        "large-v3",
        "large-v3-turbo",
        "turbo",
        "distil-small.en",
        "distil-medium.en",
        "distil-large-v2",
        "distil-large-v3",
    }
    if normalized_raw in {"whisper", "whisper-1"}:
        return "whisper", "standard", None
    whisper_size = (normalized_raw or "").removeprefix("whisper-")
    if whisper_size in whisper_sizes:
        return "whisper", "standard", whisper_size
    model_name, model_variant = normalize_model_and_variant(
        normalized_raw,
        "parakeet",
        "standard",
    )
    if model_name not in {"parakeet", "canary", "qwen3-asr"}:
        raise ValueError("Unsupported Persona Live speech model. Select Whisper, Parakeet, Canary or Qwen3-ASR.")
    return model_name, model_variant, None


def build_persona_live_stt_config(voice_runtime: dict[str, Any] | None) -> Any:
    """Build the Persona streaming configuration.

    Args:
        voice_runtime: Selected model, language and voice settings.

    Returns:
        UnifiedStreamingConfig with the existing Persona capture contract.

    Raises:
        ValueError: The requested speech identifier is unsupported.
    """
    from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Streaming_Unified import (
        UnifiedStreamingConfig,
    )

    config = UnifiedStreamingConfig()
    model_name, model_variant, whisper_model_size = normalize_persona_live_stt_model(
        (voice_runtime or {}).get("stt_model")
    )
    config.model = model_name
    config.model_variant = model_variant
    if whisper_model_size:
        config.whisper_model_size = whisper_model_size
    language = str((voice_runtime or {}).get("stt_language") or "").strip()
    config.language = language.replace("_", "-").split("-", 1)[0].lower() or None
    config.sample_rate = 16000
    config.enable_vad = False
    # Speech filtering is independent of Persona's manual/automatic commitment.
    config.vad_filter = model_name == "whisper"
    config.enable_partial = True
    config.partial_interval = 0.35
    config.min_partial_duration = 0.3
    return config


def create_persona_live_stt_transcriber(*, voice_runtime: dict[str, Any] | None) -> Any:
    """Construct the selected speech transcriber without loading its model.

    Args:
        voice_runtime: Selected model, language and voice settings.

    Returns:
        An uninitialized transcriber owned by the calling voice preparation.

    Raises:
        ValueError: The requested speech identifier is unsupported.
    """
    from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Streaming_Unified import (
        UnifiedStreamingTranscriber,
    )

    config = build_persona_live_stt_config(voice_runtime)
    if config.model == "whisper":
        from tldw_Server_API.app.core.Persona.whisper_transcriber import PersonaWhisperTranscriber

        return PersonaWhisperTranscriber(config)
    if config.model == "parakeet" and config.model_variant == "onnx":
        from tldw_Server_API.app.core.Persona.parakeet_transcriber import PersonaParakeetOnnxTranscriber

        return PersonaParakeetOnnxTranscriber(config)
    return UnifiedStreamingTranscriber(config)


def normalize_persona_live_stt_audio(audio_bytes: bytes, *, audio_format: str) -> bytes:
    """Normalize microphone PCM for the configured speech runtime.

    Args:
        audio_bytes: Input PCM samples.
        audio_format: Supported PCM16 or float32 format identifier.

    Returns:
        Normalized float32 PCM bytes.

    Raises:
        ValueError: The format or byte length is invalid.
    """
    import numpy as np

    fmt = str(audio_format or "").strip().lower()
    if fmt in {"pcm16", "pcm", "s16le"}:
        audio_np = np.frombuffer(audio_bytes, dtype="<i2").astype(np.float32, copy=False)
        if audio_np.size == 0:
            return b""
        audio_np = audio_np / 32768.0
        return audio_np.astype(np.float32, copy=False).tobytes()
    if fmt in {"float32", "f32le", "f32"}:
        if len(audio_bytes) % 4 != 0:
            raise ValueError("float32 audio size must be divisible by 4")
        return bytes(audio_bytes)
    raise ValueError(f"Unsupported live STT audio_format '{fmt}'")


def persona_live_transcript_snapshot(
    *,
    transcriber: Any,
    result: dict[str, Any],
) -> str:
    """Return the revised full transcript.

    Args:
        transcriber: Runtime exposing finalized transcript history.
        result: Latest partial or final recognition event.

    Returns:
        Complete transcript snapshot without a duplicated finalized prefix.
    """
    finalized = str(getattr(transcriber, "get_full_transcript", lambda: "")() or "").strip()
    result_text = str(result.get("text") or "").strip()
    if str(result.get("type") or "").strip().lower() == "final":
        return finalized or result_text
    if finalized and result_text:
        if result_text.startswith(finalized):
            return result_text
        return f"{finalized} {result_text}".strip()
    return result_text or finalized
