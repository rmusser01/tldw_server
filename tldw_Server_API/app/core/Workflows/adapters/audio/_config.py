"""Pydantic config models for audio adapters."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field

from tldw_Server_API.app.core.Workflows.adapters._base import BaseAdapterConfig


class NormalizationOptionsConfig(BaseAdapterConfig):
    """Options for audio normalization."""

    normalize: bool = Field(False, description="Enable loudness normalization")
    target_lufs: float = Field(-16.0, description="Target loudness in LUFS")
    true_peak_dbfs: float = Field(-1.5, description="True peak in dBFS")
    lra: float = Field(11.0, description="Loudness range target")


class PostProcessConfig(BaseAdapterConfig):
    """Post-processing options for TTS output."""

    normalize: bool = Field(False, description="Enable loudness normalization")
    target_lufs: float = Field(-16.0, description="Target LUFS for normalization")
    true_peak_dbfs: float = Field(-1.5, description="True peak in dBFS")
    lra: float = Field(11.0, description="Loudness range")


class TTSConfig(BaseAdapterConfig):
    """Config for TTS (text-to-speech) adapter."""

    input: str | None = Field(None, description="Text to synthesize (templated); defaults to last.text")
    model: str = Field("kokoro", description="TTS model (kokoro, tts-1, etc.)")
    voice: str = Field("af_heart", description="Voice identifier")
    response_format: Literal["mp3", "wav", "opus", "flac", "aac", "pcm"] = Field(
        "mp3", description="Output audio format"
    )
    speed: float = Field(1.0, ge=0.25, le=4.0, description="Speech speed multiplier")
    provider: str | None = Field(None, description="Provider hint (optional)")
    lang_code: str | None = Field(None, description="Language code hint")
    normalization_options: NormalizationOptionsConfig | None = Field(
        None, description="Input normalization options"
    )
    voice_reference: str | None = Field(None, description="Voice reference file URI")
    reference_duration_min: float | None = Field(None, description="Minimum reference duration")
    extra_params: dict[str, Any] | None = Field(None, description="Provider-specific parameters")
    provider_options: dict[str, Any] | None = Field(None, description="Additional provider options")
    output_filename_template: str | None = Field(None, description="Output filename template (Jinja)")
    artifact_metadata: dict[str, Any] | None = Field(None, description="Additional audio artifact metadata")
    post_process: PostProcessConfig | None = Field(None, description="Post-processing options")


class STTConfig(BaseAdapterConfig):
    """Config for STT (speech-to-text) adapter."""

    file_uri: str = Field(..., description="file:// path to audio/video file (required)")
    model: str = Field("large-v3", description="Whisper model name")
    language: str | None = Field(None, description="Source language code")
    hotwords: list[str] | None = Field(None, description="Hotwords for improved recognition")
    diarize: bool = Field(False, description="Enable speaker diarization")
    word_timestamps: bool = Field(False, description="Include word-level timestamps")


class AudioNormalizeConfig(BaseAdapterConfig):
    """Config for audio normalization adapter."""

    file_uri: str = Field(..., description="file:// path to input audio (required)")
    target_lufs: float = Field(-16.0, description="Target loudness in LUFS")
    true_peak_dbfs: float = Field(-1.5, description="True peak in dBFS")
    lra: float = Field(11.0, description="Loudness range target")
    output_format: str = Field("mp3", description="Output audio format")


class AudioConcatConfig(BaseAdapterConfig):
    """Config for audio concatenation adapter."""

    files: list[str] = Field(..., description="List of file:// URIs to concatenate")
    output_format: str = Field("mp3", description="Output audio format")
    crossfade_ms: int = Field(0, ge=0, description="Crossfade duration between clips in ms")


class AudioTrimConfig(BaseAdapterConfig):
    """Config for audio trimming adapter."""

    file_uri: str = Field(..., description="file:// path to input audio (required)")
    start_ms: int = Field(0, ge=0, description="Start position in milliseconds")
    end_ms: int | None = Field(None, ge=0, description="End position in milliseconds")
    duration_ms: int | None = Field(None, ge=0, description="Duration to keep in milliseconds")
    output_format: str = Field("mp3", description="Output audio format")


class AudioConvertConfig(BaseAdapterConfig):
    """Config for audio format conversion adapter."""

    file_uri: str = Field(..., description="file:// path to input audio (required)")
    output_format: Literal["mp3", "wav", "opus", "flac", "aac", "ogg"] = Field(
        "mp3", description="Target audio format"
    )
    bitrate: str | None = Field(None, description="Target bitrate (e.g., '128k')")
    sample_rate: int | None = Field(None, description="Target sample rate in Hz")
    channels: int | None = Field(None, ge=1, le=8, description="Number of audio channels")


class AudioExtractConfig(BaseAdapterConfig):
    """Config for extracting audio from video adapter."""

    file_uri: str = Field(..., description="file:// path to input video (required)")
    output_format: Literal["mp3", "wav", "opus", "flac", "aac", "ogg"] = Field(
        "mp3", description="Output audio format"
    )
    start_ms: int | None = Field(None, ge=0, description="Start position in milliseconds")
    end_ms: int | None = Field(None, ge=0, description="End position in milliseconds")


class AudioMixConfig(BaseAdapterConfig):
    """Config for audio mixing adapter."""

    files: list[str] = Field(..., description="List of file:// URIs to mix")
    volumes: list[float] | None = Field(None, description="Volume levels for each track (0.0-2.0)")
    output_format: str = Field("mp3", description="Output audio format")


class MultiVoiceTTSConfig(BaseAdapterConfig):
    """Config for multi-voice TTS adapter."""

    sections: list[dict[str, Any]] | None = Field(None, description="Sections [{voice, text}] from compose step")
    voice_assignments: dict[str, str] | None = Field(None, description="Voice marker -> Kokoro voice ID mapping")
    default_model: str = Field("kokoro", description="Default TTS model")
    default_voice: str = Field("af_heart", description="Fallback voice if assignment missing")
    response_format: Literal["mp3", "wav", "opus", "flac", "aac"] = Field(
        "mp3", description="Output audio format"
    )
    speed: float = Field(1.0, ge=0.25, le=4.0, description="Speech speed multiplier")
    pause_duration_seconds: float = Field(1.0, ge=0.0, le=5.0, description="Silence between sections")
    normalize: bool = Field(True, description="EBU R128 normalize final output")
    target_lufs: float = Field(-16.0, description="Target LUFS for normalization")
    fallback_provider: str | None = Field("openai", description="Fallback TTS provider on failure")
    fallback_voice: str = Field("nova", description="Fallback voice for fallback provider")
    background_audio_uri: str | None = Field(
        None,
        description="Optional file:// URI for background music/ambience to mix under narration",
    )
    background_volume: float = Field(
        0.15,
        ge=0.0,
        le=2.0,
        description="Background track volume multiplier",
    )
    background_delay_ms: int = Field(
        0,
        ge=0,
        le=120000,
        description="Delay before background enters, in milliseconds",
    )
    background_fade_seconds: float = Field(
        2.0,
        ge=0.0,
        le=30.0,
        description="Fade-in/out duration for background track, in seconds",
    )


class AudioDiarizeConfig(BaseAdapterConfig):
    """Config for speaker diarization adapter."""

    file_uri: str = Field(..., description="file:// path to input audio (required)")
    min_speakers: int | None = Field(None, ge=1, description="Minimum expected speakers")
    max_speakers: int | None = Field(None, ge=1, description="Maximum expected speakers")
    model: str | None = Field(None, description="Diarization model to use")
