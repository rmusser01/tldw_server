"""Core-owned speech request models, re-exported by the API schemas package (TASK-13950.15)."""

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


class NormalizationOptions(BaseModel):
    """Options for the normalization system"""

    normalize: bool = Field(
        default=True,
        description="Normalizes input text to make it easier for the model to say",
    )
    unit_normalization: bool = Field(
        default=False, description="Transforms units like 10KB to 10 kilobytes"
    )
    url_normalization: bool = Field(
        default=True,
        description="Changes urls so they can be properly pronounced by kokoro",
    )
    email_normalization: bool = Field(
        default=True,
        description="Changes emails so they can be properly pronouced by kokoro",
    )
    optional_pluralization_normalization: bool = Field(
        default=True,
        description="Replaces (s) with s so some words get pronounced correctly",
    )
    phone_normalization: bool = Field(
        default=True,
        description="Changes phone numbers so they can be properly pronouced by kokoro",
    )


class OpenAISpeechRequest(BaseModel):
    """Request schema for OpenAI-compatible speech endpoint"""

    backend: Optional[str] = Field(
        default=None,
        description="Explicit TTS backend identity. Omit to retain legacy model inference.",
    )
    model: str = Field(
        ...,
        min_length=1,
        description=(
            "The model to use for generation. This must be sent explicitly by the caller or its "
            "user-configured defaults. Supported choices include tts-1, tts-1-hd, kokoro, "
            "kitten_tts, KittenML/kitten-tts-nano-0.8, higgs, chatterbox, chatterbox-emotion, "
            "chatterbox-multilingual, chatterbox-turbo, and vibevoice."
        ),
    )
    input: str = Field(..., description="The text to generate audio for")
    voice: str = Field(
        default="af_heart",
        description="The voice to use for generation. Can be a base voice or a combined voice name.",
    )
    response_format: Literal["mp3", "opus", "aac", "flac", "wav", "pcm", "ogg", "webm", "ulaw"] = Field(
        default="mp3",
        description=(
            "The format to return audio in. Supported formats: mp3, opus, aac, flac, wav, pcm, ogg, webm, ulaw. "
            "PCM format returns raw 16-bit samples without headers."
        ),
    )
    output_format: Optional[Literal["mp3", "opus", "aac", "flac", "wav", "pcm", "ogg", "webm", "ulaw"]] = Field(
        default=None,
        description=(
            "Chatterbox compatibility alias for response_format. Used only for Chatterbox-family "
            "models when response_format was not provided."
        ),
    )
    download_format: Optional[Literal["mp3", "opus", "aac", "flac", "wav", "pcm", "ogg", "webm", "ulaw"]] = (
        Field(
            default=None,
            description=(
                "Reserved for future use. Currently ignored; the final audio format always matches response_format."
            ),
        )
    )
    speed: float = Field(
        default=1.0,
        ge=0.25,
        le=4.0,
        description="The speed of the generated audio. Select a value from 0.25 to 4.0.",
    )
    stream: bool = Field(
        default=True,  # Default to streaming for OpenAI compatibility
        description="If true (default), audio will be streamed as it's generated. Each chunk will be a complete sentence.",
    )
    allow_fallback: bool = Field(
        default=True,
        description="Allow the selected backend's administrator-configured fallback policy.",
    )
    target_sample_rate: Optional[int] = Field(
        default=None,
        ge=1,
        description=(
            "Optional target sample rate (Hz) for generated audio. The server honors this when "
            "the selected provider/adapter supports sample-rate control."
        ),
    )
    return_download_link: bool = Field(
        default=False,
        description="If true, returns a download link in X-Download-Path header after streaming completes",
    )
    lang_code: Optional[str] = Field(
        default=None,
        description="Optional language code to use for text processing. If not provided, will use first letter of voice name.",
    )
    language: Optional[str] = Field(
        default=None,
        description=(
            "Chatterbox compatibility alias for lang_code. Used only for Chatterbox-family "
            "models when lang_code was not provided."
        ),
    )
    normalization_options: Optional[NormalizationOptions] = Field(
        default=NormalizationOptions(),
        description="Options for the normalization system",
    )
    voice_reference: Optional[str] = Field(
        default=None,
        description=(
            "Base64-encoded audio data for voice cloning/reference. Supported by PocketTTS, NeuTTS, "
            "OmniVoice, Higgs (3-10s), Chatterbox (5-20s), VibeVoice, and IndexTTS2 models. "
            "OmniVoice cloning also requires `extra_params.reference_text`, whether the reference "
            "audio is uploaded directly here or loaded from a stored `custom:<voice_id>` voice."
        ),
    )
    reference_duration_min: Optional[float] = Field(
        default=None,
        ge=3.0,
        le=60.0,
        description="Minimum duration in seconds for voice reference audio. If provided, will validate reference audio length.",
    )
    extra_params: Optional[dict[str, Any]] = Field(
        default=None,
        description=(
            "Provider-specific parameters passed through to adapters (e.g., stability, clarity, cfg_scale). "
            "For Chatterbox, supported keys include exaggeration, cfg_weight, temperature, top_p, top_k, "
            "speed_factor, split_text, chunk_size, and the safe stored-voice alias "
            "`voice_mode='predefined'` plus `predefined_voice_id`; server-side reference_audio_filename "
            "paths are not dereferenced. "
            "For OmniVoice cloning, include `reference_text` describing the spoken transcript of the "
            "reference audio."
        ),
    )
