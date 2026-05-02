# audio_schemas.py
# This module defines the Pydantic schemas for audio-related data models.
#
# Imports
#
# Third-party Libraries
#
# Local Imports
#
#######################################################################################################################
from typing import Any, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field

from tldw_Server_API.app.api.v1.schemas.pagination import CursorPaginationMeta


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

    model: str = Field(
        ...,
        min_length=1,
        description=(
            "The model to use for generation. This must be sent explicitly by the caller or its "
            "user-configured defaults. Supported choices include tts-1, tts-1-hd, kokoro, "
            "kitten_tts, KittenML/kitten-tts-nano-0.8, higgs, chatterbox, and vibevoice."
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
            "For OmniVoice cloning, include `reference_text` describing the spoken transcript of the "
            "reference audio."
        ),
    )


class VoiceEncodeRequest(BaseModel):
    """Request schema for encoding stored voice references."""
    voice_id: str = Field(..., description="Stored voice ID to encode")
    provider: str = Field(default="neutts", description="Target provider for encoding artifacts")
    reference_text: Optional[str] = Field(
        default=None,
        description="Reference text associated with the stored audio (required for NeuTTS and OmniVoice)",
    )
    force: bool = Field(
        default=False,
        description="Re-encode even if artifacts already exist",
    )


class VoiceEncodeResponse(BaseModel):
    """Response schema for encoding stored voice references."""
    voice_id: str
    provider: str
    cached: bool = False
    ref_codes_len: Optional[int] = None
    reference_text: Optional[str] = None


class AudioTokenizerEncodeRequest(BaseModel):
    """Request schema for audio tokenizer encode endpoint (JSON body)."""
    audio_base64: str = Field(
        ...,
        description="Base64-encoded audio payload (no data URI prefix).",
    )
    tokenizer_model: Optional[str] = Field(
        default=None,
        description="Tokenizer model identifier (defaults to configured Qwen3 tokenizer).",
    )
    sample_rate: Optional[int] = Field(
        default=None,
        description="Optional sample rate hint when audio format does not encode it.",
    )
    token_format: Optional[Literal["list", "base64"]] = Field(
        default="list",
        description="Output token encoding: list of ints or base64-encoded bytes.",
    )


class AudioTokenizerEncodeResponse(BaseModel):
    """Response schema for audio tokenizer encode endpoint."""
    tokens: Any
    token_format: Literal["list", "base64"]
    sample_rate: int
    frame_rate: Optional[float] = None
    tokenizer_model: str
    duration_seconds: float


class AudioTokenizerDecodeRequest(BaseModel):
    """Request schema for audio tokenizer decode endpoint."""
    tokens: Any = Field(
        ...,
        description="Token payload (list[int] or base64-encoded bytes).",
    )
    tokenizer_model: Optional[str] = Field(
        default=None,
        description="Tokenizer model identifier (defaults to configured Qwen3 tokenizer).",
    )
    response_format: Literal["wav", "pcm"] = Field(
        default="wav",
        description="Desired audio output format.",
    )


class StreamingStatusFeatures(BaseModel):
    """Response schema for streaming transcription feature flags."""

    partial_results: bool = Field(True, description="Whether partial results are supported.")
    multiple_languages: bool = Field(True, description="Whether multiple languages are supported.")
    concurrent_streams: bool = Field(True, description="Whether concurrent streams are supported.")
    segment_metadata: bool = Field(True, description="Whether segment metadata is supported.")
    live_insights: bool = Field(True, description="Whether live insights are supported.")
    meeting_notes: bool = Field(True, description="Whether meeting notes are supported.")
    speaker_diarization: bool = Field(True, description="Whether speaker diarization is supported.")
    audio_persistence: bool = Field(True, description="Whether audio persistence is supported.")


class StreamingStatusResponse(BaseModel):
    """Response schema for streaming transcription availability."""

    status: Literal["available", "unavailable", "error"] = Field(
        ..., description="Availability status of streaming transcription."
    )
    available_models: list[str] = Field(
        default_factory=list,
        description="Detected streaming model variants.",
    )
    websocket_endpoint: str = Field(
        ..., description="WebSocket endpoint for streaming transcription."
    )
    supported_features: StreamingStatusFeatures = Field(
        ..., description="Feature flags for streaming transcription."
    )


class StreamingLimitsResponse(BaseModel):
    """Response schema for streaming quota and usage."""

    user_id: Union[int, str] = Field(..., description="User identifier.")
    tier: str = Field(..., description="User tier name.")
    limits: dict[str, Any] = Field(
        ..., description="Resolved limit values for the user."
    )
    used_today_minutes: float = Field(
        ..., description="Minutes already used today."
    )
    remaining_minutes: Optional[float] = Field(
        default=None,
        description="Minutes remaining today, or null if unknown/unbounded.",
    )
    active_streams: int = Field(
        ..., description="Number of currently active streams."
    )
    can_start_stream: bool = Field(
        ..., description="Whether another stream can be started."
    )
    legacy_can_start_stream: bool = Field(
        ...,
        alias="_can_start_stream",
        description="Backwards-compatible alias for can_start_stream.",
    )

    model_config = ConfigDict(populate_by_name=True)


class StreamingTestResponse(BaseModel):
    """Response schema for streaming transcription test endpoint."""

    status: Literal["success", "error"] = Field(
        ..., description="Outcome of the streaming test."
    )
    test_passed: bool = Field(..., description="Whether the test passed.")
    message: str = Field(..., description="Human-readable status message.")
    test_result: Optional[Any] = Field(
        default=None,
        description="Transcriber response or 'Buffer accumulating' when buffering.",
    )


class OpenAITranscriptionRequest(BaseModel):
    """Request schema for OpenAI-compatible transcription endpoint"""

    file: bytes = Field(..., description="The audio file to transcribe")
    model: Optional[str] = Field(
        default=None,
        description=(
            "ID of the model to use. Options: parakeet-tdt-0.6b-v3-onnx, "
            "parakeet-onnx, whisper-1, parakeet, canary, qwen2audio. "
            "Defaults to the configured STT provider when omitted."
        ),
    )
    language: Optional[str] = Field(
        default=None,
        description="The language of the input audio. Supplying the input language in ISO-639-1 format will improve accuracy and latency."
    )
    prompt: Optional[str] = Field(
        default=None,
        description="An optional text to guide the model's style or continue a previous audio segment. The prompt should match the audio language."
    )
    hotwords: Optional[list[str]] = Field(
        default=None,
        description=(
            "Optional hotwords to guide recognition. Primarily used by "
            "VibeVoice-ASR; other providers may ignore this field."
        ),
    )
    response_format: Literal["json", "text", "srt", "verbose_json", "vtt"] = Field(
        default="json",
        description="The format of the transcript output"
    )
    temperature: Optional[float] = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="The sampling temperature, between 0 and 1. Higher values make the output more random."
    )
    timestamp_granularities: Optional[list[Literal["word", "segment"]]] = Field(
        default=["segment"],
        description="The timestamp granularities to populate for this transcription"
    )


class OpenAITranscriptionResponse(BaseModel):
    """Response schema for OpenAI-compatible transcription endpoint"""

    text: str = Field(..., description="The transcribed text")
    language: Optional[str] = Field(None, description="The language of the input audio")
    duration: Optional[float] = Field(None, description="The duration of the input audio in seconds")
    words: Optional[list] = Field(None, description="Word-level timestamps if requested")
    segments: Optional[list] = Field(None, description="Segment-level timestamps if requested")


class OpenAITranslationRequest(BaseModel):
    """Request schema for OpenAI-compatible translation endpoint"""

    file: bytes = Field(..., description="The audio file to translate")
    model: Optional[str] = Field(
        default=None,
        description=(
            "ID of the model to use. Defaults to the configured STT provider when omitted "
            "(whisper-1 recommended for translations)."
        ),
    )
    prompt: Optional[str] = Field(
        default=None,
        description="An optional text to guide the model's style or continue a previous audio segment"
    )
    response_format: Literal["json", "text", "srt", "verbose_json", "vtt"] = Field(
        default="json",
        description="The format of the transcript output"
    )
    temperature: Optional[float] = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="The sampling temperature"
    )

#
# End of audio_schemas.py
#######################################################################################################################
class TranscriptUtterance(BaseModel):
    """Single utterance entry for transcript segmentation."""

    composite: str = Field(..., description="Utterance text or composite text")
    start: Optional[float] = Field(None, description="Start time in seconds")
    end: Optional[float] = Field(None, description="End time in seconds")
    speaker: Optional[str] = Field(None, description="Speaker label")
    metadata: Optional[dict[str, Any]] = Field(None, description="Arbitrary extra metadata")


class TranscriptSegmentInfo(BaseModel):
    indices: list[int]
    start_index: int
    end_index: int
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    speakers: list[str] = []
    text: str


class TranscriptSegmentationRequest(BaseModel):
    """Request schema for transcript tree segmentation."""

    entries: list[TranscriptUtterance] = Field(..., description="Transcript utterances")
    K: int = Field(6, ge=1, description="Maximum number of segments")
    min_segment_size: int = Field(5, ge=1, description="Minimum items per segment")
    lambda_balance: float = Field(0.01, ge=0.0, description="Balance penalty coefficient")
    utterance_expansion_width: int = Field(2, ge=0, description="Number of previous utterances to join per block")
    min_improvement_ratio: float = Field(0.0, ge=0.0, description="Stop splitting if relative improvement is below this threshold (0-1)")
    embeddings_provider: Optional[str] = Field(None, description="Embedding provider (if using built-in service)")
    embeddings_model: Optional[str] = Field(None, description="Embedding model (if using built-in service)")


class TranscriptSegmentationResponse(BaseModel):
    """Response schema with transitions vector and segment details."""

    transitions: list[int]
    transition_indices: list[int] = []
    segments: list[TranscriptSegmentInfo]


#######################################################################################################################
#
# Speech-to-Speech Chat (STT → LLM → TTS) Schemas


class SpeechChatSTTConfig(BaseModel):
    """Configuration options for STT in the speech chat pipeline."""

    provider: Optional[str] = Field(
        default=None,
        description="STT provider key (e.g., 'faster-whisper', 'parakeet', 'canary', 'qwen2audio'). "
                    "If omitted, the server's STT default is used.",
    )
    model: Optional[str] = Field(
        default=None,
        description="Optional STT model identifier. Semantics match existing STT configuration.",
    )
    language: Optional[str] = Field(
        default=None,
        description="Optional language code (ISO-639-1/2) to bias transcription.",
    )
    extra_params: Optional[dict[str, Any]] = Field(
        default=None,
        description="Provider-specific STT parameters passed through to underlying adapters.",
    )


class SpeechChatLLMConfig(BaseModel):
    """Configuration options for LLM in the speech chat pipeline."""

    api_provider: Optional[str] = Field(
        default=None,
        description="Target LLM provider (e.g., 'openai', 'anthropic'). If omitted, uses the server default.",
    )
    model: Optional[str] = Field(
        default=None,
        description="LLM model identifier. Required for v1; no default is inferred.",
    )
    temperature: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=2.0,
        description="Sampling temperature.",
    )
    max_tokens: Optional[int] = Field(
        default=None,
        ge=1,
        description="Maximum number of tokens to generate.",
    )
    extra_params: Optional[dict[str, Any]] = Field(
        default=None,
        description="Provider-specific LLM parameters passed through to the orchestrator.",
    )


class SpeechChatTTSConfig(BaseModel):
    """Configuration options for TTS in the speech chat pipeline."""

    provider: Optional[str] = Field(
        default=None,
        description="TTS provider hint (e.g., 'openai', 'kokoro'). If omitted, the TTS service selects a provider.",
    )
    model: Optional[str] = Field(
        default=None,
        description="Optional TTS model identifier. If omitted, the TTS provider's default is used.",
    )
    voice: Optional[str] = Field(
        default=None,
        description="Voice identifier, matching the semantics of the /audio/speech endpoint.",
    )
    response_format: Optional[Literal["mp3", "opus", "aac", "flac", "wav", "pcm"]] = Field(
        default=None,
        description="Desired audio format for the synthesized response. Defaults to 'mp3' if not provided.",
    )
    speed: Optional[float] = Field(
        default=None,
        ge=0.25,
        le=4.0,
        description="Optional speed multiplier for synthesized audio.",
    )
    extra_params: Optional[dict[str, Any]] = Field(
        default=None,
        description="Provider-specific TTS parameters passed through to adapters.",
    )


class SpeechChatRequest(BaseModel):
    """
    Request body for the non-streaming Speech-to-Speech chat endpoint.

    The client sends a single base64-encoded audio clip representing the user utterance,
    along with optional STT/LLM/TTS configuration.
    """

    session_id: Optional[str] = Field(
        default=None,
        description="Optional chat session identifier. If omitted, a new session is created.",
    )
    input_audio: str = Field(
        ...,
        description="Base64-encoded audio data for the user utterance (no data: URI prefix).",
    )
    input_audio_format: str = Field(
        ...,
        description="Declared audio format for input_audio (e.g., 'wav', 'mp3', 'ogg').",
    )
    stt_config: Optional[SpeechChatSTTConfig] = Field(
        default=None,
        description="Optional STT configuration. Defaults are used when omitted.",
    )
    llm_config: SpeechChatLLMConfig = Field(
        ...,
        description="LLM configuration for generating the assistant reply. Model is required in v1.",
    )
    tts_config: Optional[SpeechChatTTSConfig] = Field(
        default=None,
        description="Optional TTS configuration. Reasonable defaults are used when omitted.",
    )
    store_audio: Optional[bool] = Field(
        default=False,
        description="Optional hint to store raw audio alongside transcripts when enabled server-side.",
    )
    metadata: Optional[dict[str, Any]] = Field(
        default=None,
        description="Arbitrary client metadata (trace IDs, UI hints, etc.).",
    )


class SpeechChatTiming(BaseModel):
    """Timing information for each stage of the pipeline in milliseconds."""

    stt_ms: float = Field(..., description="Time spent in STT (milliseconds).")
    llm_ms: float = Field(..., description="Time spent in LLM call (milliseconds).")
    tts_ms: float = Field(..., description="Time spent in TTS synthesis (milliseconds).")


class SpeechChatTokenUsage(BaseModel):
    """Token usage summary derived from the LLM response, when available."""

    prompt_tokens: Optional[int] = Field(
        default=None,
        description="Number of prompt tokens consumed.",
    )
    completion_tokens: Optional[int] = Field(
        default=None,
        description="Number of completion tokens generated.",
    )
    total_tokens: Optional[int] = Field(
        default=None,
        description="Total tokens (prompt + completion).",
    )


class SpeechChatResponse(BaseModel):
    """
    Response body for the non-streaming Speech-to-Speech chat endpoint.

    Returns the resolved session identifier, user transcript, assistant reply text,
    and base64-encoded audio for the reply, along with timing, optional token usage,
    and an optional action_result payload when a downstream action/workflow is executed.
    """

    session_id: str = Field(..., description="Resolved chat session identifier.")
    user_transcript: str = Field(..., description="Full text transcription of the user audio turn.")
    assistant_text: str = Field(..., description="Assistant reply text produced by the LLM.")
    output_audio: str = Field(
        ...,
        description="Base64-encoded audio data for the assistant reply (no data: URI prefix).",
    )
    output_audio_mime_type: str = Field(
        ...,
        description="MIME type corresponding to output_audio (e.g., 'audio/mpeg', 'audio/wav').",
    )
    timing: SpeechChatTiming = Field(
        ...,
        description="Timing information for STT, LLM, and TTS stages.",
    )
    token_usage: Optional[SpeechChatTokenUsage] = Field(
        default=None,
        description="Optional token usage summary from the LLM response.",
    )
    metadata: Optional[dict[str, Any]] = Field(
        default=None,
        description="Optional metadata echo or server-side annotations.",
    )
    action_result: Optional[dict[str, Any]] = Field(
        default=None,
        description="Optional action/workflow execution result derived from the transcript.",
    )


class TTSHistoryListItem(BaseModel):
    """List item for TTS history."""

    id: int
    created_at: str
    has_text: bool
    text_preview: Optional[str] = None
    provider: Optional[str] = None
    model: Optional[str] = None
    voice_id: Optional[str] = None
    voice_name: Optional[str] = None
    voice_info: Optional[dict[str, Any]] = None
    duration_ms: Optional[int] = None
    format: Optional[str] = None
    status: Optional[str] = None
    favorite: bool = False
    job_id: Optional[int] = None
    output_id: Optional[int] = None
    artifact_deleted_at: Optional[str] = None


class TTSHistoryListResponse(BaseModel):
    """Response schema for listing TTS history."""

    items: list[TTSHistoryListItem]
    total: Optional[int] = None
    limit: int
    offset: int
    next_cursor: Optional[str] = None
    pagination: CursorPaginationMeta


class TTSHistoryDetailResponse(BaseModel):
    """Detail schema for a single TTS history entry."""

    id: int
    created_at: str
    has_text: bool
    text: Optional[str] = None
    text_length: Optional[int] = None
    provider: Optional[str] = None
    model: Optional[str] = None
    voice_id: Optional[str] = None
    voice_name: Optional[str] = None
    voice_info: Optional[dict[str, Any]] = None
    format: Optional[str] = None
    duration_ms: Optional[int] = None
    generation_time_ms: Optional[int] = None
    params_json: Optional[dict[str, Any]] = None
    status: Optional[str] = None
    segments_json: Optional[dict[str, Any]] = None
    favorite: bool = False
    job_id: Optional[int] = None
    output_id: Optional[int] = None
    artifact_ids: Optional[list[Any]] = None
    artifact_deleted_at: Optional[str] = None
    error_message: Optional[str] = None


class TTSHistoryFavoriteUpdate(BaseModel):
    """Update favorite status for a history entry."""

    favorite: bool = Field(..., description="Whether the entry is favorited.")
