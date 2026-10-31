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

# Core-owned since TASK-13950.15; re-exported for the API layer.
from tldw_Server_API.app.core.Streaming.speech_chat_schemas import (  # noqa: F401 - re-exported (TASK-13950.15)
    SpeechChatLLMConfig,
    SpeechChatRequest,
    SpeechChatResponse,
    SpeechChatSTTConfig,
    SpeechChatTiming,
    SpeechChatTokenUsage,
    SpeechChatTTSConfig,
)
from tldw_Server_API.app.core.TTS.speech_schemas import (  # noqa: F401 - re-exported (TASK-13950.15)
    NormalizationOptions,
    OpenAISpeechRequest,
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


class FishS2ReferenceResponse(BaseModel):
    """Response schema for one managed Fish S2 reference."""
    model_config = ConfigDict(extra="allow")

    reference_id: str = Field(..., description="Local managed reference identifier")
    voice_id: Optional[str] = Field(default=None, description="Stored voice ID backing the reference")
    remote_reference_id: Optional[str] = Field(default=None, description="Fish-hosted reference/model ID")
    reference_text: Optional[str] = Field(default=None, description="Transcript associated with the reference")
    name: Optional[str] = Field(default=None, description="Display name for the reference")
    description: Optional[str] = Field(default=None, description="Optional reference description")
    cached: Optional[bool] = Field(default=None, description="Whether an existing remote reference was reused")


class FishS2ReferenceDeleteResponse(BaseModel):
    """Response schema for deleting one managed Fish S2 reference."""
    model_config = ConfigDict(extra="allow")

    reference_id: str = Field(..., description="Deleted local managed reference identifier")
    deleted: bool = Field(..., description="Whether the managed reference was deleted")


class FishS2ReferenceListResponse(BaseModel):
    """Response schema for listing managed Fish S2 references."""
    references: list[FishS2ReferenceResponse] = Field(default_factory=list)
    count: int = Field(..., ge=0)


class FishS2ReferenceImportErrorItem(BaseModel):
    """Per-item import failure for Fish S2 reference imports."""
    index: int = Field(..., ge=0)
    message: str = Field(..., min_length=1)
    code: Optional[str] = None


class FishS2ReferenceImportResult(BaseModel):
    """Per-item success payload for Fish S2 reference imports."""
    model_config = ConfigDict(extra="allow")

    index: int = Field(..., ge=0)
    reference_id: Optional[str] = Field(default=None, description="Local managed reference identifier")
    voice_id: Optional[str] = Field(default=None, description="Stored voice ID backing the reference")
    remote_reference_id: Optional[str] = Field(default=None, description="Fish-hosted reference/model ID")
    reference_text: Optional[str] = Field(default=None, description="Transcript associated with the reference")
    name: Optional[str] = Field(default=None, description="Display name for the reference")
    description: Optional[str] = Field(default=None, description="Optional reference description")
    cached: Optional[bool] = Field(default=None, description="Whether an existing remote reference was reused")
    result: Optional[Any] = Field(default=None, description="Fallback result payload for non-dict service responses")


class FishS2ReferenceImportResponse(BaseModel):
    """Response schema for Fish S2 reference imports."""
    results: list[FishS2ReferenceImportResult] = Field(default_factory=list)
    errors: list[FishS2ReferenceImportErrorItem] = Field(default_factory=list)
    imported: int = Field(..., ge=0)
    failed: int = Field(..., ge=0)


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
    used_month_minutes: Optional[float] = Field(
        default=None, description="Minutes already used this calendar month (UTC)."
    )
    remaining_month_minutes: Optional[float] = Field(
        default=None,
        description="Minutes remaining this month, or null if there is no monthly limit.",
    )
    active_streams: Optional[int] = Field(
        default=None,
        description="Always null: the count of the user's open streams is not reported.",
    )
    can_start_stream: bool = Field(
        ...,
        description=(
            "Always true. Not a live check: this endpoint does not report open streams, so a stream "
            "over the user's limit is refused when it starts."
        ),
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
