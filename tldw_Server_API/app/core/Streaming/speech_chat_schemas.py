"""Core-owned speech-chat models, re-exported by the API schemas package (TASK-13950.15)."""

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


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
