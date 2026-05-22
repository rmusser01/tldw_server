from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


X_TLDW_SIDECAR_TOKEN_HEADER = "X-TLDW-Sidecar-" + "Token"


class OmniVoiceSidecarError(BaseModel):
    """Structured error envelope returned by the managed sidecar."""

    model_config = ConfigDict(extra="forbid")

    code: str = Field(..., min_length=1)
    message: str = Field(..., min_length=1)
    retryable: bool = False


class OmniVoiceRuntimeStatus(BaseModel):
    """Runtime status reported by the managed OmniVoice sidecar."""

    model_config = ConfigDict(extra="forbid")

    status: str = "idle_stopped"
    ready: bool = False
    provider: str = "omnivoice"
    runtime: str = "sidecar"
    model: str | None = None
    model_path: str | None = None
    sample_rate: int = 24000
    last_error_code: str | None = None


class OmniVoiceGenerationParams(BaseModel):
    """Allowlisted generation controls accepted by the sidecar."""

    model_config = ConfigDict(extra="forbid")

    num_step: int | None = Field(default=None, ge=1, le=128)
    guidance_scale: float | None = Field(default=None, ge=0.0, le=30.0)
    denoise: bool | None = None
    t_shift: float | None = None
    position_temperature: float | None = Field(default=None, ge=0.0, le=10.0)
    class_temperature: float | None = Field(default=None, ge=0.0, le=10.0)
    layer_penalty_factor: float | None = Field(default=None, ge=0.0, le=10.0)
    duration: float | None = Field(default=None, gt=0.0)
    speed: float | None = Field(default=None, gt=0.0, le=4.0)
    postprocess_output: bool | None = None
    preprocess_prompt: bool | None = None
    audio_chunk_duration: float | None = Field(default=None, gt=0.0)
    audio_chunk_threshold: float | None = Field(default=None, gt=0.0)

    def compact(self) -> dict[str, Any]:
        """Return only generation parameters explicitly supplied by the caller."""
        return self.model_dump(exclude_none=True)


class OmniVoiceSynthesizeRequest(BaseModel):
    """Internal sidecar request envelope for OmniVoice synthesis."""

    model_config = ConfigDict(extra="forbid")

    text: str = Field(..., min_length=1)
    mode: Literal["auto", "design", "clone"] = "auto"
    voice: str | None = None
    instruct: str | None = None
    language_id: str | None = None
    reference_audio_path: str | None = None
    reference_text: str | None = None
    requested_sample_rate: int | None = Field(default=None, ge=1)
    generation: OmniVoiceGenerationParams = Field(default_factory=OmniVoiceGenerationParams)

    @model_validator(mode="after")
    def validate_mode_inputs(self) -> "OmniVoiceSynthesizeRequest":
        if self.mode == "auto" and (self.instruct or self.reference_audio_path):
            raise ValueError("mode=auto cannot include instruct or reference_audio_path")
        if self.mode == "design":
            if not (self.instruct and self.instruct.strip()):
                raise ValueError("instruct is required for mode=design")
            if self.reference_audio_path:
                raise ValueError("mode=design cannot include reference_audio_path")
        if self.mode == "clone":
            if self.instruct:
                raise ValueError("mode=clone cannot include instruct")
            if not self.reference_audio_path:
                raise ValueError("reference_audio_path is required for mode=clone")
            if not (self.reference_text and self.reference_text.strip()):
                raise ValueError("reference_text is required for mode=clone")
        return self


class OmniVoiceSynthesizeResponse(BaseModel):
    """Metadata returned alongside the native WAV payload."""

    model_config = ConfigDict(extra="forbid")

    audio_format: str = "wav"
    content_type: str = "audio/wav"
    sample_rate: int = 24000
    channels: int = 1
    provider: str = "omnivoice"
    mode: str = "auto"


class OmniVoiceHealthResponse(BaseModel):
    """Minimal runtime health/status envelope for supervisor readiness checks."""

    model_config = ConfigDict(extra="forbid")

    status: str = "ok"
    ready: bool = True
    provider: str = "omnivoice"
    runtime: str = "sidecar"


def build_sidecar_auth_headers(sidecar_token: str) -> dict[str, str]:
    """Return the internal auth header required by the OmniVoice sidecar."""
    return {X_TLDW_SIDECAR_TOKEN_HEADER: sidecar_token}
