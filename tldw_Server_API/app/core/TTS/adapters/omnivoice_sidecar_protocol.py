from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator


X_TLDW_SIDECAR_TOKEN_HEADER = "X-TLDW-Sidecar-" + "Token"


class OmniVoiceSynthesizeRequest(BaseModel):
    """Minimal internal sidecar request envelope for OmniVoice synthesis."""

    model_config = ConfigDict(extra="forbid")

    text: str = Field(..., min_length=1)
    mode: str = Field(default="auto", min_length=1)
    voice: str | None = None
    reference_audio_path: str | None = None
    reference_text: str | None = None
    sample_rate: int = Field(default=24000, ge=8000, le=192000)

    @model_validator(mode="after")
    def validate_clone_inputs(self) -> "OmniVoiceSynthesizeRequest":
        if self.mode == "clone":
            if not self.reference_audio_path:
                raise ValueError("reference_audio_path is required for clone mode")
            if not self.reference_text:
                raise ValueError("reference_text is required for clone mode")
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
