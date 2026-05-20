"""Pydantic schemas for reusable TTS/STT audio presets."""

from __future__ import annotations

import re
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

AudioPresetKind = Literal["tts", "stt", "speech"]

_SECRET_KEY_NORMALIZER_RE = re.compile(r"[^a-z0-9]")


def _normalize_secret_key(value: Any) -> str:
    return _SECRET_KEY_NORMALIZER_RE.sub("", str(value or "").strip().lower())


_SECRET_CONFIG_KEY_ALIASES = frozenset(
    {
        "access_token",
        "api_key",
        "apikey",
        "apiKey",
        "authorization",
        "auth_token",
        "authToken",
        "bearer",
        "client_secret",
        "clientSecret",
        "oauth_token",
        "oauthToken",
        "password",
        "private_key",
        "refresh_token",
        "secret",
        "token",
    }
)
_SECRET_CONFIG_KEYS = frozenset(_normalize_secret_key(key) for key in _SECRET_CONFIG_KEY_ALIASES)


def _contains_secret_key(value: Any) -> bool:
    if isinstance(value, dict):
        for key, child in value.items():
            if _normalize_secret_key(key) in _SECRET_CONFIG_KEYS:
                return True
            if _contains_secret_key(child):
                return True
    if isinstance(value, list):
        return any(_contains_secret_key(item) for item in value)
    return False


def _strip_text(value: Any) -> Any:
    if isinstance(value, str):
        return value.strip()
    return value


class AudioPresetBase(BaseModel):
    """Shared preset fields accepted by create/update schemas."""

    model_config = ConfigDict(extra="forbid")

    name: Optional[str] = Field(default=None, min_length=1, max_length=120)
    description: Optional[str] = Field(default=None, max_length=500)
    favorite: Optional[bool] = None
    is_default: Optional[bool] = None
    config: Optional[dict[str, Any]] = None
    capability_assumptions: Optional[dict[str, Any]] = None

    @field_validator("name", "description", mode="before")
    @classmethod
    def _strip_strings(cls, value: Any) -> Any:
        return _strip_text(value)

    @field_validator("config")
    @classmethod
    def _reject_secret_config_keys(cls, value: Optional[dict[str, Any]]) -> Optional[dict[str, Any]]:
        if value is not None and _contains_secret_key(value):
            raise ValueError("Preset config must not contain provider credentials or secret keys.")
        return value


class AudioPresetCreateRequest(AudioPresetBase):
    """Create a reusable audio preset."""

    kind: AudioPresetKind
    name: str = Field(..., min_length=1, max_length=120)
    favorite: bool = False
    is_default: bool = False
    config: dict[str, Any] = Field(default_factory=dict)
    capability_assumptions: dict[str, Any] = Field(default_factory=dict)


class AudioPresetUpdateRequest(AudioPresetBase):
    """Patch mutable fields on a reusable audio preset."""

    @model_validator(mode="after")
    def _require_one_field(self) -> "AudioPresetUpdateRequest":
        fields_set = getattr(self, "model_fields_set", None)
        if fields_set is None:
            fields_set = getattr(self, "__fields_set__", set())
        if not fields_set:
            raise ValueError("At least one preset field must be supplied.")
        return self


class AudioPresetResponse(BaseModel):
    """Reusable audio preset returned by the API."""

    id: str
    owner_user_id: str
    kind: AudioPresetKind
    name: str
    description: Optional[str] = None
    favorite: bool = False
    is_default: bool = False
    config: dict[str, Any] = Field(default_factory=dict)
    capability_assumptions: dict[str, Any] = Field(default_factory=dict)
    created_at: str
    updated_at: str


class AudioPresetListResponse(BaseModel):
    """List response for audio presets."""

    items: list[AudioPresetResponse]
    total: int
    limit: int
    offset: int


class AudioPresetValidationWarning(BaseModel):
    """A non-blocking warning produced while validating a preset."""

    code: str
    message: str
    field: Optional[str] = None


class AudioPresetValidationResponse(BaseModel):
    """Validation result for a saved preset."""

    preset: AudioPresetResponse
    valid: bool
    warnings: list[AudioPresetValidationWarning] = Field(default_factory=list)
