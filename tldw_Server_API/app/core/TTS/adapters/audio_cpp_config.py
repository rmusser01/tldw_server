"""Configuration helpers for the audio.cpp TTS provider."""

from __future__ import annotations

import ipaddress
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from ..tts_exceptions import TTSValidationError

PROVIDER_KEY = "audio_cpp"
_SCALAR_TYPES = (str, int, float, bool)


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    return bool(value)


def _is_loopback_host(host: str | None) -> bool:
    normalized = str(host or "").strip().lower()
    if not normalized:
        return False
    if normalized == "localhost":
        return True
    try:
        return ipaddress.ip_address(normalized).is_loopback
    except ValueError:
        return False


def validate_base_url(base_url: str | None, *, allow_remote_base_url: bool = False) -> str:
    """Normalize and validate an audiocpp_server base URL."""
    normalized = str(base_url or "").strip().rstrip("/")
    parsed = urlparse(normalized)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise TTSValidationError(
            "audio.cpp base_url must be an absolute http(s) URL",
            provider=PROVIDER_KEY,
            error_code="invalid_base_url",
        )
    if not allow_remote_base_url and not _is_loopback_host(parsed.hostname):
        raise TTSValidationError(
            "audio.cpp base_url must be loopback unless allow_remote_base_url is enabled",
            provider=PROVIDER_KEY,
            error_code="remote_base_url_disabled",
        )
    return normalized


def validate_managed_host(host: str | None) -> str:
    """Normalize managed sidecar bind host and reject non-loopback addresses."""
    normalized = str(host or "127.0.0.1").strip().lower()
    if normalized == "localhost":
        return "127.0.0.1"
    try:
        parsed = ipaddress.ip_address(normalized)
    except ValueError as exc:
        raise TTSValidationError(
            "audio.cpp managed sidecar host must be a loopback address",
            provider=PROVIDER_KEY,
            error_code="invalid_managed_host",
        ) from exc
    if not parsed.is_loopback:
        raise TTSValidationError(
            "audio.cpp managed sidecar host must be a loopback address",
            provider=PROVIDER_KEY,
            error_code="invalid_managed_host",
        )
    return "127.0.0.1" if parsed.version == 4 else "::1"


def filter_request_options(
    options: dict[str, Any] | None,
    *,
    allowlist: tuple[str, ...] | list[str] | set[str],
) -> tuple[dict[str, Any], dict[str, str]]:
    """Return allowlisted scalar options and reasons for ignored options."""
    allowed = {str(item) for item in allowlist}
    filtered: dict[str, Any] = {}
    ignored: dict[str, str] = {}
    for key, value in (options or {}).items():
        normalized_key = str(key)
        if normalized_key not in allowed:
            ignored[normalized_key] = "not_allowlisted"
            continue
        if value is None:
            ignored[normalized_key] = "none_value"
            continue
        if not isinstance(value, _SCALAR_TYPES):
            ignored[normalized_key] = "non_scalar"
            continue
        filtered[normalized_key] = value
    return filtered, ignored


@dataclass(frozen=True)
class AudioCppConfig:
    """Parsed audio.cpp provider configuration."""

    base_url: str
    model: str
    model_path: str | None
    timeout: int
    managed: bool
    allow_remote_base_url: bool
    external_voice_reference_mode: str
    retain_request_artifacts: bool
    request_option_allowlist: tuple[str, ...]
    server: dict[str, Any] = field(default_factory=dict)
    repo_root: Path = field(default_factory=Path.cwd)

    @classmethod
    def from_provider_config(
        cls,
        config: dict[str, Any],
        *,
        repo_root: Path | None = None,
    ) -> AudioCppConfig:
        extra_params = dict(config.get("extra_params") or {})
        allow_remote = _as_bool(extra_params.get("allow_remote_base_url"), default=False)
        mode = str(extra_params.get("external_voice_reference_mode") or "disabled").strip().lower()
        if mode not in {"disabled", "shared_path"}:
            raise TTSValidationError(
                "audio.cpp external_voice_reference_mode must be disabled or shared_path",
                provider=PROVIDER_KEY,
                error_code="invalid_voice_reference_mode",
            )
        allowlist = extra_params.get("request_option_allowlist") or ("max_tokens", "seed")
        return cls(
            base_url=validate_base_url(
                config.get("base_url") or "http://127.0.0.1:8080",
                allow_remote_base_url=allow_remote,
            ),
            model=str(config.get("model") or "audio-cpp/pocket-tts"),
            model_path=config.get("model_path"),
            timeout=int(config.get("timeout") or 300),
            managed=_as_bool(extra_params.get("managed"), default=False),
            allow_remote_base_url=allow_remote,
            external_voice_reference_mode=mode,
            retain_request_artifacts=_as_bool(extra_params.get("retain_request_artifacts"), default=False),
            request_option_allowlist=tuple(str(item) for item in allowlist),
            server=dict(extra_params.get("server") or {}),
            repo_root=Path(repo_root or Path.cwd()).resolve(strict=False),
        )

    @property
    def models_root(self) -> Path:
        return self._resolve_repo_path(self.server.get("models_root") or "models/audio_cpp")

    @property
    def shared_scratch_dir(self) -> Path:
        configured = self.server.get("shared_scratch_dir") or "models/audio_cpp/runtime/scratch"
        scratch_dir = self._resolve_repo_path(configured)
        self._ensure_within(scratch_dir, self.models_root, "shared_scratch_dir")
        return scratch_dir

    def _resolve_repo_path(self, value: Any) -> Path:
        path = Path(str(value)).expanduser()
        if not path.is_absolute():
            path = self.repo_root / path
        return path.resolve(strict=False)

    def _ensure_within(self, path: Path, root: Path, field_name: str) -> None:
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise TTSValidationError(
                f"audio.cpp {field_name} must resolve under {root.name}",
                provider=PROVIDER_KEY,
                error_code="path_outside_audio_cpp_root",
            ) from exc

    def render_server_config(self) -> dict[str, Any]:
        """Render a single-model upstream server config dictionary."""
        host = validate_managed_host(self.server.get("host"))
        model_config = dict(self.server.get("model") or {})
        configured_model_path = model_config.get("path") or self.model_path
        model_path = self._resolve_repo_path(configured_model_path)
        self._ensure_within(model_path, self.models_root, "model.path")

        model_entry = {
            "id": str(model_config.get("id") or "pocket-tts"),
            "family": str(model_config.get("family") or "pocket_tts"),
            "path": str(model_path),
            "task": str(model_config.get("task") or "tts"),
            "mode": str(model_config.get("mode") or "offline"),
            "load_options": dict(model_config.get("load_options") or {}),
            "session_options": dict(model_config.get("session_options") or {}),
        }
        return {
            "host": host,
            "port": int(self.server.get("port") or 8080),
            "lazy_load": _as_bool(self.server.get("lazy_load"), default=True),
            "device": self.server.get("device", 0),
            "threads": int(self.server.get("threads") or 1),
            "models_root": str(self.models_root),
            "shared_scratch_dir": str(self.shared_scratch_dir),
            "models": [model_entry],
        }

    def build_reference_scratch_path(self, original_filename: str | None = None) -> Path:
        """Build a server-readable WAV scratch path without using user filenames."""
        return self.shared_scratch_dir / f"voice_ref_{uuid.uuid4().hex}.wav"
