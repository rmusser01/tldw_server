"""Safe request validation helpers for llama.cpp asset acquisition."""

from __future__ import annotations

import hashlib
import inspect
import ipaddress
import os
import re
import socket
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import SplitResult, parse_qsl, unquote, urlencode, urlsplit, urlunsplit

import httpx

from tldw_Server_API.app.api.v1.schemas.llamacpp_admin_schemas import (
    LlamaCppAsset,
    LlamaCppAssetDownloadRequest,
)
from tldw_Server_API.app.core.Local_LLM import handler_utils, llamacpp_inventory_service
from tldw_Server_API.app.core.Local_LLM.LLM_Inference_Exceptions import ServerError
from tldw_Server_API.app.core.Setup import setup_manager
from tldw_Server_API.app.core.config import load_comprehensive_config


_DOWNLOAD_SCHEMES = {"http", "https"}
_FILENAME_DELIMITERS = {",", os.pathsep}
_SECRET_QUERY_KEYS = {
    "access_token",
    "apikey",
    "api_key",
    "auth",
    "authorization",
    "key",
    "password",
    "secret",
    "sig",
    "signature",
    "token",
    "x-amz-credential",
    "x-amz-security-token",
    "x-amz-signature",
}
_SAFE_JOB_ID_RE = re.compile(r"[^a-zA-Z0-9_.-]+")
_DEFAULT_DOWNLOAD_TIMEOUT_SECONDS = 60.0
_DEFAULT_MAX_DOWNLOAD_BYTES = 50 * 1024 * 1024 * 1024

ProgressCallback = Callable[[dict[str, Any]], Awaitable[None] | None]
CancelCheck = Callable[[], bool]
DownloadStreamFactory = Callable[..., Any]


class LlamaCppDownloadCancelled(Exception):
    """Raised when a llama.cpp acquisition download observes cancellation."""


class LlamaCppDownloadError(ServerError):
    """Retryable download transport failure for llama.cpp acquisition jobs."""


@dataclass(frozen=True)
class LlamaCppValidatedDownload:
    """Sanitized llama.cpp download request ready for job creation."""

    source_url: str
    source_label: str
    destination_path: Path
    expected_sha256: str | None = None
    expected_size_bytes: int | None = None
    overwrite: bool = False
    register_asset: bool = True
    warnings: list[str] = field(default_factory=list)


def validate_download_request(
    payload: LlamaCppAssetDownloadRequest,
    saved_config: dict[str, Any] | None = None,
) -> LlamaCppValidatedDownload:
    """Validate a remote asset download request before queueing a download job."""
    config = saved_config if saved_config is not None else _read_saved_config()
    warnings: list[str] = []
    source_url = _validate_source_url(payload.url, config, warnings)
    expected_sha256 = _normalize_sha256(payload.expected_sha256)
    destination_path = resolve_download_destination(payload, config)
    if destination_path.exists() and not payload.overwrite:
        raise ServerError("Destination file already exists; set overwrite=true to replace it.")
    return LlamaCppValidatedDownload(
        source_url=source_url,
        source_label=_source_label_for_payload(payload),
        destination_path=destination_path,
        expected_sha256=expected_sha256,
        expected_size_bytes=payload.expected_size_bytes,
        overwrite=payload.overwrite,
        register_asset=payload.register_asset,
        warnings=warnings,
    )


def validate_download_payload(
    payload: dict[str, Any],
    saved_config: dict[str, Any] | None = None,
) -> LlamaCppValidatedDownload:
    """Validate a stored acquisition job payload before the worker downloads it."""
    if not isinstance(payload, dict):
        raise ServerError("Invalid llama.cpp acquisition job payload.")
    config = saved_config if saved_config is not None else _read_saved_config()
    warnings = _string_list(payload.get("warnings"))
    source_url = _validate_payload_source_url(str(payload.get("source_url") or ""), config)
    destination_raw = str(payload.get("destination_path") or "").strip()
    if not destination_raw:
        raise ServerError("Download destination path is required.")
    destination_path = Path(destination_raw).expanduser().resolve()
    _validate_download_destination_path(destination_path, config)
    expected_size_bytes = _positive_int_or_none(payload.get("expected_size_bytes"))
    return LlamaCppValidatedDownload(
        source_url=source_url,
        source_label=_sanitize_freeform_label(
            str(payload.get("source_label") or redacted_source_label(source_url))
        ),
        destination_path=destination_path,
        expected_sha256=_normalize_sha256(_optional_str(payload.get("expected_sha256"))),
        expected_size_bytes=expected_size_bytes,
        overwrite=bool(payload.get("overwrite")),
        register_asset=bool(payload.get("register_asset", True)),
        warnings=warnings,
    )


def redacted_source_label(url: str) -> str:
    """Return a display-safe URL label without userinfo or secret query values."""
    value = str(url or "").strip()
    if not value:
        return ""
    try:
        parsed = urlsplit(value)
    except ValueError:
        return _sanitize_freeform_label(value)
    if not parsed.scheme or not parsed.netloc:
        return _sanitize_freeform_label(value)

    host = parsed.hostname or ""
    netloc = host
    if ":" in host and not host.startswith("["):
        netloc = f"[{host}]"
    if parsed.port is not None:
        netloc = f"{netloc}:{parsed.port}"
    query = urlencode(
        [
            (key, val)
            for key, val in parse_qsl(parsed.query, keep_blank_values=True)
            if not _is_secret_query_key(key)
        ],
        doseq=True,
    )
    return urlunsplit((parsed.scheme, netloc, parsed.path, query, ""))


def resolve_download_destination(payload: LlamaCppAssetDownloadRequest, saved_config: dict[str, Any]) -> Path:
    """Resolve a requested download destination under configured llama.cpp roots."""
    filename = _resolve_filename(payload)
    destination_dir = _destination_dir(payload, saved_config)
    _validate_destination_dir(destination_dir, saved_config)
    destination = destination_dir / filename
    _validate_path_value(destination, "Download destination")
    allowed_bases = _allowed_bases(saved_config)
    if not allowed_bases or not handler_utils.is_path_allowed(destination, allowed_bases):
        raise ServerError("Download destination is outside allowed llama.cpp paths.")
    return destination


def partial_download_path(final_path: Path, job_id: str) -> Path:
    """Return the partial-file path colocated with the final destination."""
    safe_job_id = _SAFE_JOB_ID_RE.sub("-", str(job_id or "").strip()).strip("-._")
    if not safe_job_id:
        safe_job_id = hashlib.sha256(str(job_id).encode("utf-8")).hexdigest()[:12]
    return final_path.with_name(f"{final_path.name}.{safe_job_id}.partial")


def validate_completed_download(
    path: Path,
    expected_sha256: str | None = None,
    expected_size_bytes: int | None = None,
) -> list[str]:
    """Validate a completed local download before asset registration."""
    if not path.exists():
        raise ServerError("Downloaded file does not exist.")
    if not path.is_file():
        raise ServerError("Downloaded path is not a file.")
    if expected_size_bytes is not None and path.stat().st_size != expected_size_bytes:
        raise ServerError("Downloaded file size did not match expected size.")
    expected_digest = _normalize_sha256(expected_sha256)
    if expected_digest is not None and _sha256_file(path) != expected_digest:
        raise ServerError("Downloaded file checksum did not match expected sha256.")
    return []


def register_completed_download(path: Path) -> LlamaCppAsset:
    """Register a completed asset through the llama.cpp inventory service."""
    return LlamaCppAsset.model_validate(llamacpp_inventory_service.register_asset_path(path))


async def download_to_partial(
    validated: LlamaCppValidatedDownload,
    partial_path: Path,
    *,
    progress_callback: ProgressCallback | None = None,
    cancel_check: CancelCheck | None = None,
    timeout_seconds: float | None = None,
    max_bytes: int | None = None,
    stream_factory: DownloadStreamFactory | None = None,
) -> int:
    """Stream a remote asset into a partial file with progress and size bounds."""
    partial_path.parent.mkdir(parents=True, exist_ok=True)
    cleanup_partial_if_needed(partial_path)
    timeout = _coerce_positive_float(timeout_seconds, _download_timeout_seconds())
    byte_limit = _download_byte_limit(validated.expected_size_bytes, max_bytes=max_bytes)
    bytes_written = 0
    total_bytes: int | None = None

    try:
        stream = await _open_download_stream(
            validated.source_url,
            timeout_seconds=timeout,
            stream_factory=stream_factory,
        )
        async with stream:
            total_bytes = _stream_total_bytes(stream) or validated.expected_size_bytes
            with partial_path.open("wb") as handle:
                async for chunk in stream.aiter_bytes():
                    if _cancel_requested(cancel_check):
                        raise LlamaCppDownloadCancelled()
                    if not chunk:
                        continue
                    bytes_written += len(chunk)
                    if byte_limit is not None and bytes_written > byte_limit:
                        raise ServerError("Downloaded file exceeded the configured maximum size.")
                    handle.write(chunk)
                    await _emit_progress(
                        progress_callback,
                        bytes_downloaded=bytes_written,
                        total_bytes=total_bytes,
                    )
        if _cancel_requested(cancel_check):
            raise LlamaCppDownloadCancelled()
        return bytes_written
    except LlamaCppDownloadCancelled:
        raise
    except ServerError:
        raise
    except (httpx.HTTPError, OSError, TimeoutError) as exc:
        raise LlamaCppDownloadError("Download failed while reading remote asset.") from exc


def promote_partial_download(partial_path: Path, final_path: Path, *, overwrite: bool = False) -> Path:
    """Atomically move a validated partial file into its final destination path."""
    if not partial_path.exists():
        raise ServerError("Partial download file does not exist.")
    if final_path.exists() and not overwrite:
        raise ServerError("Destination file already exists; set overwrite=true to replace it.")
    final_path.parent.mkdir(parents=True, exist_ok=True)
    partial_path.replace(final_path)
    return final_path


def cleanup_partial_if_needed(partial_path: Path) -> None:
    """Remove a leftover partial download file if it exists."""
    try:
        if partial_path.exists():
            partial_path.unlink()
    except OSError:
        return


def _read_saved_config() -> dict[str, Any]:
    parser = load_comprehensive_config()
    section = parser["LlamaCpp"] if parser and parser.has_section("LlamaCpp") else None
    if section is None:
        return {"allowed_paths": []}
    return {
        "models_dir": _str_or_none(section.get("models_dir", fallback=None)),
        "allowed_paths": _split_list(section.get("allowed_paths", fallback="")),
        "allow_private_downloads": _config_bool(section.get("allow_private_downloads", fallback=None)),
    }


def _validate_source_url(url: str, saved_config: dict[str, Any], warnings: list[str]) -> str:
    raw_url = str(url or "").strip()
    if not raw_url:
        raise ServerError("Download URL is required.")
    try:
        parsed = urlsplit(raw_url)
    except ValueError as exc:
        raise ServerError("Download URL is invalid.") from exc
    scheme = parsed.scheme.lower()
    if scheme not in _DOWNLOAD_SCHEMES:
        raise ServerError("Download URL scheme must be http or https.")
    if not parsed.hostname:
        raise ServerError("Download URL host is required.")
    try:
        port = parsed.port
    except ValueError as exc:
        raise ServerError("Download URL is invalid.") from exc
    if parsed.username is not None or parsed.password is not None:
        raise ServerError("Download URL credentials are not supported.")
    secret_keys = sorted(
        {
            key
            for key, _val in parse_qsl(parsed.query, keep_blank_values=True)
            if _is_secret_query_key(key)
        }
    )
    if secret_keys:
        raise ServerError(f"Download URL query contains secret parameters: {', '.join(secret_keys)}.")
    allow_private = bool(_config_bool(saved_config.get("allow_private_downloads")))
    host = parsed.hostname.strip().lower()
    if _is_local_hostname(host) and not allow_private:
        raise ServerError("Download URL host resolves to a local or private network address.")
    _validate_host_addresses(host, allow_private=allow_private, warnings=warnings)
    return _normalized_source_url(parsed, port=port)


def _validate_payload_source_url(url: str, saved_config: dict[str, Any]) -> str:
    raw_url = str(url or "").strip()
    if not raw_url:
        raise ServerError("Download URL is required.")
    try:
        parsed = urlsplit(raw_url)
        port = parsed.port
    except ValueError as exc:
        raise ServerError("Download URL is invalid.") from exc
    scheme = parsed.scheme.lower()
    if scheme not in _DOWNLOAD_SCHEMES:
        raise ServerError("Download URL scheme must be http or https.")
    if not parsed.hostname:
        raise ServerError("Download URL host is required.")
    if parsed.username is not None or parsed.password is not None:
        raise ServerError("Download URL credentials are not supported.")
    if any(_is_secret_query_key(key) for key, _val in parse_qsl(parsed.query, keep_blank_values=True)):
        raise ServerError("Download URL contains unsupported sensitive query parameters.")
    allow_private = bool(_config_bool(saved_config.get("allow_private_downloads")))
    host = parsed.hostname.strip().lower()
    if _is_local_hostname(host) and not allow_private:
        raise ServerError("Download URL host resolves to a local or private network address.")
    literal = _ip_address_or_none(host)
    if literal is not None:
        _reject_private_address(literal, allow_private=allow_private)
    return _normalized_source_url(parsed, port=port)


def _validate_host_addresses(host: str, *, allow_private: bool, warnings: list[str]) -> None:
    literal = _ip_address_or_none(host)
    if literal is not None:
        _reject_private_address(literal, allow_private=allow_private)
        return
    try:
        resolved = socket.getaddrinfo(host, None, type=socket.SOCK_STREAM)
    except OSError as exc:
        if not allow_private:
            raise ServerError(f"Could not resolve download host '{host}' to verify network safety.") from exc
        warnings.append(f"Could not resolve download host '{host}': {exc.__class__.__name__}.")
        return
    for result in resolved:
        address = result[4][0]
        candidate = _ip_address_or_none(address)
        if candidate is not None:
            _reject_private_address(candidate, allow_private=allow_private)


def _reject_private_address(address: ipaddress.IPv4Address | ipaddress.IPv6Address, *, allow_private: bool) -> None:
    if allow_private:
        return
    if (
        address.is_loopback
        or address.is_private
        or address.is_link_local
        or address.is_unspecified
        or address.is_multicast
        or address.is_reserved
    ):
        raise ServerError("Download URL host resolves to a local or private network address.")


def _ip_address_or_none(host: str) -> ipaddress.IPv4Address | ipaddress.IPv6Address | None:
    try:
        return ipaddress.ip_address(host)
    except ValueError:
        return None


def _is_local_hostname(host: str) -> bool:
    return host in {"localhost"} or host.endswith(".localhost")


def _normalized_source_url(parsed: SplitResult, *, port: int | None) -> str:
    host = parsed.hostname or ""
    netloc = host
    if ":" in host and not host.startswith("["):
        netloc = f"[{host}]"
    if port is not None:
        netloc = f"{netloc}:{port}"
    return urlunsplit((parsed.scheme.lower(), netloc, parsed.path, parsed.query, ""))


def _source_label_for_payload(payload: LlamaCppAssetDownloadRequest) -> str:
    if payload.source_label:
        return _sanitize_freeform_label(payload.source_label)
    return redacted_source_label(payload.url)


def _sanitize_freeform_label(label: str) -> str:
    value = re.sub(r"//[^/@\s]+@", "//", str(label or "").strip())
    return re.sub(r"(?i)(token|secret|password|signature|api[_-]?key)=\S+", r"\1=<redacted>", value)


def _is_secret_query_key(key: str) -> bool:
    normalized = str(key or "").strip().lower().replace("-", "_")
    compact = normalized.replace("_", "")
    return (
        normalized in _SECRET_QUERY_KEYS
        or compact in _SECRET_QUERY_KEYS
        or any(token in normalized for token in ("token", "secret", "signature", "password", "credential"))
    )


def _resolve_filename(payload: LlamaCppAssetDownloadRequest) -> str:
    raw_filename = payload.filename
    if raw_filename is None:
        raw_filename = unquote(Path(urlsplit(payload.url).path).name)
    filename = str(raw_filename or "").strip()
    if not filename:
        raise ServerError("Download filename is required.")
    if filename in {".", ".."} or filename != Path(filename).name or "/" in filename or "\\" in filename:
        raise ServerError("Download filename must be a simple filename.")
    if any(delimiter in filename for delimiter in _FILENAME_DELIMITERS):
        raise ServerError("Download filename contains unsupported delimiter characters.")
    if "\x00" in filename or "\n" in filename or "\r" in filename:
        raise ServerError("Download filename contains unsupported characters.")
    if not filename.lower().endswith(".gguf"):
        raise ServerError("Download filename must end with .gguf.")
    return filename


def _destination_dir(payload: LlamaCppAssetDownloadRequest, saved_config: dict[str, Any]) -> Path:
    raw_destination = payload.destination_dir or saved_config.get("models_dir")
    if not raw_destination:
        raise ServerError("Download destination_dir is required when models_dir is not configured.")
    destination_dir = Path(str(raw_destination)).expanduser().resolve()
    _validate_path_value(destination_dir, "Download destination directory")
    if not destination_dir.exists():
        raise ServerError("Download destination directory does not exist.")
    if not destination_dir.is_dir():
        raise ServerError("Download destination is not a directory.")
    return destination_dir


def _validate_destination_dir(destination_dir: Path, saved_config: dict[str, Any]) -> None:
    allowed_bases = _allowed_bases(saved_config)
    if not allowed_bases or not handler_utils.is_path_allowed(destination_dir, allowed_bases):
        raise ServerError("Download destination directory is outside allowed llama.cpp paths.")


def _validate_download_destination_path(destination: Path, saved_config: dict[str, Any]) -> None:
    if not destination.name:
        raise ServerError("Download destination path is required.")
    if not destination.name.lower().endswith(".gguf"):
        raise ServerError("Download destination filename must end with .gguf.")
    _validate_path_value(destination, "Download destination")
    if not destination.parent.exists():
        raise ServerError("Download destination directory does not exist.")
    if not destination.parent.is_dir():
        raise ServerError("Download destination parent is not a directory.")
    allowed_bases = _allowed_bases(saved_config)
    if not allowed_bases or not handler_utils.is_path_allowed(destination, allowed_bases):
        raise ServerError("Download destination is outside allowed llama.cpp paths.")


def _validate_path_value(path: Path, label: str) -> None:
    text = str(path)
    try:
        setup_manager.validate_config_value_single_line("LlamaCpp", "models_dir", text)
    except ValueError as exc:
        raise ServerError(f"{label} contains unsupported config characters.") from exc
    if any(delimiter in text for delimiter in _FILENAME_DELIMITERS):
        raise ServerError(f"{label} contains unsupported delimiter characters.")


def _allowed_bases(saved_config: dict[str, Any]) -> list[Path]:
    models_dir = _optional_path(saved_config.get("models_dir"))
    allowed_paths = _path_list(saved_config.get("allowed_paths"))
    return handler_utils.build_allowed_paths(models_dir, allowed_paths) if models_dir else allowed_paths


def _normalize_sha256(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip().lower()
    if not normalized:
        return None
    if len(normalized) != 64 or any(char not in "0123456789abcdef" for char in normalized):
        raise ServerError("expected_sha256 must be a 64-character hexadecimal sha256 digest.")
    return normalized


def _positive_int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        raise ServerError("expected_size_bytes must be a positive integer.") from None
    if parsed <= 0:
        raise ServerError("expected_size_bytes must be a positive integer.")
    return parsed


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _optional_path(value: Any) -> Path | None:
    if value is None:
        return None
    text = str(value).strip()
    return Path(text).expanduser() if text else None


def _path_list(value: Any) -> list[Path]:
    if not value:
        return []
    if isinstance(value, str):
        values = _split_list(value)
    else:
        values = [str(item).strip() for item in value if str(item).strip()]
    return [Path(item).expanduser() for item in values]


def _split_list(raw: str | None) -> list[str]:
    if raw is None:
        return []
    return [part.strip() for part in str(raw).replace(os.pathsep, ",").split(",") if part.strip()]


def _str_or_none(raw: str | None) -> str | None:
    if raw is None:
        return None
    value = str(raw).strip()
    return value or None


def _config_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _download_timeout_seconds() -> float:
    return _coerce_positive_float(
        os.getenv("LLAMACPP_ACQUISITION_DOWNLOAD_TIMEOUT_SECONDS"),
        _DEFAULT_DOWNLOAD_TIMEOUT_SECONDS,
    )


def _download_byte_limit(expected_size_bytes: int | None, *, max_bytes: int | None = None) -> int | None:
    configured = max_bytes
    if configured is None:
        configured = _positive_int_env("LLAMACPP_ACQUISITION_MAX_DOWNLOAD_BYTES")
    if configured is None:
        configured = _DEFAULT_MAX_DOWNLOAD_BYTES
    if expected_size_bytes is not None:
        return min(int(configured), int(expected_size_bytes))
    return int(configured) if configured else None


def _positive_int_env(name: str) -> int | None:
    value = os.getenv(name)
    if value is None or str(value).strip() == "":
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _coerce_positive_float(value: Any, default: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return float(default)
    return parsed if parsed > 0 else float(default)


async def _open_download_stream(
    url: str,
    *,
    timeout_seconds: float,
    stream_factory: DownloadStreamFactory | None,
) -> Any:
    if stream_factory is not None:
        stream = stream_factory(url, timeout_seconds=timeout_seconds)
        if inspect.isawaitable(stream):
            stream = await stream
        return stream
    return _HttpxDownloadStream(url, timeout_seconds=timeout_seconds)


class _HttpxDownloadStream:
    def __init__(self, url: str, *, timeout_seconds: float) -> None:
        self._url = url
        self._timeout_seconds = timeout_seconds
        self._client: httpx.AsyncClient | None = None
        self._response_cm: Any | None = None
        self._response: httpx.Response | None = None
        self.total_bytes: int | None = None

    async def __aenter__(self) -> "_HttpxDownloadStream":
        self._client = httpx.AsyncClient(timeout=self._timeout_seconds, follow_redirects=True)
        self._response_cm = self._client.stream("GET", self._url)
        self._response = await self._response_cm.__aenter__()
        self._response.raise_for_status()
        self.total_bytes = _content_length(self._response.headers.get("content-length"))
        return self

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> bool:
        if self._response_cm is not None:
            await self._response_cm.__aexit__(exc_type, exc, tb)
        if self._client is not None:
            await self._client.aclose()
        return False

    async def aiter_bytes(self):
        if self._response is None:
            raise LlamaCppDownloadError("Download stream was not opened.")
        async for chunk in self._response.aiter_bytes():
            yield chunk


def _content_length(value: str | None) -> int | None:
    if value is None:
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _stream_total_bytes(stream: Any) -> int | None:
    value = getattr(stream, "total_bytes", None)
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _cancel_requested(cancel_check: CancelCheck | None) -> bool:
    if cancel_check is None:
        return False
    try:
        return bool(cancel_check())
    except Exception:
        return False


async def _emit_progress(
    progress_callback: ProgressCallback | None,
    *,
    bytes_downloaded: int,
    total_bytes: int | None,
) -> None:
    if progress_callback is None:
        return
    progress_percent = None
    if total_bytes:
        progress_percent = min(95.0, max(0.0, (bytes_downloaded / total_bytes) * 95.0))
    result = progress_callback(
        {
            "bytes_downloaded": bytes_downloaded,
            "total_bytes": total_bytes,
            "progress_percent": progress_percent,
            "progress_message": "downloading",
        }
    )
    if inspect.isawaitable(result):
        await result
