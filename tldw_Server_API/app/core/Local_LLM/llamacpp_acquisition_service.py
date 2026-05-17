"""Safe request validation helpers for llama.cpp asset acquisition."""

from __future__ import annotations

import hashlib
import ipaddress
import os
import re
import socket
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, unquote, urlencode, urlsplit, urlunsplit

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
    """Validate a remote asset download request without performing network I/O."""
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
    if expected_size_bytes is not None and path.stat().st_size != int(expected_size_bytes):
        raise ServerError("Downloaded file size did not match expected size.")
    expected_digest = _normalize_sha256(expected_sha256)
    if expected_digest is not None and _sha256_file(path) != expected_digest:
        raise ServerError("Downloaded file checksum did not match expected sha256.")
    return []


def register_completed_download(path: Path) -> LlamaCppAsset:
    """Register a completed asset through the llama.cpp inventory service."""
    return LlamaCppAsset.model_validate(llamacpp_inventory_service.register_asset_path(path))


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
    allow_private = bool(_config_bool(saved_config.get("allow_private_downloads")))
    host = parsed.hostname.strip().lower()
    if _is_local_hostname(host) and not allow_private:
        raise ServerError("Download URL host resolves to a local or private network address.")
    _validate_host_addresses(host, allow_private=allow_private, warnings=warnings)
    return redacted_source_label(raw_url)


def _validate_host_addresses(host: str, *, allow_private: bool, warnings: list[str]) -> None:
    literal = _ip_address_or_none(host)
    if literal is not None:
        _reject_private_address(literal, allow_private=allow_private)
        return
    try:
        resolved = socket.getaddrinfo(host, None, type=socket.SOCK_STREAM)
    except OSError as exc:
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
