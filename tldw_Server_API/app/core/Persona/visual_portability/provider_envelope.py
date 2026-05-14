"""Normalize external Persona Visual provider envelopes for review intake.

External MCP-compatible Persona Visual pack providers are untrusted review-input
sources. This module implements the pure intake boundary for their result
envelopes: it validates contract invariants, sanitizes bounded metadata, and
returns stable diagnostics without retrieving MCP resources, writing assets,
queuing jobs, or activating packs.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from itertools import islice
from typing import Any


PROVIDER_CONTRACT_VERSION = 1
CANONICAL_PERSONA_VISUAL_ARCHIVE_MEDIA_TYPE = "application/vnd.tldw.persona.visual-pack+zip"
COMPATIBLE_PERSONA_VISUAL_ARCHIVE_MEDIA_TYPES = {
    CANONICAL_PERSONA_VISUAL_ARCHIVE_MEDIA_TYPE,
    "application/zip",
}
ALLOWED_PROVIDER_RESULT_TYPES = frozenset(
    {
        "portable_archive",
        "generated_candidate",
        "manifest_patch",
        "draft_pack_request",
    }
)

_MAX_TEXT_LENGTH = 500
_MAX_INPUT_TEXT_LENGTH = 10_000
_MAX_INT_TEXT_LENGTH = 10
_MAX_COLLECTION_ITEMS = 64
_MAX_NESTING_DEPTH = 5
_SAFE_CODE_RE = re.compile(r"^[a-z][a-z0-9_:-]{0,79}$")
_INTEGER_TEXT_RE = re.compile(r"^[+-]?\d+$")
_SECRET_VALUE_RE = re.compile(
    r"(sk-[A-Za-z0-9_-]{6,}|xox[baprs]-|gh[pousr]_[A-Za-z0-9_]{8,}|"
    r"bearer\s+[A-Za-z0-9._-]{8,}|api[_-]?key|secret|token|session[_-]?cookie)",
    re.IGNORECASE,
)
_UNSAFE_PATH_RE = re.compile(
    r"(^/|^[A-Za-z]:[\\/]|\.\.|~[/\\]|file://|localhost|127\.0\.0\.1|::1)",
    re.IGNORECASE,
)
_SENSITIVE_KEY_PARTS = (
    "api_key",
    "apikey",
    "authorization",
    "bearer",
    "cookie",
    "local_path",
    "password",
    "secret",
    "session",
    "token",
)


def normalize_provider_result_envelope(raw: Any) -> dict[str, Any]:
    """Return a sanitized provider result envelope with fail-closed diagnostics."""
    blockers: list[dict[str, str]] = []
    warnings: list[dict[str, str]] = []

    if not isinstance(raw, Mapping):
        return _normalized_base(
            blockers=[_diagnostic("invalid_envelope", "Provider result envelope must be an object.")],
            warnings=[],
        )

    contract_version = _safe_int(raw.get("contract_version"))
    result_type = _safe_text(raw.get("result_type"), max_length=80)
    review_required = raw.get("review_required") is True
    activation_allowed = raw.get("activation_allowed") is True
    import_preview_required = raw.get("import_preview_required") is True

    if contract_version != PROVIDER_CONTRACT_VERSION:
        blockers.append(_diagnostic("unsupported_contract_version", "Unsupported provider contract version."))
    if result_type not in ALLOWED_PROVIDER_RESULT_TYPES:
        blockers.append(_diagnostic("unsupported_result_type", "Unsupported provider result type."))
    if not review_required:
        blockers.append(_diagnostic("review_required_missing", "Provider output must require review."))
    if activation_allowed:
        blockers.append(_diagnostic("activation_not_allowed", "Provider output cannot activate visual packs."))
    if result_type == "portable_archive" and not import_preview_required:
        blockers.append(
            _diagnostic(
                "import_preview_required_missing",
                "Portable archive provider output must require import preview.",
            )
        )

    diagnostics, diagnostics_valid = _normalize_diagnostics(raw.get("diagnostics"))
    if not diagnostics_valid:
        blockers.append(_diagnostic("malformed_diagnostics", "Diagnostics must use code/message objects."))

    provider_blockers = list(diagnostics["blockers"])
    provider_warnings = list(diagnostics["warnings"])
    blockers.extend(provider_blockers)
    warnings.extend(provider_warnings)

    unsafe_found = False
    provider, provider_unsafe = _sanitize_section(raw.get("provider"), section_name="provider")
    pack, pack_unsafe = _sanitize_section(raw.get("pack"), section_name="pack")
    provenance, provenance_unsafe = _sanitize_section(raw.get("provenance"), section_name="provenance")
    payload, payload_unsafe = _sanitize_section(raw.get("payload"), section_name="payload")
    unsafe_found = provider_unsafe or pack_unsafe or provenance_unsafe or payload_unsafe
    if unsafe_found:
        blockers.append(
            _diagnostic(
                "unsafe_provider_metadata",
                "Provider metadata contains unsafe or sensitive values.",
            )
        )

    if result_type == "portable_archive":
        _validate_portable_archive_payload(
            payload=payload,
            diagnostics_status=str(diagnostics.get("status") or ""),
            blockers=blockers,
        )

    normalized = _normalized_base(
        contract_version=contract_version or PROVIDER_CONTRACT_VERSION,
        result_type=result_type,
        review_required=review_required,
        activation_allowed=False,
        import_preview_required=import_preview_required,
        provider=provider,
        pack=pack,
        diagnostics={
            "status": diagnostics["status"],
            "blockers": _dedupe_diagnostics(blockers),
            "warnings": _dedupe_diagnostics(warnings),
        },
        provenance=provenance,
        payload=payload,
        blockers=_dedupe_diagnostics(blockers),
        warnings=_dedupe_diagnostics(warnings),
    )
    normalized["commit_eligible"] = not normalized["blockers"]
    return normalized


def _validate_portable_archive_payload(
    *,
    payload: Any,
    diagnostics_status: str,
    blockers: list[dict[str, str]],
) -> None:
    """Add portable archive blockers for missing or unsupported archive payloads."""
    if payload is None:
        if diagnostics_status.startswith("blocked") and blockers:
            return
        blockers.append(_diagnostic("archive_payload_missing", "Portable archive payload is required."))
        return
    if not isinstance(payload, Mapping) or not isinstance(payload.get("archive"), Mapping):
        blockers.append(_diagnostic("archive_payload_missing", "Portable archive payload is required."))
        return
    archive = payload["archive"]
    media_type = _safe_text(archive.get("media_type"), max_length=120)
    if media_type not in COMPATIBLE_PERSONA_VISUAL_ARCHIVE_MEDIA_TYPES:
        blockers.append(
            _diagnostic(
                "unsupported_archive_media_type",
                "Portable archive media type must be a supported zip payload type.",
            )
        )


def _normalize_diagnostics(value: Any) -> tuple[dict[str, Any], bool]:
    """Return normalized diagnostic status, blockers, and warnings."""
    if not isinstance(value, Mapping):
        return {"status": "unknown", "blockers": [], "warnings": []}, False

    valid = True
    status = _safe_text(value.get("status"), max_length=120) or "unknown"
    blockers, blockers_valid = _diagnostic_list(value.get("blockers"))
    warnings, warnings_valid = _diagnostic_list(value.get("warnings"))
    valid = valid and blockers_valid and warnings_valid
    return {"status": status, "blockers": blockers, "warnings": warnings}, valid


def _diagnostic_list(value: Any) -> tuple[list[dict[str, str]], bool]:
    """Normalize a provider diagnostic list and report malformed entries."""
    if value is None:
        return [], True
    if not isinstance(value, list):
        return [], False

    out: list[dict[str, str]] = []
    valid = True
    for item in value[:_MAX_COLLECTION_ITEMS]:
        if not isinstance(item, Mapping):
            valid = False
            continue
        code = _safe_code(item.get("code"))
        message = _safe_text(item.get("message"), max_length=300)
        if not code or not message:
            valid = False
            continue
        out.append(_diagnostic(code, message))
    if len(value) > _MAX_COLLECTION_ITEMS:
        valid = False
    return out, valid


def _sanitize_section(value: Any, *, section_name: str) -> tuple[Any, bool]:
    """Return a bounded metadata copy and whether unsafe text was removed."""
    if value is None:
        return None, False
    return _sanitize_value(value, section_name=section_name, depth=0)


def _sanitize_value(value: Any, *, section_name: str, depth: int) -> tuple[Any, bool]:
    """Recursively copy safe JSON-like provider metadata."""
    if depth > _MAX_NESTING_DEPTH:
        return None, True
    if isinstance(value, Mapping):
        out: dict[str, Any] = {}
        unsafe = len(value) > _MAX_COLLECTION_ITEMS
        for raw_key, raw_item in islice(value.items(), _MAX_COLLECTION_ITEMS):
            key = _safe_text(raw_key, max_length=80)
            if not key:
                unsafe = True
                continue
            item, item_unsafe = _sanitize_value(raw_item, section_name=section_name, depth=depth + 1)
            sensitive_key = _is_sensitive_key(key)
            unsafe = unsafe or item_unsafe or sensitive_key
            if sensitive_key:
                continue
            out[key] = item
        return out, unsafe
    if isinstance(value, list):
        out_items: list[Any] = []
        unsafe = False
        for raw_item in value[:_MAX_COLLECTION_ITEMS]:
            item, item_unsafe = _sanitize_value(raw_item, section_name=section_name, depth=depth + 1)
            unsafe = unsafe or item_unsafe
            out_items.append(item)
        if len(value) > _MAX_COLLECTION_ITEMS:
            unsafe = True
        return out_items, unsafe
    if isinstance(value, str):
        raw_text, input_too_long = _bounded_input_text(value)
        if input_too_long or _is_unsafe_text(raw_text, section_name=section_name):
            return "[redacted]", True
        text = _safe_text(raw_text, max_length=_MAX_TEXT_LENGTH)
        return text, False
    if isinstance(value, bool) or value is None:
        return value, False
    if isinstance(value, int | float):
        return value, False
    return _safe_text(value, max_length=_MAX_TEXT_LENGTH), False


def _is_sensitive_key(key: str) -> bool:
    """Return whether a metadata key name implies secret material."""
    normalized = key.strip().lower().replace("-", "_")
    return any(part in normalized for part in _SENSITIVE_KEY_PARTS)


def _is_unsafe_text(text: str, *, section_name: str) -> bool:
    """Return whether a text value looks unsafe for trace-safe review metadata."""
    if not text:
        return False
    if _SECRET_VALUE_RE.search(text):
        return True
    if _UNSAFE_PATH_RE.search(text):
        return True
    if text.startswith(("http://", "https://", "data:")):
        return True
    return False


def _safe_text(value: Any, *, max_length: int) -> str:
    """Return a stripped bounded text value."""
    if value is None:
        return ""
    raw_text = value if isinstance(value, str) else str(value)
    text, _ = _bounded_input_text(raw_text)
    if len(text) > max_length:
        return text[:max_length]
    return text


def _safe_code(value: Any) -> str:
    """Return a stable diagnostic code or an empty string."""
    text = _safe_text(value, max_length=80).lower()
    if not _SAFE_CODE_RE.match(text):
        return ""
    return text


def _safe_int(value: Any) -> int | None:
    """Return an integer value when coercion is unambiguous."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        if len(value) > _MAX_INT_TEXT_LENGTH:
            return None
        value = value.strip()
        if not _INTEGER_TEXT_RE.fullmatch(value):
            return None
        return int(value)
    return None


def _bounded_input_text(value: str) -> tuple[str, bool]:
    """Return bounded text for safety scans and whether input exceeded the limit."""
    input_too_long = len(value) > _MAX_INPUT_TEXT_LENGTH
    bounded = value[:_MAX_INPUT_TEXT_LENGTH] if input_too_long else value
    return bounded.strip(), input_too_long


def _diagnostic(code: str, message: str) -> dict[str, str]:
    """Build a stable machine-readable diagnostic object."""
    safe_code = _safe_code(code) or "invalid_diagnostic"
    safe_message = _safe_text(message, max_length=300) or "Provider envelope validation failed."
    return {"code": safe_code, "message": safe_message}


def _dedupe_diagnostics(items: Sequence[Mapping[str, str]]) -> list[dict[str, str]]:
    """Return diagnostics in first-seen order without duplicate codes."""
    seen: set[str] = set()
    out: list[dict[str, str]] = []
    for item in items:
        code = _safe_code(item.get("code"))
        if not code or code in seen:
            continue
        seen.add(code)
        out.append(_diagnostic(code, item.get("message", "")))
    return out


def _normalized_base(
    *,
    contract_version: int = PROVIDER_CONTRACT_VERSION,
    result_type: str = "",
    review_required: bool = False,
    activation_allowed: bool = False,
    import_preview_required: bool = False,
    provider: Any = None,
    pack: Any = None,
    diagnostics: Mapping[str, Any] | None = None,
    provenance: Any = None,
    payload: Any = None,
    blockers: Sequence[Mapping[str, str]] = (),
    warnings: Sequence[Mapping[str, str]] = (),
) -> dict[str, Any]:
    """Build the normalized envelope shape returned by this module."""
    normalized_blockers = _dedupe_diagnostics(blockers)
    normalized_warnings = _dedupe_diagnostics(warnings)
    diagnostic_payload = dict(diagnostics or {})
    diagnostic_payload.setdefault("status", "blocked")
    diagnostic_payload["blockers"] = normalized_blockers
    diagnostic_payload["warnings"] = normalized_warnings
    return {
        "contract_version": contract_version,
        "result_type": result_type,
        "review_required": review_required,
        "activation_allowed": activation_allowed,
        "import_preview_required": import_preview_required,
        "provider": provider,
        "pack": pack,
        "diagnostics": diagnostic_payload,
        "provenance": provenance,
        "payload": payload,
        "commit_eligible": not normalized_blockers,
        "blockers": normalized_blockers,
        "warnings": normalized_warnings,
    }
