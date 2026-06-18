"""Secret-safe Workspace runtime binding descriptor helpers."""
from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from pathlib import PurePath, PureWindowsPath
from typing import Any

from loguru import logger

from tldw_Server_API.app.core.Workspaces.models import (
    WORKSPACE_RUNTIME_BINDING_KINDS,
    WORKSPACE_RUNTIME_BINDING_OWNER_DOMAINS,
    WORKSPACE_RUNTIME_BINDING_PORTABILITY,
    WORKSPACE_RUNTIME_BINDING_STATUSES,
)

WORKSPACE_RUNTIME_BINDING_MAX_METADATA_BYTES = 16 * 1024
WORKSPACE_RUNTIME_BINDING_MAX_REDACTION_BYTES = 8 * 1024

_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
_STATUS_ALIASES = {
    "inspect_only": "inspect-only",
    "runtime_missing": "runtime-missing",
}
_PORTABILITY_ALIASES = {
    "metadata_only": "metadata-only",
    "local_only": "local-only",
}
_SECRET_KEY_RE = re.compile(
    r"(^|[_\-.])("
    r"api[_\-.]?key|access[_\-.]?key|secret|token|password|passwd|pwd|"
    r"credential|credentials|private[_\-.]?key|client[_\-.]?secret|"
    r"bearer|authorization|auth[_\-.]?header|env|environment"
    r")($|[_\-.])",
    re.IGNORECASE,
)
_SECRET_VALUE_RE = re.compile(
    r"(-----BEGIN [A-Z ]*PRIVATE KEY-----|"
    r"\bsk-[A-Za-z0-9_\-]{12,}|"
    r"\b(?:ghp|github_pat|xox[baprs])_[A-Za-z0-9_\-]{12,})",
    re.IGNORECASE,
)
_PATH_METADATA_KEY_RE = re.compile(
    r"(^|[_\-.])("
    r"absolute[_\-.]?root|root|paths?|mount[_\-.]?path|workspace[_\-.]?path|"
    r"repo[_\-.]?path|local[_\-.]?path|project[_\-.]?root|dir|directory"
    r")($|[_\-.])",
    re.IGNORECASE,
)


def _input_error(message: str) -> ValueError:
    from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import InputError

    return InputError(message)


def normalize_runtime_binding_payload(data: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize and sanitize a runtime binding descriptor for storage or response."""
    binding_id = _required_identifier(data.get("binding_id"), "binding_id")
    binding_kind = _enum_value(
        data.get("binding_kind"),
        "binding_kind",
        WORKSPACE_RUNTIME_BINDING_KINDS,
    )
    owner_domain = _enum_value(
        data.get("owner_domain"),
        "owner_domain",
        WORKSPACE_RUNTIME_BINDING_OWNER_DOMAINS,
    )
    locator_ref = _required_string(data.get("locator_ref"), "locator_ref", max_length=512)
    label = _optional_string(data.get("label"), "label", max_length=512)
    status = _enum_value(
        data.get("status"),
        "status",
        WORKSPACE_RUNTIME_BINDING_STATUSES,
        aliases=_STATUS_ALIASES,
    )
    portability = _enum_value(
        data.get("portability"),
        "portability",
        WORKSPACE_RUNTIME_BINDING_PORTABILITY,
        aliases=_PORTABILITY_ALIASES,
    )
    raw_path_hint = _optional_string(data.get("path_hint"), "path_hint", max_length=1024)
    path_hint = redacted_path_hint(raw_path_hint) if raw_path_hint is not None else None
    metadata = normalize_runtime_binding_json_object(
        data.get("metadata", data.get("metadata_json")),
        field_name="metadata",
        max_bytes=WORKSPACE_RUNTIME_BINDING_MAX_METADATA_BYTES,
        reject_secrets=True,
    )
    metadata, metadata_redacted_fields = _redact_path_like_metadata(metadata, "metadata")
    metadata, metadata_json = dump_runtime_binding_json_object(
        metadata,
        field_name="metadata",
        max_bytes=WORKSPACE_RUNTIME_BINDING_MAX_METADATA_BYTES,
        reject_secrets=True,
    )
    redaction_report = _redaction_report(data.get("redaction_report", data.get("redaction_report_json")))
    redacted_fields = list(redaction_report.get("redacted_fields") or [])
    for field_name in metadata_redacted_fields:
        if field_name not in redacted_fields:
            redacted_fields.append(field_name)
    if raw_path_hint is not None and path_hint != raw_path_hint and "path_hint" not in redacted_fields:
        redacted_fields.append("path_hint")
    redaction_report["redacted_fields"] = redacted_fields
    redaction_report["redacted"] = bool(redacted_fields or redaction_report.get("rejected_fields"))
    redaction_report, redaction_report_json = dump_runtime_binding_json_object(
        redaction_report,
        field_name="redaction_report",
        max_bytes=WORKSPACE_RUNTIME_BINDING_MAX_REDACTION_BYTES,
        reject_secrets=True,
    )

    return {
        "binding_id": binding_id,
        "binding_kind": binding_kind,
        "owner_domain": owner_domain,
        "locator_ref": locator_ref,
        "label": label,
        "status": status,
        "path_hint": path_hint,
        "portability": portability,
        "metadata": metadata,
        "metadata_json": metadata_json,
        "redaction_report": redaction_report,
        "redaction_report_json": redaction_report_json,
    }


def runtime_binding_response_payload(row: Mapping[str, Any]) -> dict[str, Any]:
    """Return a public runtime binding response payload from a DB row-like mapping."""
    item = dict(row)
    item["metadata"] = load_runtime_binding_json_object(
        item.get("metadata", item.get("metadata_json")),
        field_name="metadata_json",
    )
    item["redaction_report"] = load_runtime_binding_json_object(
        item.get("redaction_report", item.get("redaction_report_json")),
        field_name="redaction_report_json",
    ) or _default_redaction_report()
    if item.get("path_hint"):
        item["path_hint"] = redacted_path_hint(item["path_hint"])
    item["deleted"] = bool(item.get("deleted", False))
    item["version"] = int(item.get("version") or 1)
    return item


def redacted_path_hint(value: Any) -> str:
    """Reduce absolute or segmented paths to a safe display hint."""
    raw_value = str(value or "").strip()
    if not raw_value:
        return "project_root"
    windows_path = PureWindowsPath(raw_value)
    if raw_value.startswith(("/", "~", "\\\\")) or windows_path.is_absolute():
        if windows_path.is_absolute() or raw_value.startswith("\\\\"):
            return windows_path.name or "project_root"
        return PurePath(raw_value).name or "project_root"
    if "/" in raw_value or "\\" in raw_value:
        return windows_path.name or PurePath(raw_value).name or "project_root"
    return raw_value


def dump_runtime_binding_json_object(
    value: Any,
    *,
    field_name: str,
    max_bytes: int,
    reject_secrets: bool,
) -> tuple[dict[str, Any], str]:
    """Normalize, validate, and serialize a bounded descriptor JSON object."""
    normalized = normalize_runtime_binding_json_object(
        value,
        field_name=field_name,
        max_bytes=max_bytes,
        reject_secrets=reject_secrets,
    )
    try:
        dumped = json.dumps(
            normalized,
            ensure_ascii=True,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise _input_error(f"{field_name} must be JSON serializable.") from exc
    if len(dumped.encode("utf-8")) > max_bytes:
        raise _input_error(f"{field_name} exceeds {max_bytes} bytes.")
    return normalized, dumped


def normalize_runtime_binding_json_object(
    value: Any,
    *,
    field_name: str,
    max_bytes: int,
    reject_secrets: bool,
) -> dict[str, Any]:
    if value is None:
        normalized: dict[str, Any] = {}
    elif isinstance(value, str):
        if not value.strip():
            normalized = {}
        else:
            try:
                parsed = json.loads(value)
            except json.JSONDecodeError as exc:
                raise _input_error(f"{field_name} must be valid JSON.") from exc
            if not isinstance(parsed, Mapping):
                raise _input_error(f"{field_name} must be a JSON object.")
            normalized = dict(parsed)
    elif isinstance(value, Mapping):
        normalized = dict(value)
    else:
        raise _input_error(f"{field_name} must be a JSON object.")

    if reject_secrets:
        _reject_secret_like_json(normalized, field_name)
    try:
        dumped = json.dumps(
            normalized,
            ensure_ascii=True,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise _input_error(f"{field_name} must be JSON serializable.") from exc
    if len(dumped.encode("utf-8")) > max_bytes:
        raise _input_error(f"{field_name} exceeds {max_bytes} bytes.")
    return normalized


def load_runtime_binding_json_object(raw: Any, *, field_name: str) -> dict[str, Any]:
    """Load descriptor JSON with warning on corruption instead of silent loss."""
    if raw is None:
        return {}
    if isinstance(raw, Mapping):
        return dict(raw)
    if isinstance(raw, str):
        if not raw.strip():
            return {}
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            preview = raw[:120].replace("\n", "\\n")
            logger.warning(
                "Failed to decode workspace runtime binding JSON field {} ({} chars): {}",
                field_name,
                len(raw),
                preview,
            )
            return {}
        return dict(parsed) if isinstance(parsed, Mapping) else {}
    return {}


def _redaction_report(value: Any) -> dict[str, Any]:
    report = normalize_runtime_binding_json_object(
        value,
        field_name="redaction_report",
        max_bytes=WORKSPACE_RUNTIME_BINDING_MAX_REDACTION_BYTES,
        reject_secrets=True,
    )
    if not report:
        return _default_redaction_report()
    return {
        "redacted": bool(report.get("redacted", False)),
        "redacted_fields": _string_list(report.get("redacted_fields")),
        "rejected_fields": _string_list(report.get("rejected_fields")),
    }


def _default_redaction_report() -> dict[str, Any]:
    return {"redacted": False, "redacted_fields": [], "rejected_fields": []}


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        result: list[str] = []
        for item in value:
            normalized = str(item or "").strip()
            if normalized and normalized not in result:
                result.append(normalized)
        return result
    return []


def _reject_secret_like_json(value: Any, field_path: str) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            key_text = str(key)
            child_path = f"{field_path}.{key_text}"
            if _SECRET_KEY_RE.search(key_text):
                raise _input_error(f"{field_path} contains secret-looking field '{child_path}'.")
            _reject_secret_like_json(item, child_path)
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, item in enumerate(value):
            _reject_secret_like_json(item, f"{field_path}[{index}]")
        return
    if isinstance(value, str) and _SECRET_VALUE_RE.search(value):
        raise _input_error(f"{field_path} contains a secret-looking value.")


def _redact_path_like_metadata(
    value: Any,
    field_path: str,
    *,
    path_context: bool = False,
) -> tuple[Any, list[str]]:
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        redacted_fields: list[str] = []
        for key, item in value.items():
            key_text = str(key)
            child_path = f"{field_path}.{key_text}"
            redacted_item, child_fields = _redact_path_like_metadata(
                item,
                child_path,
                path_context=path_context or bool(_PATH_METADATA_KEY_RE.search(key_text)),
            )
            result[key_text] = redacted_item
            redacted_fields.extend(child_fields)
        return result, redacted_fields
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        result_list: list[Any] = []
        redacted_fields = []
        for index, item in enumerate(value):
            redacted_item, child_fields = _redact_path_like_metadata(
                item,
                f"{field_path}[{index}]",
                path_context=path_context,
            )
            result_list.append(redacted_item)
            redacted_fields.extend(child_fields)
        return result_list, redacted_fields
    if path_context and isinstance(value, str):
        redacted = redacted_path_hint(value)
        if redacted != value:
            return redacted, [field_path]
    return value, []


def _required_identifier(value: Any, field_name: str) -> str:
    normalized = _required_string(value, field_name, max_length=128)
    if not _IDENTIFIER_RE.fullmatch(normalized):
        raise _input_error(f"{field_name} contains unsupported characters.")
    return normalized


def _required_string(value: Any, field_name: str, *, max_length: int) -> str:
    normalized = str(value if value is not None else "").strip()
    if not normalized:
        raise _input_error(f"{field_name} is required.")
    if len(normalized) > max_length:
        raise _input_error(f"{field_name} exceeds {max_length} characters.")
    return normalized


def _optional_string(value: Any, field_name: str, *, max_length: int) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    if not normalized:
        return None
    if len(normalized) > max_length:
        raise _input_error(f"{field_name} exceeds {max_length} characters.")
    return normalized


def _enum_value(
    value: Any,
    field_name: str,
    allowed: frozenset[str],
    *,
    aliases: Mapping[str, str] | None = None,
) -> str:
    normalized = str(value if value is not None else "").strip().lower()
    if aliases:
        normalized = aliases.get(normalized, normalized)
    if normalized not in allowed:
        allowed_values = ", ".join(sorted(allowed))
        raise _input_error(f"{field_name} must be one of: {allowed_values}.")
    return normalized
