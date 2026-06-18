from __future__ import annotations

import hashlib
import mimetypes
from pathlib import Path
from typing import Any


MARKDOWN_MEDIA_TYPE = "text/markdown"

FEATURE_REGISTRY: set[str] = {
    "content_envelopes",
    "file_inventory",
    "integrity_metadata",
    "typed_source_refs",
    "representations",
    "lossiness_metadata",
    "schema_refs",
    "redaction_profiles",
    "external_rehydration",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def ensure_known_features(features: list[str]) -> dict[str, list[str]]:
    supported: list[str] = []
    unsupported: list[str] = []

    for feature in features:
        if feature in FEATURE_REGISTRY:
            supported.append(feature)
        else:
            unsupported.append(feature)

    return {
        "supported": supported,
        "unsupported": unsupported,
    }


def build_preview_report(manifest: Any, extract_dir: Path) -> dict[str, Any]:
    """Build a non-blocking v1.1 compatibility and integrity preview report."""
    manifest_version = _version_value(getattr(manifest, "version", None))
    features = ensure_known_features(_normalize_feature_tokens(getattr(manifest, "features_used", [])))
    failed_files: list[dict[str, Any]] = []
    verified_files = 0

    inventory = getattr(manifest, "file_inventory", []) or []
    if not isinstance(inventory, list):
        failed_files.append({"path": None, "reason": "invalid_inventory"})
        inventory = []

    for entry in inventory:
        if not isinstance(entry, dict):
            failed_files.append({"path": None, "reason": "invalid_entry"})
            continue

        relative_path = entry.get("path")
        if not isinstance(relative_path, str) or not relative_path:
            failed_files.append({"path": None, "reason": "invalid_path"})
            continue
        target_path = extract_dir / relative_path
        try:
            target_path.resolve().relative_to(extract_dir.resolve())
        except ValueError:
            failed_files.append({"path": relative_path, "reason": "unsafe_path"})
            continue
        if not target_path.is_file():
            failed_files.append({"path": relative_path, "reason": "missing_file"})
            continue

        expected_hash = _inventory_hash_value(entry)
        if not expected_hash or not _is_sha256_integrity(expected_hash):
            failed_files.append({"path": relative_path, "reason": "invalid_integrity"})
            continue

        actual_hash = sha256_file(target_path)
        if actual_hash != expected_hash:
            failed_files.append({"path": relative_path, "reason": "hash_mismatch"})
            continue
        verified_files += 1

    lossiness: dict[str, int] = {}
    source_refs: dict[str, int] = {}
    for content_item in getattr(manifest, "content_items", []) or []:
        envelope = _content_item_envelope(content_item)
        if not envelope:
            continue

        mode = _lossiness_mode(envelope)
        lossiness[mode] = lossiness.get(mode, 0) + 1

        refs = envelope.get("source_refs", []) if isinstance(envelope, dict) else []
        if not isinstance(refs, list):
            continue
        for source_ref in refs:
            status = "unknown"
            if isinstance(source_ref, dict):
                status_value = source_ref.get("resolution_status")
                if isinstance(status_value, str) and status_value:
                    status = status_value
            source_refs[status] = source_refs.get(status, 0) + 1

    warnings: list[str] = []
    errors: list[str] = []
    if features["unsupported"]:
        warnings.append("Unsupported chatbook features detected")
    if failed_files:
        errors.append("File inventory integrity failures detected")

    return {
        "compatibility": {
            "status": "compatible" if not features["unsupported"] else "partial",
            "reader_version": "1.1.0",
            "manifest_version": manifest_version,
        },
        "features": features,
        "integrity": {
            "verified_files": verified_files,
            "failed_files": failed_files,
        },
        "lossiness": lossiness,
        "source_refs": source_refs,
        "warnings": warnings,
        "errors": errors,
    }


def validate_v1_1_before_import(manifest: Any, extract_dir: Path) -> tuple[bool, list[str], list[str]]:
    """Validate v1.1 compatibility and integrity before any import writes."""
    report = build_preview_report(manifest, extract_dir)
    warnings = list(report.get("warnings") or [])
    errors: list[str] = []

    features_report = report.get("features") if isinstance(report, dict) else {}
    if not isinstance(features_report, dict):
        features_report = {}
    unsupported_features = _string_list(features_report.get("unsupported"))
    unsupported_behavior = _unsupported_feature_behavior(getattr(manifest, "compatibility", {}))

    if unsupported_features:
        feature_list = ", ".join(unsupported_features)
        if unsupported_behavior == "reject_import":
            errors.append(
                "Unsupported chatbook features rejected before import: "
                f"{feature_list}"
            )
        else:
            _append_unique(
                warnings,
                "Unsupported chatbook features detected before import "
                f"({unsupported_behavior}): {feature_list}",
            )

    integrity_report = report.get("integrity") if isinstance(report, dict) else {}
    failed_files: list[Any] = []
    if isinstance(integrity_report, dict):
        raw_failed_files = integrity_report.get("failed_files")
        if isinstance(raw_failed_files, list):
            failed_files = raw_failed_files
    for failed_file in failed_files:
        errors.append(_inventory_failure_error(failed_file))

    for report_error in report.get("errors") or []:
        if isinstance(report_error, str):
            _append_unique(errors, report_error)

    return not errors, warnings, errors


def build_content_envelope(
    *,
    format_id: str,
    schema_version: int | str,
    media_type: str,
    structured_path: str,
    integrity_value: str | None,
    lossiness_mode: str = "lossless",
    rendered: list[dict[str, Any]] | None = None,
    source_refs: list[dict[str, Any]] | None = None,
    redaction_profile: dict[str, Any] | None = None,
) -> dict[str, Any]:
    representations = [
        {
            "kind": "structured",
            "path": structured_path,
            "media_type": media_type,
            "primary": True,
            "role": "restore_payload",
        }
    ]
    representations.extend(rendered or [])

    return {
        "format": format_id,
        "schema_version": schema_version,
        "media_type": media_type,
        "representations": representations,
        "integrity": {
            "status": "verified" if integrity_value else "unsupported",
            "algorithm": "sha256" if integrity_value else None,
            "value": integrity_value,
            "scope": "primary_payload",
        },
        "lossiness": {"mode": lossiness_mode, "reasons": []},
        "provenance": {},
        "source_refs": source_refs or [],
        "attachments": [],
        "redaction_profile": redaction_profile,
    }


def build_file_inventory(work_dir: Path) -> list[dict[str, Any]]:
    inventory: list[dict[str, Any]] = []

    for path in sorted(
        work_dir.rglob("*"),
        key=lambda item: item.relative_to(work_dir).as_posix(),
    ):
        if not path.is_file():
            continue

        relative_path = path.relative_to(work_dir).as_posix()
        if relative_path == "manifest.json" or path.name.endswith(".sha256"):
            continue

        media_type = _guess_media_type(relative_path)
        inventory.append(
            {
                "path": relative_path,
                "media_type": media_type,
                "size_bytes": path.stat().st_size,
                "integrity": {
                    "status": "verified",
                    "algorithm": "sha256",
                    "value": sha256_file(path),
                },
                "role": _role_for_path(relative_path),
                "content_item_ids": [],
            }
        )

    return inventory


def _guess_media_type(relative_path: str) -> str:
    media_type, _ = mimetypes.guess_type(relative_path)
    if media_type:
        return media_type
    if Path(relative_path).suffix.lower() in {".md", ".markdown"}:
        return MARKDOWN_MEDIA_TYPE
    return "application/octet-stream"


def _role_for_path(relative_path: str) -> str:
    if relative_path == "README.md":
        return "readme"
    if relative_path.startswith("rendered/"):
        return "rendered"
    if relative_path.startswith("schemas/"):
        return "schema"
    if relative_path.startswith("content/"):
        return "payload"
    return "payload"


def _version_value(version: Any) -> str | None:
    if version is None:
        return None
    value = getattr(version, "value", version)
    return str(value)


def _normalize_feature_tokens(features: Any) -> list[str]:
    if features is None:
        return []
    if not isinstance(features, list):
        return [_stringify_malformed_token(features)]
    return [
        feature if isinstance(feature, str) else _stringify_malformed_token(feature)
        for feature in features
    ]


def _stringify_malformed_token(value: Any) -> str:
    try:
        return str(value)
    except Exception:
        return f"<malformed:{type(value).__name__}>"


def _inventory_hash_value(entry: dict[str, Any]) -> str | None:
    integrity = entry.get("integrity")
    if isinstance(integrity, dict):
        value = integrity.get("value")
        return value if isinstance(value, str) else None
    return None


def _is_sha256_integrity(value: str) -> bool:
    prefix = "sha256:"
    if not value.startswith(prefix):
        return False
    digest = value[len(prefix):]
    return len(digest) == 64 and all(char in "0123456789abcdefABCDEF" for char in digest)


def _content_item_envelope(content_item: Any) -> dict[str, Any] | None:
    metadata = getattr(content_item, "metadata", None)
    if isinstance(metadata, dict):
        envelope = metadata.get("envelope")
        return envelope if isinstance(envelope, dict) else None
    if isinstance(content_item, dict):
        item_metadata = content_item.get("metadata")
        if isinstance(item_metadata, dict):
            envelope = item_metadata.get("envelope")
            return envelope if isinstance(envelope, dict) else None
    return None


def _lossiness_mode(envelope: dict[str, Any]) -> str:
    lossiness = envelope.get("lossiness")
    if isinstance(lossiness, dict):
        mode = lossiness.get("mode")
        if isinstance(mode, str) and mode:
            return mode
    return "unknown"


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str)]


def _unsupported_feature_behavior(compatibility: Any) -> str:
    if isinstance(compatibility, dict):
        behavior = compatibility.get("unsupported_feature_behavior")
        if behavior in {"warn_and_skip", "warn_lossy_import", "reject_import"}:
            return behavior
    return "warn_and_skip"


def _append_unique(items: list[str], value: str) -> None:
    if value not in items:
        items.append(value)


def _inventory_failure_error(failed_file: Any) -> str:
    path = None
    reason = "invalid_entry"
    if isinstance(failed_file, dict):
        path = failed_file.get("path")
        raw_reason = failed_file.get("reason")
        if isinstance(raw_reason, str) and raw_reason:
            reason = raw_reason
    path_label = path if isinstance(path, str) and path else "<unknown>"
    reason_label = reason.replace("_", " ")
    if reason == "hash_mismatch":
        return f"Checksum validation failed for {path_label}: hash mismatch"
    return f"File inventory validation failed for {path_label}: {reason_label}"
