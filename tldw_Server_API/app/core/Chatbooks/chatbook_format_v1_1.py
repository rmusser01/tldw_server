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
