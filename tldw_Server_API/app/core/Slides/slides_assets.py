"""Asset resolution helpers for Slides metadata."""

from __future__ import annotations

import base64
import json
import mimetypes
import re
from pathlib import Path
from typing import Any

from tldw_Server_API.app.services.outputs_service import _resolve_output_path_for_user

_OUTPUT_ASSET_REF_RE = re.compile(r"^output:(?P<output_id>\d+)$")
MAX_RESOLVED_SLIDE_ASSET_BYTES = 5 * 1024 * 1024


class SlidesAssetError(ValueError):
    """Raised when a slide asset reference cannot be resolved."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def parse_slide_asset_ref(asset_ref: Any) -> tuple[str, int]:
    """Validate and parse a supported slide asset reference."""

    if not isinstance(asset_ref, str):
        raise SlidesAssetError("slide_asset_ref_invalid")
    value = asset_ref.strip()
    if not value:
        raise SlidesAssetError("slide_asset_ref_invalid")
    match = _OUTPUT_ASSET_REF_RE.match(value)
    if match:
        return "output", int(match.group("output_id"))
    raise SlidesAssetError("slide_asset_ref_invalid")


def _parse_metadata_json(raw: Any) -> dict[str, Any]:
    if not raw:
        return {}
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str):
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _guess_mime_type(*, path: Path, format_hint: Any, metadata: dict[str, Any]) -> str:
    for key in ("mime", "mime_type", "content_type"):
        value = metadata.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip().lower()

    guesses = []
    if isinstance(format_hint, str) and format_hint.strip():
        guesses.append(f"asset.{format_hint.strip().lower()}")
    guesses.append(path.name)

    for candidate in guesses:
        guessed, _ = mimetypes.guess_type(candidate)
        if guessed:
            return guessed.lower()
    return "application/octet-stream"


def resolve_slide_asset(
    asset_ref: str,
    *,
    collections_db: Any | None = None,
    user_id: int | None = None,
    max_bytes: int | None = MAX_RESOLVED_SLIDE_ASSET_BYTES,
) -> dict[str, Any]:
    """Resolve a slide asset reference into file-backed bytes and metadata."""

    asset_kind, asset_id = parse_slide_asset_ref(asset_ref)
    if asset_kind != "output":
        raise SlidesAssetError("slide_asset_ref_invalid")
    if collections_db is None or user_id is None:
        raise SlidesAssetError("slide_asset_context_required")
    try:
        row = collections_db.get_output_artifact(asset_id)
    except KeyError as exc:
        raise SlidesAssetError("slide_asset_not_found") from exc

    storage_path = getattr(row, "storage_path", None)
    if not storage_path:
        raise SlidesAssetError("slide_asset_storage_missing")
    normalized_storage = collections_db.resolve_output_storage_path(storage_path)
    file_path = _resolve_output_path_for_user(int(user_id), normalized_storage)
    if not file_path.exists():
        raise SlidesAssetError("slide_asset_file_missing")
    if isinstance(max_bytes, int) and max_bytes > 0:
        if file_path.stat().st_size > max_bytes:
            raise SlidesAssetError("slide_asset_too_large")

    metadata = _parse_metadata_json(getattr(row, "metadata_json", None))
    mime = _guess_mime_type(
        path=file_path,
        format_hint=getattr(row, "format", None),
        metadata=metadata,
    )
    raw_bytes = file_path.read_bytes()
    if isinstance(max_bytes, int) and max_bytes > 0 and len(raw_bytes) > max_bytes:
        raise SlidesAssetError("slide_asset_too_large")
    return {
        "asset_ref": asset_ref,
        "mime": mime,
        "data_b64": base64.b64encode(raw_bytes).decode("ascii"),
        "title": getattr(row, "title", None),
        "type": getattr(row, "type", None),
        "format": getattr(row, "format", None),
        "storage_path": normalized_storage,
        "filename": file_path.name,
        "download_path": f"/api/v1/outputs/{asset_id}/download",
    }
