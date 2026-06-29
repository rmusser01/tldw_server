"""Import helpers for Fish S2 managed reference files."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


class FishS2ReferenceImportError(ValueError):
    """Raised when a Fish S2 reference import file cannot be parsed."""


@dataclass(frozen=True)
class FishS2ReferenceImportItem:
    """Normalized Fish S2 reference import item."""

    voice_id: str | None = None
    reference_text: str | None = None
    name: str | None = None
    description: str | None = None
    filename: str | None = None
    audio_base64: str | None = None
    force: bool | None = None
    source_index: int = 0


@dataclass(frozen=True)
class FishS2ReferenceImportIssue:
    """Indexed validation issue for an import item."""

    index: int
    message: str


@dataclass(frozen=True)
class FishS2ReferenceImportParseResult:
    """Detailed Fish S2 reference import parse result."""

    items: list[FishS2ReferenceImportItem]
    errors: list[FishS2ReferenceImportIssue]


SUPPORTED_FISH_S2_IMPORT_EXTENSIONS = {".json", ".md", ".markdown"}
FISH_S2_REFERENCE_IMPORT_MAX_BYTES = 75 * 1024 * 1024
FISH_S2_REFERENCE_IMPORT_MAX_ITEMS = 25
FISH_S2_REFERENCE_IMPORT_MAX_DECODED_AUDIO_BYTES = 50 * 1024 * 1024


def parse_fish_s2_reference_import(
    *,
    filename: str,
    content: bytes,
) -> list[FishS2ReferenceImportItem]:
    """Parse JSON or Markdown Fish S2 reference imports into normalized items."""
    result = parse_fish_s2_reference_import_result(filename=filename, content=content)
    if result.errors:
        first_error = result.errors[0]
        raise FishS2ReferenceImportError(first_error.message)
    return result.items


def parse_fish_s2_reference_import_result(
    *,
    filename: str,
    content: bytes,
    max_items: int = FISH_S2_REFERENCE_IMPORT_MAX_ITEMS,
    max_bytes: int = FISH_S2_REFERENCE_IMPORT_MAX_BYTES,
) -> FishS2ReferenceImportParseResult:
    """Parse a Fish S2 reference import and collect indexed item errors."""
    if len(content) > max_bytes:
        raise FishS2ReferenceImportError(
            f"Fish S2 import file exceeds the maximum size of {max_bytes} bytes"
        )

    suffix = Path(filename or "").suffix.lower()
    if suffix not in SUPPORTED_FISH_S2_IMPORT_EXTENSIONS:
        raise FishS2ReferenceImportError(
            "Unsupported Fish S2 import file type. Use .json, .md, or .markdown."
        )

    try:
        text = content.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise FishS2ReferenceImportError("Fish S2 import file must be UTF-8 text") from exc

    if suffix == ".json":
        records = _load_json_records(text)
    else:
        records = [_load_markdown_record(text)]

    if not records:
        raise FishS2ReferenceImportError("Fish S2 import file contains no references")
    if len(records) > max_items:
        raise FishS2ReferenceImportError(
            f"Fish S2 import file contains {len(records)} references; at most {max_items} are allowed"
        )

    items: list[FishS2ReferenceImportItem] = []
    errors: list[FishS2ReferenceImportIssue] = []
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            errors.append(FishS2ReferenceImportIssue(index=index, message="Each Fish S2 JSON import item must be an object"))
            continue
        try:
            items.append(_normalize_record(record, index))
        except FishS2ReferenceImportError as exc:
            errors.append(FishS2ReferenceImportIssue(index=index, message=str(exc)))
    return FishS2ReferenceImportParseResult(items=items, errors=errors)


def _load_json_records(text: str) -> list[dict[str, Any]]:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise FishS2ReferenceImportError(f"Invalid Fish S2 JSON import: {exc.msg}") from exc

    if isinstance(payload, list):
        records = payload
    elif isinstance(payload, dict):
        nested = payload.get("references")
        records = nested if isinstance(nested, list) else [payload]
    else:
        raise FishS2ReferenceImportError("Fish S2 JSON import must be an object or array")

    return records


def _load_markdown_record(text: str) -> dict[str, Any]:
    metadata, body = _split_markdown_frontmatter(text)
    record = dict(metadata)
    if body and not _first_text(record, "reference_text", "text", "transcript"):
        record["reference_text"] = body
    return record


def _split_markdown_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    lines = text.splitlines(keepends=True)
    if lines and lines[0].strip() == "---":
        closing_index = None
        for index, line in enumerate(lines[1:], start=1):
            if line.strip() == "---":
                closing_index = index
                break
        if closing_index is None:
            raise FishS2ReferenceImportError("Markdown frontmatter is missing a closing --- marker")

        frontmatter = "".join(lines[1:closing_index])
        body = "".join(lines[closing_index + 1 :]).strip()
        try:
            metadata = yaml.safe_load(frontmatter) or {}
        except yaml.YAMLError as exc:
            raise FishS2ReferenceImportError("Invalid Markdown frontmatter for Fish S2 import") from exc
        if not isinstance(metadata, dict):
            raise FishS2ReferenceImportError("Markdown frontmatter must be a mapping")
        return metadata, body

    return {}, text.strip()


def _normalize_record(record: dict[str, Any], source_index: int) -> FishS2ReferenceImportItem:
    voice_id = _first_text(record, "voice_id", "reference_id")
    audio_base64 = _first_text(record, "audio_base64", "audio_b64")
    reference_text = _first_text(record, "reference_text", "text", "transcript")
    filename = _first_text(record, "filename", "audio_filename")
    name = _first_text(record, "name", "title")
    description = _first_text(record, "description")
    force = _optional_bool(record.get("force"))

    if voice_id and audio_base64:
        raise FishS2ReferenceImportError("Provide either voice_id or audio_base64, not both")
    if not voice_id and not audio_base64:
        raise FishS2ReferenceImportError("voice_id or audio_base64 is required for Fish S2 imports")
    if audio_base64 and not (filename and name and reference_text):
        raise FishS2ReferenceImportError(
            "filename, name, and reference_text are required when audio_base64 is provided"
        )

    return FishS2ReferenceImportItem(
        voice_id=voice_id,
        reference_text=reference_text,
        name=name,
        description=description,
        filename=filename,
        audio_base64=audio_base64,
        force=force,
        source_index=source_index,
    )


def _first_text(record: dict[str, Any], *keys: str) -> str | None:
    for key in keys:
        value = record.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return None


def _optional_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    raise FishS2ReferenceImportError("force must be a boolean when provided")
