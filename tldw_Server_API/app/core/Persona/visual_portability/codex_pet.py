"""Codex pet archive adapter for Persona Visual import flows.

Petdex/Codex pets are distributed as a small JSON manifest plus a single
spritesheet atlas. This adapter validates that archive shape without extracting
files, translates the fixed 8x9 Codex atlas rows into Persona Visual sprite-frame
states, and keeps the original package importable through the same preview and
commit pipeline used by native persona visual packs.
"""

from __future__ import annotations

import io
import json
import zipfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from loguru import logger
from PIL import Image, UnidentifiedImageError

from tldw_Server_API.app.core.Persona.visuals import validate_visual_manifest

from .archive import (
    DEFAULT_MAX_ARCHIVE_SIZE_BYTES,
    DEFAULT_MAX_MEMBER_SIZE_BYTES,
    normalize_member_name,
)
from .constants import ASSET_BYTES_STATUS_PRESENT
from .fingerprints import sha256_bytes


CODEX_PET_SCHEMA_VERSION = "codex.pet.v1"
CODEX_PET_ASSET_ID = "codex-pet-spritesheet"
CODEX_PET_FRAME_WIDTH = 192
CODEX_PET_FRAME_HEIGHT = 208
CODEX_PET_COLUMNS = 8
CODEX_PET_ROWS = 9
CODEX_PET_SHEET_WIDTH = CODEX_PET_FRAME_WIDTH * CODEX_PET_COLUMNS
CODEX_PET_SHEET_HEIGHT = CODEX_PET_FRAME_HEIGHT * CODEX_PET_ROWS
CODEX_PET_FRAME_DURATION_MS = 120

_MANIFEST_FILENAMES = frozenset({"pet.json", "petjson.json"})
_SUPPORTED_SPRITE_MIME_TYPES = {
    ".png": "image/png",
    ".webp": "image/webp",
}
_IMAGE_VALIDATION_ERRORS = (
    OSError,
    ValueError,
    UnidentifiedImageError,
)


@dataclass(frozen=True)
class CodexPetArchivePayload:
    """Normalized Codex pet metadata ready for preview or import commit."""

    manifest: dict[str, Any]
    pack: dict[str, Any]
    assets: list[dict[str, Any]]
    asset_files: dict[str, bytes]


@dataclass(frozen=True)
class _CodexPetSprite:
    path: str
    content: bytes
    mime_type: str
    width: int
    height: int


def is_codex_pet_archive(archive_path: Path) -> bool:
    """Return whether the ZIP appears to contain a Codex pet manifest."""
    try:
        with zipfile.ZipFile(archive_path, "r") as archive:
            return any(
                not info.is_dir()
                and PurePosixPath(normalize_member_name(info.filename)).name
                in _MANIFEST_FILENAMES
                for info in archive.infolist()
            )
    except (ValueError, zipfile.BadZipFile):
        return False


def load_codex_pet_archive(archive_path: Path) -> CodexPetArchivePayload:
    """Validate and translate one Codex pet archive into Persona Visual records."""
    archive_path = Path(archive_path)
    logger.debug("Loading Codex pet archive: {}", archive_path)
    validate_codex_pet_archive_members(archive_path)
    logger.debug("Validated Codex pet archive member constraints: {}", archive_path)
    with zipfile.ZipFile(archive_path, "r") as archive:
        members = _archive_members_by_normalized_name(archive)
        pet_manifest_path = _codex_pet_manifest_path(members)
        logger.debug("Resolved Codex pet manifest path: {}", pet_manifest_path)
        pet_payload = _read_json_member(
            archive,
            members[pet_manifest_path],
            path=pet_manifest_path,
        )
        if not isinstance(pet_payload, Mapping):
            raise ValueError(f"malformed_codex_pet_manifest: {pet_manifest_path}")

        sprite = _read_codex_pet_sprite(
            archive,
            members=members,
            pet_payload=pet_payload,
            pet_manifest_path=pet_manifest_path,
        )
        logger.debug(
            "Resolved Codex pet sprite {} ({}x{})",
            sprite.path,
            sprite.width,
            sprite.height,
        )

    title = _codex_pet_pack_title(pet_payload)
    pet_id = _safe_pet_id(pet_payload.get("id"))
    visual_manifest = _codex_pet_visual_manifest(
        pet_id=pet_id,
        display_name=title,
        sprite_path=sprite.path,
    )
    validate_visual_manifest(
        visual_manifest,
        available_asset_ids={CODEX_PET_ASSET_ID},
        available_asset_dimensions={
            CODEX_PET_ASSET_ID: (sprite.width, sprite.height),
        },
        require_activatable=False,
    )
    logger.debug("Validated Codex pet visual manifest for pet_id={}", pet_id)
    pack = {
        "title": title,
        "renderer_type": "sprite_frames",
        "source_format": "codex_pet",
        "visual_manifest": visual_manifest,
    }
    asset = {
        "source_asset_id": CODEX_PET_ASSET_ID,
        "asset_role": "sprite_sheet",
        "asset_group": "animation_atlas",
        "asset_bytes_status": ASSET_BYTES_STATUS_PRESENT,
        "asset_path": sprite.path,
        "asset_sha256": sha256_bytes(sprite.content),
        "mime_type": sprite.mime_type,
        "width": sprite.width,
        "height": sprite.height,
        "original_filename": PurePosixPath(sprite.path).name,
    }
    logger.info("Loaded Codex pet archive '{}' ({})", title, pet_id)
    return CodexPetArchivePayload(
        manifest={
            "schema_version": CODEX_PET_SCHEMA_VERSION,
            "pack_title": title,
            "renderer_type": "sprite_frames",
        },
        pack=pack,
        assets=[asset],
        asset_files={sprite.path: sprite.content},
    )


def validate_codex_pet_archive_members(
    archive_path: Path,
    *,
    max_member_size_bytes: int = DEFAULT_MAX_MEMBER_SIZE_BYTES,
    max_archive_size_bytes: int = DEFAULT_MAX_ARCHIVE_SIZE_BYTES,
) -> list[str]:
    """Validate Codex pet archive members without extracting files."""
    normalized_members: list[str] = []
    seen_members: set[str] = set()
    total_uncompressed_size = 0

    try:
        with zipfile.ZipFile(archive_path, "r") as archive:
            for info in archive.infolist():
                raw_member_name = info.filename
                if info.is_dir():
                    _validate_directory_member_name(raw_member_name)
                    continue
                normalized_name = normalize_member_name(raw_member_name)
                if normalized_name in seen_members:
                    raise ValueError(f"duplicate_archive_member: {normalized_name}")
                if _is_zip_symlink(info):
                    raise ValueError(f"unsafe_archive_member: symlink:{normalized_name}")
                if info.flag_bits & 0x1:
                    raise ValueError(
                        f"unsupported_archive_member: encrypted:{normalized_name}"
                    )
                if info.file_size > max_member_size_bytes:
                    raise ValueError(f"archive_member_too_large: {normalized_name}")

                total_uncompressed_size += info.file_size
                if total_uncompressed_size > max_archive_size_bytes:
                    raise ValueError("archive_too_large")

                seen_members.add(normalized_name)
                normalized_members.append(normalized_name)
    except zipfile.BadZipFile as exc:
        raise ValueError("invalid_archive: bad_zip_file") from exc

    if not any(PurePosixPath(member).name in _MANIFEST_FILENAMES for member in seen_members):
        raise ValueError("missing_required_archive_member: pet.json")
    return normalized_members


def _archive_members_by_normalized_name(archive: zipfile.ZipFile) -> dict[str, zipfile.ZipInfo]:
    members: dict[str, zipfile.ZipInfo] = {}
    for info in archive.infolist():
        if info.is_dir():
            continue
        members[normalize_member_name(info.filename)] = info
    return members


def _codex_pet_manifest_path(members: Mapping[str, zipfile.ZipInfo]) -> str:
    top_level_matches = [
        path for path in sorted(members) if path in _MANIFEST_FILENAMES
    ]
    if top_level_matches:
        return top_level_matches[0]

    matches = [
        path
        for path in sorted(members)
        if PurePosixPath(path).name in _MANIFEST_FILENAMES
    ]
    if not matches:
        raise ValueError("missing_required_archive_member: pet.json")
    if len(matches) > 1:
        raise ValueError("ambiguous_codex_pet_manifest")
    return matches[0]


def _read_json_member(
    archive: zipfile.ZipFile,
    info: zipfile.ZipInfo,
    *,
    path: str,
) -> Any:
    try:
        return json.loads(archive.read(info).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"malformed_codex_pet_manifest: {path}") from exc


def _read_codex_pet_sprite(
    archive: zipfile.ZipFile,
    *,
    members: Mapping[str, zipfile.ZipInfo],
    pet_payload: Mapping[str, Any],
    pet_manifest_path: str,
) -> _CodexPetSprite:
    sprite_path = _sprite_member_path(
        members,
        sprite_reference=pet_payload.get("spritesheetPath"),
        pet_manifest_path=pet_manifest_path,
    )
    content = archive.read(members[sprite_path])
    mime_type = _sprite_mime_type(sprite_path)
    try:
        with Image.open(io.BytesIO(content)) as image:
            image.load()
            width, height = image.size
    except _IMAGE_VALIDATION_ERRORS as exc:
        raise ValueError(f"malformed_codex_pet_sprite: {sprite_path}") from exc
    if (width, height) != (CODEX_PET_SHEET_WIDTH, CODEX_PET_SHEET_HEIGHT):
        raise ValueError("unsupported_codex_pet_spritesheet_dimensions")
    return _CodexPetSprite(
        path=sprite_path,
        content=content,
        mime_type=mime_type,
        width=width,
        height=height,
    )


def _sprite_member_path(
    members: Mapping[str, zipfile.ZipInfo],
    *,
    sprite_reference: Any,
    pet_manifest_path: str,
) -> str:
    declared = str(sprite_reference or "").strip()
    manifest_dir = PurePosixPath(pet_manifest_path).parent
    if declared:
        candidate = normalize_member_name(
            str(manifest_dir / declared) if str(manifest_dir) != "." else declared
        )
        if candidate in members:
            return candidate

    fallback_names = (
        "spritesheet.webp",
        "spritesheet.png",
        "sprite.webp",
        "sprite.png",
    )
    for name in fallback_names:
        candidate = normalize_member_name(
            str(manifest_dir / name) if str(manifest_dir) != "." else name
        )
        if candidate in members:
            return candidate

    image_members = [
        path
        for path in sorted(members)
        if PurePosixPath(path).suffix.casefold() in _SUPPORTED_SPRITE_MIME_TYPES
    ]
    if len(image_members) == 1:
        return image_members[0]
    raise ValueError("missing_required_archive_member: spritesheet")


def _sprite_mime_type(sprite_path: str) -> str:
    suffix = PurePosixPath(sprite_path).suffix.casefold()
    mime_type = _SUPPORTED_SPRITE_MIME_TYPES.get(suffix)
    if mime_type is None:
        raise ValueError("unsupported_codex_pet_sprite_type")
    return mime_type


def _codex_pet_pack_title(pet_payload: Mapping[str, Any]) -> str:
    for field in ("displayName", "display_name", "id"):
        value = _safe_display_text(pet_payload.get(field), max_length=120)
        if value:
            return value
    return "Imported Codex pet"


def _safe_pet_id(value: Any) -> str | None:
    pet_id = _safe_display_text(value, max_length=80)
    return pet_id or None


def _safe_display_text(value: Any, *, max_length: int) -> str:
    if not isinstance(value, str):
        return ""
    stripped = value.strip()
    if not stripped:
        return ""
    return "".join(char for char in stripped if char >= " ")[:max_length]


def _codex_pet_visual_manifest(
    *,
    pet_id: str | None,
    display_name: str,
    sprite_path: str,
) -> dict[str, Any]:
    animations = {
        "idle-loop": _row_animation(0),
        "codex-moving-right-loop": _row_animation(1),
        "codex-moving-left-loop": _row_animation(2),
        "codex-waving-loop": _row_animation(3),
        "codex-jumping-loop": _row_animation(4),
        "codex-failed-loop": _row_animation(5),
        "codex-waiting-loop": _row_animation(6),
        "codex-running-loop": _row_animation(7),
        "codex-review-loop": _row_animation(8),
    }
    state_catalog = {
        "moving_right": {
            "label": "Moving right",
            "kind": "live_variant",
            "description": "Used while the buddy is being moved to the right.",
            "tags": ["movement", "codex-pet"],
        },
        "moving_left": {
            "label": "Moving left",
            "kind": "live_variant",
            "description": "Used while the buddy is being moved to the left.",
            "tags": ["movement", "codex-pet"],
        },
        "codex.waving": _codex_state_catalog_entry("Waving"),
        "codex.jumping": _codex_state_catalog_entry("Jumping"),
        "codex.failed": _codex_state_catalog_entry("Failed"),
        "codex.waiting": _codex_state_catalog_entry("Waiting"),
        "codex.running": _codex_state_catalog_entry("Running"),
        "codex.review": _codex_state_catalog_entry("Review"),
    }
    return {
        "manifest_version": 1,
        "renderer_type": "sprite_frames",
        "source_format": {
            "kind": "codex_pet",
            "pet_id": pet_id,
            "display_name": display_name,
            "spritesheet_path": sprite_path,
            "atlas": {
                "columns": CODEX_PET_COLUMNS,
                "rows": CODEX_PET_ROWS,
                "frame_width": CODEX_PET_FRAME_WIDTH,
                "frame_height": CODEX_PET_FRAME_HEIGHT,
            },
        },
        "state_catalog": state_catalog,
        "states": {
            "idle": {"animation_id": "idle-loop"},
            "listening": {"animation_id": "codex-waiting-loop"},
            "thinking": {"animation_id": "codex-review-loop"},
            "speaking": {"animation_id": "codex-waving-loop"},
            "tool_running": {"animation_id": "codex-running-loop"},
            "approval_needed": {"animation_id": "codex-review-loop"},
            "wake_armed": {"animation_id": "codex-waving-loop"},
            "error": {"animation_id": "codex-failed-loop"},
            "offline": {"animation_id": "idle-loop"},
            "moving_right": {"animation_id": "codex-moving-right-loop"},
            "moving_left": {"animation_id": "codex-moving-left-loop"},
            "codex.waving": {"animation_id": "codex-waving-loop"},
            "codex.jumping": {"animation_id": "codex-jumping-loop"},
            "codex.failed": {"animation_id": "codex-failed-loop"},
            "codex.waiting": {"animation_id": "codex-waiting-loop"},
            "codex.running": {"animation_id": "codex-running-loop"},
            "codex.review": {"animation_id": "codex-review-loop"},
        },
        "fallbacks": {
            "moving_right": ["tool_running", "thinking", "idle"],
            "moving_left": ["tool_running", "thinking", "idle"],
            "codex.waving": ["speaking", "idle"],
            "codex.jumping": ["idle"],
            "codex.failed": ["error", "idle"],
            "codex.waiting": ["listening", "idle"],
            "codex.running": ["tool_running", "thinking", "idle"],
            "codex.review": ["thinking", "idle"],
        },
        "animations": animations,
    }


def _codex_state_catalog_entry(label: str) -> dict[str, Any]:
    return {
        "label": label,
        "kind": "pack_private",
        "description": f"Imported Codex pet {label.casefold()} animation.",
        "tags": ["codex-pet"],
    }


def _row_animation(row: int) -> dict[str, Any]:
    return {
        "frames": [
            {
                "asset_id": CODEX_PET_ASSET_ID,
                "region": {
                    "x": column * CODEX_PET_FRAME_WIDTH,
                    "y": row * CODEX_PET_FRAME_HEIGHT,
                    "width": CODEX_PET_FRAME_WIDTH,
                    "height": CODEX_PET_FRAME_HEIGHT,
                },
                "duration_ms": CODEX_PET_FRAME_DURATION_MS,
            }
            for column in range(CODEX_PET_COLUMNS)
        ],
        "frame_rate": 8,
        "preview_frame": 0,
        "alignment": {"x": 0.5, "y": 1.0},
    }


def _validate_directory_member_name(member_name: str) -> None:
    if not member_name.endswith("/"):
        raise ValueError("unsafe_archive_member: invalid_directory")
    normalize_member_name(member_name[:-1])


def _is_zip_symlink(info: zipfile.ZipInfo) -> bool:
    unix_file_type = (info.external_attr >> 16) & 0o170000
    return unix_file_type == 0o120000
