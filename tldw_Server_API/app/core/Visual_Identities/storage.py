"""Safe storage and validation helpers for visual identity expression assets."""

from __future__ import annotations

import hashlib
import io
import os
import re
import stat
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from loguru import logger
from PIL import Image, UnidentifiedImageError

from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths
from tldw_Server_API.app.core.Visual_Identities import constraints

VISUAL_IDENTITY_INVALID_STORAGE_PATH = "invalid_storage_path"
VISUAL_IDENTITY_UNSUPPORTED_MIME_TYPE = "unsupported_mime_type"
VISUAL_IDENTITY_MIME_MISMATCH = "mime_mismatch"
VISUAL_IDENTITY_INVALID_IMAGE = "invalid_image"
VISUAL_IDENTITY_FILE_TOO_LARGE = "file_too_large"
VISUAL_IDENTITY_DIMENSIONS_EXCEED_LIMIT = "image_dimensions_exceed_limit"
VISUAL_IDENTITY_FRAME_COUNT_EXCEEDS_LIMIT = "image_frame_count_exceeds_limit"
VISUAL_IDENTITY_UNSUPPORTED_EXTENSION = "unsupported_extension"
VISUAL_IDENTITY_EXTENSION_MISMATCH = "extension_mismatch"
VISUAL_IDENTITY_GENERATED_FILE_NOT_FOUND = "generated_file_not_found"
VISUAL_IDENTITY_GENERATED_FILE_NOT_IMAGE = "generated_file_not_image"
VISUAL_IDENTITY_STORED_ASSET_HASH_MISMATCH = "stored_asset_hash_mismatch"

_IMAGE_VALIDATION_ERRORS = (OSError, ValueError, UnidentifiedImageError)
_SAFE_COMPONENT_RE = re.compile(r"[^A-Za-z0-9_.-]+")
_MIME_EXTENSION_CHOICES = {
    "image/png": (".png",),
    "image/jpeg": (".jpg", ".jpeg"),
    "image/webp": (".webp",),
    "image/gif": (".gif",),
    constraints.AVIF_MIME_TYPE: (".avif",),
}
_MIME_DEFAULT_EXTENSION = {
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/webp": ".webp",
    "image/gif": ".gif",
    constraints.AVIF_MIME_TYPE: ".avif",
}
_ALL_ALLOWED_EXTENSIONS = frozenset(
    extension for extensions in _MIME_EXTENSION_CHOICES.values() for extension in extensions
)


@dataclass(frozen=True)
class VisualIdentityStoredAsset:
    """Metadata returned after a visual identity asset is validated and stored."""

    relpath: str
    content_type: str
    bytes: int
    sha256: str
    width: int
    height: int
    is_animated: bool
    frame_count: int
    duration_ms: int | None
    preview_relpath: str | None


def validate_and_store_visual_identity_asset(
    *,
    source_path: str | Path,
    owner_user_id: int,
    expression_key: str,
    storage_root: str | Path | None = None,
    content_type: str | None = None,
    pack_id: int | str | None = None,
) -> VisualIdentityStoredAsset:
    """Validate a raster expression asset and store its original bytes safely."""
    source = Path(source_path)
    if not source.is_file():
        raise ValueError(VISUAL_IDENTITY_INVALID_IMAGE)

    byte_count = source.stat().st_size
    if byte_count <= 0:
        raise ValueError(VISUAL_IDENTITY_INVALID_IMAGE)
    if byte_count > constraints.MAX_EXPRESSION_ASSET_BYTES:
        raise ValueError(VISUAL_IDENTITY_FILE_TOO_LARGE)

    content = source.read_bytes()
    byte_count = len(content)
    if byte_count <= 0:
        raise ValueError(VISUAL_IDENTITY_INVALID_IMAGE)
    if byte_count > constraints.MAX_EXPRESSION_ASSET_BYTES:
        raise ValueError(VISUAL_IDENTITY_FILE_TOO_LARGE)

    sha256 = hashlib.sha256(content).hexdigest()
    sniffed_mime = _sniff_mime_type(content)
    _ensure_supported_mime_type(sniffed_mime)
    _validate_declared_mime_type(content_type, sniffed_mime)

    image_metadata = _validate_image_content(content, expected_mime_type=sniffed_mime)
    extension = _validated_storage_extension(source, sniffed_mime)
    root = _visual_identity_storage_root(owner_user_id, storage_root=storage_root)
    asset_relpath = _asset_relpath(
        sha256=sha256,
        extension=extension,
        expression_key=expression_key,
        pack_id=pack_id,
    )
    target_path = resolve_visual_identity_asset_path(
        owner_user_id=owner_user_id,
        relpath=asset_relpath,
        storage_root=root,
    )
    target_path.parent.mkdir(parents=True, exist_ok=True)
    _write_once(target_path, content, expected_sha256=sha256)

    preview_relpath = _create_first_frame_preview(
        content,
        owner_user_id=owner_user_id,
        storage_root=root,
        sha256=sha256,
        expression_key=expression_key,
        pack_id=pack_id,
    )

    return VisualIdentityStoredAsset(
        relpath=asset_relpath,
        content_type=sniffed_mime,
        bytes=byte_count,
        sha256=sha256,
        width=image_metadata["width"],
        height=image_metadata["height"],
        is_animated=image_metadata["is_animated"],
        frame_count=image_metadata["frame_count"],
        duration_ms=image_metadata["duration_ms"],
        preview_relpath=preview_relpath,
    )


def resolve_visual_identity_asset_path(
    *,
    owner_user_id: int,
    relpath: str | Path,
    storage_root: str | Path | None = None,
) -> Path:
    """Resolve a stored visual identity relpath under the user's storage root."""
    raw_relpath = str(relpath or "")
    if not raw_relpath:
        raise ValueError(VISUAL_IDENTITY_INVALID_STORAGE_PATH)
    if "\\" in raw_relpath:
        raise ValueError(VISUAL_IDENTITY_INVALID_STORAGE_PATH)

    relative_path = Path(raw_relpath)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise ValueError(VISUAL_IDENTITY_INVALID_STORAGE_PATH)

    base = _visual_identity_storage_root(owner_user_id, storage_root=storage_root)
    target = (base / relative_path).resolve(strict=False)
    if not target.is_relative_to(base):
        raise ValueError(VISUAL_IDENTITY_INVALID_STORAGE_PATH)
    return target


def copy_generated_file_record_to_expression_asset(
    *,
    owner_user_id: int,
    pack_id: int | str,
    expression_key: str,
    generated_file_record: Mapping[str, Any],
    source_feature: str | None = None,
    storage_root: str | Path | None = None,
) -> VisualIdentityStoredAsset:
    """Copy an already-loaded generated-file record into expression storage.

    Stage 11 can wrap this lower-level helper with AuthnzGeneratedFilesRepo lookup
    and VisualIdentityRepository asset-row creation.
    """
    _validate_generated_file_record(
        generated_file_record,
        owner_user_id=owner_user_id,
        source_feature=source_feature,
    )
    resolved_source_path = _resolve_generated_file_record_path(owner_user_id, generated_file_record)
    declared_content_type = str(generated_file_record.get("mime_type") or "").strip() or None
    return validate_and_store_visual_identity_asset(
        source_path=resolved_source_path,
        owner_user_id=owner_user_id,
        expression_key=expression_key,
        storage_root=storage_root,
        content_type=declared_content_type,
        pack_id=pack_id,
    )


def _visual_identity_storage_root(
    owner_user_id: int,
    *,
    storage_root: str | Path | None,
) -> Path:
    if storage_root is None:
        return DatabasePaths.get_user_visual_identities_dir(owner_user_id).resolve(strict=False)
    root = Path(storage_root).resolve(strict=False)
    root.mkdir(parents=True, exist_ok=True)
    return root


def _asset_relpath(
    *,
    sha256: str,
    extension: str,
    expression_key: str,
    pack_id: int | str | None,
) -> str:
    filename = f"{sha256}{extension}"
    if pack_id is not None:
        return (
            Path("packs")
            / _safe_storage_component(str(pack_id), prefix="pack")
            / _safe_storage_component(expression_key, prefix="expression")
            / filename
        ).as_posix()
    return (Path("assets") / _safe_storage_component(expression_key, prefix="expression") / filename).as_posix()


def _preview_relpath(
    *,
    sha256: str,
    expression_key: str,
    pack_id: int | str | None,
) -> str:
    filename = f"{sha256}.png"
    if pack_id is not None:
        return (
            Path("previews")
            / "packs"
            / _safe_storage_component(str(pack_id), prefix="pack")
            / _safe_storage_component(expression_key, prefix="expression")
            / filename
        ).as_posix()
    return (
        Path("previews") / "assets" / _safe_storage_component(expression_key, prefix="expression") / filename
    ).as_posix()


def _safe_storage_component(value: str, *, prefix: str) -> str:
    raw = str(value or "").strip()
    if not raw or raw in {".", ".."}:
        digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
        return f"{prefix}_{digest}"
    cleaned = _SAFE_COMPONENT_RE.sub("_", raw).strip("._-")
    if not cleaned:
        digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
        return f"{prefix}_{digest}"
    if cleaned in {".", ".."}:
        digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
        return f"{prefix}_{digest}"
    return cleaned[:120]


def _sniff_mime_type(content: bytes) -> str:
    if content.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if content.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if content.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif"
    if len(content) >= 12 and content[:4] == b"RIFF" and content[8:12] == b"WEBP":
        return "image/webp"
    if len(content) >= 12 and content[4:8] == b"ftyp":
        major_brand = content[8:12]
        compatible_brands = content[16:64]
        if major_brand in {b"avif", b"avis"} or b"avif" in compatible_brands or b"avis" in compatible_brands:
            return constraints.AVIF_MIME_TYPE
    raise ValueError(VISUAL_IDENTITY_UNSUPPORTED_MIME_TYPE)


def _ensure_supported_mime_type(mime_type: str) -> None:
    if mime_type == constraints.AVIF_MIME_TYPE and not constraints.supports_avif():
        raise ValueError(VISUAL_IDENTITY_UNSUPPORTED_MIME_TYPE)
    if mime_type not in constraints.supported_visual_identity_mime_types():
        raise ValueError(VISUAL_IDENTITY_UNSUPPORTED_MIME_TYPE)


def _normalize_content_type(content_type: str | None) -> str:
    return str(content_type or "").split(";", 1)[0].strip().lower()


def _validate_declared_mime_type(content_type: str | None, detected_mime_type: str) -> None:
    normalized = _normalize_content_type(content_type)
    if normalized and normalized != detected_mime_type:
        raise ValueError(VISUAL_IDENTITY_MIME_MISMATCH)


def _validate_image_content(content: bytes, *, expected_mime_type: str) -> dict[str, Any]:
    try:
        with Image.open(io.BytesIO(content)) as image:
            width, height = image.size
            pillow_mime = Image.MIME.get(image.format or "")
            image.verify()
    except _IMAGE_VALIDATION_ERRORS as exc:
        raise ValueError(VISUAL_IDENTITY_INVALID_IMAGE) from exc

    if pillow_mime and pillow_mime.lower() != expected_mime_type:
        raise ValueError(VISUAL_IDENTITY_MIME_MISMATCH)
    _validate_dimensions(width, height)

    try:
        with Image.open(io.BytesIO(content)) as image:
            frame_count = max(int(getattr(image, "n_frames", 1) or 1), 1)
            if frame_count > constraints.MAX_EXPRESSION_FRAME_COUNT:
                raise ValueError(VISUAL_IDENTITY_FRAME_COUNT_EXCEEDS_LIMIT)
            is_animated = bool(getattr(image, "is_animated", False)) or frame_count > 1
            duration_ms = _duration_ms_for_frames(image, frame_count=frame_count) if is_animated else None
    except ValueError:
        raise
    except _IMAGE_VALIDATION_ERRORS as exc:
        raise ValueError(VISUAL_IDENTITY_INVALID_IMAGE) from exc

    return {
        "width": int(width),
        "height": int(height),
        "is_animated": is_animated,
        "frame_count": frame_count,
        "duration_ms": duration_ms,
    }


def _validate_dimensions(width: int, height: int) -> None:
    if width <= 0 or height <= 0:
        raise ValueError(VISUAL_IDENTITY_INVALID_IMAGE)
    if width > constraints.MAX_EXPRESSION_IMAGE_DIMENSION or height > constraints.MAX_EXPRESSION_IMAGE_DIMENSION:
        raise ValueError(VISUAL_IDENTITY_DIMENSIONS_EXCEED_LIMIT)


def _duration_ms_for_frames(image: Image.Image, *, frame_count: int) -> int | None:
    total_ms = 0
    saw_duration = False
    for frame_index in range(frame_count):
        try:
            image.seek(frame_index)
        except EOFError:
            break
        raw_duration = image.info.get("duration")
        if raw_duration is None:
            continue
        try:
            total_ms += max(int(raw_duration), 0)
            saw_duration = True
        except (TypeError, ValueError):
            continue
    return total_ms if saw_duration else None


def _validated_storage_extension(source_path: Path, detected_mime_type: str) -> str:
    source_extension = source_path.suffix.lower()
    allowed_for_mime = _MIME_EXTENSION_CHOICES[detected_mime_type]
    if not source_extension:
        return _MIME_DEFAULT_EXTENSION[detected_mime_type]
    if source_extension in allowed_for_mime:
        return source_extension
    if source_extension in _ALL_ALLOWED_EXTENSIONS:
        raise ValueError(VISUAL_IDENTITY_EXTENSION_MISMATCH)
    raise ValueError(VISUAL_IDENTITY_UNSUPPORTED_EXTENSION)


def _write_once(target_path: Path, content: bytes, *, expected_sha256: str) -> None:
    expected_size = len(content)
    try:
        _verify_existing_hash_target(
            target_path,
            expected_size=expected_size,
            expected_sha256=expected_sha256,
        )
        return
    except FileNotFoundError:
        pass

    temp_path = _write_same_dir_temp_file(target_path, content)
    try:
        try:
            os.link(temp_path, target_path)
        except FileExistsError:
            _verify_existing_hash_target(
                target_path,
                expected_size=expected_size,
                expected_sha256=expected_sha256,
            )
        except OSError:
            if target_path.exists():
                _verify_existing_hash_target(
                    target_path,
                    expected_size=expected_size,
                    expected_sha256=expected_sha256,
                )
            else:
                temp_path.replace(target_path)
                _verify_existing_hash_target(
                    target_path,
                    expected_size=expected_size,
                    expected_sha256=expected_sha256,
                )
    finally:
        temp_path.unlink(missing_ok=True)


def _write_same_dir_temp_file(target_path: Path, content: bytes) -> Path:
    fd, raw_temp_path = tempfile.mkstemp(
        prefix=f".{target_path.name}.",
        suffix=".tmp",
        dir=target_path.parent,
    )
    temp_path = Path(raw_temp_path)
    try:
        with os.fdopen(fd, "wb") as file_obj:
            file_obj.write(content)
            file_obj.flush()
            os.fsync(file_obj.fileno())
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise
    return temp_path


def _verify_existing_hash_target(
    target_path: Path,
    *,
    expected_size: int,
    expected_sha256: str,
) -> None:
    try:
        target_stat = target_path.lstat()
    except FileNotFoundError:
        raise
    if not stat.S_ISREG(target_stat.st_mode):
        raise ValueError(VISUAL_IDENTITY_STORED_ASSET_HASH_MISMATCH)
    if target_stat.st_size != expected_size:
        raise ValueError(VISUAL_IDENTITY_STORED_ASSET_HASH_MISMATCH)
    if _sha256_file(target_path) != expected_sha256:
        raise ValueError(VISUAL_IDENTITY_STORED_ASSET_HASH_MISMATCH)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _create_first_frame_preview(
    content: bytes,
    *,
    owner_user_id: int,
    storage_root: Path,
    sha256: str,
    expression_key: str,
    pack_id: int | str | None,
) -> str | None:
    relpath = _preview_relpath(
        sha256=sha256,
        expression_key=expression_key,
        pack_id=pack_id,
    )
    try:
        preview_path = resolve_visual_identity_asset_path(
            owner_user_id=owner_user_id,
            relpath=relpath,
            storage_root=storage_root,
        )
        with Image.open(io.BytesIO(content)) as image:
            image.seek(0)
            frame = image.convert("RGBA")
            preview_path.parent.mkdir(parents=True, exist_ok=True)
            if not preview_path.exists():
                frame.save(preview_path, format="PNG")
    except _IMAGE_VALIDATION_ERRORS as exc:
        logger.debug("Skipping visual identity preview extraction: {}", exc)
        return None
    return relpath


def _validate_generated_file_record(
    generated_file_record: Mapping[str, Any],
    *,
    owner_user_id: int,
    source_feature: str | None,
) -> None:
    try:
        record_user_id = int(generated_file_record.get("user_id") or 0)
    except (TypeError, ValueError) as exc:
        raise ValueError(VISUAL_IDENTITY_GENERATED_FILE_NOT_FOUND) from exc
    if record_user_id != int(owner_user_id) or bool(generated_file_record.get("is_deleted")):
        raise ValueError(VISUAL_IDENTITY_GENERATED_FILE_NOT_FOUND)

    file_category = str(generated_file_record.get("file_category") or "").strip().lower()
    if file_category and file_category != "image":
        raise ValueError(VISUAL_IDENTITY_GENERATED_FILE_NOT_IMAGE)

    expected_source_feature = str(source_feature or "").strip().lower()
    if expected_source_feature:
        record_source_feature = str(generated_file_record.get("source_feature") or "").strip().lower()
        if record_source_feature != expected_source_feature:
            raise ValueError(VISUAL_IDENTITY_GENERATED_FILE_NOT_FOUND)


def _resolve_generated_file_record_path(
    owner_user_id: int,
    generated_file_record: Mapping[str, Any],
) -> Path:
    raw_storage_path = str(generated_file_record.get("storage_path") or "")
    if not raw_storage_path or "\\" in raw_storage_path:
        raise ValueError(VISUAL_IDENTITY_GENERATED_FILE_NOT_FOUND)

    relative_path = Path(raw_storage_path)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise ValueError(VISUAL_IDENTITY_GENERATED_FILE_NOT_FOUND)

    base = DatabasePaths.get_user_outputs_dir(owner_user_id).resolve(strict=False)
    resolved = (base / relative_path).resolve(strict=False)
    if not resolved.is_relative_to(base):
        raise ValueError(VISUAL_IDENTITY_GENERATED_FILE_NOT_FOUND)
    return resolved


__all__ = [
    "VisualIdentityStoredAsset",
    "copy_generated_file_record_to_expression_asset",
    "resolve_visual_identity_asset_path",
    "validate_and_store_visual_identity_asset",
]
