"""Helper functions for registering generated files with the storage tracking system.

This module provides easy-to-use functions for saving and registering generated files
from various sources (TTS, image generation, voice clones, etc.).

Usage:
    from tldw_Server_API.app.core.Storage.generated_file_helpers import (
        save_and_register_tts_audio,
        save_and_register_image,
    )

    # Save TTS audio
    file_info = await save_and_register_tts_audio(
        user_id=user.id,
        audio_bytes=audio_data,
        audio_format="mp3",
        original_text="Hello world",
    )

    # Save generated image
    file_info = await save_and_register_image(
        user_id=user.id,
        image_bytes=image_data,
        image_format="png",
        source_prompt="A cat",
    )
"""
from __future__ import annotations

import contextlib
import hashlib
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import aiofiles
from loguru import logger

from tldw_Server_API.app.core.AuthNZ.repos.generated_files_repo import (
    FILE_CATEGORY_IMAGE,
    FILE_CATEGORY_MINDMAP,
    FILE_CATEGORY_SPREADSHEET,
    FILE_CATEGORY_TTS_AUDIO,
    FILE_CATEGORY_VOICE_CLONE,
    SOURCE_FEATURE_DATA_TABLES,
    SOURCE_FEATURE_IMAGE_GEN,
    SOURCE_FEATURE_MINDMAP,
    SOURCE_FEATURE_TTS,
    SOURCE_FEATURE_VN_ASSETS,
    SOURCE_FEATURE_VOICE_STUDIO,
)
from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths
from tldw_Server_API.app.core.Utils.Utils import sanitize_filename
from tldw_Server_API.app.services.storage_quota_service import get_storage_service

# MIME type mappings
AUDIO_MIME_TYPES = {
    "mp3": "audio/mpeg",
    "wav": "audio/wav",
    "flac": "audio/flac",
    "ogg": "audio/ogg",
    "opus": "audio/opus",
    "aac": "audio/aac",
    "pcm": "audio/pcm",
}

IMAGE_MIME_TYPES = {
    "png": "image/png",
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "webp": "image/webp",
    "gif": "image/gif",
    "bmp": "image/bmp",
}

SPREADSHEET_MIME_TYPES = {
    "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "csv": "text/csv",
    "xls": "application/vnd.ms-excel",
}


def _compute_checksum(data: bytes) -> str:
    """Compute SHA-256 checksum of data."""
    return hashlib.sha256(data).hexdigest()


def _normalize_extension(file_format: str) -> str:
    """Normalize a file extension to a safe, alphanumeric lowercase token."""
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "", str(file_format)).lower()
    return cleaned or "bin"


def _generate_filename(prefix: str, file_format: str) -> str:
    """Generate a unique, sanitized filename."""
    file_uuid = uuid.uuid4().hex[:12]
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    safe_prefix = sanitize_filename(prefix, max_total_length=80).replace(" ", "_")
    safe_ext = _normalize_extension(file_format)
    return f"{safe_prefix}_{timestamp}_{file_uuid}.{safe_ext}"


def _get_date_folder() -> str:
    """Get date-based subfolder name (YYYY/MM/DD)."""
    now = datetime.now(timezone.utc)
    return f"{now.year}/{now.month:02d}/{now.day:02d}"


async def _save_file(
    user_id: int,
    data: bytes,
    category_folder: str,
    filename: str,
) -> Path:
    """
    Save file to user's outputs directory.

    Returns:
        Full path to saved file
    """
    outputs_dir = DatabasePaths.get_user_outputs_dir(user_id)
    date_folder = _get_date_folder()

    # Create category/date directory structure
    target_dir = outputs_dir / category_folder / date_folder
    target_dir.mkdir(parents=True, exist_ok=True)

    file_path = target_dir / filename

    async with aiofiles.open(file_path, "wb") as f:
        await f.write(data)

    return file_path


async def save_and_register_tts_audio(
    *,
    user_id: int,
    audio_bytes: bytes,
    audio_format: str = "mp3",
    original_text: str | None = None,
    voice_name: str | None = None,
    model_name: str | None = None,
    org_id: int | None = None,
    team_id: int | None = None,
    folder_tag: str | None = None,
    tags: list[str] | None = None,
    is_transient: bool = False,
    expires_at: datetime | None = None,
    check_quota: bool = True,
) -> dict[str, Any]:
    """
    Save TTS audio and register with storage tracking.

    Args:
        user_id: Owner user ID
        audio_bytes: Audio data bytes
        audio_format: Audio format (mp3, wav, etc.)
        original_text: Original text that was synthesized (for reference)
        voice_name: Voice used for synthesis
        model_name: TTS model used
        org_id: Optional organization ID
        team_id: Optional team ID
        folder_tag: Virtual folder tag
        tags: Additional tags
        is_transient: Whether file is temporary
        expires_at: Expiration timestamp
        check_quota: Whether to check quota before saving

    Returns:
        File record dict with id, uuid, storage_path, etc.
    """
    # Generate filename
    filename = _generate_filename("tts_audio", audio_format)
    category_folder = "tts_audio"
    _get_date_folder()

    # Save file
    file_path = await _save_file(user_id, audio_bytes, category_folder, filename)

    # Compute relative storage path
    outputs_dir = DatabasePaths.get_user_outputs_dir(user_id)
    relative_path = str(file_path.relative_to(outputs_dir))

    # Build source reference with metadata
    source_ref_parts = []
    if voice_name:
        source_ref_parts.append(f"voice:{voice_name}")
    if model_name:
        source_ref_parts.append(f"model:{model_name}")
    source_ref = ";".join(source_ref_parts) if source_ref_parts else None

    # Build tags
    file_tags = list(tags) if tags else []
    if voice_name:
        file_tags.append(f"voice:{voice_name}")
    if model_name:
        file_tags.append(f"model:{model_name}")

    # Get MIME type
    mime_type = AUDIO_MIME_TYPES.get(audio_format.lower(), "audio/mpeg")

    # Register with storage service
    try:
        service = await get_storage_service()
        file_record = await service.register_generated_file(
            user_id=user_id,
            filename=filename,
            storage_path=relative_path,
            file_category=FILE_CATEGORY_TTS_AUDIO,
            source_feature=SOURCE_FEATURE_TTS,
            file_size_bytes=len(audio_bytes),
            org_id=org_id,
            team_id=team_id,
            original_filename=f"speech.{audio_format}",
            mime_type=mime_type,
            checksum=_compute_checksum(audio_bytes),
            source_ref=source_ref,
            folder_tag=folder_tag,
            tags=file_tags if file_tags else None,
            is_transient=is_transient,
            expires_at=expires_at,
            check_quota=check_quota,
        )

        logger.info(f"Registered TTS audio: {filename} ({len(audio_bytes)} bytes) for user {user_id}")
        return file_record

    except Exception:
        # Cleanup file if registration fails
        with contextlib.suppress(Exception):
            file_path.unlink()
        raise


async def save_and_register_image(
    *,
    user_id: int,
    image_bytes: bytes,
    image_format: str = "png",
    source_prompt: str | None = None,
    model_name: str | None = None,
    org_id: int | None = None,
    team_id: int | None = None,
    folder_tag: str | None = None,
    tags: list[str] | None = None,
    is_transient: bool = False,
    expires_at: datetime | None = None,
    check_quota: bool = True,
) -> dict[str, Any]:
    """
    Save generated image and register with storage tracking.

    Args:
        user_id: Owner user ID
        image_bytes: Image data bytes
        image_format: Image format (png, jpg, webp, etc.)
        source_prompt: Prompt used for generation
        model_name: Model used for generation
        org_id: Optional organization ID
        team_id: Optional team ID
        folder_tag: Virtual folder tag
        tags: Additional tags
        is_transient: Whether file is temporary
        expires_at: Expiration timestamp
        check_quota: Whether to check quota before saving

    Returns:
        File record dict
    """
    filename = _generate_filename("image", image_format)
    category_folder = "images"

    file_path = await _save_file(user_id, image_bytes, category_folder, filename)

    outputs_dir = DatabasePaths.get_user_outputs_dir(user_id)
    relative_path = str(file_path.relative_to(outputs_dir))

    source_ref = f"model:{model_name}" if model_name else None

    file_tags = list(tags) if tags else []
    if model_name:
        file_tags.append(f"model:{model_name}")

    mime_type = IMAGE_MIME_TYPES.get(image_format.lower(), "image/png")

    try:
        service = await get_storage_service()
        file_record = await service.register_generated_file(
            user_id=user_id,
            filename=filename,
            storage_path=relative_path,
            file_category=FILE_CATEGORY_IMAGE,
            source_feature=SOURCE_FEATURE_IMAGE_GEN,
            file_size_bytes=len(image_bytes),
            org_id=org_id,
            team_id=team_id,
            original_filename=f"generated.{image_format}",
            mime_type=mime_type,
            checksum=_compute_checksum(image_bytes),
            source_ref=source_ref,
            folder_tag=folder_tag,
            tags=file_tags if file_tags else None,
            is_transient=is_transient,
            expires_at=expires_at,
            check_quota=check_quota,
        )

        logger.info(f"Registered image: {filename} ({len(image_bytes)} bytes) for user {user_id}")
        return file_record

    except Exception:
        with contextlib.suppress(Exception):
            file_path.unlink()
        raise


async def save_and_register_vn_asset_image(
    *,
    user_id: int,
    image_bytes: bytes,
    image_format: str = "png",
    pack_id: int,
    item_id: int,
    asset_type: str,
    labels: dict[str, Any] | None = None,
    org_id: int | None = None,
    team_id: int | None = None,
    tags: list[str] | None = None,
    is_transient: bool = False,
    expires_at: datetime | None = None,
    check_quota: bool = True,
) -> dict[str, Any]:
    """
    Save a generated VN asset image and register it as durable AuthNZ storage.

    VN asset metadata remains in the user's ChaChaNotes database; the image bytes
    and quota accounting are tracked through generated-file storage.
    """
    filename = _generate_filename(f"vn_asset_{item_id}", image_format)
    category_folder = "vn_assets"

    file_path = await _save_file(user_id, image_bytes, category_folder, filename)
    outputs_dir = DatabasePaths.get_user_outputs_dir(user_id)
    relative_path = str(file_path.relative_to(outputs_dir))

    file_tags = list(tags) if tags else []
    file_tags.extend(
        [
            f"vn_pack:{pack_id}",
            f"vn_item:{item_id}",
            f"asset_type:{asset_type}",
        ]
    )
    for key, value in (labels or {}).items():
        file_tags.append(f"label:{key}:{value}")

    mime_type = IMAGE_MIME_TYPES.get(image_format.lower(), "image/png")

    try:
        service = await get_storage_service()
        file_record = await service.register_generated_file(
            user_id=user_id,
            filename=filename,
            storage_path=relative_path,
            file_category=FILE_CATEGORY_IMAGE,
            source_feature=SOURCE_FEATURE_VN_ASSETS,
            file_size_bytes=len(image_bytes),
            org_id=org_id,
            team_id=team_id,
            original_filename=f"vn_asset_{item_id}.{image_format}",
            mime_type=mime_type,
            checksum=_compute_checksum(image_bytes),
            source_ref=f"vn_asset_item:{item_id}",
            folder_tag=f"vn-assets/{pack_id}",
            tags=file_tags,
            is_transient=is_transient,
            expires_at=expires_at,
            check_quota=check_quota,
        )

        logger.info(f"Registered VN asset image: {filename} ({len(image_bytes)} bytes) for user {user_id}")
        return file_record

    except Exception:
        with contextlib.suppress(Exception):
            file_path.unlink()
        raise


async def save_and_register_voice_clone(
    *,
    user_id: int,
    voice_data: bytes,
    voice_format: str = "bin",
    voice_name: str,
    provider: str | None = None,
    org_id: int | None = None,
    team_id: int | None = None,
    folder_tag: str | None = None,
    tags: list[str] | None = None,
    check_quota: bool = True,
) -> dict[str, Any]:
    """
    Save voice clone data and register with storage tracking.

    Args:
        user_id: Owner user ID
        voice_data: Voice clone data bytes
        voice_format: Data format
        voice_name: Name of the cloned voice
        provider: TTS provider for this voice clone
        org_id: Optional organization ID
        team_id: Optional team ID
        folder_tag: Virtual folder tag
        tags: Additional tags
        check_quota: Whether to check quota before saving

    Returns:
        File record dict
    """
    filename = _generate_filename(f"voice_{voice_name}", voice_format)
    category_folder = "voice_clones"

    file_path = await _save_file(user_id, voice_data, category_folder, filename)

    outputs_dir = DatabasePaths.get_user_outputs_dir(user_id)
    relative_path = str(file_path.relative_to(outputs_dir))

    source_ref = f"provider:{provider}" if provider else None

    file_tags = list(tags) if tags else []
    file_tags.append(f"voice:{voice_name}")
    if provider:
        file_tags.append(f"provider:{provider}")

    service = await get_storage_service()
    try:
        file_record = await service.register_generated_file(
            user_id=user_id,
            filename=filename,
            storage_path=relative_path,
            file_category=FILE_CATEGORY_VOICE_CLONE,
            source_feature=SOURCE_FEATURE_VOICE_STUDIO,
            file_size_bytes=len(voice_data),
            org_id=org_id,
            team_id=team_id,
            original_filename=f"{voice_name}.{voice_format}",
            mime_type="application/octet-stream",
            checksum=_compute_checksum(voice_data),
            source_ref=source_ref,
            folder_tag=folder_tag,
            tags=file_tags,
            is_transient=False,  # Voice clones are typically permanent
            check_quota=check_quota,
        )

        logger.info(f"Registered voice clone: {voice_name} ({len(voice_data)} bytes) for user {user_id}")
        return file_record

    except Exception:
        with contextlib.suppress(Exception):
            file_path.unlink()
        raise


async def save_and_register_spreadsheet(
    *,
    user_id: int,
    spreadsheet_bytes: bytes,
    spreadsheet_format: str = "xlsx",
    original_filename: str | None = None,
    source_ref: str | None = None,
    org_id: int | None = None,
    team_id: int | None = None,
    folder_tag: str | None = None,
    tags: list[str] | None = None,
    is_transient: bool = False,
    expires_at: datetime | None = None,
    check_quota: bool = True,
) -> dict[str, Any]:
    """
    Save generated spreadsheet and register with storage tracking.

    Args:
        user_id: Owner user ID
        spreadsheet_bytes: Spreadsheet data bytes
        spreadsheet_format: Format (xlsx, csv, etc.)
        original_filename: Original filename
        source_ref: Reference to source data
        org_id: Optional organization ID
        team_id: Optional team ID
        folder_tag: Virtual folder tag
        tags: Additional tags
        is_transient: Whether file is temporary
        expires_at: Expiration timestamp
        check_quota: Whether to check quota before saving

    Returns:
        File record dict
    """
    filename = _generate_filename("spreadsheet", spreadsheet_format)
    category_folder = "spreadsheets"

    file_path = await _save_file(user_id, spreadsheet_bytes, category_folder, filename)

    outputs_dir = DatabasePaths.get_user_outputs_dir(user_id)
    relative_path = str(file_path.relative_to(outputs_dir))

    mime_type = SPREADSHEET_MIME_TYPES.get(spreadsheet_format.lower(), "application/octet-stream")

    service = await get_storage_service()
    try:
        file_record = await service.register_generated_file(
            user_id=user_id,
            filename=filename,
            storage_path=relative_path,
            file_category=FILE_CATEGORY_SPREADSHEET,
            source_feature=SOURCE_FEATURE_DATA_TABLES,
            file_size_bytes=len(spreadsheet_bytes),
            org_id=org_id,
            team_id=team_id,
            original_filename=original_filename or f"export.{spreadsheet_format}",
            mime_type=mime_type,
            checksum=_compute_checksum(spreadsheet_bytes),
            source_ref=source_ref,
            folder_tag=folder_tag,
            tags=tags,
            is_transient=is_transient,
            expires_at=expires_at,
            check_quota=check_quota,
        )

        logger.info(f"Registered spreadsheet: {filename} ({len(spreadsheet_bytes)} bytes) for user {user_id}")
        return file_record

    except Exception:
        with contextlib.suppress(Exception):
            file_path.unlink()
        raise


async def save_and_register_mindmap(
    *,
    user_id: int,
    mindmap_bytes: bytes,
    mindmap_format: str = "json",
    title: str | None = None,
    source_ref: str | None = None,
    org_id: int | None = None,
    team_id: int | None = None,
    folder_tag: str | None = None,
    tags: list[str] | None = None,
    is_transient: bool = False,
    expires_at: datetime | None = None,
    check_quota: bool = True,
) -> dict[str, Any]:
    """
    Save generated mindmap and register with storage tracking.

    Args:
        user_id: Owner user ID
        mindmap_bytes: Mindmap data bytes
        mindmap_format: Format (json, svg, png, etc.)
        title: Mindmap title
        source_ref: Reference to source content
        org_id: Optional organization ID
        team_id: Optional team ID
        folder_tag: Virtual folder tag
        tags: Additional tags
        is_transient: Whether file is temporary
        expires_at: Expiration timestamp
        check_quota: Whether to check quota before saving

    Returns:
        File record dict
    """
    prefix = f"mindmap_{title.replace(' ', '_')[:20]}" if title else "mindmap"
    filename = _generate_filename(prefix, mindmap_format)
    category_folder = "mindmaps"

    file_path = await _save_file(user_id, mindmap_bytes, category_folder, filename)

    outputs_dir = DatabasePaths.get_user_outputs_dir(user_id)
    relative_path = str(file_path.relative_to(outputs_dir))

    # Determine MIME type
    mime_map = {
        "json": "application/json",
        "svg": "image/svg+xml",
        "png": "image/png",
        "html": "text/html",
    }
    mime_type = mime_map.get(mindmap_format.lower(), "application/octet-stream")

    service = await get_storage_service()
    try:
        file_record = await service.register_generated_file(
            user_id=user_id,
            filename=filename,
            storage_path=relative_path,
            file_category=FILE_CATEGORY_MINDMAP,
            source_feature=SOURCE_FEATURE_MINDMAP,
            file_size_bytes=len(mindmap_bytes),
            org_id=org_id,
            team_id=team_id,
            original_filename=f"{title or 'mindmap'}.{mindmap_format}",
            mime_type=mime_type,
            checksum=_compute_checksum(mindmap_bytes),
            source_ref=source_ref,
            folder_tag=folder_tag,
            tags=tags,
            is_transient=is_transient,
            expires_at=expires_at,
            check_quota=check_quota,
        )

        logger.info(f"Registered mindmap: {filename} ({len(mindmap_bytes)} bytes) for user {user_id}")
        return file_record

    except Exception:
        with contextlib.suppress(Exception):
            file_path.unlink()
        raise
