"""Read complete native archives for the independent Buddy snapshot store."""

from __future__ import annotations

import os
import zipfile
import zlib
from dataclasses import dataclass
from typing import Any, BinaryIO

from tldw_Server_API.app.core.exceptions import BuddyConflictError
from tldw_Server_API.app.core.Persona.visual_artwork import artwork_manifest_for_import
from tldw_Server_API.app.core.Persona.visual_portability.archive import normalize_member_name, validate_archive_members
from tldw_Server_API.app.core.Persona.visual_portability.fingerprints import sha256_stream
from tldw_Server_API.app.core.Persona.visual_portability.preview import read_native_visual_contents
from tldw_Server_API.app.core.Persona.visual_service import MAX_VISUAL_UPLOAD_BYTES

MAX_BUDDY_ASSETS = 256
MAX_BUDDY_ART_BYTES = 64 * 1024 * 1024
MAX_BUDDY_ARCHIVE_BYTES = MAX_BUDDY_ART_BYTES + 4 * 1024 * 1024


@dataclass(frozen=True)
class BuddyArchive:
    """Validated archive payload before raster and runtime validation."""

    name: str
    sha256: str
    manifest: dict[str, Any]
    assets: list[tuple[str, bytes, str]]


def read_buddy_archive(source: BinaryIO, *, reviewed_sha256: str | None = None) -> BuddyArchive:
    """Validate a bounded ZIP and read its artwork without extracting archive paths."""
    if not 0 < source.seek(0, os.SEEK_END) <= MAX_BUDDY_ARCHIVE_BYTES:
        raise ValueError("Choose a nonempty avatar pack no larger than 68 MiB.")
    source.seek(0)
    digest = sha256_stream(source)
    if reviewed_sha256 is not None and digest != reviewed_sha256:
        raise BuddyConflictError("The avatar pack changed. Choose the file again to review it.")

    try:
        validate_archive_members(
            source,
            max_member_size_bytes=MAX_VISUAL_UPLOAD_BYTES,
            max_archive_size_bytes=MAX_BUDDY_ARCHIVE_BYTES,
            max_members=MAX_BUDDY_ASSETS * 4,
        )
        with zipfile.ZipFile(source) as archive:
            contents = read_native_visual_contents(
                archive, max_asset_items=MAX_BUDDY_ASSETS, max_asset_bytes=MAX_BUDDY_ART_BYTES
            )
            if not contents.assets or contents.asset_summary["missing_asset_items"]:
                raise ValueError("The avatar pack must contain all its artwork within the import limits.")
            assets = []
            ids: set[str] = set()
            for asset in contents.assets:
                asset_id = asset["source_asset_id"]
                if not isinstance(asset_id, str) or asset_id in ids:
                    raise ValueError("The avatar pack has invalid or duplicate artwork IDs.")
                ids.add(asset_id)
                member = contents.members[normalize_member_name(str(asset["asset_path"]))]
                assets.append((asset_id, archive.read(member), str(asset.get("mime_type") or "")))
    except (zipfile.BadZipFile, NotImplementedError, RecursionError, zlib.error) as exc:
        raise ValueError("The avatar pack could not be read as a native archive.") from exc

    title = contents.pack.get("title")
    name = title.strip()[:120].rstrip() if isinstance(title, str) else ""
    if "\x00" in name:
        raise ValueError("The avatar pack has an invalid name.")
    try:
        name.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ValueError("The avatar pack has an invalid name.") from exc
    return BuddyArchive(name or "Imported avatar", digest, artwork_manifest_for_import(contents.pack), assets)
