"""Pure canonical filename and media-type policy for Notes attachments."""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path

from tldw_Server_API.app.core.exceptions import NoteAttachmentPolicyError
from tldw_Server_API.app.core.Utils.Utils import sanitize_filename

NOTE_ATTACHMENT_MAX_FILENAME_LEN = 180
NOTE_ATTACHMENT_ALLOWED_EXTENSIONS = frozenset(
    {
        ".bmp",
        ".csv",
        ".doc",
        ".docx",
        ".gif",
        ".gz",
        ".jpeg",
        ".jpg",
        ".json",
        ".md",
        ".mp3",
        ".mp4",
        ".m4a",
        ".mov",
        ".ogg",
        ".pdf",
        ".png",
        ".ppt",
        ".pptx",
        ".svg",
        ".tar.gz",
        ".txt",
        ".wav",
        ".webm",
        ".webp",
        ".xlsx",
        ".xls",
        ".yaml",
        ".yml",
        ".zip",
    }
)
_MEDIA_TYPE_RE = re.compile(r"[a-z0-9!#$%&'*+.^_`|~-]+/[a-z0-9!#$%&'*+.^_`|~-]+\Z")


def canonicalize_note_attachment_file_name(raw_name: object) -> tuple[str, str]:
    """Return the canonical display filename and bounded comparison key."""

    input_name = raw_name.strip() if isinstance(raw_name, str) else ""
    if not input_name:
        raise NoteAttachmentPolicyError("Attachment filename is required")
    normalized_input = unicodedata.normalize("NFKC", input_name)
    if (
        "/" in normalized_input
        or "\\" in normalized_input
        or any(ord(character) < 32 for character in normalized_input)
    ):
        raise NoteAttachmentPolicyError("Invalid attachment filename")
    basename = Path(normalized_input).name
    if basename != normalized_input:
        raise NoteAttachmentPolicyError("Invalid attachment filename")

    suffixes = [suffix.lower() for suffix in Path(basename).suffixes]
    full_extension = "".join(suffixes)
    if full_extension in NOTE_ATTACHMENT_ALLOWED_EXTENSIONS:
        extension = full_extension
    elif suffixes and suffixes[-1] in NOTE_ATTACHMENT_ALLOWED_EXTENSIONS:
        extension = suffixes[-1]
    else:
        allowed = ", ".join(sorted(NOTE_ATTACHMENT_ALLOWED_EXTENSIONS))
        raise NoteAttachmentPolicyError(
            f"Unsupported attachment type. Allowed extensions: {allowed}"
        )

    stem = basename[: -len(extension)] if len(extension) < len(basename) else "attachment"
    max_stem_len = max(1, NOTE_ATTACHMENT_MAX_FILENAME_LEN - len(extension))
    safe_stem = (
        sanitize_filename(stem, max_total_length=max_stem_len)
        .replace(" ", "_")
        .strip("._")
    )
    if not safe_stem:
        safe_stem = "attachment"
    display_name = f"{safe_stem[:max_stem_len]}{extension}"
    normalized_key = unicodedata.normalize("NFKC", display_name).casefold()
    if not 1 <= len(normalized_key) <= NOTE_ATTACHMENT_MAX_FILENAME_LEN:
        raise NoteAttachmentPolicyError(
            "The normalized attachment filename exceeds its safe basename boundary"
        )
    return display_name, normalized_key


def sanitize_note_attachment_file_name(raw_name: object) -> str:
    """Return the canonical path-safe Notes attachment display filename."""

    return canonicalize_note_attachment_file_name(raw_name)[0]


def validate_note_attachment_original_file_name(value: object) -> str:
    """Return an unchanged original filename after safe-basename validation."""

    if not isinstance(value, str) or not value or value != value.strip():
        raise NoteAttachmentPolicyError(
            "Attachment original_file_name must be a non-empty safe basename"
        )
    normalized = unicodedata.normalize("NFKC", value)
    if (
        len(value) > 255
        or normalized in {".", ".."}
        or "/" in normalized
        or "\\" in normalized
        or Path(normalized).name != normalized
        or any(ord(character) < 32 for character in normalized)
    ):
        raise NoteAttachmentPolicyError(
            "Attachment original_file_name exceeds its safe basename boundary"
        )
    try:
        encoded = value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise NoteAttachmentPolicyError(
            "Attachment original_file_name exceeds its UTF-8 byte boundary"
        ) from exc
    if len(encoded) > 1024:
        raise NoteAttachmentPolicyError(
            "Attachment original_file_name exceeds its UTF-8 byte boundary"
        )
    return value


def validate_note_attachment_content_type(value: object) -> str:
    """Return an already-canonical lowercase ``type/subtype`` media type."""

    if (
        not isinstance(value, str)
        or not 1 <= len(value) <= 255
        or value != value.strip()
        or value != value.lower()
        or _MEDIA_TYPE_RE.fullmatch(value) is None
    ):
        raise NoteAttachmentPolicyError(
            "Attachment content_type must be a canonical normalized media type"
        )
    return value


__all__ = [
    "NOTE_ATTACHMENT_ALLOWED_EXTENSIONS",
    "NOTE_ATTACHMENT_MAX_FILENAME_LEN",
    "canonicalize_note_attachment_file_name",
    "sanitize_note_attachment_file_name",
    "validate_note_attachment_content_type",
    "validate_note_attachment_original_file_name",
]
