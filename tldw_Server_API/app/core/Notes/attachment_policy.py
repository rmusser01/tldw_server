"""Pure canonical filename and media-type policy for Notes attachments."""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path

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


class NoteAttachmentPolicyError(ValueError):
    """Raised when attachment metadata is outside the canonical Notes policy."""


def sanitize_note_attachment_file_name(raw_name: object) -> str:
    """Return the canonical path-safe Notes attachment filename."""

    input_name = str(raw_name or "").strip()
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
    return f"{safe_stem[:max_stem_len]}{extension}"


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
    "NoteAttachmentPolicyError",
    "sanitize_note_attachment_file_name",
    "validate_note_attachment_content_type",
]
