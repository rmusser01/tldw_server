"""Pure canonical filename and media-type policy for Notes attachments."""

from __future__ import annotations

import json
import re
import unicodedata
import zipfile
from io import BytesIO
from pathlib import Path

from defusedxml import ElementTree

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
_NOTE_ATTACHMENT_MEDIA_TYPES = {
    ".bmp": frozenset({"image/bmp"}),
    ".csv": frozenset({"text/csv"}),
    ".doc": frozenset({"application/msword"}),
    ".docx": frozenset(
        {"application/vnd.openxmlformats-officedocument.wordprocessingml.document"}
    ),
    ".gif": frozenset({"image/gif"}),
    ".gz": frozenset({"application/gzip", "application/x-gzip"}),
    ".jpeg": frozenset({"image/jpeg"}),
    ".jpg": frozenset({"image/jpeg"}),
    ".json": frozenset({"application/json"}),
    ".m4a": frozenset({"audio/mp4", "audio/x-m4a"}),
    ".md": frozenset({"text/markdown", "text/plain"}),
    ".mov": frozenset({"video/quicktime"}),
    ".mp3": frozenset({"audio/mpeg"}),
    ".mp4": frozenset({"video/mp4"}),
    ".ogg": frozenset({"audio/ogg", "application/ogg"}),
    ".pdf": frozenset({"application/pdf"}),
    ".png": frozenset({"image/png"}),
    ".ppt": frozenset({"application/vnd.ms-powerpoint"}),
    ".pptx": frozenset(
        {"application/vnd.openxmlformats-officedocument.presentationml.presentation"}
    ),
    ".svg": frozenset({"image/svg+xml"}),
    ".tar.gz": frozenset({"application/gzip", "application/x-gzip"}),
    ".txt": frozenset({"text/plain"}),
    ".wav": frozenset({"audio/wav", "audio/x-wav", "audio/wave"}),
    ".webm": frozenset({"video/webm"}),
    ".webp": frozenset({"image/webp"}),
    ".xls": frozenset({"application/vnd.ms-excel"}),
    ".xlsx": frozenset(
        {"application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"}
    ),
    ".yaml": frozenset({"application/yaml", "text/yaml", "text/x-yaml"}),
    ".yml": frozenset({"application/yaml", "text/yaml", "text/x-yaml"}),
    ".zip": frozenset({"application/zip", "application/x-zip-compressed"}),
}
_NOTE_ATTACHMENT_TEXT_EXTENSIONS = frozenset(
    {".csv", ".json", ".md", ".svg", ".txt", ".yaml", ".yml"}
)
_NOTE_ATTACHMENT_ZIP_EXTENSIONS = frozenset({".docx", ".pptx", ".xlsx", ".zip"})
_NOTE_ATTACHMENT_OFFICE_PREFIXES = {
    ".docx": "word/",
    ".pptx": "ppt/",
    ".xlsx": "xl/",
}
_NOTE_ATTACHMENT_CFB_EXTENSIONS = frozenset({".doc", ".ppt", ".xls"})
_NOTE_ATTACHMENT_PREFERRED_MEDIA_TYPES = {
    ".gz": "application/gzip",
    ".md": "text/markdown",
    ".ogg": "audio/ogg",
    ".tar.gz": "application/gzip",
    ".wav": "audio/wav",
    ".yaml": "application/yaml",
    ".yml": "application/yaml",
    ".zip": "application/zip",
}


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


def _note_attachment_extension(file_name: str) -> str:
    """Return the canonical allow-listed extension for a safe attachment name."""

    suffixes = [suffix.lower() for suffix in Path(file_name).suffixes]
    full_extension = "".join(suffixes)
    if full_extension in NOTE_ATTACHMENT_ALLOWED_EXTENSIONS:
        return full_extension
    if suffixes and suffixes[-1] in NOTE_ATTACHMENT_ALLOWED_EXTENSIONS:
        return suffixes[-1]
    raise NoteAttachmentPolicyError("Unsupported attachment type")


def _validate_note_attachment_text(payload: bytes, extension: str) -> None:
    """Reject malformed structured text and active SVG content."""

    try:
        decoded = payload.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise NoteAttachmentPolicyError(
            "Attachment content does not match its filename type"
        ) from exc
    if "\x00" in decoded:
        raise NoteAttachmentPolicyError(
            "Attachment content does not match its filename type"
        )
    if extension == ".json":
        try:
            json.loads(decoded)
        except (TypeError, ValueError) as exc:
            raise NoteAttachmentPolicyError("Attachment JSON content is invalid") from exc
    if extension != ".svg":
        return
    try:
        root = ElementTree.fromstring(decoded)
    except ElementTree.ParseError as exc:
        raise NoteAttachmentPolicyError("Attachment SVG content is invalid") from exc
    if root.tag.rsplit("}", 1)[-1].lower() != "svg":
        raise NoteAttachmentPolicyError("Attachment SVG content is invalid")
    for element in root.iter():
        if element.tag.rsplit("}", 1)[-1].lower() == "script":
            raise NoteAttachmentPolicyError("Attachment SVG active content is not allowed")
        for attribute, value in element.attrib.items():
            local_attribute = attribute.rsplit("}", 1)[-1].lower()
            if local_attribute.startswith("on") or str(value).strip().lower().startswith(
                "javascript:"
            ):
                raise NoteAttachmentPolicyError(
                    "Attachment SVG active content is not allowed"
                )


def _validate_note_attachment_zip(payload: bytes, extension: str) -> None:
    """Validate ZIP structure and the expected OOXML container family."""

    try:
        with zipfile.ZipFile(BytesIO(payload)) as archive:
            members = archive.infolist()
    except (OSError, zipfile.BadZipFile, zipfile.LargeZipFile) as exc:
        raise NoteAttachmentPolicyError(
            "Attachment content does not match its filename type"
        ) from exc
    if len(members) > 10_000 or sum(member.file_size for member in members) > 1_073_741_824:
        raise NoteAttachmentPolicyError("Attachment archive metadata exceeds safe limits")
    member_names = {member.filename for member in members}
    office_prefix = _NOTE_ATTACHMENT_OFFICE_PREFIXES.get(extension)
    if office_prefix is not None and (
        "[Content_Types].xml" not in member_names
        or not any(name.startswith(office_prefix) for name in member_names)
    ):
        raise NoteAttachmentPolicyError(
            "Attachment content does not match its filename type"
        )


def validate_note_attachment_upload_content(
    *,
    file_name: str,
    declared_content_type: object,
    payload: bytes,
) -> str:
    """Validate one-shot upload bytes and return a server-derived media type."""

    canonical_name, _ = canonicalize_note_attachment_file_name(file_name)
    extension = _note_attachment_extension(canonical_name)
    declared = validate_note_attachment_content_type(declared_content_type)
    allowed_media_types = _NOTE_ATTACHMENT_MEDIA_TYPES.get(extension)
    if allowed_media_types is None or declared not in allowed_media_types:
        raise NoteAttachmentPolicyError(
            "Attachment content_type does not match its filename type"
        )
    if extension in _NOTE_ATTACHMENT_TEXT_EXTENSIONS:
        _validate_note_attachment_text(payload, extension)
    elif extension in _NOTE_ATTACHMENT_ZIP_EXTENSIONS:
        _validate_note_attachment_zip(payload, extension)
    elif extension in _NOTE_ATTACHMENT_CFB_EXTENSIONS:
        if not payload.startswith(b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"):
            raise NoteAttachmentPolicyError(
                "Attachment content does not match its filename type"
            )
    elif extension in {".gz", ".tar.gz"}:
        if not payload.startswith(b"\x1f\x8b"):
            raise NoteAttachmentPolicyError(
                "Attachment content does not match its filename type"
            )
    else:
        signatures = {
            ".bmp": payload.startswith(b"BM"),
            ".gif": payload.startswith((b"GIF87a", b"GIF89a")),
            ".jpeg": payload.startswith(b"\xff\xd8\xff"),
            ".jpg": payload.startswith(b"\xff\xd8\xff"),
            ".m4a": len(payload) >= 12 and payload[4:8] == b"ftyp",
            ".mov": len(payload) >= 12 and payload[4:8] == b"ftyp",
            ".mp3": payload.startswith(b"ID3")
            or (len(payload) >= 2 and payload[0] == 0xFF and payload[1] & 0xE0 == 0xE0),
            ".mp4": len(payload) >= 12 and payload[4:8] == b"ftyp",
            ".ogg": payload.startswith(b"OggS"),
            ".pdf": payload.startswith(b"%PDF-"),
            ".png": payload.startswith(b"\x89PNG\r\n\x1a\n"),
            ".wav": len(payload) >= 12
            and payload.startswith(b"RIFF")
            and payload[8:12] == b"WAVE",
            ".webm": payload.startswith(b"\x1a\x45\xdf\xa3"),
            ".webp": len(payload) >= 12
            and payload.startswith(b"RIFF")
            and payload[8:12] == b"WEBP",
        }
        if not signatures.get(extension, False):
            raise NoteAttachmentPolicyError(
                "Attachment content does not match its filename type"
            )
    return _NOTE_ATTACHMENT_PREFERRED_MEDIA_TYPES.get(
        extension,
        min(allowed_media_types, key=len),
    )


__all__ = [
    "NOTE_ATTACHMENT_ALLOWED_EXTENSIONS",
    "NOTE_ATTACHMENT_MAX_FILENAME_LEN",
    "canonicalize_note_attachment_file_name",
    "sanitize_note_attachment_file_name",
    "validate_note_attachment_content_type",
    "validate_note_attachment_original_file_name",
    "validate_note_attachment_upload_content",
]
