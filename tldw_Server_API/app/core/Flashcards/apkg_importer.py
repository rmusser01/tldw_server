"""Utilities for importing Anki APKG files into flashcard rows."""

from __future__ import annotations

import contextlib
import io
import json
import mimetypes
import os
import re
import sqlite3
import tempfile
import zipfile
from datetime import datetime, timezone
from typing import Any

from tldw_Server_API.app.core.Flashcards.asset_refs import build_flashcard_asset_markdown


class APKGImportError(ValueError):
    """Raised when APKG content cannot be parsed into importable rows."""


DEFAULT_MAX_ARCHIVE_MEMBERS = 100_000
DEFAULT_MAX_COLLECTION_BYTES = 128 * 1024 * 1024
DEFAULT_MAX_MEDIA_MAPPING_BYTES = 5 * 1024 * 1024
DEFAULT_MAX_TOTAL_MEDIA_BYTES = 256 * 1024 * 1024

_IMG_TAG_RE = re.compile(
    r"<img\b(?P<attrs>[^>]*?)\bsrc\s*=\s*(['\"])(?P<src>.+?)\2(?P<tail>[^>]*)>",
    re.IGNORECASE,
)
_ALT_ATTR_RE = re.compile(r"\balt\s*=\s*(['\"])(?P<alt>.*?)\1", re.IGNORECASE)


def _to_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _to_iso_utc(seconds: int) -> str:
    dt = datetime.fromtimestamp(seconds, tz=timezone.utc)
    return dt.isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _day_index_to_epoch_seconds(day_value: int, col_crt_days: int) -> int | None:
    if day_value <= 0:
        return None
    # In Anki exports, review due may be absolute day index or days offset.
    if col_crt_days > 0 and day_value < col_crt_days:
        return (col_crt_days + day_value) * 86400
    return day_value * 86400


def _normalize_col_crt_days(raw_crt_value: int) -> int:
    if raw_crt_value <= 0:
        return 0
    # Some collections store crt as day index, others as epoch seconds.
    if raw_crt_value > 1_000_000_000:
        return raw_crt_value // 86400
    return raw_crt_value


def _derive_due_at(
    *,
    card_type: int,
    queue: int,
    due: int,
    col_crt_days: int,
) -> str | None:
    if due <= 0:
        return None

    # Learning cards often store due as epoch seconds.
    if queue in (1, 3) or card_type == 1:
        if due > 1_000_000_000:
            return _to_iso_utc(due)
        day_due = _day_index_to_epoch_seconds(due, col_crt_days)
        return _to_iso_utc(day_due) if day_due else None

    # Review cards typically store day index.
    if queue == 2 or card_type == 2:
        day_due = _day_index_to_epoch_seconds(due, col_crt_days)
        return _to_iso_utc(day_due) if day_due else None

    return None


def _parse_note_tags(tags_raw: Any) -> list[str]:
    if not isinstance(tags_raw, str):
        return []
    return [tag.strip() for tag in tags_raw.split(" ") if tag and tag.strip()]


def _validate_field_lengths(
    *,
    index: int,
    deck_name: str,
    front: str,
    back: str,
    notes: str | None,
    extra: str | None,
    tags: list[str],
    max_field_length: int,
    errors: list[dict[str, Any]],
) -> bool:
    fields = {
        "Deck": deck_name,
        "Front": front,
        "Back": back,
        "Extra": extra or "",
        "Notes": notes or "",
        "Tags": " ".join(tags),
    }
    for field_name, value in fields.items():
        if len((value or "").encode("utf-8")) > max_field_length:
            errors.append(
                {
                    "index": index,
                    "error": f"Field too long: {field_name} (> {max_field_length} bytes)",
                }
            )
            return False
    return True


def _guess_media_mime(filename: str, content: bytes) -> str:
    guessed, _ = mimetypes.guess_type(filename)
    if guessed:
        return guessed
    if content[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if content[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if content[:4] == b"RIFF" and content[8:12] == b"WEBP":
        return "image/webp"
    return "application/octet-stream"


def _read_zip_info_with_limit(
    zf: zipfile.ZipFile,
    info: zipfile.ZipInfo,
    *,
    max_bytes: int,
    label: str,
) -> bytes:
    """Read one zip member after verifying its declared uncompressed size is bounded."""
    if info.file_size > max_bytes:
        raise APKGImportError(f"APKG {label} exceeds max size of {max_bytes} bytes")
    return zf.read(info)


def _rewrite_media_html_to_markdown(
    value: str | None,
    *,
    media_files: dict[str, bytes],
    asset_importer,
    seen_media_names: set[str],
    total_media_bytes: list[int],
    max_total_media_bytes: int | None,
) -> str | None:
    if not value or asset_importer is None:
        return value

    per_note_asset_cache: dict[str, str] = {}

    def repl(match: re.Match[str]) -> str:
        src = match.group("src").strip()
        if src not in media_files:
            return match.group(0)

        if src not in seen_media_names:
            total_media_bytes[0] += len(media_files[src])
            if max_total_media_bytes is not None and total_media_bytes[0] > max_total_media_bytes:
                raise APKGImportError(
                    f"APKG media exceeds max total size of {max_total_media_bytes} bytes"
                )
            seen_media_names.add(src)

        if src not in per_note_asset_cache:
            asset_uuid = asset_importer(media_files[src], _guess_media_mime(src, media_files[src]), src)
            per_note_asset_cache[src] = str(asset_uuid)

        alt_match = _ALT_ATTR_RE.search(match.group(0))
        alt_text = alt_match.group("alt") if alt_match else os.path.splitext(src)[0]
        return build_flashcard_asset_markdown(per_note_asset_cache[src], alt_text)

    return _IMG_TAG_RE.sub(repl, value)


def import_rows_from_apkg_bytes(
    apkg_bytes: bytes,
    *,
    max_notes: int,
    max_field_length: int,
    max_total_media_bytes: int | None = DEFAULT_MAX_TOTAL_MEDIA_BYTES,
    asset_importer=None,
    max_collection_bytes: int = DEFAULT_MAX_COLLECTION_BYTES,
    max_archive_members: int = DEFAULT_MAX_ARCHIVE_MEMBERS,
    max_media_mapping_bytes: int = DEFAULT_MAX_MEDIA_MAPPING_BYTES,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Parse APKG bytes into normalized flashcard rows.

    Returns `(rows, errors)` where each row includes:
    `deck_name, front, back, extra, tags, model_type, reverse, is_cloze,
    ef, interval_days, repetitions, lapses, due_at`.
    """
    errors: list[dict[str, Any]] = []
    if not apkg_bytes:
        raise APKGImportError("Empty APKG upload")

    try:
        zf = zipfile.ZipFile(io.BytesIO(apkg_bytes))
    except zipfile.BadZipFile as exc:
        raise APKGImportError("Invalid APKG archive") from exc

    with zf:
        archive_members = zf.infolist()
        if len(archive_members) > max_archive_members:
            raise APKGImportError(f"APKG archive exceeds max entry count of {max_archive_members}")

        collection_info = next(
            (
                info
                for info in archive_members
                if os.path.basename(info.filename) in {"collection.anki2", "collection.anki21"}
            ),
            None,
        )
        if not collection_info:
            raise APKGImportError("APKG is missing collection database")
        collection_bytes = _read_zip_info_with_limit(
            zf,
            collection_info,
            max_bytes=max_collection_bytes,
            label="collection database",
        )
        media_files: dict[str, bytes] = {}
        media_infos = {info.filename: info for info in archive_members}
        with contextlib.suppress(KeyError, json.JSONDecodeError):
            media_index_info = media_infos["media"]
            media_mapping = json.loads(
                _read_zip_info_with_limit(
                    zf,
                    media_index_info,
                    max_bytes=max_media_mapping_bytes,
                    label="media mapping",
                ).decode("utf-8")
            )
            if isinstance(media_mapping, dict):
                total_mapped_media_bytes = 0
                for archive_name, original_name in media_mapping.items():
                    media_info = media_infos.get(str(archive_name))
                    if media_info is None:
                        continue
                    total_mapped_media_bytes += int(media_info.file_size)
                    if max_total_media_bytes is not None and total_mapped_media_bytes > max_total_media_bytes:
                        raise APKGImportError(
                            f"APKG media exceeds max total size of {max_total_media_bytes} bytes"
                        )
                    media_files[str(original_name)] = zf.read(media_info)

    with tempfile.TemporaryDirectory() as tmp_dir:
        collection_path = os.path.join(tmp_dir, "collection.anki2")
        with open(collection_path, "wb") as f:
            f.write(collection_bytes)

        try:
            conn = sqlite3.connect(collection_path)
            conn.row_factory = sqlite3.Row
        except sqlite3.Error as exc:
            raise APKGImportError("Failed to open APKG collection database") from exc
        try:
            with conn:
                col_row = conn.execute("SELECT crt, models, decks FROM col LIMIT 1").fetchone()
                if not col_row:
                    raise APKGImportError("APKG collection metadata is missing")

                col_crt_days = _normalize_col_crt_days(_to_int(col_row["crt"], 0))
                try:
                    models = json.loads(col_row["models"] or "{}")
                except json.JSONDecodeError:
                    models = {}
                try:
                    decks = json.loads(col_row["decks"] or "{}")
                except json.JSONDecodeError:
                    decks = {}

                mid_to_model = {
                    _to_int(mid): model
                    for mid, model in (models.items() if isinstance(models, dict) else [])
                    if isinstance(model, dict)
                }
                did_to_name = {
                    _to_int(did): str(deck.get("name") or "Default")
                    for did, deck in (decks.items() if isinstance(decks, dict) else [])
                    if isinstance(deck, dict)
                }

                safe_max_notes = max(0, int(max_notes))
                note_rows = conn.execute(
                    "SELECT id, mid, tags, flds FROM notes ORDER BY id ASC LIMIT ?",
                    (safe_max_notes + 1,),
                ).fetchall()
                parsed_rows: list[dict[str, Any]] = []
                processed = 0
                seen_media_names: set[str] = set()
                total_media_bytes = [0]

                for note_index, note_row in enumerate(note_rows, start=1):
                    if processed >= safe_max_notes:
                        errors.append(
                            {
                                "index": note_index,
                                "error": f"Maximum import item limit reached ({max_notes})",
                            }
                        )
                        break

                    nid = _to_int(note_row["id"], 0)
                    mid = _to_int(note_row["mid"], 0)
                    tags = _parse_note_tags(note_row["tags"])
                    raw_fields = str(note_row["flds"] or "")
                    fields = raw_fields.split("\x1f")

                    card_rows = conn.execute(
                        """
                        SELECT did, ord, type, queue, due, ivl, factor, reps, lapses
                        FROM cards
                        WHERE nid = ?
                        ORDER BY ord ASC, id ASC
                        """,
                        (nid,),
                    ).fetchall()
                    if not card_rows:
                        errors.append({"index": note_index, "error": "Skipped note without cards"})
                        continue

                    primary_card = next(
                        (row for row in card_rows if _to_int(row["ord"], -1) == 0),
                        card_rows[0],
                    )
                    deck_name = did_to_name.get(_to_int(primary_card["did"], 0), "Default")

                    model = mid_to_model.get(mid, {})
                    model_name = str(model.get("name") or "").lower()
                    model_is_cloze = bool(_to_int(model.get("type"), 0) == 1 or "cloze" in model_name)

                    if model_is_cloze:
                        front = fields[0] if len(fields) > 0 else ""
                        back = ""
                        extra = fields[1] if len(fields) > 1 else None
                        notes = fields[2] if len(fields) > 2 else None
                        model_type = "cloze"
                        reverse = False
                    else:
                        front = fields[0] if len(fields) > 0 else ""
                        back = fields[1] if len(fields) > 1 else ""
                        extra = fields[2] if len(fields) > 2 else None
                        notes = fields[3] if len(fields) > 3 else None
                        reverse = any(_to_int(row["ord"], -1) == 1 for row in card_rows)
                        model_type = "basic_reverse" if reverse else "basic"

                    front = str(front or "").strip()
                    back = str(back or "").strip()
                    extra = str(extra or "").strip() or None
                    notes = str(notes or "").strip() or None
                    if not front:
                        errors.append({"index": note_index, "error": "Missing required field: Front"})
                        continue

                    if not _validate_field_lengths(
                        index=note_index,
                        deck_name=deck_name,
                        front=front,
                        back=back,
                        notes=notes,
                        extra=extra,
                        tags=tags,
                        max_field_length=max_field_length,
                        errors=errors,
                    ):
                        continue

                    front = _rewrite_media_html_to_markdown(
                        front,
                        media_files=media_files,
                        asset_importer=asset_importer,
                        seen_media_names=seen_media_names,
                        total_media_bytes=total_media_bytes,
                        max_total_media_bytes=max_total_media_bytes,
                    ) or ""
                    back = _rewrite_media_html_to_markdown(
                        back,
                        media_files=media_files,
                        asset_importer=asset_importer,
                        seen_media_names=seen_media_names,
                        total_media_bytes=total_media_bytes,
                        max_total_media_bytes=max_total_media_bytes,
                    ) or ""
                    extra = _rewrite_media_html_to_markdown(
                        extra,
                        media_files=media_files,
                        asset_importer=asset_importer,
                        seen_media_names=seen_media_names,
                        total_media_bytes=total_media_bytes,
                        max_total_media_bytes=max_total_media_bytes,
                    )
                    notes = _rewrite_media_html_to_markdown(
                        notes,
                        media_files=media_files,
                        asset_importer=asset_importer,
                        seen_media_names=seen_media_names,
                        total_media_bytes=total_media_bytes,
                        max_total_media_bytes=max_total_media_bytes,
                    )

                    card_type = _to_int(primary_card["type"], 0)
                    queue = _to_int(primary_card["queue"], 0)
                    due = _to_int(primary_card["due"], 0)
                    interval_days = max(0, _to_int(primary_card["ivl"], 0))
                    repetitions = max(0, _to_int(primary_card["reps"], 0))
                    lapses = max(0, _to_int(primary_card["lapses"], 0))
                    factor = _to_int(primary_card["factor"], 0)
                    ef = _to_float(factor / 1000.0, 2.5) if factor > 0 else 2.5
                    due_at = _derive_due_at(
                        card_type=card_type,
                        queue=queue,
                        due=due,
                        col_crt_days=col_crt_days,
                    )

                    parsed_rows.append(
                        {
                            "deck_name": deck_name,
                            "front": front,
                            "back": back,
                            "notes": notes,
                            "extra": extra,
                            "tags": tags,
                            "model_type": model_type,
                            "reverse": reverse,
                            "is_cloze": model_type == "cloze",
                            "ef": ef,
                            "interval_days": interval_days,
                            "repetitions": repetitions,
                            "lapses": lapses,
                            "due_at": due_at,
                        }
                    )
                    processed += 1

                return parsed_rows, errors
        except sqlite3.Error as exc:
            raise APKGImportError("Invalid APKG collection schema") from exc
        finally:
            conn.close()
