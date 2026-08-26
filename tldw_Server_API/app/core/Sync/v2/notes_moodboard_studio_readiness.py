from __future__ import annotations

"""Pure validation helpers for dormant moodboard and Studio readiness metadata."""

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TypeAlias
from uuid import RFC_4122, UUID

NOTES_MOODBOARD_STUDIO_READINESS_KEYS = frozenset(
    {
        "state",
        "source_cursor",
        "source_count",
        "source_fingerprint",
        "reason_code",
        "resume_phase",
    }
)
NOTES_MOODBOARD_STUDIO_READINESS_STATES = frozenset(
    {
        "not_enrolled",
        "enrolling",
        "bootstrapping",
        "verifying",
        "ready",
        "blocked",
    }
)
NOTES_MOODBOARD_STUDIO_READINESS_REASON_CODES_BY_KEY = {
    "notes_moodboard_v1": frozenset(
        {
            "notes_moodboard_source_invalid",
            "notes_moodboard_source_changed",
            "notes_moodboard_source_scope_invalid",
            "notes_moodboard_source_catalog_invalid",
            "notes_moodboard_verification_failed",
        }
    ),
    "notes_moodboard_note_v1": frozenset(
        {
            "notes_moodboard_note_source_invalid",
            "notes_moodboard_note_source_changed",
            "notes_moodboard_note_source_scope_invalid",
            "notes_moodboard_note_source_catalog_invalid",
            "notes_moodboard_note_verification_failed",
        }
    ),
    "notes_studio_document_v1": frozenset(
        {
            "notes_studio_document_source_invalid",
            "notes_studio_document_source_changed",
            "notes_studio_document_source_scope_invalid",
            "notes_studio_document_source_catalog_invalid",
            "notes_studio_document_verification_failed",
        }
    ),
}
NOTES_MOODBOARD_STUDIO_SERVER_METADATA_KEYS = frozenset(
    {
        *NOTES_MOODBOARD_STUDIO_READINESS_REASON_CODES_BY_KEY,
        "moodboard_capture_enabled",
        "studio_document_capture_enabled",
    }
)
NOTES_MOODBOARD_STUDIO_READINESS_COUNT_MAX = 9_223_372_036_854_775_807
_RESUME_PHASES = frozenset({"bootstrapping", "verifying"})
_FINGERPRINT = re.compile(r"[0-9a-f]{64}")

CursorOrderKey: TypeAlias = UUID | tuple[UUID, UUID]


@dataclass(frozen=True, slots=True)
class NotesMoodboardStudioReadinessRecord:
    """Validated readiness record plus its domain-specific cursor order key."""

    state: str
    source_cursor: str | None
    source_count: int
    source_fingerprint: str | None
    reason_code: str | None
    resume_phase: str | None
    source_cursor_key: CursorOrderKey | None

    def as_metadata(self) -> dict[str, object]:
        """Return the exact persisted metadata representation."""

        return {
            "state": self.state,
            "source_cursor": self.source_cursor,
            "source_count": self.source_count,
            "source_fingerprint": self.source_fingerprint,
            "reason_code": self.reason_code,
            "resume_phase": self.resume_phase,
        }


@dataclass(frozen=True, slots=True)
class NotesMoodboardStudioReadinessParseResult:
    """Total parser outcome with a stable, non-sensitive error code."""

    record: NotesMoodboardStudioReadinessRecord | None = None
    error_code: str | None = None


def default_notes_moodboard_studio_readiness_record() -> (
    NotesMoodboardStudioReadinessRecord
):
    """Return the implicit readiness record for an absent domain key."""

    return NotesMoodboardStudioReadinessRecord(
        state="not_enrolled",
        source_cursor=None,
        source_count=0,
        source_fingerprint=None,
        reason_code=None,
        resume_phase=None,
        source_cursor_key=None,
    )


def redact_notes_moodboard_studio_server_metadata(
    metadata: Mapping[str, object],
) -> dict[str, object]:
    """Return public dataset metadata without internal readiness state."""

    return {
        key: value
        for key, value in metadata.items()
        if key not in NOTES_MOODBOARD_STUDIO_SERVER_METADATA_KEYS
    }


def parse_notes_moodboard_studio_readiness_record(
    raw: object,
    *,
    readiness_key: str,
) -> NotesMoodboardStudioReadinessParseResult:
    """Parse one exact dormant moodboard/Studio readiness record without raising."""

    if not isinstance(raw, dict) or set(raw) != NOTES_MOODBOARD_STUDIO_READINESS_KEYS:
        return _error("notes_moodboard_studio_readiness_state_invalid")
    reason_codes = NOTES_MOODBOARD_STUDIO_READINESS_REASON_CODES_BY_KEY.get(
        readiness_key
    )
    if reason_codes is None:
        return _error("notes_moodboard_studio_readiness_state_invalid")

    state = raw.get("state")
    source_cursor = raw.get("source_cursor")
    source_count = raw.get("source_count")
    source_fingerprint = raw.get("source_fingerprint")
    reason_code = raw.get("reason_code")
    resume_phase = raw.get("resume_phase")

    if (
        not isinstance(state, str)
        or state not in NOTES_MOODBOARD_STUDIO_READINESS_STATES
    ):
        return _error("notes_moodboard_studio_readiness_state_invalid")
    if (
        isinstance(source_count, bool)
        or not isinstance(source_count, int)
        or not 0 <= source_count <= NOTES_MOODBOARD_STUDIO_READINESS_COUNT_MAX
    ):
        return _error("notes_moodboard_studio_readiness_progress_invalid")
    if source_fingerprint is not None and (
        not isinstance(source_fingerprint, str)
        or _FINGERPRINT.fullmatch(source_fingerprint) is None
    ):
        return _error("notes_moodboard_studio_readiness_fingerprint_invalid")
    if reason_code is not None and (
        not isinstance(reason_code, str) or reason_code not in reason_codes
    ):
        return _error("notes_moodboard_studio_readiness_reason_invalid")
    if state == "blocked":
        if reason_code is None:
            return _error("notes_moodboard_studio_readiness_reason_invalid")
        if not isinstance(resume_phase, str) or resume_phase not in _RESUME_PHASES:
            return _error("notes_moodboard_studio_readiness_state_invalid")
        if resume_phase == "verifying" and source_fingerprint is None:
            return _error("notes_moodboard_studio_readiness_state_invalid")
    elif reason_code is not None:
        return _error("notes_moodboard_studio_readiness_reason_invalid")
    elif resume_phase is not None:
        return _error("notes_moodboard_studio_readiness_state_invalid")

    cursor_key = _parse_cursor(source_cursor, readiness_key)
    if source_cursor is not None and cursor_key is None:
        return _error("notes_moodboard_studio_readiness_cursor_invalid")
    if state in {"not_enrolled", "enrolling"} and (
        source_cursor is not None
        or source_count != 0
        or source_fingerprint is not None
    ):
        return _error("notes_moodboard_studio_readiness_progress_invalid")
    if (source_cursor is None) != (source_count == 0):
        return _error("notes_moodboard_studio_readiness_progress_invalid")
    if state in {"verifying", "ready"} and source_fingerprint is None:
        return _error("notes_moodboard_studio_readiness_fingerprint_invalid")
    if source_cursor is not None and source_fingerprint is None:
        return _error("notes_moodboard_studio_readiness_progress_invalid")

    return NotesMoodboardStudioReadinessParseResult(
        record=NotesMoodboardStudioReadinessRecord(
            state=state,
            source_cursor=source_cursor,
            source_count=source_count,
            source_fingerprint=source_fingerprint,
            reason_code=reason_code,
            resume_phase=resume_phase,
            source_cursor_key=cursor_key,
        )
    )


def _parse_cursor(value: object, readiness_key: str) -> CursorOrderKey | None:
    if value is None:
        return None
    if not isinstance(value, str):
        return None
    if readiness_key in {"notes_moodboard_v1", "notes_studio_document_v1"}:
        return _parse_uuid(value)
    if readiness_key != "notes_moodboard_note_v1":
        return None
    try:
        board_id_text, note_id_text = value.split("|", 1)
    except ValueError:
        return None
    board_id = _parse_uuid(board_id_text)
    note_id = _parse_uuid(note_id_text)
    if board_id is None or note_id is None:
        return None
    return board_id, note_id


def _parse_uuid(value: str) -> UUID | None:
    try:
        parsed = UUID(value)
    except (AttributeError, ValueError):
        return None
    if parsed.version != 4 or parsed.variant != RFC_4122 or str(parsed) != value:
        return None
    return parsed


def _error(error_code: str) -> NotesMoodboardStudioReadinessParseResult:
    return NotesMoodboardStudioReadinessParseResult(error_code=error_code)
