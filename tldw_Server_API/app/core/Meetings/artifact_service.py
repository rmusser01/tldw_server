"""Artifact-level domain logic for Meetings."""

from __future__ import annotations

import re
from typing import Any

from tldw_Server_API.app.core.DB_Management.Meetings_DB import MeetingsDatabase

_DEFAULT_FINAL_KINDS: tuple[str, ...] = ("summary", "action_items", "decisions", "speaker_stats")
_FINALIZABLE_KINDS = set(_DEFAULT_FINAL_KINDS)


class MeetingArtifactService:
    """High-level operations for meeting artifacts."""

    def __init__(self, db: MeetingsDatabase) -> None:
        self._db = db

    def create_artifact(
        self,
        *,
        session_id: str,
        kind: str,
        format: str,
        payload_json: dict[str, Any],
        version: int = 1,
    ) -> dict[str, Any]:
        artifact_id = self._db.create_artifact(
            session_id=session_id,
            kind=kind,
            format=format,
            payload_json=payload_json,
            version=version,
        )
        return self.get_artifact(artifact_id=artifact_id)

    def get_artifact(self, *, artifact_id: str) -> dict[str, Any]:
        row = self._db.get_artifact(artifact_id=artifact_id)
        if row is None:
            raise KeyError(f"meeting artifact not found: {artifact_id}")
        return row

    def list_artifacts(self, *, session_id: str) -> list[dict[str, Any]]:
        return self._db.list_artifacts(session_id=session_id)

    def generate_final_artifacts(
        self,
        *,
        session_id: str,
        transcript_text: str,
        include: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        clean_transcript = str(transcript_text).strip()
        if not clean_transcript:
            raise ValueError("transcript_text is required")
        if self._db.get_session(session_id=session_id) is None:
            raise KeyError(f"meeting session not found: {session_id}")

        payloads = self._build_finalize_payloads(clean_transcript)
        requested_kinds = self._normalize_requested_kinds(include=include)
        unsupported = [kind for kind in requested_kinds if kind not in _FINALIZABLE_KINDS]
        if unsupported:
            raise ValueError(f"meeting artifact kinds are not finalizable: {', '.join(unsupported)}")

        artifact_specs = [
            {
                "kind": kind,
                "format": "json",
                "payload_json": payloads[kind],
                "version": 1,
            }
            for kind in requested_kinds
        ]
        replace_kinds = requested_kinds
        if include is not None and not requested_kinds:
            replace_kinds = list(_DEFAULT_FINAL_KINDS)
        artifact_ids = self._db.replace_artifacts(
            session_id=session_id,
            artifacts=artifact_specs,
            replace_kinds=replace_kinds,
            replace_version=1,
        )
        return [self.get_artifact(artifact_id=artifact_id) for artifact_id in artifact_ids]

    @staticmethod
    def _normalize_requested_kinds(*, include: list[str] | None) -> list[str]:
        """Return ordered, lower-cased, de-duplicated final artifact kinds.

        `None` requests the default final artifact set. An explicit list,
        including an empty list, is preserved as the caller's requested scope
        after trimming blank values and dropping duplicates.
        """
        raw_kinds = list(_DEFAULT_FINAL_KINDS) if include is None else include
        requested_kinds: list[str] = []
        seen: set[str] = set()
        for kind in raw_kinds:
            normalized_kind = str(kind).strip().lower()
            if not normalized_kind or normalized_kind in seen:
                continue
            seen.add(normalized_kind)
            requested_kinds.append(normalized_kind)
        return requested_kinds

    @staticmethod
    def _build_finalize_payloads(transcript_text: str) -> dict[str, dict[str, Any]]:
        summary = MeetingArtifactService._build_summary(transcript_text)
        action_items = MeetingArtifactService._extract_action_items(transcript_text)
        decisions = MeetingArtifactService._extract_decisions(transcript_text)
        speaker_stats = {
            "word_count": len([token for token in transcript_text.split() if token.strip()]),
            "line_count": len([line for line in transcript_text.splitlines() if line.strip()]),
        }
        return {
            "summary": {"text": summary},
            "action_items": {"items": action_items},
            "decisions": {"items": decisions},
            "speaker_stats": speaker_stats,
        }

    @staticmethod
    def _build_summary(transcript_text: str) -> str:
        collapsed = " ".join(part.strip() for part in transcript_text.splitlines() if part.strip())
        if len(collapsed) <= 240:
            return collapsed
        return f"{collapsed[:237].rstrip()}..."

    @staticmethod
    def _extract_action_items(transcript_text: str) -> list[str]:
        matches = re.findall(r"(?:^|\b)(?:TODO|ACTION)[:\-]\s*([^\.\n]+)", transcript_text, flags=re.IGNORECASE)
        items = [match.strip() for match in matches if match.strip()]
        if items:
            return items
        return []

    @staticmethod
    def _extract_decisions(transcript_text: str) -> list[str]:
        matches = re.findall(r"(?:^|\b)(?:DECISION|DECIDED)[:\-]\s*([^\.\n]+)", transcript_text, flags=re.IGNORECASE)
        decisions = [match.strip() for match in matches if match.strip()]
        if decisions:
            return decisions
        return []
