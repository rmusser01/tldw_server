"""Deterministic, operation-owned cloning for shared Workspaces."""
from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from loguru import logger

from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.DB_Management.media_db.api import (
    OperationOwnedMediaReadiness,
    OperationOwnedMediaResult,
    hash_media_clone_snapshot,
)
from tldw_Server_API.app.core.DB_Management.media_db.native_class import MediaDatabase
from tldw_Server_API.app.core.Sharing.clone_models import (
    CloneCancelled,
    CloneCopyCounts,
    ClonePersistenceError,
    CloneRetrievalReadiness,
    CloneSnapshotUnavailable,
    CloneWarning,
    MediaCloneSnapshot,
    WorkspaceCloneRequest,
    WorkspaceCloneResult,
    WorkspaceCloneSnapshot,
)

_PROGRESS_PHASES = frozenset(
    {
        "queued",
        "authorizing",
        "preparing",
        "sources",
        "notes",
        "artifacts",
        "finalizing",
    }
)
_RESOURCE_ID_MAX_LENGTH = 255
_POSITIVE_INTEGER_PATTERN = re.compile(r"[0-9]+")
_SUPPORTED_MEMBERSHIP_TYPES = frozenset(
    {"media", "workspace_source", "workspace_artifact", "workspace_note"}
)
_ARTIFACT_SAFE_FIELDS = (
    "artifact_type",
    "title",
    "status",
    "content",
    "total_tokens",
    "total_cost_usd",
    "completed_at",
    "content_type",
    "preview_text",
    "summary",
    "review_state",
    "producer_metadata",
    "source_lineage",
    "version_metadata",
    "export_refs",
    "redaction",
    "schema_version",
)


def _safe_exception_type(exc: BaseException) -> str:
    """Return a log-safe exception type label without exception details."""
    exc_type = exc.__class__.__name__
    if exc_type and all(character.isalnum() or character == "_" for character in exc_type):
        return exc_type
    return "Exception"


def _thaw_json(value: Any) -> Any:
    """Convert immutable snapshot containers into JSON-compatible containers."""
    if isinstance(value, Mapping):
        return {str(key): _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    if isinstance(value, frozenset):
        return sorted((_thaw_json(item) for item in value), key=repr)
    return value


class _CancellationRequested(Exception):
    """Internal control-flow marker for cooperative cancellation."""


class _PublicationPending(Exception):
    """Internal marker for an exact reservation already awaiting publication."""


class _FatalClone(Exception):
    """Internal bounded fatal failure classification."""

    def __init__(
        self,
        code: str,
        cause: BaseException | None = None,
        *,
        cleanup_ambiguous: bool = False,
    ) -> None:
        super().__init__(code)
        self.code = code
        self.cause = cause
        self.cleanup_ambiguous = cleanup_ambiguous


@dataclass(slots=True)
class _ProgressReporter:
    callback: Callable[[str, float], None] | None
    last_fraction: float = field(init=False, default=0.0)

    def emit(self, phase: str, fraction: float) -> None:
        if phase not in _PROGRESS_PHASES:
            raise ValueError("unsupported clone progress phase")
        if not 0.0 <= fraction <= 1.0 or fraction < self.last_fraction:
            raise ValueError("clone progress must be monotonic and bounded")
        self.last_fraction = fraction
        if self.callback is not None:
            self.callback(phase, fraction)


@dataclass(slots=True)
class _PreparedSnapshot:
    workspace: WorkspaceCloneSnapshot
    media_ids: tuple[int, ...]
    source_media_ids: tuple[int | None, ...]
    membership_media_ids: tuple[int | None, ...]
    media_snapshots: Mapping[int, MediaCloneSnapshot]
    media_hashes: Mapping[int, str]


@dataclass(slots=True)
class _TrackedMedia:
    source_media_id: int
    source_identity: str
    content_hash: str
    result: OperationOwnedMediaResult
    reference_count: int = 0


@dataclass(slots=True)
class _CopyState:
    warnings: dict[str, int] = field(default_factory=dict)
    tracked_media: dict[int, _TrackedMedia] = field(default_factory=dict)
    media_failed_once: set[int] = field(default_factory=set)
    copied_source_ids: set[str] = field(default_factory=set)
    copied_source_media_ids: dict[str, int] = field(default_factory=dict)
    copied_artifact_ids: set[str] = field(default_factory=set)
    note_id_map: dict[str, str] = field(default_factory=dict)
    successful_source_media_ids: set[int] = field(default_factory=set)
    sources_copied: int = 0
    sources_failed: int = 0
    notes_copied: int = 0
    notes_failed: int = 0
    artifacts_copied: int = 0
    artifacts_failed: int = 0

    def warn(self, code: str, count: int = 1) -> None:
        self.warnings[code] = self.warnings.get(code, 0) + count


class CloneService:
    """Copy immutable source snapshots into deterministic staged targets."""

    def __init__(
        self,
        source_chacha_db: CharactersRAGDB,
        source_media_db: MediaDatabase,
        target_chacha_db: CharactersRAGDB,
        target_media_db: MediaDatabase,
        *,
        vector_retrieval_configured: bool = False,
    ) -> None:
        if not isinstance(vector_retrieval_configured, bool):
            raise TypeError("vector_retrieval_configured must be a boolean")
        self._src_chacha = source_chacha_db
        self._src_media = source_media_db
        self._tgt_chacha = target_chacha_db
        self._tgt_media = target_media_db
        self._vector_retrieval_configured = vector_retrieval_configured

    def clone_workspace(
        self,
        request: WorkspaceCloneRequest,
        *,
        should_cancel: Callable[[], bool],
        on_progress: Callable[[str, float], None] | None = None,
    ) -> WorkspaceCloneResult:
        """Clone one Workspace snapshot and leave its target publication pending."""
        if not isinstance(request, WorkspaceCloneRequest):
            raise TypeError("request must be a WorkspaceCloneRequest")
        if not callable(should_cancel):
            raise TypeError("should_cancel must be callable")
        if on_progress is not None and not callable(on_progress):
            raise TypeError("on_progress must be callable or None")

        reporter = _ProgressReporter(on_progress)
        state = _CopyState()
        target_reserved = False

        try:
            reporter.emit("queued", 0.0)
            reporter.emit("authorizing", 0.05)
            self._cancel_if_requested(should_cancel)

            reporter.emit("preparing", 0.1)
            prepared = self._prepare_source_snapshot(request)
            reporter.emit("preparing", 0.2)
            self._cancel_if_requested(should_cancel)

            description, workspace_profile = self._reservation_fields(prepared.workspace)
            try:
                reservation = self._tgt_chacha.reserve_clone_target(
                    workspace_id=request.target_workspace_id,
                    operation_id=request.operation_id,
                    request_fingerprint=request.request_fingerprint,
                    name=request.name,
                    description=description,
                    workspace_profile=workspace_profile,
                )
            except Exception as exc:
                self._log_failure("Workspace clone target reservation failed", request, exc)
                raise _FatalClone(
                    "clone_reservation_failed",
                    exc,
                    cleanup_ambiguous=True,
                ) from None
            if not self._reservation_matches(
                reservation,
                request,
                description=description,
                workspace_profile=workspace_profile,
            ):
                raise _FatalClone(
                    "clone_validation_failed",
                    cleanup_ambiguous=True,
                )
            if reservation["system_operation_state"] == "publication_pending":
                raise _PublicationPending
            target_reserved = True

            self._copy_sources(
                request,
                prepared,
                state,
                should_cancel=should_cancel,
                reporter=reporter,
            )
            self._copy_notes(
                request,
                prepared.workspace.notes,
                state,
                should_cancel=should_cancel,
                reporter=reporter,
            )
            self._copy_artifacts(
                request,
                prepared.workspace.artifacts,
                state,
                should_cancel=should_cancel,
                reporter=reporter,
            )
            self._copy_memberships(request, prepared, state)
            self._validate_copy_state(prepared, state)
            text_ready = self._read_target_text_readiness(request, state)

            result = self._build_result(
                request,
                prepared,
                state,
                text_ready=text_ready,
            )
            reporter.emit("finalizing", 0.95)
            self._cancel_if_requested(should_cancel)
            try:
                publication = self._tgt_chacha.publish_clone_target(
                    workspace_id=request.target_workspace_id,
                    operation_id=request.operation_id,
                )
            except Exception as exc:
                self._log_failure("Workspace clone publication failed", request, exc)
                raise _FatalClone("clone_publication_failed", exc) from None
            if not self._publication_matches(publication, request):
                raise _FatalClone("clone_validation_failed")
            reporter.emit("finalizing", 1.0)
            logger.bind(
                operation_id=request.operation_id,
                target_workspace_id=request.target_workspace_id,
                outcome=result.outcome,
            ).info("Workspace clone reached publication pending")
            return result
        except CloneSnapshotUnavailable:
            raise
        except _PublicationPending:
            raise ClonePersistenceError(
                code="clone_publication_pending",
                cleanup_state="pending",
            ) from None
        except _CancellationRequested:
            cleanup_state = self._cleanup(
                request,
                state,
                target_reserved=target_reserved,
            )
            raise CloneCancelled(cleanup_state=cleanup_state) from None
        except _FatalClone as exc:
            cleanup_state = self._cleanup(
                request,
                state,
                target_reserved=target_reserved,
            )
            if exc.cleanup_ambiguous:
                cleanup_state = "pending"
            raise ClonePersistenceError(
                code=exc.code,
                cleanup_state=cleanup_state,
            ) from None
        except Exception as exc:
            self._log_failure("Workspace clone persistence failed", request, exc)
            cleanup_state = self._cleanup(
                request,
                state,
                target_reserved=target_reserved,
            )
            raise ClonePersistenceError(
                code="clone_persistence_failed",
                cleanup_state=cleanup_state,
            ) from None

    @staticmethod
    def _cancel_if_requested(should_cancel: Callable[[], bool]) -> None:
        try:
            decision = should_cancel()
        except Exception as exc:
            raise _FatalClone("clone_validation_failed", exc) from None
        if not isinstance(decision, bool):
            raise _FatalClone("clone_validation_failed")
        if decision:
            raise _CancellationRequested

    def _prepare_source_snapshot(self, request: WorkspaceCloneRequest) -> _PreparedSnapshot:
        try:
            workspace = self._src_chacha.read_workspace_clone_snapshot(
                request.source_workspace_id
            )
            if not isinstance(workspace, WorkspaceCloneSnapshot):
                raise CloneSnapshotUnavailable(cleanup_state="complete")
            media_ids, source_media_ids, membership_media_ids = self._collect_media_ids(
                workspace
            )
            media_snapshots = self._src_media.read_media_clone_snapshots(media_ids)
        except _FatalClone:
            raise
        except CloneSnapshotUnavailable:
            raise
        except Exception as exc:
            self._log_failure("Workspace clone source snapshot failed", request, exc)
            raise CloneSnapshotUnavailable(cleanup_state="complete") from None

        if not isinstance(media_snapshots, Mapping):
            raise CloneSnapshotUnavailable(cleanup_state="complete")
        if set(media_snapshots) != set(media_ids) or len(media_snapshots) != len(media_ids):
            raise CloneSnapshotUnavailable(cleanup_state="complete")
        normalized_snapshots: dict[int, MediaCloneSnapshot] = {}
        hashes: dict[int, str] = {}
        for media_id in media_ids:
            snapshot = media_snapshots.get(media_id)
            if not isinstance(snapshot, MediaCloneSnapshot):
                raise CloneSnapshotUnavailable(cleanup_state="complete")
            normalized_snapshots[media_id] = snapshot
            try:
                hashes[media_id] = hash_media_clone_snapshot(snapshot)
            except Exception as exc:
                self._log_failure("Workspace clone source validation failed", request, exc)
                raise _FatalClone("clone_validation_failed", exc) from None

        return _PreparedSnapshot(
            workspace=workspace,
            media_ids=media_ids,
            source_media_ids=source_media_ids,
            membership_media_ids=membership_media_ids,
            media_snapshots=normalized_snapshots,
            media_hashes=hashes,
        )

    @classmethod
    def _collect_media_ids(
        cls,
        workspace: WorkspaceCloneSnapshot,
    ) -> tuple[tuple[int, ...], tuple[int | None, ...], tuple[int | None, ...]]:
        ordered_ids: dict[int, None] = {}
        source_media_ids: list[int | None] = []
        for source in workspace.sources:
            media_id = cls._normalize_media_reference(source.get("media_id"))
            source_media_ids.append(media_id)
            if media_id is not None:
                ordered_ids.setdefault(media_id, None)

        membership_media_ids: list[int | None] = []
        for membership in workspace.memberships:
            if membership.get("deleted") in (True, 1):
                membership_media_ids.append(None)
                continue
            if membership.get("resource_type") != "media":
                membership_media_ids.append(None)
                continue
            media_id = cls._normalize_active_membership_media_reference(
                membership.get("resource_id")
            )
            membership_media_ids.append(media_id)
            ordered_ids.setdefault(media_id, None)
        return tuple(ordered_ids), tuple(source_media_ids), tuple(membership_media_ids)

    @staticmethod
    def _normalize_media_reference(value: Any) -> int | None:
        if value is None:
            return None
        if isinstance(value, bool):
            if value is False:
                return None
            raise _FatalClone("clone_validation_failed")
        if isinstance(value, int):
            if value == 0:
                return None
            if value > 0:
                return value
            raise _FatalClone("clone_validation_failed")
        if isinstance(value, str):
            normalized = value.strip()
            if not normalized:
                return None
            if _POSITIVE_INTEGER_PATTERN.fullmatch(normalized) is None:
                raise _FatalClone("clone_validation_failed")
            parsed = int(normalized)
            return parsed if parsed > 0 else None
        raise _FatalClone("clone_validation_failed")

    @classmethod
    def _normalize_active_membership_media_reference(cls, value: Any) -> int:
        media_id = cls._normalize_media_reference(value)
        if media_id is None:
            raise _FatalClone("clone_validation_failed")
        return media_id

    @staticmethod
    def _reservation_fields(snapshot: WorkspaceCloneSnapshot) -> tuple[str | None, str]:
        description = snapshot.workspace.get("description")
        if description is not None and not isinstance(description, str):
            raise _FatalClone("clone_validation_failed")
        profile = snapshot.workspace.get("workspace_profile") or "research"
        if not isinstance(profile, str) or not profile.strip():
            raise _FatalClone("clone_validation_failed")
        return description, profile

    @staticmethod
    def _reservation_matches(
        reservation: Any,
        request: WorkspaceCloneRequest,
        *,
        description: str | None,
        workspace_profile: str,
    ) -> bool:
        if not isinstance(reservation, Mapping):
            return False
        if str(reservation.get("id") or "") != request.target_workspace_id:
            return False
        if reservation.get("system_operation_state") not in {
            "staged",
            "publication_pending",
        }:
            return False
        if reservation.get("deleted") in (True, 1):
            return False

        expected_fields = {
            "system_operation_id": request.operation_id,
            "system_request_fingerprint": request.request_fingerprint,
            "name": request.name,
            "description": description,
            "workspace_profile": workspace_profile,
        }
        return all(
            field_name in reservation and reservation[field_name] == expected
            for field_name, expected in expected_fields.items()
        )

    @staticmethod
    def _publication_matches(
        publication: Any,
        request: WorkspaceCloneRequest,
    ) -> bool:
        return (
            isinstance(publication, Mapping)
            and str(publication.get("id") or "") == request.target_workspace_id
            and publication.get("system_operation_state") == "publication_pending"
        )

    def _copy_sources(
        self,
        request: WorkspaceCloneRequest,
        prepared: _PreparedSnapshot,
        state: _CopyState,
        *,
        should_cancel: Callable[[], bool],
        reporter: _ProgressReporter,
    ) -> None:
        sources = prepared.workspace.sources
        reporter.emit("sources", 0.25 if sources else 0.6)
        for index, (source, source_media_id) in enumerate(
            zip(sources, prepared.source_media_ids, strict=True)
        ):
            self._cancel_if_requested(should_cancel)
            tracked = None
            if source_media_id is not None:
                tracked = self._ensure_media(request, prepared, state, source_media_id)
                if tracked is None:
                    state.sources_failed += 1
                    state.warn("source_copy_failed")
                    reporter.emit("sources", 0.25 + 0.35 * ((index + 1) / len(sources)))
                    continue

            source_key = self._resource_key(source.get("id"))
            if source_key is None:
                raise _FatalClone("clone_validation_failed")
            payload = self._source_payload(source, tracked)
            try:
                copied = self._tgt_chacha.add_workspace_source(
                    request.target_workspace_id,
                    payload,
                )
            except Exception as exc:
                self._log_failure(
                    "Workspace clone source write response unavailable",
                    request,
                    exc,
                )
                try:
                    copied = self._tgt_chacha.get_workspace_source(
                        request.target_workspace_id,
                        source_key,
                    )
                except Exception as lookup_exc:
                    self._log_failure(
                        "Workspace clone source reconciliation failed",
                        request,
                        lookup_exc,
                    )
                    raise _FatalClone("clone_validation_failed", lookup_exc) from None
                if copied is None:
                    state.sources_failed += 1
                    state.warn("source_copy_failed")
                    if tracked is not None and tracked.reference_count == 0:
                        self._delete_unreferenced_media(request, state, tracked)
                    reporter.emit(
                        "sources",
                        0.25 + 0.35 * ((index + 1) / len(sources)),
                    )
                    continue
            if not self._source_copy_matches(copied, payload):
                raise _FatalClone("clone_validation_failed")

            state.copied_source_ids.add(source_key)
            state.copied_source_media_ids[source_key] = (
                tracked.result.media_id if tracked is not None else 0
            )
            state.sources_copied += 1
            if tracked is not None:
                tracked.reference_count += 1
                state.successful_source_media_ids.add(tracked.source_media_id)
            reporter.emit("sources", 0.25 + 0.35 * ((index + 1) / len(sources)))

    @staticmethod
    def _source_payload(
        source: Mapping[str, Any],
        tracked: _TrackedMedia | None,
    ) -> dict[str, Any]:
        return {
            "id": source.get("id"),
            "media_id": tracked.result.media_id if tracked is not None else 0,
            "source_type": source.get("source_type", ""),
            "title": source.get("title", ""),
            "url": source.get("url"),
            "position": source.get("position", 0),
            "selected": bool(source.get("selected", True)),
            "review_state": "unset",
        }

    @classmethod
    def _source_copy_matches(cls, copied: Any, payload: Mapping[str, Any]) -> bool:
        if not isinstance(copied, Mapping):
            return False
        try:
            copied_media_id = int(copied.get("media_id") or 0)
            expected_media_id = int(payload.get("media_id") or 0)
            copied_position = int(copied.get("position") or 0)
            expected_position = int(payload.get("position") or 0)
        except (TypeError, ValueError):
            return False
        return (
            cls._resource_key(copied.get("id")) == cls._resource_key(payload.get("id"))
            and copied_media_id == expected_media_id
            and copied.get("source_type", "") == payload.get("source_type", "")
            and copied.get("title", "") == payload.get("title", "")
            and copied.get("url") == payload.get("url")
            and copied_position == expected_position
            and bool(copied.get("selected", True)) is bool(payload.get("selected", True))
            and copied.get("review_state") == "unset"
            and copied.get("reviewed_at") is None
            and copied.get("reviewed_by_user_id") is None
        )

    def _copy_notes(
        self,
        request: WorkspaceCloneRequest,
        notes: Sequence[Mapping[str, Any]],
        state: _CopyState,
        *,
        should_cancel: Callable[[], bool],
        reporter: _ProgressReporter,
    ) -> None:
        reporter.emit("notes", 0.65 if notes else 0.75)
        for index, note in enumerate(notes):
            self._cancel_if_requested(should_cancel)
            source_key = self._resource_key(note.get("id"))
            if source_key is None:
                raise _FatalClone("clone_validation_failed")
            try:
                payload = {
                    "title": note.get("title", ""),
                    "content": note.get("content", ""),
                    "keywords": self._decode_note_keywords(note),
                }
            except Exception as exc:
                state.notes_failed += 1
                state.warn("note_copy_failed")
                self._log_failure("Workspace clone note copy failed", request, exc)
                reporter.emit("notes", 0.65 + 0.1 * ((index + 1) / len(notes)))
                continue
            try:
                copied = self._tgt_chacha.add_workspace_note(
                    request.target_workspace_id,
                    payload,
                )
            except Exception as exc:
                self._log_failure(
                    "Workspace clone note write response unavailable",
                    request,
                    exc,
                )
                copied = self._reconcile_note_write(
                    request,
                    payload,
                    claimed_target_ids=set(state.note_id_map.values()),
                )
                if copied is None:
                    state.notes_failed += 1
                    state.warn("note_copy_failed")
                    reporter.emit("notes", 0.65 + 0.1 * ((index + 1) / len(notes)))
                    continue
            target_key = (
                self._resource_key(copied.get("id"))
                if isinstance(copied, Mapping)
                else None
            )
            if target_key is None or not self._note_copy_matches(copied, payload):
                raise _FatalClone("clone_validation_failed")
            state.note_id_map[source_key] = target_key
            state.notes_copied += 1
            reporter.emit("notes", 0.65 + 0.1 * ((index + 1) / len(notes)))

    def _reconcile_note_write(
        self,
        request: WorkspaceCloneRequest,
        payload: Mapping[str, Any],
        *,
        claimed_target_ids: set[str],
    ) -> Mapping[str, Any] | None:
        try:
            rows = self._tgt_chacha.list_workspace_notes(request.target_workspace_id)
        except Exception as exc:
            self._log_failure("Workspace clone note reconciliation failed", request, exc)
            raise _FatalClone("clone_validation_failed", exc) from None
        if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
            raise _FatalClone("clone_validation_failed")

        unclaimed: list[Mapping[str, Any]] = []
        for row in rows:
            if not isinstance(row, Mapping):
                raise _FatalClone("clone_validation_failed")
            target_key = self._resource_key(row.get("id"))
            if target_key is None:
                raise _FatalClone("clone_validation_failed")
            if target_key not in claimed_target_ids:
                unclaimed.append(row)
        if not unclaimed:
            return None
        if len(unclaimed) != 1 or not self._note_copy_matches(unclaimed[0], payload):
            raise _FatalClone("clone_validation_failed")
        return unclaimed[0]

    @classmethod
    def _note_copy_matches(cls, copied: Any, payload: Mapping[str, Any]) -> bool:
        if not isinstance(copied, Mapping) or copied.get("deleted") in (True, 1):
            return False
        try:
            copied_keywords = cls._decode_note_keywords(copied)
        except (TypeError, ValueError, json.JSONDecodeError):
            return False
        return (
            copied.get("title", "") == payload.get("title", "")
            and copied.get("content", "") == payload.get("content", "")
            and copied_keywords == payload.get("keywords", [])
        )

    @staticmethod
    def _decode_note_keywords(note: Mapping[str, Any]) -> list[str]:
        raw = note.get("keywords", note.get("keywords_json"))
        if raw is None or raw == "":
            return []
        if isinstance(raw, str):
            parsed = json.loads(raw)
        elif isinstance(raw, (list, tuple)):
            parsed = list(raw)
        else:
            raise ValueError("note keywords must be a JSON list")
        if not isinstance(parsed, list) or any(not isinstance(item, str) for item in parsed):
            raise ValueError("note keywords must be a JSON string list")
        return list(parsed)

    def _copy_artifacts(
        self,
        request: WorkspaceCloneRequest,
        artifacts: Sequence[Mapping[str, Any]],
        state: _CopyState,
        *,
        should_cancel: Callable[[], bool],
        reporter: _ProgressReporter,
    ) -> None:
        reporter.emit("artifacts", 0.8 if artifacts else 0.9)
        for index, artifact in enumerate(artifacts):
            self._cancel_if_requested(should_cancel)
            artifact_id = self._resource_key(artifact.get("id"))
            if artifact_id is None:
                raise _FatalClone("clone_validation_failed")
            payload: dict[str, Any] = {"id": artifact_id}
            for field_name in _ARTIFACT_SAFE_FIELDS:
                if field_name in artifact:
                    payload[field_name] = _thaw_json(artifact[field_name])
            try:
                copied = self._tgt_chacha.add_workspace_artifact(
                    request.target_workspace_id,
                    payload,
                )
            except Exception as exc:
                self._log_failure(
                    "Workspace clone artifact write response unavailable",
                    request,
                    exc,
                )
                try:
                    copied = self._tgt_chacha.get_workspace_artifact(
                        request.target_workspace_id,
                        artifact_id,
                    )
                except Exception as lookup_exc:
                    self._log_failure(
                        "Workspace clone artifact reconciliation failed",
                        request,
                        lookup_exc,
                    )
                    raise _FatalClone("clone_validation_failed", lookup_exc) from None
                if copied is None:
                    state.artifacts_failed += 1
                    state.warn("artifact_copy_failed")
                    reporter.emit(
                        "artifacts",
                        0.8 + 0.1 * ((index + 1) / len(artifacts)),
                    )
                    continue
            if not self._artifact_copy_matches(copied, payload):
                raise _FatalClone("clone_validation_failed")
            state.copied_artifact_ids.add(artifact_id)
            state.artifacts_copied += 1
            reporter.emit("artifacts", 0.8 + 0.1 * ((index + 1) / len(artifacts)))

    @classmethod
    def _artifact_copy_matches(cls, copied: Any, payload: Mapping[str, Any]) -> bool:
        if not isinstance(copied, Mapping):
            return False
        if cls._resource_key(copied.get("id")) != cls._resource_key(payload.get("id")):
            return False
        return all(copied.get(field_name) == expected for field_name, expected in payload.items())

    def _copy_memberships(
        self,
        request: WorkspaceCloneRequest,
        prepared: _PreparedSnapshot,
        state: _CopyState,
    ) -> None:
        provenance = {
            "kind": "shared_workspace_clone",
            "operation_id": request.operation_id,
            "source_workspace_id": request.source_workspace_id,
        }
        for membership, membership_media_id in zip(
            prepared.workspace.memberships,
            prepared.membership_media_ids,
            strict=True,
        ):
            if membership.get("deleted") in (True, 1):
                continue
            resource_type = membership.get("resource_type")
            if resource_type not in _SUPPORTED_MEMBERSHIP_TYPES:
                state.warn("membership_skipped")
                continue

            tracked: _TrackedMedia | None = None
            mapped_resource_id: str | None
            if resource_type == "media":
                if membership_media_id is None:
                    state.warn("membership_skipped")
                    continue
                tracked = self._ensure_media(
                    request,
                    prepared,
                    state,
                    membership_media_id,
                )
                mapped_resource_id = (
                    str(tracked.result.media_id) if tracked is not None else None
                )
            else:
                source_resource_id = self._resource_key(membership.get("resource_id"))
                if resource_type == "workspace_source":
                    mapped_resource_id = (
                        source_resource_id
                        if source_resource_id in state.copied_source_ids
                        else None
                    )
                elif resource_type == "workspace_artifact":
                    mapped_resource_id = (
                        source_resource_id
                        if source_resource_id in state.copied_artifact_ids
                        else None
                    )
                else:
                    mapped_resource_id = state.note_id_map.get(source_resource_id or "")

            if mapped_resource_id is None:
                state.warn("membership_skipped")
                continue
            payload = {
                "resource_type": resource_type,
                "resource_id": mapped_resource_id,
                "role": membership.get("role") or "member",
                "label": membership.get("label"),
                "transfer_policy": "copy",
                "provenance": provenance,
                "metadata": {},
            }
            try:
                copied = self._tgt_chacha.add_workspace_resource_membership(
                    request.target_workspace_id,
                    payload,
                )
            except Exception as exc:
                self._log_failure(
                    "Workspace clone membership write response unavailable",
                    request,
                    exc,
                )
                try:
                    copied = self._tgt_chacha.get_workspace_resource_membership(
                        request.target_workspace_id,
                        resource_type,
                        mapped_resource_id,
                    )
                except Exception as lookup_exc:
                    self._log_failure(
                        "Workspace clone membership reconciliation failed",
                        request,
                        lookup_exc,
                    )
                    raise _FatalClone("clone_validation_failed", lookup_exc) from None
                if copied is None:
                    state.warn("membership_skipped")
                    if tracked is not None and tracked.reference_count == 0:
                        self._delete_unreferenced_media(request, state, tracked)
                    continue
            if not self._membership_copy_matches(copied, payload):
                raise _FatalClone("clone_validation_failed")
            if tracked is not None:
                tracked.reference_count += 1

    @staticmethod
    def _membership_copy_matches(copied: Any, payload: Mapping[str, Any]) -> bool:
        return (
            isinstance(copied, Mapping)
            and copied.get("resource_type") == payload.get("resource_type")
            and str(copied.get("resource_id") or "")
            == str(payload.get("resource_id") or "")
            and copied.get("role") == payload.get("role")
            and copied.get("label") == payload.get("label")
            and copied.get("transfer_policy") == "copy"
            and copied.get("provenance") == payload.get("provenance")
            and copied.get("metadata") == payload.get("metadata")
        )

    def _ensure_media(
        self,
        request: WorkspaceCloneRequest,
        prepared: _PreparedSnapshot,
        state: _CopyState,
        source_media_id: int,
    ) -> _TrackedMedia | None:
        existing = state.tracked_media.get(source_media_id)
        if existing is not None:
            return existing
        if source_media_id in state.media_failed_once:
            return None

        source_identity = f"media:{source_media_id}"
        content_hash = prepared.media_hashes[source_media_id]
        insert_kwargs = {
            "snapshot": prepared.media_snapshots[source_media_id],
            "operation_id": request.operation_id,
            "source_identity": source_identity,
            "expected_content_hash": content_hash,
        }
        try:
            result = self._tgt_media.insert_operation_owned_clone_media(**insert_kwargs)
        except Exception as exc:
            self._log_failure(
                "Workspace clone Media write response unavailable",
                request,
                exc,
            )
            try:
                result = self._tgt_media.insert_operation_owned_clone_media(**insert_kwargs)
            except Exception as retry_exc:
                self._log_failure("Workspace clone Media retry failed", request, retry_exc)
                try:
                    deleted = self._tgt_media.delete_operation_owned_clone_media(
                        operation_id=request.operation_id,
                        source_identity=source_identity,
                        expected_content_hash=content_hash,
                    )
                except Exception as cleanup_exc:
                    self._log_failure(
                        "Workspace clone ambiguous Media cleanup failed",
                        request,
                        cleanup_exc,
                    )
                    raise _FatalClone(
                        "clone_cleanup_incomplete",
                        cleanup_exc,
                        cleanup_ambiguous=True,
                    ) from None
                if (
                    isinstance(deleted, bool)
                    or not isinstance(deleted, int)
                    or deleted not in {0, 1}
                ):
                    raise _FatalClone(
                        "clone_cleanup_incomplete",
                        cleanup_ambiguous=True,
                    ) from None
                state.media_failed_once.add(source_media_id)
                state.warn("media_copy_failed")
                return None

        if not isinstance(result, OperationOwnedMediaResult):
            self._delete_possible_media_identity(
                request,
                source_identity=source_identity,
                content_hash=content_hash,
            )
            raise _FatalClone("clone_validation_failed")
        tracked = _TrackedMedia(
            source_media_id=source_media_id,
            source_identity=source_identity,
            content_hash=content_hash,
            result=result,
        )
        state.tracked_media[source_media_id] = tracked
        return tracked

    def _delete_unreferenced_media(
        self,
        request: WorkspaceCloneRequest,
        state: _CopyState,
        tracked: _TrackedMedia,
    ) -> None:
        if tracked.reference_count != 0:
            return
        try:
            deleted = self._tgt_media.delete_operation_owned_clone_media(
                operation_id=request.operation_id,
                source_identity=tracked.source_identity,
                expected_content_hash=tracked.content_hash,
            )
        except Exception as exc:
            self._log_failure("Workspace clone immediate Media cleanup failed", request, exc)
            raise _FatalClone("clone_cleanup_incomplete", exc) from None
        if deleted != 1:
            raise _FatalClone("clone_cleanup_incomplete")
        state.tracked_media.pop(tracked.source_media_id, None)

    def _delete_possible_media_identity(
        self,
        request: WorkspaceCloneRequest,
        *,
        source_identity: str,
        content_hash: str,
    ) -> None:
        """Delete an exact identity after a persistence return-contract violation."""
        try:
            deleted = self._tgt_media.delete_operation_owned_clone_media(
                operation_id=request.operation_id,
                source_identity=source_identity,
                expected_content_hash=content_hash,
            )
        except Exception as exc:
            self._log_failure("Workspace clone invalid Media cleanup failed", request, exc)
            raise _FatalClone(
                "clone_cleanup_incomplete",
                exc,
                cleanup_ambiguous=True,
            ) from None
        if deleted != 1:
            raise _FatalClone(
                "clone_cleanup_incomplete",
                cleanup_ambiguous=True,
            )

    @staticmethod
    def _resource_key(value: Any) -> str | None:
        if value is None or isinstance(value, bool):
            return None
        normalized = str(value).strip()
        if (
            not normalized
            or len(normalized) > _RESOURCE_ID_MAX_LENGTH
            or not normalized.isascii()
            or any(ord(character) < 0x21 or ord(character) > 0x7E for character in normalized)
        ):
            return None
        return normalized

    @staticmethod
    def _validate_copy_state(prepared: _PreparedSnapshot, state: _CopyState) -> None:
        if any(tracked.reference_count <= 0 for tracked in state.tracked_media.values()):
            raise _FatalClone("clone_validation_failed")
        if not state.successful_source_media_ids <= set(state.tracked_media):
            raise _FatalClone("clone_validation_failed")
        if not set(state.tracked_media) <= set(prepared.media_ids):
            raise _FatalClone("clone_validation_failed")

    def _build_result(
        self,
        request: WorkspaceCloneRequest,
        prepared: _PreparedSnapshot,
        state: _CopyState,
        *,
        text_ready: bool,
    ) -> WorkspaceCloneResult:
        media_copied = len(state.tracked_media)
        media_failed = len(prepared.media_ids) - media_copied
        operation_owned_media = sum(
            1 for tracked in state.tracked_media.values() if tracked.result.created
        )

        if not self._vector_retrieval_configured:
            vector_readiness = "not_configured"
        else:
            vector_readiness = "needs_indexing"
            state.warn("vector_index_not_generated")

        counts = CloneCopyCounts(
            sources_attempted=len(prepared.workspace.sources),
            sources_copied=state.sources_copied,
            sources_failed=state.sources_failed,
            notes_attempted=len(prepared.workspace.notes),
            notes_copied=state.notes_copied,
            notes_failed=state.notes_failed,
            artifacts_attempted=len(prepared.workspace.artifacts),
            artifacts_copied=state.artifacts_copied,
            artifacts_failed=state.artifacts_failed,
            media_attempted=len(prepared.media_ids),
            media_copied=media_copied,
            media_failed=media_failed,
            operation_owned_media_count=operation_owned_media,
        )
        warnings = tuple(
            CloneWarning(code=code, count=count) for code, count in state.warnings.items()
        )
        has_partial_copy = bool(warnings) or any(
            (
                counts.sources_failed,
                counts.notes_failed,
                counts.artifacts_failed,
                counts.media_failed,
            )
        )
        readiness_value = "ready" if text_ready else "unavailable"
        return WorkspaceCloneResult(
            workspace_id=request.target_workspace_id,
            name=request.name,
            outcome="partial" if has_partial_copy else "complete",
            publication_confirmed=False,
            counts=counts,
            readiness=CloneRetrievalReadiness(
                text_search=readiness_value,
                citations=readiness_value,
                vector_search=vector_readiness,
            ),
            warnings=warnings,
        )

    def _read_target_text_readiness(
        self,
        request: WorkspaceCloneRequest,
        state: _CopyState,
    ) -> bool:
        try:
            rows = self._tgt_chacha.list_workspace_sources(request.target_workspace_id)
        except Exception as exc:
            self._log_failure("Workspace clone target source validation failed", request, exc)
            raise _FatalClone("clone_validation_failed", exc) from None
        if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
            raise _FatalClone("clone_validation_failed")

        target_source_media_ids: dict[str, int] = {}
        tracked_target_media_ids = {
            tracked.result.media_id for tracked in state.tracked_media.values()
        }
        for row in rows:
            if not isinstance(row, Mapping):
                raise _FatalClone("clone_validation_failed")
            source_id = self._resource_key(row.get("id"))
            if source_id is None or source_id in target_source_media_ids:
                raise _FatalClone("clone_validation_failed")
            try:
                raw_media_id = row.get("media_id")
                if isinstance(raw_media_id, bool):
                    raise ValueError
                media_id = int(raw_media_id or 0)
            except (TypeError, ValueError):
                raise _FatalClone("clone_validation_failed") from None
            if media_id < 0 or (media_id and media_id not in tracked_target_media_ids):
                raise _FatalClone("clone_validation_failed")
            target_source_media_ids[source_id] = media_id

        if target_source_media_ids != state.copied_source_media_ids:
            raise _FatalClone("clone_validation_failed")

        ordered_source_media_ids = tuple(sorted(state.tracked_media))
        expected_by_source_identity = {
            state.tracked_media[source_media_id].source_identity: state.tracked_media[
                source_media_id
            ].result.media_id
            for source_media_id in ordered_source_media_ids
        }
        if len(expected_by_source_identity) != len(ordered_source_media_ids):
            raise _FatalClone("clone_validation_failed")
        readiness_items = tuple(
            (
                state.tracked_media[source_media_id].source_identity,
                state.tracked_media[source_media_id].content_hash,
            )
            for source_media_id in ordered_source_media_ids
        )
        try:
            readiness = self._tgt_media.read_operation_owned_clone_media_readiness(
                operation_id=request.operation_id,
                items=readiness_items,
            )
        except Exception as exc:
            self._log_failure("Workspace clone target media validation failed", request, exc)
            raise _FatalClone("clone_validation_failed", exc) from None
        if not isinstance(readiness, Mapping) or set(readiness) != set(
            expected_by_source_identity
        ):
            raise _FatalClone("clone_validation_failed")
        for source_identity, item in readiness.items():
            if (
                not isinstance(item, OperationOwnedMediaReadiness)
                or item.source_identity != source_identity
                or item.media_id != expected_by_source_identity[source_identity]
            ):
                raise _FatalClone("clone_validation_failed")
        source_readiness_identities = {
            state.tracked_media[source_media_id].source_identity
            for source_media_id in state.successful_source_media_ids
        }
        return any(
            readiness[source_identity].has_chunks
            for source_identity in source_readiness_identities
        )

    def _cleanup(
        self,
        request: WorkspaceCloneRequest,
        state: _CopyState,
        *,
        target_reserved: bool,
    ) -> str:
        cleanup_complete = True
        for tracked in tuple(state.tracked_media.values()):
            try:
                deleted = self._tgt_media.delete_operation_owned_clone_media(
                    operation_id=request.operation_id,
                    source_identity=tracked.source_identity,
                    expected_content_hash=tracked.content_hash,
                )
            except Exception as exc:
                cleanup_complete = False
                self._log_failure("Workspace clone Media cleanup failed", request, exc)
                continue
            if deleted != 1:
                cleanup_complete = False

        if target_reserved:
            try:
                discarded = self._tgt_chacha.discard_clone_target(
                    workspace_id=request.target_workspace_id,
                    operation_id=request.operation_id,
                )
            except Exception as exc:
                cleanup_complete = False
                self._log_failure("Workspace clone target cleanup failed", request, exc)
            else:
                if not discarded:
                    cleanup_complete = False
        return "complete" if cleanup_complete else "pending"

    @staticmethod
    def _log_failure(
        message: str,
        request: WorkspaceCloneRequest,
        exc: BaseException,
    ) -> None:
        logger.bind(
            operation_id=request.operation_id,
            target_workspace_id=request.target_workspace_id,
            exception_type=_safe_exception_type(exc),
        ).warning(message)
