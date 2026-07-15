"""Service layer for owner-scoped asynchronous playlist preflight resources."""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import timedelta
from typing import Any
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse
from uuid import uuid4

from loguru import logger
from pydantic import BaseModel, ValidationError

from tldw_Server_API.app.api.v1.schemas.media_playlist_ingest import (
    DirectUrlInput,
    DuplicateEvidence,
    DuplicatePolicy,
    FileStubInput,
    MaterializedPlaylistItemInput,
    PlaylistIngestRunCreateRequest,
    ReviewOverride,
    ReviewRequiredItem,
)
from tldw_Server_API.app.core.DB_Management.media_db.api import get_media_by_id, get_media_by_urls
from tldw_Server_API.app.core.DB_Management.media_db.dedupe_urls import (
    media_dedupe_url_candidates,
    normalize_media_dedupe_url,
)
from tldw_Server_API.app.core.DB_Management.media_db.errors import ConflictError, InputError
from tldw_Server_API.app.core.Ingestion_Media_Processing.Video.playlist_ingest_store import (
    _PREFLIGHT_JOB_SENTINEL,
    MediaIngestRunItemRecord,
    MediaIngestRunRecord,
    PlaylistIngestConflictError,
    PlaylistIngestNotFoundError,
    PlaylistIngestStore,
    PlaylistItemRecord,
    PlaylistMaterializationRecord,
    PlaylistPage,
    PlaylistPreflightCapacityError,
    PlaylistPreflightRecord,
)
from tldw_Server_API.app.core.Ingestion_Media_Processing.Video.playlist_preflight import (
    PlaylistUrlClassification,
    canonical_youtube_video_url,
    classify_playlist_url,
)

_SENSITIVE_QUERY_KEY = re.compile(
    r"(?:auth|cookie|credential|key|password|secret|signature|token)",
    re.IGNORECASE,
)
_YOUTUBE_ID = re.compile(r"[A-Za-z0-9_-]{1,200}")
_SAFE_BLOCKED_ERROR_CODES = {
    "playlist_not_found",
    "playlist_private_or_auth_required",
    "playlist_metadata_unavailable",
    "playlist_too_large",
    "preflight_busy",
    "preflight_timeout",
    "preflight_cancelled",
}
_WORKER_ERROR_MAP = {
    "playlist_too_large": "playlist_too_large",
    "playlist_preflight_result_too_large": "playlist_too_large",
    "playlist_preflight_capacity_unavailable": "preflight_busy",
    "playlist_preflight_timeout": "preflight_timeout",
    "playlist_preflight_cancelled": "preflight_cancelled",
    "playlist_private_or_auth_required": "playlist_private_or_auth_required",
    "playlist_not_found": "playlist_not_found",
}


class PlaylistPreflightBusyError(RuntimeError):
    """Raised when no transactional preflight reservation is available."""


class InvalidPlaylistUrlError(ValueError):
    """Raised for input outside the trusted YouTube playlist boundary."""


class PlaylistPreflightUnavailableError(RuntimeError):
    """Raised when the resource cannot be durably bound to an internal job."""


class PlaylistPreflightIncompleteError(RuntimeError):
    """Raised when materialization is requested before a complete snapshot."""


class PlaylistSelectionError(ValueError):
    """Raised for an invalid server occurrence selection."""


class PlaylistRunValidationError(ValueError):
    """Raised with a stable code when a run request is invalid."""


class PlaylistRunPendingError(PlaylistIngestConflictError):
    """Raised when an existing run still has an ambiguous zero-job action."""

    def __init__(self, run_id: str) -> None:
        self.run_id = str(run_id)
        super().__init__("duplicate_action_pending")


class PlaylistRunStatusUnavailableError(PlaylistIngestConflictError):
    """Raised when retry reconciliation cannot establish current owner media state."""


class PlaylistPreflightRequiredError(PlaylistRunValidationError):
    """Raised when a direct URL must first use playlist preflight."""


class ReviewRequiredError(RuntimeError):
    """Raised before persistence when refreshed duplicate evidence changed Review."""

    def __init__(self, items: Sequence[ReviewRequiredItem]) -> None:
        self.items = tuple(items)
        super().__init__("review_required")


@dataclass(frozen=True, slots=True)
class CreatedPlaylistPreflight:
    """Accepted preflight plus the safe admission limits used for it."""

    record: PlaylistPreflightRecord
    max_items: int
    global_capacity: int
    owner_capacity: int

    @property
    def preflight_id(self) -> str:
        return self.record.preflight_id


@dataclass(frozen=True, slots=True)
class CreatedPlaylistMaterialization:
    """Materialization record and its bounded authoritative identities."""

    record: PlaylistMaterializationRecord
    items: tuple[PlaylistItemRecord, ...]


def _bounded_env_int(name: str, default: int, *, minimum: int, maximum: int) -> int:
    try:
        value = int((os.getenv(name) or "").strip() or default)
    except ValueError:
        value = default
    return min(maximum, max(minimum, value))


def _playlist_preflight_queue() -> str:
    return (
        (os.getenv("MEDIA_INGEST_JOBS_QUEUE") or "").strip()
        or (os.getenv("MEDIA_INGEST_JOBS_DEFAULT_QUEUE") or "").strip()
        or "default"
    )


def _request_fingerprint(request: PlaylistIngestRunCreateRequest) -> str:
    """Hash the canonical validated request body without its replay key."""
    payload = request.model_dump(mode="json", exclude={"client_request_id"})
    encoded = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _owner_media_db(owner_user_id: str) -> Any:
    """Open the existing owner Media DB without touching AuthNZ storage."""
    from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths
    from tldw_Server_API.app.core.DB_Management.media_db.api import create_media_database

    db_path = DatabasePaths.get_media_db_path(owner_user_id)
    return create_media_database(client_id=f"playlist_ingest:{owner_user_id}", db_path=str(db_path))


def _owner_collections_db(owner_user_id: str) -> Any:
    """Open the existing owner Collections DB adapter."""
    from tldw_Server_API.app.core.DB_Management.Collections_DB import CollectionsDatabase

    return CollectionsDatabase.for_user(owner_user_id)


def _trusted_youtube_playlist(url: str) -> tuple[str, PlaylistUrlClassification]:
    """Validate and canonicalize only a credential-free YouTube playlist URL."""
    try:
        parsed = urlparse(str(url or "").strip())
        _ = parsed.port
    except ValueError as exc:
        raise InvalidPlaylistUrlError("invalid_playlist_url") from exc
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.netloc
        or parsed.username is not None
        or parsed.password is not None
        or parsed.fragment
        or parsed.port is not None
    ):
        raise InvalidPlaylistUrlError("invalid_playlist_url")

    query = parse_qs(parsed.query, keep_blank_values=True)
    if any(_SENSITIVE_QUERY_KEY.search(key) for key in query):
        raise InvalidPlaylistUrlError("invalid_playlist_url")
    playlist_values = query.get("list") or []
    if len(playlist_values) != 1 or _YOUTUBE_ID.fullmatch(playlist_values[0]) is None:
        raise InvalidPlaylistUrlError("invalid_playlist_url")

    try:
        classified = classify_playlist_url(url)
    except ValueError as exc:
        raise InvalidPlaylistUrlError("invalid_playlist_url") from exc
    if (
        not classified.is_playlist
        or classified.source_kind not in {"youtube_playlist", "youtube_watch_playlist"}
        or classified.playlist_id != playlist_values[0]
    ):
        raise InvalidPlaylistUrlError("invalid_playlist_url")

    canonical_query: dict[str, str] = {"list": classified.playlist_id}
    if classified.video_id is not None:
        if _YOUTUBE_ID.fullmatch(classified.video_id) is None:
            raise InvalidPlaylistUrlError("invalid_playlist_url")
        canonical_query = {"v": classified.video_id, **canonical_query}
        canonical_path = "/watch"
    else:
        canonical_path = "/playlist"
    canonical_url = urlunparse(
        (
            "https",
            "www.youtube.com",
            canonical_path,
            "",
            urlencode(canonical_query),
            "",
        )
    )
    return canonical_url, classified


class PlaylistIngestService:
    """Coordinate the reviewed store and existing Jobs manager."""

    def __init__(
        self,
        job_manager,
        *,
        media_db_factory: Callable[[str], Any] | None = None,
        collections_db_factory: Callable[[str], Any] | None = None,
    ) -> None:
        self._jobs = job_manager
        self._store = PlaylistIngestStore(job_manager)
        self._media_db_factory = media_db_factory or _owner_media_db
        self._collections_db_factory = collections_db_factory or _owner_collections_db

    def _cleanup_expired_resources(self, owner_user_id: str) -> None:
        """Best-effort bounded cleanup at owner-scoped mutation seams."""
        limit = _bounded_env_int(
            "PLAYLIST_INGEST_CLEANUP_LIMIT",
            20,
            minimum=1,
            maximum=100,
        )
        try:
            deleted = self._store.cleanup_expired_resources(owner_user_id, limit=limit)
        except Exception as exc:  # noqa: BLE001 - cleanup cannot block an unrelated mutation
            logger.bind(error_type=type(exc).__name__).warning("Playlist ingest cleanup failed")
            return
        counts = {f"deleted_{name}": int(count) for name, count in deleted.items()}
        if any(counts.values()):
            logger.bind(**counts).info("Playlist ingest cleanup completed")

    @staticmethod
    def public_error(error: dict | None) -> dict[str, str] | None:
        """Return only a stable public code from stored worker error state."""
        if not error:
            return None
        code = _WORKER_ERROR_MAP.get(str(error.get("code") or ""), "playlist_metadata_unavailable")
        if code not in _SAFE_BLOCKED_ERROR_CODES:
            code = "playlist_metadata_unavailable"
        return {"code": code}

    def create_preflight(
        self,
        owner_user_id: str,
        *,
        url: str,
        max_items: int,
        timeout_seconds: int,
    ) -> CreatedPlaylistPreflight:
        """Reserve capacity, enqueue non-acquirable work, then bind and publish it."""
        self._cleanup_expired_resources(owner_user_id)
        canonical_url, classified = _trusted_youtube_playlist(url)
        global_capacity = _bounded_env_int(
            "PLAYLIST_PREFLIGHT_GLOBAL_CAPACITY",
            2,
            minimum=1,
            maximum=100,
        )
        owner_capacity = _bounded_env_int(
            "PLAYLIST_PREFLIGHT_OWNER_CAPACITY",
            1,
            minimum=1,
            maximum=100,
        )
        ttl_seconds = _bounded_env_int(
            "PLAYLIST_PREFLIGHT_TTL_SECONDS",
            1800,
            minimum=60,
            maximum=86400,
        )
        queue = _playlist_preflight_queue()
        orphan_grace_seconds = _bounded_env_int(
            "PLAYLIST_PREFLIGHT_ORPHAN_GRACE_SECONDS",
            30,
            minimum=1,
            maximum=3600,
        )
        orphan_limit = _bounded_env_int(
            "PLAYLIST_PREFLIGHT_ORPHAN_LIMIT",
            20,
            minimum=1,
            maximum=100,
        )
        for orphan_job_id, orphan_preflight_id in self._store.list_orphaned_preflight_jobs(
            owner_user_id,
            queue=queue,
            grace_seconds=orphan_grace_seconds,
            limit=orphan_limit,
        ):
            if not self._store.claim_orphaned_preflight_job(
                owner_user_id,
                preflight_id=orphan_preflight_id,
                job_id=orphan_job_id,
                queue=queue,
                grace_seconds=orphan_grace_seconds,
            ):
                continue
            try:
                self._jobs.cancel_job(orphan_job_id, reason="playlist_preflight_orphaned")
            except Exception:  # noqa: BLE001 - fenced sentinel remains non-acquirable for retry
                logger.warning("Failed to cancel a fenced orphan playlist preflight job")
        try:
            preflight = self._store.reserve_preflight(
                owner_user_id,
                source_url=canonical_url,
                source_kind=classified.source_kind,
                playlist_id=classified.playlist_id,
                ttl_seconds=ttl_seconds,
                global_capacity=global_capacity,
                owner_capacity=owner_capacity,
            )
        except PlaylistPreflightCapacityError as exc:
            raise PlaylistPreflightBusyError("preflight_busy") from exc

        job: dict | None = None
        try:
            payload = {
                "preflight_id": preflight.preflight_id,
                "max_items": int(max_items),
                "timeout_seconds": int(timeout_seconds),
            }
            job = self._jobs.create_job(
                domain="media_ingest",
                queue=queue,
                job_type="playlist_preflight",
                payload=payload,
                owner_user_id=str(owner_user_id),
                priority=5,
                max_retries=0,
                available_at=_PREFLIGHT_JOB_SENTINEL,
                idempotency_key=f"playlist_preflight:{owner_user_id}:{preflight.preflight_id}",
            )
            bound = self._store.bind_preflight_job(
                owner_user_id,
                preflight.preflight_id,
                int(job["id"]),
                expected_queue=queue,
                expected_payload=payload,
            )
        except Exception as exc:
            if job is not None and job.get("id") is not None:
                try:
                    self._jobs.cancel_job(int(job["id"]), reason="playlist_preflight_bind_failed")
                except Exception:  # noqa: BLE001 - best-effort cleanup must preserve safe failure mapping
                    logger.warning("Failed to cancel an unpublished playlist preflight job")
            try:
                self._store.expire_preflight(owner_user_id, preflight.preflight_id, status="blocked")
            except (PlaylistIngestNotFoundError, RuntimeError, ValueError):
                logger.warning("Failed to expire an unbound playlist preflight reservation")
            raise PlaylistPreflightUnavailableError("preflight_unavailable") from exc

        return CreatedPlaylistPreflight(
            record=bound,
            max_items=int(max_items),
            global_capacity=global_capacity,
            owner_capacity=owner_capacity,
        )

    def get_preflight(self, owner_user_id: str, preflight_id: str) -> PlaylistPreflightRecord:
        """Return one owner-scoped unexpired preflight summary."""
        return self._store.get_preflight(owner_user_id, preflight_id)

    def list_preflight_items(
        self,
        owner_user_id: str,
        preflight_id: str,
        *,
        limit: int,
        cursor: str | None,
    ) -> PlaylistPage[PlaylistItemRecord]:
        """Return a bounded owner-scoped immutable occurrence page."""
        if cursor is not None and (not cursor.strip() or len(cursor) > 4096):
            raise PlaylistIngestNotFoundError("playlist resource not found")
        return self._store.list_preflight_items(
            owner_user_id,
            preflight_id,
            limit=limit,
            cursor=cursor,
        )

    def create_materialization(
        self,
        owner_user_id: str,
        preflight_id: str,
        occurrence_ids: list[str],
    ) -> CreatedPlaylistMaterialization:
        """Copy only selected authoritative identities from a ready snapshot."""
        self._cleanup_expired_resources(owner_user_id)
        preflight = self._store.get_preflight(owner_user_id, preflight_id)
        if preflight.status != "ready":
            raise PlaylistPreflightIncompleteError("preflight_incomplete")
        ttl_seconds = _bounded_env_int(
            "PLAYLIST_MATERIALIZATION_TTL_SECONDS",
            604800,
            minimum=60,
            maximum=2592000,
        )
        try:
            materialization = self._store.create_materialization(
                owner_user_id,
                preflight_id=preflight_id,
                occurrence_ids=occurrence_ids,
                expires_at=self._store._now() + timedelta(seconds=ttl_seconds),
            )
        except ValueError as exc:
            raise PlaylistSelectionError("invalid_occurrence_selection") from exc
        page = self._store.list_materialization_items(
            owner_user_id,
            materialization.materialization_id,
            limit=len(occurrence_ids),
        )
        return CreatedPlaylistMaterialization(materialization, tuple(page.items))

    def cancel_preflight(self, owner_user_id: str, preflight_id: str) -> None:
        """Fence the resource first, then request cancellation of its linked job."""
        job_id = self._store.expire_preflight(owner_user_id, preflight_id, status="cancelled")
        if job_id is None:
            return
        try:
            self._jobs.cancel_job(job_id, reason="playlist_preflight_cancelled")
        except Exception:  # noqa: BLE001 - resource fencing remains authoritative if Jobs is unavailable
            logger.warning("Playlist preflight job cancellation request failed")

    @staticmethod
    def _direct_identity(item: DirectUrlInput) -> dict[str, Any]:
        try:
            parsed = urlparse(item.url)
            _ = parsed.port
        except ValueError as exc:
            raise PlaylistRunValidationError("invalid_direct_url") from exc
        if (
            parsed.scheme.lower() not in {"http", "https"}
            or not parsed.netloc
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or bool(parsed.fragment)
        ):
            raise PlaylistRunValidationError("invalid_direct_url")
        query = parse_qs(parsed.query, keep_blank_values=True)
        if any(_SENSITIVE_QUERY_KEY.search(key) for key in query):
            raise PlaylistRunValidationError("invalid_direct_url")
        try:
            classified = classify_playlist_url(item.url)
        except ValueError as exc:
            raise PlaylistRunValidationError("invalid_direct_url") from exc
        if classified.is_playlist:
            raise PlaylistPreflightRequiredError("playlist_preflight_required")
        source_url = (
            canonical_youtube_video_url(classified.video_id)
            if classified.video_id is not None and classified.source_kind == "youtube_video"
            else item.url
        )
        canonical_url = normalize_media_dedupe_url(source_url)
        if canonical_url is None or len(canonical_url) > 8192:
            raise PlaylistRunValidationError("invalid_direct_url")
        try:
            canonical_classification = classify_playlist_url(canonical_url)
        except ValueError as exc:
            raise PlaylistRunValidationError("invalid_direct_url") from exc
        if canonical_classification.is_playlist:
            raise PlaylistPreflightRequiredError("playlist_preflight_required")
        if (
            classified.source_kind == "youtube_video"
            and classified.video_id is not None
            and _YOUTUBE_ID.fullmatch(classified.video_id) is None
        ):
            if "list" in classified.video_id.lower():
                raise PlaylistPreflightRequiredError("playlist_preflight_required")
            raise PlaylistRunValidationError("invalid_direct_url")
        return {
            "occurrence_id": item.occurrence_id,
            "input_kind": item.input_kind,
            "materialization_id": None,
            "source_url": canonical_url,
            "lookup_urls": list(dict.fromkeys((item.url, canonical_url))),
            "normalized_source_id": (
                classified.normalized_source_id if classified.source_kind == "youtube_video" else f"url:{canonical_url}"
            ),
            "source_kind": classified.source_kind,
            "display_metadata": item.display_metadata.model_dump(exclude_none=True),
            "state": "staged",
        }

    def _resolve_run_inputs(
        self,
        owner_user_id: str,
        inputs: Sequence[MaterializedPlaylistItemInput | DirectUrlInput | FileStubInput],
    ) -> list[dict[str, Any]]:
        materialized_pairs = [
            (item.materialization_id, item.occurrence_id)
            for item in inputs
            if isinstance(item, MaterializedPlaylistItemInput)
        ]
        authoritative_items = iter(
            self._store.resolve_materialization_occurrences(owner_user_id, materialized_pairs)
            if materialized_pairs
            else ()
        )
        resolved: list[dict[str, Any]] = []
        for item in inputs:
            if isinstance(item, MaterializedPlaylistItemInput):
                authoritative = next(authoritative_items)
                if authoritative.source_url is None:
                    raise PlaylistIngestNotFoundError("playlist resource not found")
                resolved.append(
                    {
                        "occurrence_id": item.occurrence_id,
                        "input_kind": item.input_kind,
                        "materialization_id": item.materialization_id,
                        "source_url": authoritative.source_url,
                        "lookup_urls": [authoritative.source_url],
                        "normalized_source_id": authoritative.normalized_source_id or f"url:{authoritative.source_url}",
                        "source_kind": authoritative.source_kind,
                        "display_metadata": authoritative.display_metadata,
                        "state": "staged",
                    }
                )
            elif isinstance(item, DirectUrlInput):
                resolved.append(self._direct_identity(item))
            else:
                display = item.display_metadata.model_dump(exclude_none=True)
                display.update(
                    {
                        "name": item.name,
                        "content_type": item.content_type,
                        "size_bytes": item.size_bytes,
                    }
                )
                resolved.append(
                    {
                        "occurrence_id": item.occurrence_id,
                        "input_kind": item.input_kind,
                        "materialization_id": None,
                        "source_url": None,
                        "normalized_source_id": None,
                        "source_kind": "file",
                        "display_metadata": {key: value for key, value in display.items() if value is not None},
                        "state": "awaiting_upload",
                    }
                )
        return resolved

    def _fresh_duplicate_evidence(
        self,
        owner_user_id: str,
        items: Sequence[dict[str, Any]],
    ) -> list[DuplicateEvidence]:
        urls = list(
            dict.fromkeys(str(candidate) for item in items for candidate in item.get("lookup_urls", ()) if candidate)
        )
        library_by_url: dict[str, int] = {}
        if urls:
            media_db = None
            try:
                media_db = self._media_db_factory(owner_user_id)
                rows = get_media_by_urls(media_db, urls)
            except Exception as exc:
                raise PlaylistRunValidationError("library_lookup_failed") from exc
            finally:
                if media_db is not None:
                    with contextlib.suppress(Exception):
                        media_db.close_connection()
            for row in rows:
                media_id = row.get("id")
                if type(media_id) is not int or media_id < 1:
                    continue
                for candidate in media_dedupe_url_candidates(row.get("url")):
                    library_by_url.setdefault(candidate, media_id)

        seen: dict[str, str] = {}
        evidence: list[DuplicateEvidence] = []
        for item in items:
            source_url = item.get("source_url")
            if source_url is None:
                evidence.append(DuplicateEvidence(kind="none"))
                continue
            source_key = str(item.get("normalized_source_id") or f"url:{source_url}")
            first_occurrence = seen.get(source_key)
            if first_occurrence is not None:
                evidence.append(
                    DuplicateEvidence(
                        kind="in_run",
                        duplicate_of_occurrence_id=first_occurrence,
                    )
                )
            else:
                evidence_candidates = dict.fromkeys(
                    candidate
                    for lookup_url in item.get("lookup_urls", (source_url,))
                    for candidate in media_dedupe_url_candidates(str(lookup_url))
                )
                media_id = next(
                    (library_by_url[candidate] for candidate in evidence_candidates if candidate in library_by_url),
                    None,
                )
                evidence.append(
                    DuplicateEvidence(kind="library", existing_media_id=media_id)
                    if media_id is not None
                    else DuplicateEvidence(kind="none")
                )
            seen.setdefault(source_key, str(item["occurrence_id"]))
        return evidence

    @staticmethod
    def _review_required_item(
        occurrence_id: str,
        reason: str,
        evidence: DuplicateEvidence,
        allowed_actions: Sequence[DuplicatePolicy] | None = None,
    ) -> ReviewRequiredItem:
        return ReviewRequiredItem(
            occurrence_id=occurrence_id,
            reason=reason,
            evidence=evidence,
            allowed_actions=(
                list(allowed_actions)
                if allowed_actions is not None
                else (list(DuplicatePolicy) if evidence.kind != "none" else [])
            ),
        )

    def _renew_run_initialization_or_pending(
        self,
        owner_user_id: str,
        run_id: str,
        initialization_token: str,
        initialization_lease_seconds: int,
    ) -> None:
        if not self._store.renew_run_initialization(
            owner_user_id,
            run_id,
            initialization_token=initialization_token,
            initialization_lease_seconds=initialization_lease_seconds,
        ):
            raise PlaylistRunPendingError(run_id)

    @staticmethod
    def _manifest_from_run_items(
        records: Sequence[MediaIngestRunItemRecord],
    ) -> list[dict[str, Any]]:
        return [
            {
                "occurrence_id": item.occurrence_id,
                "input_kind": item.input_kind,
                "materialization_id": item.materialization_id,
                "source_url": item.source_url,
                "normalized_source_id": item.normalized_source_id,
                "source_kind": item.source_kind,
                "display_metadata": item.display_metadata,
                "state": item.state,
                "action": item.action,
                "metadata_patch": item.metadata_patch,
                "media_id": item.media_id,
                "planned_collection_item_id": item.planned_collection_item_id,
            }
            for item in records
        ]

    def _resume_run_initialization(
        self,
        owner_user_id: str,
        run: MediaIngestRunRecord,
        request: PlaylistIngestRunCreateRequest,
        *,
        initialization_token: str,
        initialization_lease_seconds: int,
    ) -> MediaIngestRunRecord:
        records = list(self._store.list_run_items(owner_user_id, run.run_id, limit=500))
        if not records:
            raise PlaylistRunPendingError(run.run_id)
        manifest = self._manifest_from_run_items(records)
        collections_db = None
        try:
            if request.new_collection is not None and run.collection_id is None:
                try:
                    collections_db = self._collections_db_factory(owner_user_id)
                except Exception as exc:
                    raise PlaylistRunValidationError("collection_planning_failed") from exc
                run = self._create_and_attach_collection_plan(
                    owner_user_id,
                    run.run_id,
                    manifest,
                    request.new_collection,
                    collections_db,
                    initialization_token=initialization_token,
                    initialization_lease_seconds=initialization_lease_seconds,
                )
            if collections_db is None and any(
                item["action"] in {"include_existing", "update_metadata_only"}
                and item["planned_collection_item_id"] is not None
                and record.state != "terminal"
                for item, record in zip(manifest, records, strict=True)
            ):
                collections_db = self._collections_db_factory(owner_user_id)
            try:
                self._resolve_nonprocessing_actions(
                    owner_user_id,
                    run.run_id,
                    manifest,
                    collections_db=collections_db,
                    initialization_token=initialization_token,
                    initialization_lease_seconds=initialization_lease_seconds,
                )
            except PlaylistRunPendingError:
                raise
            except Exception as exc:
                raise PlaylistRunPendingError(run.run_id) from exc
            self._resolved_run_or_pending(owner_user_id, run.run_id)
            self._renew_run_initialization_or_pending(
                owner_user_id,
                run.run_id,
                initialization_token,
                initialization_lease_seconds,
            )
            try:
                return self._store.complete_run_initialization(
                    owner_user_id,
                    run.run_id,
                    initialization_token=initialization_token,
                )
            except PlaylistIngestConflictError as exc:
                raise PlaylistRunPendingError(run.run_id) from exc
        finally:
            if collections_db is not None:
                with contextlib.suppress(Exception):
                    collections_db.close()

    def create_run(
        self,
        owner_user_id: str,
        *,
        client_request_id: str | None = None,
        inputs: Sequence[Mapping[str, Any] | BaseModel],
        review_overrides: Mapping[str, Mapping[str, Any] | BaseModel],
        processing_options: Mapping[str, Any] | None = None,
        playlist_summaries: Sequence[Mapping[str, Any]] | None = None,
        collection_id: int | None = None,
        new_collection: Mapping[str, Any] | BaseModel | None = None,
    ) -> MediaIngestRunRecord:
        """Validate refreshed evidence, then atomically persist a mixed run manifest."""
        if collection_id is not None:
            raise PlaylistRunValidationError("invalid_run_request")
        request_id = client_request_id if client_request_id is not None else f"internal:{uuid4().hex}"
        try:
            raw_inputs = [item.model_dump() if isinstance(item, BaseModel) else dict(item) for item in inputs]
            raw_overrides = {
                key: value.model_dump() if isinstance(value, BaseModel) else dict(value)
                for key, value in review_overrides.items()
            }
            request = PlaylistIngestRunCreateRequest.model_validate(
                {
                    "client_request_id": request_id,
                    "inputs": raw_inputs,
                    "review_overrides": raw_overrides,
                    "processing_options": dict(processing_options) if processing_options is not None else None,
                    "playlist_summaries": (
                        [dict(summary) for summary in playlist_summaries] if playlist_summaries is not None else None
                    ),
                    "new_collection": (
                        new_collection.model_dump()
                        if isinstance(new_collection, BaseModel)
                        else (dict(new_collection) if new_collection is not None else None)
                    ),
                }
            )
        except (AttributeError, TypeError, ValueError, ValidationError) as exc:
            raise PlaylistRunValidationError("invalid_run_request") from exc

        owner = self._store._owner(owner_user_id)
        fingerprint = _request_fingerprint(request)
        initialization_token = uuid4().hex
        initialization_lease_seconds = _bounded_env_int(
            "PLAYLIST_RUN_INITIALIZATION_LEASE_SECONDS",
            30,
            minimum=1,
            maximum=900,
        )
        existing = self._store.get_run_by_client_request_id(owner, request.client_request_id)
        if existing is not None:
            if existing.request_fingerprint != fingerprint:
                raise PlaylistIngestConflictError("request fingerprint does not match")
            if existing.initialization_token is None:
                return existing
            claimed = self._store.claim_run_initialization(
                owner,
                client_request_id=request.client_request_id,
                request_fingerprint=fingerprint,
                initialization_token=initialization_token,
                initialization_lease_seconds=initialization_lease_seconds,
            )
            if claimed.initialization_token is None:
                return claimed
            if claimed.initialization_token != initialization_token:
                raise PlaylistRunPendingError(claimed.run_id)
            return self._resume_run_initialization(
                owner,
                claimed,
                request,
                initialization_token=initialization_token,
                initialization_lease_seconds=initialization_lease_seconds,
            )

        self._cleanup_expired_resources(owner)
        resolved = self._resolve_run_inputs(owner, request.inputs)
        evidence = self._fresh_duplicate_evidence(owner, resolved)
        occurrence_ids = {item["occurrence_id"] for item in resolved}

        review_items: list[ReviewRequiredItem] = []
        for item, current_evidence in zip(resolved, evidence, strict=True):
            occurrence_id = str(item["occurrence_id"])
            override = request.review_overrides.get(occurrence_id)
            if current_evidence.kind == "none":
                if override is not None:
                    review_items.append(
                        self._review_required_item(
                            occurrence_id,
                            "duplicate_no_longer_present",
                            current_evidence,
                        )
                    )
                item["action"] = "ingest"
                item["metadata_patch"] = None
                continue
            if override is None:
                review_items.append(
                    self._review_required_item(
                        occurrence_id,
                        "duplicate_action_required",
                        current_evidence,
                    )
                )
                continue
            try:
                validated_override = ReviewOverride.model_validate(override.model_dump())
            except ValidationError:
                review_items.append(
                    self._review_required_item(
                        occurrence_id,
                        "invalid_duplicate_override",
                        current_evidence,
                    )
                )
                continue
            target_omitted = (
                validated_override.existing_media_id is None
                and validated_override.duplicate_of_occurrence_id is None
            )
            target_matches = target_omitted or (
                current_evidence.kind == "library"
                and validated_override.existing_media_id == current_evidence.existing_media_id
                and validated_override.duplicate_of_occurrence_id is None
            ) or (
                current_evidence.kind == "in_run"
                and validated_override.duplicate_of_occurrence_id == current_evidence.duplicate_of_occurrence_id
                and validated_override.existing_media_id is None
            )
            if not target_matches:
                review_items.append(
                    self._review_required_item(
                        occurrence_id,
                        "duplicate_target_changed",
                        current_evidence,
                    )
                )
                continue
            if current_evidence.kind == "in_run" and validated_override.duplicate_policy in {
                DuplicatePolicy.INCLUDE_EXISTING,
                DuplicatePolicy.UPDATE_METADATA_ONLY,
            }:
                review_items.append(
                    self._review_required_item(
                        occurrence_id,
                        "in_run_duplicate_requires_processing_or_skip",
                        current_evidence,
                        [DuplicatePolicy.SKIP, DuplicatePolicy.OVERWRITE],
                    )
                )
                continue
            item["action"] = validated_override.duplicate_policy.value
            item["existing_media_id"] = current_evidence.existing_media_id
            item["metadata_patch"] = (
                validated_override.metadata_patch.model_dump(exclude_none=True)
                if validated_override.metadata_patch is not None
                else None
            )
        for occurrence_id in sorted(set(request.review_overrides) - occurrence_ids):
            review_items.append(
                self._review_required_item(
                    occurrence_id,
                    "unknown_review_override",
                    DuplicateEvidence(kind="none"),
                )
            )
        if review_items:
            raise ReviewRequiredError(review_items)

        manifest = [
            {
                **{key: value for key, value in item.items() if key not in {"lookup_urls", "existing_media_id"}},
                **(
                    {"media_id": item["existing_media_id"]}
                    if item.get("existing_media_id") is not None
                    and item.get("action") in {"skip", "include_existing", "update_metadata_only"}
                    else {}
                ),
            }
            for item in resolved
        ]
        created = self._store.create_validated_run(
            owner,
            items=manifest,
            processing_options=request.processing_options,
            playlist_summaries=request.playlist_summaries,
            collection_id=None,
            client_request_id=request.client_request_id,
            request_fingerprint=fingerprint,
            initialization_token=initialization_token,
            initialization_lease_seconds=initialization_lease_seconds,
        )
        if created.request_fingerprint != fingerprint:
            raise PlaylistIngestConflictError("request fingerprint does not match")
        if created.initialization_token is None:
            return created
        if created.initialization_token != initialization_token:
            claimed = self._store.claim_run_initialization(
                owner,
                client_request_id=request.client_request_id,
                request_fingerprint=fingerprint,
                initialization_token=initialization_token,
                initialization_lease_seconds=initialization_lease_seconds,
            )
            if claimed.initialization_token is None:
                return claimed
            if claimed.initialization_token != initialization_token:
                raise PlaylistRunPendingError(claimed.run_id)
            created = claimed
        return self._resume_run_initialization(
            owner,
            created,
            request,
            initialization_token=initialization_token,
            initialization_lease_seconds=initialization_lease_seconds,
        )

    def reconcile_nonprocessing_actions(
        self,
        owner_user_id: str,
        run_id: str,
    ) -> MediaIngestRunRecord:
        """Resume one owner-scoped run's preparing actions for status/SSE callers."""
        owner = self._store._owner(owner_user_id)
        collections_db = None
        try:
            run = self._store.get_run(owner, run_id)
            records = list(self._store.list_run_items(owner, run.run_id, limit=500))
            items = [
                {
                    "occurrence_id": item.occurrence_id,
                    "action": item.action,
                    "media_id": item.media_id,
                    "metadata_patch": item.metadata_patch,
                    "planned_collection_item_id": item.planned_collection_item_id,
                }
                for item in records
            ]
            if any(
                item["action"] in {"include_existing", "update_metadata_only"}
                and item["planned_collection_item_id"] is not None
                and record.state != "terminal"
                for item, record in zip(items, records, strict=True)
            ):
                collections_db = self._collections_db_factory(owner)
            self._resolve_nonprocessing_actions(
                owner,
                run.run_id,
                items,
                collections_db=collections_db,
            )
        except PlaylistIngestNotFoundError:
            raise
        except Exception as exc:
            raise PlaylistRunPendingError(run_id) from exc
        finally:
            if collections_db is not None:
                with contextlib.suppress(Exception):
                    collections_db.close()
        return self._resolved_run_or_pending(owner, run.run_id)

    @staticmethod
    def _safe_progress_message(value: object) -> str | None:
        if not isinstance(value, str):
            return None
        message = value.strip()
        return message[:1000] if message else None

    def _exact_run_item_job_binding(
        self,
        owner_user_id: str,
        run_id: str,
        item: MediaIngestRunItemRecord,
        job: Mapping[str, Any] | None,
        *,
        expected_job_id: int,
    ) -> dict[str, Any] | None:
        """Return a Jobs binding only when every stored run authority matches."""
        binding = self._jobs.normalize_job_binding_view(job, owner_user_id=owner_user_id)
        payload = binding.get("payload") if binding is not None else None
        if not (
            binding is not None
            and binding.get("id") == expected_job_id
            and binding.get("domain") == "media_ingest"
            and binding.get("queue") == item.submission_queue
            and binding.get("job_type") == "media_ingest_item"
            and binding.get("batch_group") == item.batch_id
            and binding.get("idempotency_key") == item.idempotency_identity
            and isinstance(payload, Mapping)
            and payload.get("run_id") == run_id
            and payload.get("occurrence_id") == item.occurrence_id
            and type(payload.get("attempt")) is int
            and payload.get("attempt") == item.attempt
        ):
            return None
        return binding

    def reconcile_run_jobs(
        self,
        owner_user_id: str,
        run_id: str,
    ) -> MediaIngestRunRecord:
        """Reconcile every currently bound run occurrence from its exact Jobs row."""
        owner = self._store._owner(owner_user_id)
        with contextlib.suppress(PlaylistRunPendingError):
            self.reconcile_nonprocessing_actions(owner, run_id)
        run = self._store.get_run(owner, run_id)
        items = list(self._store.list_run_items(owner, run.run_id, limit=500))
        for item in items:
            if item.job_id is None or item.state == "terminal":
                continue
            try:
                job = self._jobs.get_job_or_archived(int(item.job_id), domain="media_ingest")
                binding = self._exact_run_item_job_binding(
                    owner,
                    run.run_id,
                    item,
                    job,
                    expected_job_id=int(item.job_id),
                )
            except Exception:  # noqa: BLE001 - a temporary status read failure is recoverable
                job = None
                binding = None
            progress_percent = None
            progress_message = None
            media_id = None
            retryable = False
            outcome = None
            if binding is None or not isinstance(job, Mapping):
                state = "status_unavailable"
            else:
                raw_progress = job.get("progress_percent")
                if type(raw_progress) in {int, float} and 0 <= float(raw_progress) <= 100:
                    progress_percent = float(raw_progress)
                progress_message = self._safe_progress_message(job.get("progress_message"))
                job_status = str(binding.get("status") or "").lower()
                if job_status == "queued":
                    state = "cancellation_requested" if item.state == "cancellation_requested" else "queued"
                elif job_status == "processing":
                    state = "cancellation_requested" if item.state == "cancellation_requested" else "running"
                elif job_status == "cancelled":
                    state = "terminal"
                    outcome = "cancelled"
                elif job_status in {"failed", "quarantined"}:
                    state = "terminal"
                    outcome = "processing_failed"
                    retryable = True
                elif job_status == "completed":
                    result = job.get("result")
                    exact_result = (
                        isinstance(result, Mapping)
                        and result.get("run_id") == run.run_id
                        and result.get("occurrence_id") == item.occurrence_id
                        and type(result.get("attempt")) is int
                        and result.get("attempt") == item.attempt
                    )
                    result_media_id = result.get("media_id") if exact_result else None
                    if type(result_media_id) is int and result_media_id > 0:
                        state = "terminal"
                        outcome = "completed"
                        media_id = result_media_id
                    else:
                        state = "terminal"
                        outcome = "processing_failed"
                        retryable = True
                else:
                    state = "status_unavailable"
            self._store.reconcile_run_item_job(
                owner,
                run.run_id,
                item.occurrence_id,
                expected_job_id=int(item.job_id),
                expected_attempt=item.attempt,
                state=state,
                outcome=outcome,
                progress_percent=progress_percent,
                progress_message=progress_message,
                retryable=retryable,
                media_id=media_id,
            )
        return self._store.get_run(owner, run.run_id)

    def cancel_run(
        self,
        owner_user_id: str,
        run_id: str,
        *,
        occurrence_ids: Sequence[str] | None,
        reason: str | None,
    ) -> MediaIngestRunRecord:
        """Cancel selected occurrences, or every nonterminal occurrence when omitted."""
        owner = self._store._owner(owner_user_id)
        self.reconcile_run_jobs(owner, run_id)
        items = list(self._store.list_run_items(owner, run_id, limit=500))
        by_occurrence = {item.occurrence_id: item for item in items}
        selected = list(by_occurrence) if occurrence_ids is None else list(occurrence_ids)
        if any(occurrence_id not in by_occurrence for occurrence_id in selected):
            raise PlaylistSelectionError("invalid_occurrence_selection")
        for occurrence_id in selected:
            item = by_occurrence[occurrence_id]
            job = None
            try:
                if item.job_id is not None:
                    job = self._jobs.get_job_or_archived(int(item.job_id), domain="media_ingest")
                elif item.idempotency_identity and item.batch_id and item.submission_queue:
                    job = self._jobs.get_job_by_idempotency(
                        domain="media_ingest",
                        queue=item.submission_queue,
                        job_type="media_ingest_item",
                        idempotency_key=item.idempotency_identity,
                        owner_user_id=owner,
                        batch_group=item.batch_id,
                    )
            except Exception:  # noqa: BLE001 - durable item fencing remains authoritative
                job = None
            changed = self._store.request_run_item_cancellation(
                owner,
                run_id,
                occurrence_id,
                expected_attempt=item.attempt,
            )
            if changed.state == "cancellation_requested" or (item.job_id is None and isinstance(job, Mapping)):
                try:
                    job_id = int(job.get("id")) if isinstance(job, Mapping) else int(changed.job_id)
                    fresh_job = self._jobs.get_job_or_archived(job_id, domain="media_ingest")
                    binding = self._exact_run_item_job_binding(
                        owner,
                        run_id,
                        changed,
                        fresh_job,
                        expected_job_id=job_id,
                    )
                    if binding is None:
                        continue
                    self._jobs.cancel_job(job_id, reason=reason)
                except Exception:  # noqa: BLE001 - the next reconciliation retries status observation
                    logger.warning(
                        "Playlist occurrence cancellation could not confirm Jobs state for run {} item {}",
                        run_id,
                        occurrence_id,
                    )
        return self.reconcile_run_jobs(owner, run_id)

    def retry_run_items(
        self,
        owner_user_id: str,
        run_id: str,
        occurrence_ids: Sequence[str],
    ) -> tuple[MediaIngestRunItemRecord, ...]:
        """Media-first reconcile eligible failures, then CAS new attempts once."""
        owner = self._store._owner(owner_user_id)
        self.reconcile_run_jobs(owner, run_id)
        items = {item.occurrence_id: item for item in self._store.list_run_items(owner, run_id, limit=500)}
        selected = list(occurrence_ids)
        if any(occurrence_id not in items for occurrence_id in selected):
            raise PlaylistSelectionError("invalid_occurrence_selection")
        eligible = [
            items[occurrence_id]
            for occurrence_id in selected
            if items[occurrence_id].state == "terminal" and items[occurrence_id].retryable
        ]
        if not eligible:
            return ()

        urls = list(dict.fromkeys(item.source_url for item in eligible if item.source_url))
        media_db = None
        collections_db = None
        try:
            media_db = self._media_db_factory(owner)
            rows = get_media_by_urls(media_db, urls) if urls else []
            by_url: dict[str, int] = {}
            for row in rows:
                media_id = row.get("id")
                if type(media_id) is not int or media_id < 1:
                    continue
                for candidate in media_dedupe_url_candidates(row.get("url")):
                    by_url.setdefault(candidate, media_id)

            retried: list[MediaIngestRunItemRecord] = []
            for item in eligible:
                resolved_media_id = next(
                    (
                        by_url[candidate]
                        for candidate in media_dedupe_url_candidates(item.source_url)
                        if candidate in by_url
                    ),
                    None,
                )
                if resolved_media_id is None and item.planned_collection_item_id is not None:
                    if collections_db is None:
                        collections_db = self._collections_db_factory(owner)
                    planned = collections_db.get_media_collection_item(item.planned_collection_item_id)
                    planned_media_id = getattr(planned, "media_id", None)
                    if type(planned_media_id) is int and planned_media_id > 0:
                        media = get_media_by_id(media_db, planned_media_id)
                        if isinstance(media, Mapping) and type(media.get("id")) is int:
                            resolved_media_id = int(media["id"])
                updated, changed = self._store.retry_run_item(
                    owner,
                    run_id,
                    item.occurrence_id,
                    expected_attempt=item.attempt,
                    resolved_media_id=resolved_media_id,
                )
                if changed and resolved_media_id is None:
                    retried.append(updated)
            return tuple(retried)
        except (PlaylistIngestNotFoundError, PlaylistSelectionError):
            raise
        except Exception as exc:
            raise PlaylistRunStatusUnavailableError("run_status_unavailable") from exc
        finally:
            if collections_db is not None:
                with contextlib.suppress(Exception):
                    collections_db.close()
            if media_db is not None:
                with contextlib.suppress(Exception):
                    media_db.close_connection()

    def _resolved_run_or_pending(
        self,
        owner_user_id: str,
        run_id: str,
    ) -> MediaIngestRunRecord:
        """Return a run only when every zero-job action is terminal."""
        try:
            items = self._store.list_run_items(owner_user_id, run_id, limit=500)
        except PlaylistIngestNotFoundError:
            raise
        except Exception as exc:
            raise PlaylistRunPendingError(run_id) from exc
        if any(
            item.action in {"skip", "include_existing", "update_metadata_only"} and item.state != "terminal"
            for item in items
        ):
            raise PlaylistRunPendingError(run_id)
        try:
            return self._store.get_run(owner_user_id, run_id)
        except PlaylistIngestNotFoundError:
            raise
        except Exception as exc:
            raise PlaylistRunPendingError(run_id) from exc

    def _create_and_attach_collection_plan(
        self,
        owner_user_id: str,
        run_id: str,
        items: list[dict[str, Any]],
        collection_request: BaseModel,
        collections_db: Any,
        *,
        initialization_token: str,
        initialization_lease_seconds: int,
    ) -> MediaIngestRunRecord:
        """Create one non-skip plan, then compensate if run attachment fails."""
        planned_items: list[dict[str, Any]] = []
        planned_occurrences: list[str] = []
        for ordinal, item in enumerate(items, start=1):
            if item.get("action") == "skip":
                continue
            display = dict(item.get("display_metadata") or {})
            occurrence_id = str(item["occurrence_id"])
            planned_occurrences.append(occurrence_id)
            planned_items.append(
                {
                    "source_url": item.get("source_url") or f"urn:tldw:file-stub:{occurrence_id}",
                    "normalized_source_id": item.get("normalized_source_id"),
                    "source_kind": item.get("source_kind"),
                    "ordinal": ordinal,
                    "title": display.get("title"),
                    "speaker": display.get("channel_or_uploader"),
                    "published_at": display.get("published_at"),
                    "duplicate_status": "existing" if item.get("action") != "ingest" else "unknown",
                }
            )
        self._renew_run_initialization_or_pending(
            owner_user_id,
            run_id,
            initialization_token,
            initialization_lease_seconds,
        )
        created_new = False
        try:
            collection = collections_db.create_media_collection_with_items(
                name=collection_request.name,
                kind="playlist_ingest",
                description=collection_request.description,
                source_url=collection_request.source_url,
                metadata={"playlist_ingest_run_id": run_id},
                default_tags=collection_request.default_tags,
                items=planned_items,
            )
            created_new = True
        except Exception as exc:  # noqa: BLE001 - reconcile a possible commit before deciding
            try:
                collection = collections_db.get_playlist_ingest_collection_for_run(run_id)
            except KeyError:
                raise PlaylistRunValidationError("collection_planning_failed") from exc
            except Exception as reconciliation_exc:
                raise PlaylistRunValidationError("collection_planning_reconciliation_failed") from reconciliation_exc

        collection_items = list(collection.items)
        collection_item_ids = [int(item.id) for item in collection_items]
        try:
            self._renew_run_initialization_or_pending(
                owner_user_id,
                run_id,
                initialization_token,
                initialization_lease_seconds,
            )
        except PlaylistRunPendingError:
            if created_new:
                try:
                    collections_db.discard_media_collection(
                        int(collection.id),
                        expected_item_ids=collection_item_ids,
                    )
                except Exception as cleanup_exc:
                    raise PlaylistRunValidationError("collection_planning_cleanup_failed") from cleanup_exc
            raise
        if len(collection_items) != len(planned_occurrences):
            try:
                collections_db.discard_media_collection(
                    int(collection.id),
                    expected_item_ids=collection_item_ids,
                )
            except Exception as cleanup_exc:
                raise PlaylistRunValidationError("collection_planning_cleanup_failed") from cleanup_exc
            raise PlaylistRunValidationError("collection_planning_failed")
        mapping = {
            occurrence_id: int(planned_item.id)
            for occurrence_id, planned_item in zip(planned_occurrences, collection_items, strict=True)
        }
        try:
            attached = self._store.attach_collection_plan(
                owner_user_id,
                run_id,
                collection_id=int(collection.id),
                planned_item_ids=mapping,
                initialization_token=initialization_token,
            )
        except Exception as exc:
            try:
                reconciled_run = self._store.get_run(owner_user_id, run_id)
                reconciled_items = self._store.list_run_items(
                    owner_user_id,
                    run_id,
                    limit=max(1, len(items)),
                )
                reconciled_mapping = {
                    item.occurrence_id: item.planned_collection_item_id
                    for item in reconciled_items
                    if item.action != "skip"
                }
            except Exception as reconciliation_exc:
                raise PlaylistRunValidationError("collection_planning_reconciliation_failed") from reconciliation_exc
            if reconciled_run.collection_id == int(collection.id) and reconciled_mapping == mapping:
                attached = reconciled_run
            elif reconciled_run.collection_id is not None or any(
                item_id is not None for item_id in reconciled_mapping.values()
            ):
                raise PlaylistRunValidationError("collection_planning_reconciliation_failed") from exc
            else:
                try:
                    collections_db.discard_media_collection(
                        int(collection.id),
                        expected_item_ids=collection_item_ids,
                    )
                except Exception as cleanup_exc:
                    raise PlaylistRunValidationError("collection_planning_cleanup_failed") from cleanup_exc
                raise PlaylistRunValidationError("collection_planning_failed") from exc
        for item in items:
            item["planned_collection_item_id"] = mapping.get(str(item["occurrence_id"]))
        return attached

    def _resolve_nonprocessing_actions(
        self,
        owner_user_id: str,
        run_id: str,
        items: Sequence[Mapping[str, Any]],
        *,
        collections_db: Any | None = None,
        initialization_token: str | None = None,
        initialization_lease_seconds: int = 30,
    ) -> None:
        """Finish reviewed library-duplicate actions without creating media jobs."""
        metadata_db = None
        try:
            for item in items:
                action = item.get("action")
                media_id = item.get("media_id")
                if action not in {"skip", "include_existing", "update_metadata_only"}:
                    continue
                if action != "skip" and media_id is None:
                    continue
                if initialization_token is not None:
                    self._renew_run_initialization_or_pending(
                        owner_user_id,
                        run_id,
                        initialization_token,
                        initialization_lease_seconds,
                    )
                prepared = self._store.prepare_nonprocessing_run_item(
                    owner_user_id,
                    run_id,
                    str(item["occurrence_id"]),
                )
                if prepared.state == "terminal":
                    continue
                if action == "skip":
                    outcome = "skipped_existing"
                elif action == "include_existing":
                    outcome = "included_existing"
                else:
                    if initialization_token is not None:
                        self._renew_run_initialization_or_pending(
                            owner_user_id,
                            run_id,
                            initialization_token,
                            initialization_lease_seconds,
                        )
                    try:
                        if metadata_db is None:
                            metadata_db = self._media_db_factory(owner_user_id)
                        metadata_db.apply_media_metadata_patch(
                            int(media_id),
                            **dict(item.get("metadata_patch") or {}),
                        )
                    except (ConflictError, InputError):
                        if initialization_token is not None:
                            self._renew_run_initialization_or_pending(
                                owner_user_id,
                                run_id,
                                initialization_token,
                                initialization_lease_seconds,
                            )
                        outcome = "metadata_update_failed"
                    except Exception:  # noqa: BLE001 - an exact retry reconciles ambiguous commit state
                        if initialization_token is not None:
                            self._renew_run_initialization_or_pending(
                                owner_user_id,
                                run_id,
                                initialization_token,
                                initialization_lease_seconds,
                            )
                        try:
                            metadata_db.apply_media_metadata_patch(
                                int(media_id),
                                **dict(item.get("metadata_patch") or {}),
                            )
                        except Exception as reconciliation_exc:  # noqa: BLE001 - unknown state remains recoverable
                            if initialization_token is not None:
                                self._renew_run_initialization_or_pending(
                                    owner_user_id,
                                    run_id,
                                    initialization_token,
                                    initialization_lease_seconds,
                                )
                            logger.warning(
                                "Playlist metadata reconciliation remains ambiguous for run {} item {} ({})",
                                run_id,
                                item.get("occurrence_id"),
                                type(reconciliation_exc).__name__,
                            )
                            continue
                        else:
                            if initialization_token is not None:
                                self._renew_run_initialization_or_pending(
                                    owner_user_id,
                                    run_id,
                                    initialization_token,
                                    initialization_lease_seconds,
                                )
                            outcome = "metadata_updated"
                    else:
                        if initialization_token is not None:
                            self._renew_run_initialization_or_pending(
                                owner_user_id,
                                run_id,
                                initialization_token,
                                initialization_lease_seconds,
                            )
                        outcome = "metadata_updated"
                planned_item_id = item.get("planned_collection_item_id")
                resolved_membership = None
                if (
                    collections_db is not None
                    and planned_item_id is not None
                    and outcome in {"included_existing", "metadata_updated"}
                ):
                    if initialization_token is not None:
                        self._renew_run_initialization_or_pending(
                            owner_user_id,
                            run_id,
                            initialization_token,
                            initialization_lease_seconds,
                        )
                    try:
                        resolved_membership = collections_db.resolve_media_collection_item(
                            int(planned_item_id),
                            media_id=int(media_id),
                            status="skipped_existing" if outcome == "included_existing" else "completed",
                        )
                    except Exception:  # noqa: BLE001 - reconcile a possible commit before deciding
                        if initialization_token is not None:
                            self._renew_run_initialization_or_pending(
                                owner_user_id,
                                run_id,
                                initialization_token,
                                initialization_lease_seconds,
                            )
                        try:
                            current_membership = collections_db.get_media_collection_item(int(planned_item_id))
                        except Exception as reconciliation_exc:  # noqa: BLE001 - unknown state remains recoverable
                            logger.warning(
                                "Playlist membership reconciliation remains ambiguous for run {} item {} ({})",
                                run_id,
                                item.get("occurrence_id"),
                                type(reconciliation_exc).__name__,
                            )
                            continue
                        expected_status = "skipped_existing" if outcome == "included_existing" else "completed"
                        if (
                            current_membership.status != expected_status
                            or current_membership.media_id != int(media_id)
                            or current_membership.content_item_id is not None
                            or current_membership.latest_job_id is not None
                            or current_membership.latest_run_id is not None
                        ):
                            outcome = "metadata_update_failed"
                        else:
                            resolved_membership = current_membership
                    else:
                        if initialization_token is not None:
                            self._renew_run_initialization_or_pending(
                                owner_user_id,
                                run_id,
                                initialization_token,
                                initialization_lease_seconds,
                            )
                if initialization_token is not None:
                    self._renew_run_initialization_or_pending(
                        owner_user_id,
                        run_id,
                        initialization_token,
                        initialization_lease_seconds,
                    )
                try:
                    self._store.resolve_nonprocessing_run_item(
                        owner_user_id,
                        run_id,
                        str(item["occurrence_id"]),
                        outcome=outcome,
                        media_id=int(media_id) if media_id is not None else None,
                    )
                except Exception as exc:
                    try:
                        current_item = self._store.get_run_item(
                            owner_user_id,
                            run_id,
                            str(item["occurrence_id"]),
                        )
                    except Exception as reconciliation_exc:
                        raise PlaylistRunValidationError(
                            "duplicate_action_reconciliation_failed"
                        ) from reconciliation_exc
                    if (
                        current_item.state == "terminal"
                        and current_item.outcome == outcome
                        and current_item.media_id == (int(media_id) if media_id is not None else None)
                    ):
                        continue
                    if current_item.state != "preparing":
                        raise PlaylistRunValidationError("duplicate_action_reconciliation_failed") from exc
                    if resolved_membership is not None:
                        if initialization_token is not None:
                            self._renew_run_initialization_or_pending(
                                owner_user_id,
                                run_id,
                                initialization_token,
                                initialization_lease_seconds,
                            )
                        try:
                            collections_db.restore_media_collection_item_plan(
                                int(resolved_membership.id),
                                expected_media_id=int(resolved_membership.media_id),
                                expected_status=str(resolved_membership.status),
                                expected_updated_at=str(resolved_membership.updated_at),
                            )
                        except Exception as cleanup_exc:
                            raise PlaylistRunValidationError("collection_action_cleanup_failed") from cleanup_exc
                        if initialization_token is not None:
                            self._renew_run_initialization_or_pending(
                                owner_user_id,
                                run_id,
                                initialization_token,
                                initialization_lease_seconds,
                            )
                    continue
        finally:
            if metadata_db is not None:
                with contextlib.suppress(Exception):
                    metadata_db.close_connection()


__all__ = [
    "CreatedPlaylistMaterialization",
    "CreatedPlaylistPreflight",
    "InvalidPlaylistUrlError",
    "PlaylistIngestService",
    "PlaylistPreflightRequiredError",
    "PlaylistPreflightBusyError",
    "PlaylistPreflightIncompleteError",
    "PlaylistPreflightUnavailableError",
    "PlaylistRunPendingError",
    "PlaylistRunStatusUnavailableError",
    "PlaylistRunValidationError",
    "PlaylistSelectionError",
    "ReviewRequiredError",
]
