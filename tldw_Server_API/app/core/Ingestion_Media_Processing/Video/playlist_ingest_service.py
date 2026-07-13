"""Service layer for owner-scoped asynchronous playlist preflight resources."""

from __future__ import annotations

import contextlib
import os
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import timedelta
from typing import Any
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse

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
from tldw_Server_API.app.core.DB_Management.media_db.api import get_media_by_urls
from tldw_Server_API.app.core.DB_Management.media_db.dedupe_urls import (
    media_dedupe_url_candidates,
    normalize_media_dedupe_url,
)
from tldw_Server_API.app.core.Ingestion_Media_Processing.Video.playlist_ingest_store import (
    _PREFLIGHT_JOB_SENTINEL,
    MediaIngestRunRecord,
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


def _owner_media_db(owner_user_id: str) -> Any:
    """Open the existing owner Media DB without touching AuthNZ storage."""
    from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths
    from tldw_Server_API.app.core.DB_Management.media_db.api import create_media_database

    db_path = DatabasePaths.get_media_db_path(owner_user_id)
    return create_media_database(client_id=f"playlist_ingest:{owner_user_id}", db_path=str(db_path))


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

    def __init__(self, job_manager, *, media_db_factory: Callable[[str], Any] | None = None) -> None:
        self._jobs = job_manager
        self._store = PlaylistIngestStore(job_manager)
        self._media_db_factory = media_db_factory or _owner_media_db

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
    ) -> ReviewRequiredItem:
        return ReviewRequiredItem(
            occurrence_id=occurrence_id,
            reason=reason,
            evidence=evidence,
            allowed_actions=list(DuplicatePolicy) if evidence.kind != "none" else [],
        )

    def create_run(
        self,
        owner_user_id: str,
        *,
        inputs: Sequence[Mapping[str, Any] | BaseModel],
        review_overrides: Mapping[str, Mapping[str, Any] | BaseModel],
        processing_options: Mapping[str, Any] | None = None,
        playlist_summaries: Sequence[Mapping[str, Any]] | None = None,
        collection_id: int | None = None,
    ) -> MediaIngestRunRecord:
        """Validate refreshed evidence, then atomically persist a mixed run manifest."""
        try:
            raw_inputs = [item.model_dump() if isinstance(item, BaseModel) else dict(item) for item in inputs]
            raw_overrides = {
                key: value.model_dump() if isinstance(value, BaseModel) else dict(value)
                for key, value in review_overrides.items()
            }
            request = PlaylistIngestRunCreateRequest.model_validate(
                {
                    "inputs": raw_inputs,
                    "review_overrides": raw_overrides,
                    "processing_options": dict(processing_options) if processing_options is not None else None,
                    "playlist_summaries": (
                        [dict(summary) for summary in playlist_summaries] if playlist_summaries is not None else None
                    ),
                    "collection_id": collection_id,
                }
            )
        except (AttributeError, TypeError, ValueError, ValidationError) as exc:
            raise PlaylistRunValidationError("invalid_run_request") from exc

        owner = self._store._owner(owner_user_id)
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
            target_matches = (
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
            item["action"] = validated_override.duplicate_policy.value
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

        manifest = [{key: value for key, value in item.items() if key != "lookup_urls"} for item in resolved]
        return self._store.create_validated_run(
            owner,
            items=manifest,
            processing_options=request.processing_options,
            playlist_summaries=request.playlist_summaries,
            collection_id=request.collection_id,
        )


__all__ = [
    "CreatedPlaylistMaterialization",
    "CreatedPlaylistPreflight",
    "InvalidPlaylistUrlError",
    "PlaylistIngestService",
    "PlaylistPreflightRequiredError",
    "PlaylistPreflightBusyError",
    "PlaylistPreflightIncompleteError",
    "PlaylistPreflightUnavailableError",
    "PlaylistRunValidationError",
    "PlaylistSelectionError",
    "ReviewRequiredError",
]
