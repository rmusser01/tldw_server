"""Service layer for owner-scoped asynchronous playlist preflight resources."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from datetime import timedelta
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse

from loguru import logger

from tldw_Server_API.app.core.Ingestion_Media_Processing.Video.playlist_ingest_store import (
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

    def __init__(self, job_manager) -> None:
        self._jobs = job_manager
        self._store = PlaylistIngestStore(job_manager)

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
        try:
            preflight = self._store.reserve_preflight(
                owner_user_id,
                source_url=canonical_url,
                source_kind=classified.source_kind,
                playlist_id=classified.playlist_id,
                expires_at=self._store._now() + timedelta(seconds=ttl_seconds),
                global_capacity=global_capacity,
                owner_capacity=owner_capacity,
            )
        except PlaylistPreflightCapacityError as exc:
            raise PlaylistPreflightBusyError("preflight_busy") from exc

        job: dict | None = None
        try:
            queue = (os.getenv("MEDIA_INGEST_JOBS_DEFAULT_QUEUE") or "default").strip() or "default"
            job = self._jobs.create_job(
                domain="media_ingest",
                queue=queue,
                job_type="playlist_preflight",
                payload={
                    "preflight_id": preflight.preflight_id,
                    "max_items": int(max_items),
                    "timeout_seconds": int(timeout_seconds),
                },
                owner_user_id=str(owner_user_id),
                priority=5,
                max_retries=0,
                available_at=self._store._now() + timedelta(days=1),
                idempotency_key=f"playlist_preflight:{owner_user_id}:{preflight.preflight_id}",
            )
            bound = self._store.bind_preflight_job(
                owner_user_id,
                preflight.preflight_id,
                int(job["id"]),
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


__all__ = [
    "CreatedPlaylistMaterialization",
    "CreatedPlaylistPreflight",
    "InvalidPlaylistUrlError",
    "PlaylistIngestService",
    "PlaylistPreflightBusyError",
    "PlaylistPreflightIncompleteError",
    "PlaylistPreflightUnavailableError",
    "PlaylistSelectionError",
]
