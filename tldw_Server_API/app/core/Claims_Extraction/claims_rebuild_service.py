from __future__ import annotations

import contextlib
import sqlite3
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from queue import Empty, Queue
from typing import Any

from loguru import logger

from tldw_Server_API.app.core.Chunking import chunk_for_embedding
from tldw_Server_API.app.core.Claims_Extraction.budget_guard import resolve_claims_job_budget
from tldw_Server_API.app.core.Claims_Extraction.ingestion_claims import (
    extract_claims_for_chunks,
    store_claims,
)
from tldw_Server_API.app.core.Claims_Extraction.monitoring import (
    record_claims_rebuild_metrics,
)
from tldw_Server_API.app.core.config import settings
from tldw_Server_API.app.core.DB_Management.db_path_utils import get_user_media_db_path
from tldw_Server_API.app.core.DB_Management.media_db.api import managed_media_database

_CLAIMS_REBUILD_NONCRITICAL_EXCEPTIONS: tuple[type[BaseException], ...] = (
    AssertionError,
    AttributeError,
    ConnectionError,
    LookupError,
    OSError,
    RuntimeError,
    TimeoutError,
    TypeError,
    ValueError,
    sqlite3.Error,
)

_TIMESTAMP_PARSE_EXCEPTIONS: tuple[type[BaseException], ...] = (
    OSError,
    OverflowError,
    TypeError,
    ValueError,
)


@dataclass
class ClaimsRebuildTask:
    media_id: int
    db_path: str


def _claims_monitoring_system_user_id() -> int:
    try:
        return int(settings.get("CLAIMS_MONITORING_SYSTEM_USER_ID", 0))
    except (TypeError, ValueError):
        return 0


def _format_timestamp(ts: float | None) -> str | None:
    if not ts:
        return None
    try:
        return datetime.fromtimestamp(float(ts), tz=timezone.utc).strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'
    except _TIMESTAMP_PARSE_EXCEPTIONS:
        return None


class ClaimsRebuildService:
    """Background service to rebuild claims for media items."""

    def __init__(self, worker_threads: int = 1):
        self._queue: Queue[ClaimsRebuildTask] = Queue()
        self._threads: list[threading.Thread] = []
        self._stop = threading.Event()
        self._worker_threads = max(1, int(worker_threads))
        # Stats
        self._stats_lock = threading.Lock()
        self._stats = {"enqueued": 0, "processed": 0, "failed": 0}
        self._last_heartbeat_ts = 0.0
        self._last_processed_ts = 0.0
        self._last_failure: dict[str, Any] | None = None
        self._last_health_persist_ts = 0.0
        self._last_health_persist_queue: int | None = None
        self._health_db_initialized = False

    def start(self) -> None:
        if self._threads:
            return
        for i in range(self._worker_threads):
            t = threading.Thread(target=self._worker_loop, name=f"claims-rebuild-{i}", daemon=True)
            t.start()
            self._threads.append(t)
        logger.info(f"ClaimsRebuildService started with {self._worker_threads} thread(s)")

    def stop(self) -> None:
        self._stop.set()
        for _ in self._threads:
            self._queue.put_nowait(ClaimsRebuildTask(media_id=-1, db_path=""))
        for t in self._threads:
            with contextlib.suppress(_CLAIMS_REBUILD_NONCRITICAL_EXCEPTIONS):
                t.join(timeout=1.0)
        self._threads.clear()
        self._stop.clear()
        stats = self.get_stats()
        logger.info(
            "ClaimsRebuildService stopped (enqueued={enq}, processed={proc}, failed={fail}, queue_len={q})",
            enq=stats.get("enqueued", 0),
            proc=stats.get("processed", 0),
            fail=stats.get("failed", 0),
            q=self._queue.qsize(),
        )

    def submit(self, media_id: int, db_path: str) -> None:
        self._queue.put_nowait(ClaimsRebuildTask(media_id=int(media_id), db_path=str(db_path)))
        with self._stats_lock:
            self._stats["enqueued"] += 1
        record_claims_rebuild_metrics(queue_size=self._queue.qsize())
        self._persist_health()

    def _persist_health(self, *, force: bool = False) -> None:
        now = time.time()
        queue_size = self._queue.qsize()
        if not force:
            if self._last_health_persist_queue == queue_size and (now - self._last_health_persist_ts) < 5.0:
                return
        self._last_health_persist_ts = now
        self._last_health_persist_queue = queue_size
        user_id = _claims_monitoring_system_user_id()
        db_path = None
        try:
            db_path = get_user_media_db_path(user_id)
        except _CLAIMS_REBUILD_NONCRITICAL_EXCEPTIONS as exc:
            logger.debug("Claims rebuild health persistence skipped: {}", exc)
            return
        should_initialize = not self._health_db_initialized
        try:
            with managed_media_database(
                client_id=str(settings.get("SERVER_CLIENT_ID", "SERVER_API_V1")),
                db_path=db_path,
                initialize=should_initialize,
                suppress_close_exceptions=_CLAIMS_REBUILD_NONCRITICAL_EXCEPTIONS,
            ) as db:
                if should_initialize:
                    self._health_db_initialized = True
                last_failure_reason = None
                last_failure_at = None
                if self._last_failure:
                    last_failure_reason = self._last_failure.get("error")
                    last_failure_at = _format_timestamp(self._last_failure.get("timestamp"))
                db.upsert_claims_monitoring_health(
                    user_id=str(user_id),
                    queue_size=queue_size,
                    worker_count=self.get_worker_count(),
                    last_worker_heartbeat=_format_timestamp(self._last_heartbeat_ts),
                    last_processed_at=_format_timestamp(self._last_processed_ts) if self._last_processed_ts else None,
                    last_failure_at=last_failure_at,
                    last_failure_reason=last_failure_reason,
                )
        except _CLAIMS_REBUILD_NONCRITICAL_EXCEPTIONS as exc:
            if should_initialize and not self._health_db_initialized:
                logger.debug("Claims rebuild health persistence DB setup failed: {}", exc)
            else:
                logger.debug("Claims rebuild health persistence failed: {}", exc)

    def _worker_loop(self) -> None:
        while not self._stop.is_set():
            self._touch_heartbeat()
            try:
                task = self._queue.get(timeout=0.5)
            except Empty:
                continue
            if task.media_id < 0:
                # sentinel
                self._queue.task_done()
                continue
            start_time = time.time()
            try:
                logger.debug(
                    "Processing claims rebuild task media_id={mid}, queue_size={qsize}",
                    mid=task.media_id,
                    qsize=self._queue.qsize(),
                )
                self._process_task(task)
                with self._stats_lock:
                    self._stats["processed"] += 1
                self._last_processed_ts = time.time()
                record_claims_rebuild_metrics(
                    processed=1,
                    duration_s=time.time() - start_time,
                    queue_size=self._queue.qsize(),
                )
                self._persist_health()
            except _CLAIMS_REBUILD_NONCRITICAL_EXCEPTIONS as e:
                logger.error(f"Claims rebuild failed for media_id={task.media_id}: {e}")
                with self._stats_lock:
                    self._stats["failed"] += 1
                self._last_failure = {
                    "media_id": task.media_id,
                    "error": str(e),
                    "timestamp": time.time(),
                }
                record_claims_rebuild_metrics(
                    failed=1,
                    duration_s=time.time() - start_time,
                    queue_size=self._queue.qsize(),
                )
                self._persist_health(force=True)
            finally:
                self._queue.task_done()

    def _process_task(self, task: ClaimsRebuildTask) -> None:
        with managed_media_database(
            client_id=str(settings.get("SERVER_CLIENT_ID", "SERVER_API_V1")),
            db_path=task.db_path,
            initialize=False,
            suppress_close_exceptions=_CLAIMS_REBUILD_NONCRITICAL_EXCEPTIONS,
        ) as db:
            media = db.get_media_by_id(task.media_id, include_deleted=False, include_trash=False)
            if not media:
                logger.warning(f"Claims rebuild: media_id={task.media_id} not found")
                return
            content = media.get("content") or ""
            title = media.get("title") or f"media_{task.media_id}.txt"
            # Chunk content
            chunks = chunk_for_embedding(content, file_name=title)
            # Extract
            max_per = int(settings.get("CLAIMS_MAX_PER_CHUNK", 3))
            mode = str(settings.get("CLAIM_EXTRACTOR_MODE", "heuristic"))
            budget = resolve_claims_job_budget(settings=settings)
            claims = extract_claims_for_chunks(
                chunks,
                extractor_mode=mode,
                max_per_chunk=max_per,
                budget=budget,
            )
            if not claims:
                logger.info(f"Claims rebuild: no claims extracted for media_id={task.media_id}")
                return
            # Build map
            chunk_text_map: dict[int, str] = {}
            for ch in chunks:
                meta = (ch or {}).get("metadata", {}) or {}
                idx = int(meta.get("chunk_index") or meta.get("index") or 0)
                chunk_text_map[idx] = (ch or {}).get("text") or (ch or {}).get("content") or ""
            # Replace old claims atomically so failed writes keep the previous claim set active.
            with db.transaction():
                deleted = db.soft_delete_claims_for_media(task.media_id)
                inserted = store_claims(db, media_id=task.media_id, chunk_texts_by_index=chunk_text_map, claims=claims)
                if inserted <= 0:
                    raise RuntimeError(f"Claims rebuild stored zero replacement claims for media_id={task.media_id}")
            logger.info(f"Claims rebuild: media_id={task.media_id} deleted={deleted} inserted={inserted}")

    def get_stats(self) -> dict[str, int]:
        with self._stats_lock:
            return dict(self._stats)

    def get_queue_length(self) -> int:
        return self._queue.qsize()

    def get_worker_count(self) -> int:
        return len(self._threads)

    def get_health(self) -> dict[str, Any]:
        return {
            "queue_length": self.get_queue_length(),
            "workers": self.get_worker_count(),
            "last_heartbeat_ts": self._last_heartbeat_ts,
            "last_processed_ts": self._last_processed_ts or None,
            "last_failure": self._last_failure,
        }

    def _touch_heartbeat(self) -> None:
        self._last_heartbeat_ts = time.time()
        record_claims_rebuild_metrics(heartbeat_ts=self._last_heartbeat_ts)
        self._persist_health()


# Module-level singleton for convenience
_service_singleton: ClaimsRebuildService | None = None


def get_claims_rebuild_service() -> ClaimsRebuildService:
    global _service_singleton
    if _service_singleton is None:
        _service_singleton = ClaimsRebuildService(worker_threads=1)
        _service_singleton.start()
    return _service_singleton
