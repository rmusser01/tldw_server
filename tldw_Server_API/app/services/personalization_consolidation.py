"""
Personalization Consolidation Service

Periodic job to embed recent events, update topic profiles, and distill memories.
Stage 1 scaffold: topic scoring from event tags, no embedding integration yet.
"""
from __future__ import annotations

import asyncio
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone

from loguru import logger

from tldw_Server_API.app.core.config import settings
from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths
from tldw_Server_API.app.core.DB_Management.Personalization_DB import PersonalizationDB
from tldw_Server_API.app.core.Metrics import get_metrics_registry
from tldw_Server_API.app.core.Personalization.companion_derivations import derive_companion_knowledge_cards
from tldw_Server_API.app.core.Personalization.companion_user_ids import resolve_companion_storage_user_id

_PERSONALIZATION_CONSOLIDATION_NONCRITICAL_EXCEPTIONS = (
    asyncio.CancelledError,
    AttributeError,
    ConnectionError,
    KeyError,
    OSError,
    RuntimeError,
    TimeoutError,
    TypeError,
    ValueError,
)


def _resolve_user_storage_id(user_id: str) -> str:
    """Return the shared personalization storage id for logical user ids."""
    return resolve_companion_storage_user_id(user_id)


@dataclass
class ConsolidationConfig:
    interval_seconds: int = 1800  # default 30 minutes


class PersonalizationConsolidationService:
    def __init__(self, config: ConsolidationConfig | None = None):
        self.config = config or ConsolidationConfig()
        self._task: asyncio.Task | None = None
        self._shutdown = asyncio.Event()
        self._last_tick: dict[str, str] = {}

    async def start(self) -> asyncio.Task | None:
        if self._task and not self._task.done():
            return self._task
        self._task = asyncio.create_task(self._run_loop(), name="personalization_consolidation_loop")
        logger.info("Personalization consolidation service started (scaffold)")
        return self._task

    async def stop(self) -> None:
        self._shutdown.set()
        if self._task:
            try:
                await self._task
            except _PERSONALIZATION_CONSOLIDATION_NONCRITICAL_EXCEPTIONS as e:
                logger.debug(f"Personalization consolidation stop wait failed: {e}")
                try:
                    get_metrics_registry().increment(
                        "app_warning_events_total",
                        labels={"component": "personalization", "event": "stop_wait_failed"},
                    )
                except _PERSONALIZATION_CONSOLIDATION_NONCRITICAL_EXCEPTIONS:
                    logger.debug("metrics increment failed for personalization stop_wait_failed")
        logger.info("Personalization consolidation service stopped (scaffold)")

    async def trigger_consolidation(self, user_id: str | None = None) -> bool:
        """One-off consolidation for a user."""
        try:
            if not user_id:
                user_id = str(settings.get("SINGLE_USER_FIXED_ID", "1"))
            self._consolidate_user(user_id)
            return True
        except _PERSONALIZATION_CONSOLIDATION_NONCRITICAL_EXCEPTIONS as e:
            logger.debug(f"Consolidation trigger failed: {e}")
            try:
                get_metrics_registry().increment(
                    "app_warning_events_total",
                    labels={"component": "personalization", "event": "trigger_failed"},
                )
            except _PERSONALIZATION_CONSOLIDATION_NONCRITICAL_EXCEPTIONS:
                logger.debug("metrics increment failed for personalization trigger_failed")
            return False

    async def _run_loop(self) -> None:
        while not self._shutdown.is_set():
            try:
                logger.debug("Consolidation tick")
                user_ids = self._enumerate_user_ids()
                for uid in user_ids:
                    if self._shutdown.is_set():
                        break
                    self._consolidate_user(str(uid))
            except _PERSONALIZATION_CONSOLIDATION_NONCRITICAL_EXCEPTIONS as e:
                logger.warning(f"Consolidation loop error (scaffold): {e}")
                try:
                    get_metrics_registry().increment(
                        "app_exception_events_total",
                        labels={"component": "personalization", "event": "loop_error"},
                    )
                except _PERSONALIZATION_CONSOLIDATION_NONCRITICAL_EXCEPTIONS:
                    logger.debug("metrics increment failed for personalization loop_error")
            try:
                await asyncio.wait_for(self._shutdown.wait(), timeout=self.config.interval_seconds)
            except asyncio.TimeoutError:
                continue

    def _consolidate_user(self, user_id: str) -> None:
        """Consolidate per-user topics from recent events."""
        db = self._get_user_db(user_id)
        # Use public thread-safe method instead of bypassing the lock
        events = db.list_recent_events(user_id)
        scores = self._score_topics_from_events(events)
        for label, score in scores.items():
            try:
                db.upsert_topic(user_id, label, score)
            except _PERSONALIZATION_CONSOLIDATION_NONCRITICAL_EXCEPTIONS:
                try:
                    get_metrics_registry().increment(
                        "app_warning_events_total",
                        labels={"component": "personalization", "event": "upsert_topic_failed"},
                    )
                except _PERSONALIZATION_CONSOLIDATION_NONCRITICAL_EXCEPTIONS:
                    logger.debug("metrics increment failed for personalization upsert_topic_failed")

        try:
            cards = derive_companion_knowledge_cards(db, user_id=user_id)
            db.delete_companion_knowledge_cards(user_id=user_id)
            for card in cards:
                db.upsert_companion_knowledge_card(user_id=user_id, **card)
        except _PERSONALIZATION_CONSOLIDATION_NONCRITICAL_EXCEPTIONS:
            try:
                get_metrics_registry().increment(
                    "app_warning_events_total",
                    labels={"component": "personalization", "event": "upsert_companion_card_failed"},
                )
            except _PERSONALIZATION_CONSOLIDATION_NONCRITICAL_EXCEPTIONS:
                logger.debug("metrics increment failed for personalization upsert_companion_card_failed")

        self._last_tick[str(user_id)] = datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _enumerate_user_ids() -> list[int]:
        """Scan user_databases/ for per-user subdirectories.

        Falls back to ``DatabasePaths.get_single_user_id()`` when no
        directories are found (single-user mode).  Matches the canonical
        pattern used by ``outputs_purge_scheduler`` and other services.
        """
        try:
            base = DatabasePaths.get_user_db_base_dir()
        except _PERSONALIZATION_CONSOLIDATION_NONCRITICAL_EXCEPTIONS as exc:
            logger.debug(f"personalization: failed to resolve user db base dir: {exc}")
            try:
                get_metrics_registry().increment(
                    "app_warning_events_total",
                    labels={"component": "personalization", "event": "user_db_dir_read_failed"},
                )
            except _PERSONALIZATION_CONSOLIDATION_NONCRITICAL_EXCEPTIONS:
                logger.debug("metrics increment failed for personalization user_db_dir_read_failed")
            return []

        uids: list[int] = []
        for p in base.iterdir():
            if p.is_dir():
                try:
                    uids.append(int(p.name))
                except (TypeError, ValueError) as exc:
                    logger.debug(f"personalization: skipping non-int user dir {p.name}: {exc}")
                    try:
                        get_metrics_registry().increment(
                            "app_warning_events_total",
                            labels={"component": "personalization", "event": "invalid_user_dir_name"},
                        )
                    except _PERSONALIZATION_CONSOLIDATION_NONCRITICAL_EXCEPTIONS:
                        logger.debug("metrics increment failed for personalization invalid_user_dir_name")

        if not uids:
            try:
                uids = [DatabasePaths.get_single_user_id()]
            except _PERSONALIZATION_CONSOLIDATION_NONCRITICAL_EXCEPTIONS as exc:
                logger.debug(f"personalization: failed to derive single_user_id: {exc}")
                try:
                    get_metrics_registry().increment(
                        "app_warning_events_total",
                        labels={"component": "personalization", "event": "single_user_id_fallback_failed"},
                    )
                except _PERSONALIZATION_CONSOLIDATION_NONCRITICAL_EXCEPTIONS:
                    logger.debug("metrics increment failed for personalization single_user_id_fallback_failed")
                uids = []

        return sorted(set(uids))

    def get_status(self) -> dict:
        """Return service status including last consolidation ticks."""
        running = bool(self._task and not self._task.done())
        last = dict(self._last_tick)
        return {"running": running, "last_ticks": last, "user_count": len(self._last_tick)}

    @staticmethod
    def _get_user_db(user_id: str) -> PersonalizationDB:
        storage_user_id = _resolve_user_storage_id(user_id)
        return PersonalizationDB.for_user(storage_user_id)

    @staticmethod
    def _score_topics_from_events(events: list[dict]) -> dict[str, float]:
        tags: list[str] = []
        for e in events:
            tags.extend([t for t in (e.get("tags") or []) if isinstance(t, str)])
        c = Counter(tags)
        if not c:
            return {}
        maxc = max(c.values()) or 1
        return {k: v / maxc for k, v in c.items()}


_singleton: PersonalizationConsolidationService | None = None


def get_consolidation_service() -> PersonalizationConsolidationService:
    global _singleton
    if _singleton is None:
        _singleton = PersonalizationConsolidationService()
    return _singleton
