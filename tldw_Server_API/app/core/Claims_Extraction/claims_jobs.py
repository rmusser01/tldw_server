"""Claims enqueue helpers backed by the shared Jobs module."""

from __future__ import annotations

import hashlib
import os
from collections.abc import Mapping
from typing import Any

from tldw_Server_API.app.core.config import settings
from tldw_Server_API.app.core.Jobs.manager import JobManager
from tldw_Server_API.app.core.Jobs.worker_utils import coerce_int, jobs_manager_from_env

from .claims_job_contracts import (
    CLAIMS_DELIVER_ALERT_JOB_TYPE,
    CLAIMS_DELIVER_REVIEW_NOTIFICATION_JOB_TYPE,
    CLAIMS_GENERATE_ANALYTICS_EXPORT_JOB_TYPE,
    CLAIMS_JOB_PAYLOAD_VERSION,
    CLAIMS_JOBS_DEFAULT_QUEUE,
    CLAIMS_JOBS_DOMAIN,
    CLAIMS_REBUILD_MEDIA_JOB_TYPE,
    validate_alert_delivery_payload,
    validate_analytics_export_payload,
    validate_rebuild_media_payload,
    validate_review_notification_payload,
)


def _settings_map(settings_obj: Mapping[str, Any] | None = None) -> Mapping[str, Any]:
    """Return the injected settings mapping or the process-wide settings object."""
    return settings_obj if settings_obj is not None else settings


def _setting_value(
    key: str,
    default: Any = None,
    settings_obj: Mapping[str, Any] | None = None,
) -> Any:
    """Resolve a Claims Jobs setting with environment variables taking precedence."""
    if settings_obj is not None:
        return settings_obj.get(key, default)
    env_value = os.getenv(key)
    if env_value is not None:
        return env_value
    return _settings_map().get(key, default)


def _truthy(value: Any) -> bool:
    """Interpret common enabled values from environment and config sources."""
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def claims_jobs_enabled(settings_obj: Mapping[str, Any] | None = None) -> bool:
    """Return whether Claims background work should enqueue through Jobs."""
    return _truthy(_setting_value("CLAIMS_JOBS_ENABLED", False, settings_obj))


def claims_analytics_export_jobs_enabled(
    settings_obj: Mapping[str, Any] | None = None,
) -> bool:
    """Return whether Claims analytics exports should enqueue through Jobs."""
    return claims_jobs_enabled(settings_obj) and _truthy(
        _setting_value("CLAIMS_ANALYTICS_EXPORT_JOBS_ENABLED", False, settings_obj)
    )


def claims_jobs_worker_enabled(settings_obj: Mapping[str, Any] | None = None) -> bool:
    """Return whether the Claims Jobs worker should start during app lifecycle."""
    return _truthy(_setting_value("CLAIMS_JOBS_WORKER_ENABLED", False, settings_obj))


def claims_jobs_queue(settings_obj: Mapping[str, Any] | None = None) -> str:
    """Resolve the queue name used by Claims Jobs producers and workers."""
    queue = str(_setting_value("CLAIMS_JOBS_QUEUE", CLAIMS_JOBS_DEFAULT_QUEUE, settings_obj)).strip()
    return queue or CLAIMS_JOBS_DEFAULT_QUEUE


def _max_retries(
    key: str,
    default: int = 3,
    settings_obj: Mapping[str, Any] | None = None,
) -> int:
    """Resolve a non-negative max retry count for a Claims job type."""
    retries = coerce_int(_setting_value(key, None, settings_obj), default)
    return int(default) if retries < 0 else retries


def _manager(job_manager: JobManager | None = None) -> JobManager:
    """Return the provided Jobs manager or construct one from environment config."""
    return job_manager or jobs_manager_from_env()


def _refresh(manager: JobManager, created: dict[str, Any]) -> dict[str, Any]:
    """Reload a created job so callers receive the persisted Jobs row shape."""
    job_id = created.get("id")
    if job_id is None:
        return created
    return manager.get_job(int(job_id)) or created


def _hash_ids(values: list[int]) -> str:
    """Build a stable digest for a set of integer identifiers."""
    joined = ",".join(str(v) for v in sorted(set(values)))
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def enqueue_claims_rebuild_media(
    *,
    media_id: int,
    owner_user_id: str,
    idempotency_scope: str | None = None,
    job_manager: JobManager | None = None,
    settings_obj: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Create a Jobs row for rebuilding claims on one media item."""
    payload = validate_rebuild_media_payload(
        {
            "version": CLAIMS_JOB_PAYLOAD_VERSION,
            "owner_user_id": owner_user_id,
            "media_id": media_id,
        }
    )
    manager = _manager(job_manager)
    scope = str(idempotency_scope).strip() if idempotency_scope is not None else ""
    idempotency_key = f"claims:rebuild:{payload['owner_user_id']}:{payload['media_id']}:{scope}" if scope else None
    created = manager.create_job(
        domain=CLAIMS_JOBS_DOMAIN,
        queue=claims_jobs_queue(settings_obj),
        job_type=CLAIMS_REBUILD_MEDIA_JOB_TYPE,
        payload=payload,
        owner_user_id=payload["owner_user_id"],
        priority=5,
        max_retries=_max_retries("CLAIMS_JOBS_MAX_RETRIES_REBUILD", 3, settings_obj),
        idempotency_key=idempotency_key,
    )
    return _refresh(manager, created)


def enqueue_claims_analytics_export(
    *,
    owner_user_id: str,
    export_id: str,
    job_manager: JobManager | None = None,
    settings_obj: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Create one analytics export Job and return its acceptance result directly."""
    payload = validate_analytics_export_payload(
        {
            "version": CLAIMS_JOB_PAYLOAD_VERSION,
            "owner_user_id": owner_user_id,
            "export_id": export_id,
        }
    )
    manager = _manager(job_manager)
    return manager.create_job(
        domain=CLAIMS_JOBS_DOMAIN,
        queue=claims_jobs_queue(settings_obj),
        job_type=CLAIMS_GENERATE_ANALYTICS_EXPORT_JOB_TYPE,
        payload=payload,
        owner_user_id=payload["owner_user_id"],
        priority=5,
        max_retries=_max_retries(
            "CLAIMS_JOBS_MAX_RETRIES_ANALYTICS_EXPORT",
            3,
            settings_obj,
        ),
        batch_group=f"claims-analytics-export:{payload['export_id']}",
        idempotency_key=(
            f"claims:analytics_export:{payload['owner_user_id']}:"
            f"{payload['export_id']}"
        ),
    )


def enqueue_claims_review_notification(
    *,
    owner_user_id: str,
    notification_ids: list[int],
    job_manager: JobManager | None = None,
    settings_obj: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Create a Jobs row for delivering pending claim-review notifications."""
    payload = validate_review_notification_payload(
        {
            "version": CLAIMS_JOB_PAYLOAD_VERSION,
            "owner_user_id": owner_user_id,
            "notification_ids": notification_ids,
        }
    )
    manager = _manager(job_manager)
    created = manager.create_job(
        domain=CLAIMS_JOBS_DOMAIN,
        queue=claims_jobs_queue(settings_obj),
        job_type=CLAIMS_DELIVER_REVIEW_NOTIFICATION_JOB_TYPE,
        payload=payload,
        owner_user_id=payload["owner_user_id"],
        priority=5,
        max_retries=_max_retries(
            "CLAIMS_JOBS_MAX_RETRIES_REVIEW_NOTIFICATION",
            3,
            settings_obj,
        ),
        idempotency_key=(
            f"claims:notify_review:{payload['owner_user_id']}:" f"{_hash_ids(payload['notification_ids'])}"
        ),
    )
    return _refresh(manager, created)


def enqueue_claims_alert_delivery(
    *,
    owner_user_id: str,
    event_id: int,
    alert_id: int,
    channel: str,
    job_manager: JobManager | None = None,
    settings_obj: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Create a Jobs row for one alert delivery channel."""
    payload = validate_alert_delivery_payload(
        {
            "version": CLAIMS_JOB_PAYLOAD_VERSION,
            "owner_user_id": owner_user_id,
            "event_id": event_id,
            "alert_id": alert_id,
            "channel": channel,
        }
    )
    manager = _manager(job_manager)
    created = manager.create_job(
        domain=CLAIMS_JOBS_DOMAIN,
        queue=claims_jobs_queue(settings_obj),
        job_type=CLAIMS_DELIVER_ALERT_JOB_TYPE,
        payload=payload,
        owner_user_id=payload["owner_user_id"],
        priority=5,
        max_retries=_max_retries("CLAIMS_JOBS_MAX_RETRIES_ALERT", 3, settings_obj),
        idempotency_key=(
            f"claims:alert:{payload['owner_user_id']}:{payload['event_id']}:"
            f"{payload['alert_id']}:{payload['channel']}"
        ),
    )
    return _refresh(manager, created)


def claims_jobs_summary(
    *,
    job_manager: JobManager | None = None,
    owner_user_id: str | None = None,
) -> dict[str, Any]:
    """Return read-only Claims Jobs counts for dashboard analytics."""
    manager = _manager(job_manager)
    return {
        "domain": CLAIMS_JOBS_DOMAIN,
        "counts": manager.summarize_by_status(
            domain=CLAIMS_JOBS_DOMAIN,
            owner_user_id=str(owner_user_id) if owner_user_id else None,
        ),
    }
