"""
Prompt Studio Jobs worker (Phase 2):

- Consumes core Jobs entries for Prompt Studio jobs.
- Executes prompt studio job handlers via JobProcessor.
- Updates Jobs status/result via the core JobManager.

Job contract (domain/queue/job_type):
- domain = "prompt_studio"
- queue = os.getenv("PROMPT_STUDIO_JOBS_QUEUE", "default")
- job_type = "optimization" | "evaluation" | "generation"

Payload fields (examples):
- optimization_id / evaluation_id / project_id (generation)
- entity_id (generic id for job processor)
- prompt_id, test_case_ids, model_configs, optimization_config, optimizer_type
- request_id (optional)

Usage:
  python -m tldw_Server_API.app.core.Prompt_Management.prompt_studio.services.jobs_worker
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import threading
from collections import OrderedDict
from datetime import datetime
from typing import Any

from loguru import logger

from tldw_Server_API.app.core.AuthNZ.byok_runtime import ByokResolutionError
from tldw_Server_API.app.core.AuthNZ.llm_provider_overrides import (
    capture_provider_override_call_snapshot,
    refresh_llm_provider_overrides,
    shutdown_llm_provider_override_recovery,
    start_llm_provider_override_refresh_service,
)
from tldw_Server_API.app.core.AuthNZ.orgs_teams import (
    list_active_team_memberships_for_user,
    list_org_memberships_for_user,
)
from tldw_Server_API.app.core.AuthNZ.provider_credential_runtime import (
    ProviderCredentialRuntime,
    mark_provider_credential_used,
)
from tldw_Server_API.app.core.AuthNZ.repos.users_repo import AuthnzUsersRepo
from tldw_Server_API.app.core.AuthNZ.settings import get_settings as get_auth_settings
from tldw_Server_API.app.core.Chat.bounded_daemon import (
    DaemonCapacityError,
    await_owned_worker,
)
from tldw_Server_API.app.core.Chat.Chat_Deps import (
    ChatConfigurationError,
    ChatProviderError,
    SanitizedProviderStreamError,
)
from tldw_Server_API.app.core.DB_Management.DB_Manager import (
    create_prompt_studio_database,
    get_content_backend_instance,
)
from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths
from tldw_Server_API.app.core.DB_Management.PromptStudioDatabase import DatabaseError
from tldw_Server_API.app.core.exceptions import raise_detached_error
from tldw_Server_API.app.core.Jobs.manager import JobManager
from tldw_Server_API.app.core.Jobs.worker_sdk import WorkerConfig, WorkerSDK
from tldw_Server_API.app.core.LLM_Calls.adapter_utils import (
    provider_auth_is_resolved,
)
from tldw_Server_API.app.core.LLM_Calls.provider_metadata import (
    provider_requires_api_key,
)
from tldw_Server_API.app.core.Prompt_Management.optimization_model_config import (
    normalize_durable_optimization_config,
    runtime_model_config,
    strip_sensitive_durable_mapping,
    strip_sensitive_optimization_config,
)
from tldw_Server_API.app.core.Prompt_Management.prompt_studio.job_processor import JobProcessor
from tldw_Server_API.app.core.Prompt_Management.prompt_studio.quota_config import (
    apply_prompt_studio_quota_defaults,
    apply_prompt_studio_quota_policy,
)

_PROMPT_STUDIO_DOMAIN = "prompt_studio"

if os.getenv("PROMPT_STUDIO_JOBS_BACKEND") not in {"", "core"}:
    logger.warning("PROMPT_STUDIO_JOBS_BACKEND is not core; forcing core backend for prompt studio jobs worker")
    os.environ["PROMPT_STUDIO_JOBS_BACKEND"] = "core"


class PromptStudioJobError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        retryable: bool = False,
        backoff_seconds: int | None = None,
        failure_code: str = "prompt_studio_job_failed",
    ) -> None:
        super().__init__(message)
        self.retryable = retryable
        self.failure_code = failure_code
        if backoff_seconds is not None:
            self.backoff_seconds = backoff_seconds


_DB_CACHE: OrderedDict[str, Any] = OrderedDict()
_PROCESSOR_CACHE: OrderedDict[str, JobProcessor] = OrderedDict()
_ACTIVE_USER_COUNTS: dict[str, int] = {}
_ACTIVE_THREAD_USER_COUNTS: dict[tuple[int, str], int] = {}
_PENDING_CLOSE: dict[str, list[Any]] = {}
_CACHE_LOCK = threading.RLock()


def _jobs_manager() -> JobManager:
    apply_prompt_studio_quota_defaults()
    db_url = (os.getenv("JOBS_DB_URL") or "").strip()
    if not db_url:
        return JobManager()
    backend = "postgres" if db_url.startswith("postgres") else None
    return JobManager(backend=backend, db_url=db_url)


def _coerce_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


_MAX_CACHE_ENTRIES = max(1, _coerce_int(os.getenv("PROMPT_STUDIO_JOBS_CACHE_MAX_USERS"), 20))


def _normalize_user_id(value: Any, *, allow_default: bool = False) -> str:
    if value is None or str(value).strip() == "":
        if not allow_default:
            raise PromptStudioJobError("Missing owner_user_id for prompt studio job", retryable=False)
        return str(DatabasePaths.get_single_user_id())
    return str(value)


def _auth_mode() -> str:
    """Return the configured auth mode or fail closed with bounded metadata."""

    lookup_failure: PromptStudioJobError | None = None
    mode = ""
    try:
        mode = str(get_auth_settings().AUTH_MODE).strip().lower()
    except Exception as exc:  # noqa: BLE001 - settings detail may contain paths
        logger.bind(error_type=type(exc).__name__).warning(
            "Prompt Studio job auth-mode lookup failed"
        )
        lookup_failure = PromptStudioJobError(
            "Prompt Studio job owner state is temporarily unavailable",
            retryable=True,
            failure_code="credential_store_unavailable",
        )
    if lookup_failure is not None:
        raise lookup_failure
    return mode


async def _ensure_job_owner_active(user_id: str, *, auth_mode: str) -> None:
    """Revalidate a queued owner before any membership or credential lookup."""

    if auth_mode == "single_user":
        return
    owner_id = 0
    invalid_owner = False
    try:
        owner_id = int(user_id)
    except (TypeError, ValueError):
        invalid_owner = True
    if invalid_owner or owner_id <= 0:
        raise PromptStudioJobError(
            "Prompt Studio job owner is unavailable",
            retryable=False,
            failure_code="credential_scope_revoked",
        )

    lookup_failure: PromptStudioJobError | None = None
    user: Any = None
    try:
        users_repo = await AuthnzUsersRepo.from_pool()
        user = await users_repo.get_user_by_id(owner_id)
    except asyncio.CancelledError:
        raise
    except Exception as exc:  # noqa: BLE001 - owner store failures are retryable
        logger.bind(error_type=type(exc).__name__).warning(
            "Prompt Studio job owner revalidation failed"
        )
        lookup_failure = PromptStudioJobError(
            "Prompt Studio job owner state is temporarily unavailable",
            retryable=True,
            failure_code="credential_store_unavailable",
        )
    if lookup_failure is not None:
        raise lookup_failure
    if not isinstance(user, dict) or user.get("is_active") is not True:
        raise PromptStudioJobError(
            "Prompt Studio job owner is unavailable",
            retryable=False,
            failure_code="credential_scope_revoked",
        )


def _close_db(db: Any, *, user_id: str | None = None) -> None:
    for method_name in ("close_connection", "close"):
        method = getattr(db, method_name, None)
        if callable(method):
            try:
                method()
            except Exception as exc:
                logger.opt(exception=exc).warning(
                    "Failed to close Prompt Studio DB for user {} via {}",
                    user_id or "<unknown>",
                    method_name,
                )
            return


def _evict_cache_entries_if_needed() -> None:
    while len(_DB_CACHE) > _MAX_CACHE_ENTRIES:
        user_id, db = _DB_CACHE.popitem(last=False)
        _PROCESSOR_CACHE.pop(user_id, None)
        if _ACTIVE_USER_COUNTS.get(user_id, 0) > 0:
            _PENDING_CLOSE.setdefault(user_id, []).append(db)
            logger.debug("Deferred Prompt Studio DB close for active user {}", user_id)
            continue
        _close_db(db, user_id=user_id)


@contextlib.contextmanager
def _active_user_cache_scope(user_id: str):
    thread_key = (threading.get_ident(), user_id)
    with _CACHE_LOCK:
        _ACTIVE_USER_COUNTS[user_id] = _ACTIVE_USER_COUNTS.get(user_id, 0) + 1
        _ACTIVE_THREAD_USER_COUNTS[thread_key] = (
            _ACTIVE_THREAD_USER_COUNTS.get(thread_key, 0) + 1
        )
    try:
        yield
    finally:
        pending_close: list[Any] = []
        thread_databases: list[Any] = []
        with _CACHE_LOCK:
            thread_remaining = _ACTIVE_THREAD_USER_COUNTS.get(thread_key, 0) - 1
            if thread_remaining > 0:
                _ACTIVE_THREAD_USER_COUNTS[thread_key] = thread_remaining
            else:
                _ACTIVE_THREAD_USER_COUNTS.pop(thread_key, None)
                cached = _DB_CACHE.get(user_id)
                if cached is not None:
                    thread_databases.append(cached)
                thread_databases.extend(_PENDING_CLOSE.get(user_id, []))

            remaining = _ACTIVE_USER_COUNTS.get(user_id, 0) - 1
            if remaining > 0:
                _ACTIVE_USER_COUNTS[user_id] = remaining
            else:
                _ACTIVE_USER_COUNTS.pop(user_id, None)
                pending_close = _PENDING_CLOSE.pop(user_id, [])

        closed_ids: set[int] = set()
        for db in (*thread_databases, *pending_close):
            identity = id(db)
            if identity in closed_ids:
                continue
            closed_ids.add(identity)
            _close_db(db, user_id=user_id)


def _drain_tenant_db_cache() -> None:
    """Detach every cached tenant DB and close it once no scope is active."""

    to_close: list[tuple[str, Any]] = []
    with _CACHE_LOCK:
        cached = list(_DB_CACHE.items())
        _DB_CACHE.clear()
        _PROCESSOR_CACHE.clear()
        for user_id, db in cached:
            if _ACTIVE_USER_COUNTS.get(user_id, 0) > 0:
                pending = _PENDING_CLOSE.setdefault(user_id, [])
                if all(candidate is not db for candidate in pending):
                    pending.append(db)
            else:
                to_close.append((user_id, db))

        for user_id in list(_PENDING_CLOSE):
            if _ACTIVE_USER_COUNTS.get(user_id, 0) > 0:
                continue
            to_close.extend(
                (user_id, db) for db in _PENDING_CLOSE.pop(user_id, [])
            )

    closed_ids: set[int] = set()
    for user_id, db in to_close:
        identity = id(db)
        if identity in closed_ids:
            continue
        closed_ids.add(identity)
        _close_db(db, user_id=user_id)


def _normalize_payload(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _build_worker_config(*, worker_id: str, queue: str) -> WorkerConfig:
    lease_seconds = _coerce_int(os.getenv("PROMPT_STUDIO_JOBS_LEASE_SECONDS"), 60)
    renew_jitter_seconds = _coerce_int(os.getenv("PROMPT_STUDIO_JOBS_RENEW_JITTER_SECONDS"), 5)
    renew_threshold_seconds = _coerce_int(os.getenv("PROMPT_STUDIO_JOBS_RENEW_THRESHOLD_SECONDS"), 10)

    heartbeat_raw = (os.getenv("TLDW_PS_HEARTBEAT_SECONDS") or "").strip()
    if heartbeat_raw:
        heartbeat_seconds = _coerce_int(heartbeat_raw, 0)
        if heartbeat_seconds > 0:
            max_threshold = max(1, lease_seconds - 1) if lease_seconds > 1 else 1
            desired_threshold = max(1, lease_seconds - heartbeat_seconds)
            renew_threshold_seconds = min(max_threshold, desired_threshold)

    return WorkerConfig(
        domain=_PROMPT_STUDIO_DOMAIN,
        queue=queue,
        worker_id=worker_id,
        lease_seconds=lease_seconds,
        renew_jitter_seconds=renew_jitter_seconds,
        renew_threshold_seconds=renew_threshold_seconds,
        backoff_base_seconds=_coerce_int(os.getenv("PROMPT_STUDIO_JOBS_BACKOFF_BASE_SECONDS"), 2),
        backoff_max_seconds=_coerce_int(os.getenv("PROMPT_STUDIO_JOBS_BACKOFF_MAX_SECONDS"), 30),
        retry_on_exception=True,
        retry_backoff_seconds=_coerce_int(os.getenv("PROMPT_STUDIO_JOBS_RETRY_BACKOFF_SECONDS"), 10),
    )


def _create_db(user_id: str):
    """Create one uncached tenant Prompt Studio database facade."""

    backend = get_content_backend_instance()
    db_path = DatabasePaths.get_prompt_studio_db_path(user_id)
    client_id = f"prompt_studio_jobs_worker:{user_id}"
    db = create_prompt_studio_database(
        client_id=client_id,
        db_path=db_path,
        tenant_user_id=str(user_id),
        backend=backend,
    )
    db.user_id = str(user_id)
    return db


def _get_db(user_id: str):
    with _CACHE_LOCK:
        cached = _DB_CACHE.get(user_id)
        if cached is not None:
            _DB_CACHE.move_to_end(user_id)
            return cached
        db = _create_db(user_id)
        _DB_CACHE[user_id] = db
        _DB_CACHE.move_to_end(user_id)
        _evict_cache_entries_if_needed()
        return db


def _get_processor(user_id: str) -> JobProcessor:
    with _CACHE_LOCK:
        cached = _PROCESSOR_CACHE.get(user_id)
        if cached is not None:
            _PROCESSOR_CACHE.move_to_end(user_id)
            if user_id in _DB_CACHE:
                _DB_CACHE.move_to_end(user_id, last=True)
            return cached
        db = _get_db(user_id)
        processor = JobProcessor(db)
        _PROCESSOR_CACHE[user_id] = processor
        _PROCESSOR_CACHE.move_to_end(user_id)
        _evict_cache_entries_if_needed()
        return processor


def _create_reconciliation_processor(user_id: str) -> JobProcessor:
    """Create a tenant processor isolated from the worker's shared LRU cache."""

    db = _create_db(user_id)
    try:
        return JobProcessor(db)
    except BaseException:
        _close_db(db, user_id=user_id)
        raise


def _resolve_entity_id(job_type: str, payload: dict[str, Any]) -> int:
    if job_type == "optimization":
        value = payload.get("optimization_id") or payload.get("entity_id")
    elif job_type == "evaluation":
        value = payload.get("evaluation_id") or payload.get("entity_id")
    elif job_type == "generation":
        value = payload.get("project_id") or payload.get("entity_id")
    else:
        value = payload.get("entity_id")
    if value is None:
        raise PromptStudioJobError(f"Missing entity id for {job_type} job", retryable=False)
    return _coerce_int(value, 0)


def _replace_live_job_payload(
    job: dict[str, Any],
    payload: dict[str, Any],
    *,
    job_manager: JobManager | None = None,
) -> None:
    """Persist a scrubbed payload for an exact live or archived Jobs row."""
    if "domain" not in job:
        return
    if str(job.get("domain") or "").strip() != _PROMPT_STUDIO_DOMAIN:
        raise PromptStudioJobError(
            "Prompt Studio job identity is unavailable",
            retryable=False,
            failure_code="job_identity_invalid",
        )
    archived = job.get("archived") is True
    job_uuid = str(job.get("uuid") or "").strip()
    archive_locator = job.get("_archive_locator")
    if (not archived and not job_uuid) or (
        archived and not job_uuid and archive_locator is None
    ):
        raise PromptStudioJobError(
            "Prompt Studio job identity is unavailable",
            retryable=False,
            failure_code="job_identity_invalid",
        )
    try:
        job_id = int(job["id"])
        manager = job_manager or _jobs_manager()
        replace_payload = (
            manager.replace_archived_job_payload
            if archived
            else manager.replace_job_payload
        )
        replace_kwargs: dict[str, Any] = {
            "payload": payload,
            "expected_uuid": job_uuid or None,
            "expected_domain": _PROMPT_STUDIO_DOMAIN,
        }
        if archived:
            replace_kwargs["expected_archive_locator"] = archive_locator
        replaced = replace_payload(job_id, **replace_kwargs)
    except PromptStudioJobError:
        raise
    except Exception:  # noqa: BLE001 - map Jobs storage details to one code
        raise PromptStudioJobError(
            "Prompt Studio job payload could not be secured",
            retryable=True,
            failure_code="job_store_unavailable",
        ) from None
    if not replaced:
        raise PromptStudioJobError(
            "Prompt Studio job payload could not be secured",
            retryable=False,
            failure_code="job_identity_invalid",
        )


def _scrub_optimization_job_payload_snapshot(
    *,
    job: dict[str, Any],
    payload: dict[str, Any],
    job_manager: JobManager | None,
) -> dict[str, Any]:
    """Persist secret removal before any identity or tenant-store lookup."""

    secured_payload = strip_sensitive_durable_mapping(payload)
    if secured_payload != payload or job.get(
        "_archive_payload_rewrite_required"
    ) is True:
        _replace_live_job_payload(
            job,
            secured_payload,
            job_manager=job_manager,
        )
    if "domain" in job:
        job["payload"] = secured_payload
    return secured_payload


def _secure_optimization_job_payload(
    *,
    job: dict[str, Any],
    payload: dict[str, Any],
    job_manager: JobManager | None,
    require_valid_config: bool,
    fallback_config: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Scrub one Jobs payload without consulting tenant Prompt state."""

    payload = _scrub_optimization_job_payload_snapshot(
        job=job,
        payload=payload,
        job_manager=job_manager,
    )
    raw_config = payload.get("optimization_config")
    if not isinstance(raw_config, dict):
        raw_config = _normalize_payload(fallback_config)

    validation_error: ValueError | None = None
    try:
        durable_config = normalize_durable_optimization_config(
            raw_config,
            reject_sensitive=False,
        )
    except ValueError as exc:
        durable_config = strip_sensitive_optimization_config(raw_config)
        validation_error = exc

    secured_payload = dict(payload)
    secured_payload["optimization_config"] = durable_config
    secured_payload = strip_sensitive_durable_mapping(secured_payload)
    if secured_payload != payload or (
        job_manager is None and "domain" in job
    ):
        _replace_live_job_payload(
            job,
            secured_payload,
            job_manager=job_manager,
        )
    if "domain" in job:
        job["payload"] = secured_payload
    if require_valid_config and validation_error is not None:
        raise validation_error
    return secured_payload, durable_config


def _secure_optimization_durable_state(
    *,
    processor: JobProcessor,
    optimization_id: int,
    job: dict[str, Any],
    payload: dict[str, Any],
    job_manager: JobManager | None,
    require_valid_config: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Scrub independent Jobs and Prompt snapshots before terminal return."""

    expected_uuid = str(payload.get("optimization_uuid") or "").strip() or None
    optimization_row = processor.db.get_optimization(
        optimization_id,
        include_deleted=True,
    )
    if expected_uuid is not None and (
        optimization_row is None
        or str(optimization_row.get("uuid") or "") != expected_uuid
    ):
        raise PromptStudioJobError(
            "Prompt Studio optimization identity changed",
            retryable=False,
            failure_code="job_identity_invalid",
        )
    stored_config = _normalize_payload(
        (optimization_row or {}).get("optimization_config")
    )
    secured_payload, durable_config = _secure_optimization_job_payload(
        job=job,
        payload=payload,
        job_manager=job_manager,
        require_valid_config=require_valid_config,
        fallback_config=stored_config,
    )
    if optimization_row is not None:
        try:
            secured_stored_config = normalize_durable_optimization_config(
                stored_config,
                reject_sensitive=False,
            )
        except ValueError:
            secured_stored_config = strip_sensitive_optimization_config(
                stored_config
            )
        if stored_config != secured_stored_config:
            updated = processor.db.update_optimization(
                optimization_id,
                {"optimization_config": secured_stored_config},
                expected_uuid=expected_uuid,
                _return_transition_applied=True,
            )
            if (
                expected_uuid is not None
                and isinstance(updated, tuple)
                and len(updated) == 2
                and not bool(updated[1])
            ):
                raise PromptStudioJobError(
                    "Prompt Studio optimization identity changed",
                    retryable=False,
                    failure_code="job_identity_invalid",
                )
    return secured_payload, durable_config


def _cancelled_job_error() -> PromptStudioJobError:
    """Return the bounded control-flow error used for terminal Jobs cancellation."""

    return PromptStudioJobError(
        "Optimization was cancelled",
        retryable=False,
        failure_code="job_cancelled",
    )


async def _owner_membership_scope(user_id: int) -> tuple[list[int], list[int]]:
    team_rows, org_rows = await asyncio.gather(
        list_active_team_memberships_for_user(user_id),
        list_org_memberships_for_user(user_id),
    )
    team_ids = sorted(
        {
            int(row["team_id"])
            for row in team_rows
            if isinstance(row, dict) and row.get("team_id") is not None
        }
    )
    org_ids = {
        int(row["org_id"])
        for row in org_rows
        if isinstance(row, dict)
        and row.get("org_id") is not None
        and isinstance(row.get("status"), str)
        and str(row["status"]).strip().lower() == "active"
    }
    return team_ids, sorted(org_ids)


def _bounded_optimization_error(exc: BaseException) -> PromptStudioJobError:
    if isinstance(exc, PromptStudioJobError):
        return exc
    if isinstance(exc, ByokResolutionError):
        code = str(getattr(exc, "policy_code", None) or exc.code)
        return PromptStudioJobError(
            "Provider credentials are temporarily unavailable"
            if code == "credential_store_unavailable"
            else "Provider credentials are unavailable",
            retryable=code == "credential_store_unavailable",
            failure_code=code,
        )
    if isinstance(exc, SanitizedProviderStreamError):
        code = str(exc.code)
        return PromptStudioJobError(
            "Optimization provider execution failed",
            retryable=code == "provider_unavailable",
            failure_code=code,
        )
    if isinstance(exc, DatabaseError):
        return PromptStudioJobError(
            "Prompt Studio job store temporarily unavailable",
            retryable=True,
            failure_code="job_store_unavailable",
        )
    if isinstance(exc, (ChatConfigurationError, ValueError)):
        return PromptStudioJobError(
            "Optimization provider configuration is invalid",
            retryable=False,
            failure_code="provider_configuration_invalid",
        )
    if isinstance(exc, (DaemonCapacityError, TimeoutError, ChatProviderError, RuntimeError)):
        return PromptStudioJobError(
            "Optimization provider is temporarily unavailable",
            retryable=True,
            failure_code="provider_unavailable",
        )
    return PromptStudioJobError(
        "Optimization job failed",
        retryable=False,
    )


def _mark_failed_safely(
    processor: JobProcessor,
    optimization_id: int,
    error: PromptStudioJobError,
    *,
    expected_uuid: str | None,
) -> None:
    try:
        updated = processor.db.update_optimization(
            optimization_id,
            {
                "status": "failed",
                "error_message": str(error),
            },
            expected_statuses=("pending", "running"),
            expected_uuid=expected_uuid,
            set_completed_at=True,
            _return_transition_applied=True,
        )
        if isinstance(updated, tuple) and len(updated) == 2 and not bool(updated[1]):
            logger.warning(
                "Skipped Prompt Studio failure repair after row identity changed"
            )
    except Exception as exc:  # noqa: BLE001 - preserve the bounded primary failure
        logger.warning(
            "Failed to mark Prompt Studio optimization failed; error_type={}",
            type(exc).__name__,
        )


def _retry_attempt_remains(
    job: dict[str, Any],
    error: PromptStudioJobError,
) -> bool:
    """Predict whether Jobs will retry instead of failing or quarantining."""

    if not error.retryable:
        return False
    try:
        retry_count = int(job.get("retry_count") or 0)
        max_retries = int(job.get("max_retries") or 0)
        failure_streak_count = int(job.get("failure_streak_count") or 0)
        quarantine_threshold = int(
            os.getenv("JOBS_QUARANTINE_THRESHOLD", "2") or "2"
        )
    except (TypeError, ValueError):
        return False
    if (
        retry_count < 0
        or max_retries < 0
        or failure_streak_count < 0
        or quarantine_threshold <= 0
        or retry_count >= max_retries
    ):
        return False

    previous_code = str(job.get("failure_streak_code") or "")
    next_streak = (
        failure_streak_count + 1
        if previous_code == error.failure_code
        else 1
    )
    return next_streak < quarantine_threshold


def _mark_retry_pending_safely(
    processor: JobProcessor,
    optimization_id: int,
    error: PromptStudioJobError,
    *,
    expected_uuid: str | None,
) -> None:
    """Keep an optimization nonterminal while its Jobs retry is pending."""

    try:
        updated = processor.db.update_optimization(
            optimization_id,
            {
                "status": "pending",
                "error_message": str(error),
                "started_at": None,
                "completed_at": None,
            },
            expected_statuses=("pending", "running"),
            expected_uuid=expected_uuid,
            _return_transition_applied=True,
        )
        if isinstance(updated, tuple) and len(updated) == 2 and not bool(updated[1]):
            logger.warning(
                "Skipped Prompt Studio retry repair after row identity changed"
            )
    except Exception as exc:  # noqa: BLE001 - preserve the bounded primary failure
        logger.warning(
            "Failed to mark Prompt Studio optimization retry pending; error_type={}",
            type(exc).__name__,
        )


def _read_exact_job_state(
    job: dict[str, Any],
    job_manager: JobManager,
) -> dict[str, Any]:
    """Read the exact Jobs row and verify its immutable identity."""

    try:
        job_id = int(job["id"])
        expected_uuid = str(job["uuid"])
    except (KeyError, TypeError, ValueError):
        raise PromptStudioJobError(
            "Prompt Studio job identity is unavailable",
            retryable=False,
            failure_code="job_identity_invalid",
        ) from None

    try:
        live_job = job_manager.get_job(job_id)
    except Exception:  # noqa: BLE001 - hide Jobs backend details
        raise PromptStudioJobError(
            "Prompt Studio job state is temporarily unavailable",
            retryable=True,
            failure_code="job_store_unavailable",
        ) from None
    if not isinstance(live_job, dict):
        raise PromptStudioJobError(
            "Prompt Studio job state is temporarily unavailable",
            retryable=True,
            failure_code="job_store_unavailable",
        )
    if (
        str(live_job.get("uuid") or "") != expected_uuid
        or str(live_job.get("domain") or "") != _PROMPT_STUDIO_DOMAIN
        or str(live_job.get("job_type") or "")
        != str(job.get("job_type") or "")
    ):
        raise PromptStudioJobError(
            "Prompt Studio job identity changed during execution",
            retryable=False,
            failure_code="job_identity_invalid",
        )
    return live_job


def _read_live_job_state(job: dict[str, Any], job_manager: JobManager) -> dict[str, Any]:
    """Read the exact leased Jobs row or fail closed before Prompt completion."""

    live_job = _read_exact_job_state(job, job_manager)

    live_status = str(live_job.get("status") or "").lower()
    cancellation_requested = (
        live_status == "cancelled"
        or live_job.get("cancel_requested_at") is not None
    )
    if not cancellation_requested:
        expected_lease = str(job.get("lease_id") or "")
        live_lease = str(live_job.get("lease_id") or "")
        if live_status != "processing" or (
            expected_lease and live_lease != expected_lease
        ):
            raise PromptStudioJobError(
                "Prompt Studio job lease is no longer authoritative",
                retryable=True,
                failure_code="job_state_unavailable",
            )
    return live_job


def _required_optimization_uuid(job: dict[str, Any]) -> str:
    """Return durable Prompt row identity or fail closed."""

    payload = _normalize_payload(job.get("payload"))
    expected_uuid = str(payload.get("optimization_uuid") or "").strip()
    if not expected_uuid:
        raise PromptStudioJobError(
            "Prompt Studio optimization identity is unavailable",
            retryable=False,
            failure_code="job_identity_invalid",
        )
    return expected_uuid


def _sync_live_job_cancellation(
    *,
    processor: JobProcessor,
    optimization_id: int,
    job: dict[str, Any],
    job_manager: JobManager,
) -> bool:
    """Mirror a terminal Jobs cancellation into the Prompt optimization row."""

    expected_uuid = _required_optimization_uuid(job)
    current = processor.db.get_optimization(
        optimization_id,
        include_deleted=True,
    ) or {}
    if not _optimization_identity_matches(current, job, require_identity=True):
        raise PromptStudioJobError(
            "Prompt Studio optimization identity changed",
            retryable=False,
            failure_code="job_identity_invalid",
        )
    live_job = _read_live_job_state(job, job_manager)
    live_status = str(live_job.get("status") or "").lower()
    if live_status != "cancelled" and live_job.get("cancel_requested_at") is None:
        return False
    reason = str(live_job.get("cancellation_reason") or "Jobs runtime cancellation")
    updated = processor.db.update_optimization(
        optimization_id,
        {
            "status": "cancelled",
            "error_message": reason,
        },
        expected_statuses=(
            "pending",
            "queued",
            "running",
            "completed",
            "failed",
            "cancelled",
        ),
        set_completed_at=True,
        expected_uuid=expected_uuid,
        _return_transition_applied=True,
    )
    if isinstance(updated, tuple) and len(updated) == 2 and not bool(updated[1]):
        raise PromptStudioJobError(
            "Prompt Studio optimization identity changed",
            retryable=False,
            failure_code="job_identity_invalid",
        )
    return True


def _cancel_live_job_for_prompt_outcome(
    *,
    processor: JobProcessor,
    optimization_id: int,
    job: dict[str, Any],
    job_manager: JobManager,
) -> None:
    """Cancel and verify the exact core Jobs row for a cancelled Prompt result."""

    live_job = _read_live_job_state(job, job_manager)
    live_status = str(live_job.get("status") or "").lower()
    if live_status != "cancelled" and live_job.get("cancel_requested_at") is None:
        try:
            cancelled = job_manager.cancel_job(
                int(live_job["id"]),
                reason="Prompt Studio optimization cancelled",
                expected_uuid=str(live_job["uuid"]),
                expected_domain=_PROMPT_STUDIO_DOMAIN,
                expected_job_type=str(live_job["job_type"]),
            )
        except Exception:  # noqa: BLE001 - hide Jobs backend details
            raise PromptStudioJobError(
                "Prompt Studio job state is temporarily unavailable",
                retryable=True,
                failure_code="job_store_unavailable",
            ) from None
        if not cancelled:
            live_job = _read_live_job_state(job, job_manager)
            live_status = str(live_job.get("status") or "").lower()
            if (
                live_status != "cancelled"
                and live_job.get("cancel_requested_at") is None
            ):
                raise PromptStudioJobError(
                    "Prompt Studio job cancellation could not be verified",
                    retryable=True,
                    failure_code="job_state_unavailable",
                )
    if not _sync_live_job_cancellation(
        processor=processor,
        optimization_id=optimization_id,
        job=job,
        job_manager=job_manager,
    ):
        raise PromptStudioJobError(
            "Prompt Studio job cancellation could not be verified",
            retryable=True,
            failure_code="job_state_unavailable",
        )


def _created_at_cursor(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        return datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError:
        return None


def _archive_cursor_from_job(
    job: dict[str, Any],
) -> tuple[datetime, int, str, str | int] | None:
    """Extract one complete immutable archive keyset position."""

    created_at = _created_at_cursor(job.get("_archive_cursor_created_at"))
    try:
        job_id = int(job["id"])
    except (KeyError, TypeError, ValueError):
        return None
    archive_locator = job.get("_archive_locator")
    if created_at is None or archive_locator is None:
        return None
    return (
        created_at,
        job_id,
        str(job.get("_archive_cursor_uuid") or ""),
        archive_locator,
    )


def _cancelled_job_should_override_prompt(
    _cancelled_job: dict[str, Any],
    optimization: dict[str, Any],
) -> bool:
    """Keep the core Jobs cancellation authoritative across terminal races."""

    status = str(optimization.get("status") or "").lower()
    return status in {"pending", "queued", "running", "completed", "failed"}


def _optimization_identity_matches(
    optimization: dict[str, Any],
    job: dict[str, Any],
    *,
    require_identity: bool,
) -> bool:
    """Verify immutable row identity before replaying a terminal snapshot."""

    payload = _normalize_payload(job.get("payload"))
    expected_uuid = str(payload.get("optimization_uuid") or "").strip()
    current_uuid = str(optimization.get("uuid") or "").strip()
    if expected_uuid:
        matches = bool(current_uuid) and current_uuid == expected_uuid
        if not matches:
            logger.warning(
                "Prompt Studio terminal reconciliation rejected stale optimization identity"
            )
        return matches
    if require_identity:
        logger.warning(
            "Prompt Studio terminal reconciliation skipped row without optimization identity"
        )
        return False
    return True


def _terminal_prompt_target(job: dict[str, Any]) -> str | None:
    """Map terminal core Jobs states to their Prompt Studio terminal state."""

    status = str(job.get("status") or "").strip().lower()
    if status == "cancelled":
        return "cancelled"
    if status in {"failed", "quarantined"}:
        return "failed"
    return None


def _converge_terminal_prompt_state(
    *,
    processor: JobProcessor,
    optimization_id: int,
    job: dict[str, Any],
    job_manager: JobManager,
    archived: bool,
    require_identity: bool = False,
) -> bool:
    """Converge one immutable/live Jobs terminal snapshot into its tenant row."""

    target = _terminal_prompt_target(job)
    if target is None:
        return False
    current = processor.db.get_optimization(
        optimization_id,
        include_deleted=True,
    ) or {}
    if not _optimization_identity_matches(
        current,
        job,
        require_identity=require_identity,
    ):
        return False
    current_status = str(current.get("status") or "").strip().lower()

    if target == "cancelled":
        if not _cancelled_job_should_override_prompt(job, current):
            return False
        if not archived:
            return _sync_live_job_cancellation(
                processor=processor,
                optimization_id=optimization_id,
                job=job,
                job_manager=job_manager,
            )
        expected_statuses = (
            "pending",
            "queued",
            "running",
            "completed",
            "failed",
        )
        error_message = str(
            job.get("cancellation_reason") or "Jobs runtime cancellation"
        )
    else:
        if current_status not in {"pending", "queued", "running"}:
            return False
        if not archived:
            live_job = _read_exact_job_state(job, job_manager)
            if _terminal_prompt_target(live_job) != "failed":
                return False
        expected_statuses = ("pending", "queued", "running")
        error_message = (
            "Jobs runtime quarantined repeated failures"
            if str(job.get("status") or "").strip().lower() == "quarantined"
            else "Jobs runtime exhausted all retry attempts"
        )

    updated = processor.db.update_optimization(
        optimization_id,
        {
            "status": target,
            "error_message": error_message,
        },
        expected_statuses=expected_statuses,
        expected_uuid=(
            str(_normalize_payload(job.get("payload")).get("optimization_uuid") or "").strip()
            or None
        ),
        set_completed_at=True,
        _return_transition_applied=True,
    )
    if isinstance(updated, tuple) and len(updated) == 2:
        return bool(updated[1])
    latest = processor.db.get_optimization(
        optimization_id,
        include_deleted=True,
    ) or {}
    return (
        current_status != target
        and str(latest.get("status") or "").strip().lower() == target
    )


class _CancellationReconciliationState:
    """Persistent bounded state for recurring terminal reconciliation."""

    def __init__(self, *, stop_event: threading.Event | None = None) -> None:
        self.processors: OrderedDict[str, JobProcessor] = OrderedDict()
        self.archive_cursor: tuple[datetime, int, str, str | int] | None = None
        self.stop_event = stop_event or threading.Event()


async def _run_reconciliation_thread(
    func: Any,
    /,
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Own a thread call through cancellation so shutdown can join it."""

    return await await_owned_worker(asyncio.to_thread(func, *args, **kwargs))


async def _reconcile_cancelled_optimization_jobs(
    job_manager: JobManager,
    *,
    include_archived: bool = False,
    state: _CancellationReconciliationState | None = None,
) -> int:
    """Reconcile terminal Jobs rows into tenant Prompt stores."""

    page_size = min(
        1000,
        max(
            1,
            _coerce_int(
                os.getenv("PROMPT_STUDIO_CANCEL_RECONCILE_PAGE_SIZE"),
                200,
            ),
        ),
    )
    archive_page_budget = min(
        100,
        max(
            1,
            _coerce_int(
                os.getenv("PROMPT_STUDIO_CANCEL_RECONCILE_ARCHIVE_PAGES"),
                1,
            ),
        ),
    )
    reconciled = 0
    reconciliation_processors = (
        state.processors
        if state is not None
        else OrderedDict()
    )

    def _stopping() -> bool:
        return state is not None and state.stop_event.is_set()

    def _processor_for_reconciliation(user_id: str) -> JobProcessor:
        """Return a bounded processor without mutating the worker-thread LRU."""

        cached = reconciliation_processors.get(user_id)
        if cached is not None:
            reconciliation_processors.move_to_end(user_id)
            return cached
        processor = _create_reconciliation_processor(user_id)
        reconciliation_processors[user_id] = processor
        reconciliation_processors.move_to_end(user_id)
        while len(reconciliation_processors) > _MAX_CACHE_ENTRIES:
            evicted_user_id, evicted_processor = (
                reconciliation_processors.popitem(last=False)
            )
            # Every row closes its executor-thread handle in ``finally``;
            # eviction is a defensive no-op for well-behaved facades.
            _close_db(evicted_processor.db, user_id=evicted_user_id)
        return processor

    # Live rows run first so an archive outage cannot suppress live repair.
    sources: list[tuple[bool, Any, str | None]] = [
        (False, job_manager.list_jobs, status)
        for status in ("cancelled", "failed", "quarantined")
    ]
    if include_archived:
        sources.append((True, job_manager.list_archived_jobs, None))

    for archived, list_jobs, status in sources:
        if _stopping():
            break
        created_before: datetime | None = None
        before_id: int | None = None
        before_uuid: str | None = None
        before_archive_locator: str | int | None = None
        if archived and state is not None and state.archive_cursor is not None:
            (
                created_before,
                before_id,
                before_uuid,
                before_archive_locator,
            ) = state.archive_cursor
        archive_pages = 0

        while not _stopping():
            list_kwargs: dict[str, Any] = {
                "domain": _PROMPT_STUDIO_DOMAIN,
                "queue": None,
                "status": status,
                "job_type": "optimization",
                "created_before": created_before,
                "before_id": before_id,
                "limit": page_size,
            }
            if archived:
                list_kwargs.update(
                    before_uuid=before_uuid,
                    before_archive_locator=before_archive_locator,
                    fail_on_decryption_error=True,
                )
            else:
                list_kwargs.update(
                    sort_by="created_at",
                    sort_order="desc",
                )
            jobs = await _run_reconciliation_thread(list_jobs, **list_kwargs)
            if _stopping():
                break

            for terminal_job in jobs:
                if _stopping():
                    break

                def _process_row(
                    selected_job: dict[str, Any] = terminal_job,
                    is_archived: bool = archived,
                ) -> bool:
                    rewrite_required = selected_job.get(
                        "_archive_payload_rewrite_required"
                    ) is True
                    original_payload = _normalize_payload(
                        selected_job.get("payload")
                    )
                    payload = _scrub_optimization_job_payload_snapshot(
                        job=selected_job,
                        payload=original_payload,
                        job_manager=job_manager,
                    )
                    payload_changed = payload != original_payload or rewrite_required
                    target = _terminal_prompt_target(selected_job)
                    if target is None:
                        return payload_changed

                    owner_user_id = selected_job.get("owner_user_id")
                    if is_archived and (
                        owner_user_id is None
                        or not str(owner_user_id).strip()
                    ):
                        return payload_changed
                    optimization_id = _resolve_entity_id(
                        "optimization",
                        payload,
                    )
                    auth_mode = _auth_mode()
                    user_id = _normalize_user_id(
                        owner_user_id,
                        allow_default=auth_mode == "single_user",
                    )
                    processor = _processor_for_reconciliation(user_id)
                    require_identity = True
                    try:
                        current = processor.db.get_optimization(
                            optimization_id,
                            include_deleted=True,
                        ) or {}
                        if not _optimization_identity_matches(
                            current,
                            selected_job,
                            require_identity=require_identity,
                        ):
                            return payload_changed
                        payload, _durable_config = (
                            _secure_optimization_durable_state(
                                processor=processor,
                                optimization_id=optimization_id,
                                job=selected_job,
                                payload=payload,
                                job_manager=job_manager,
                                require_valid_config=False,
                            )
                        )
                        payload_changed = (
                            payload_changed or payload != original_payload
                        )
                        transitioned = _converge_terminal_prompt_state(
                            processor=processor,
                            optimization_id=optimization_id,
                            job=selected_job,
                            job_manager=job_manager,
                            archived=is_archived,
                            require_identity=require_identity,
                        )
                    finally:
                        # This function runs in an executor thread. Release
                        # that thread's cached SQLite/PostgreSQL connection
                        # before the owning await can finish or be cancelled.
                        _close_db(processor.db, user_id=user_id)
                    return payload_changed or transitioned

                try:
                    if await _run_reconciliation_thread(_process_row):
                        reconciled += 1
                except asyncio.CancelledError:
                    raise
                except Exception as exc:  # noqa: BLE001 - isolate malformed rows
                    if archived and (
                        isinstance(exc, DatabaseError)
                        or (
                            isinstance(exc, PromptStudioJobError)
                            and exc.retryable
                        )
                    ):
                        if state is not None:
                            failed_cursor = _archive_cursor_from_job(terminal_job)
                            if failed_cursor is not None:
                                # The cursor is a scan position, not an ack. Skip
                                # this row for the current sweep, then revisit it
                                # after the existing cyclic EOF reset.
                                state.archive_cursor = failed_cursor
                            else:
                                logger.warning(
                                    "Prompt Studio archive reconciliation could not "
                                    "advance past a retryable row"
                                )
                        raise
                    logger.bind(error_type=type(exc).__name__).warning(
                        "Prompt Studio cancelled-job reconciliation skipped one row"
                    )

            if _stopping():
                break
            archive_pages += int(archived)
            if len(jobs) < page_size:
                if archived and state is not None:
                    # A cyclic reset is race-safe for late PostgreSQL commits and
                    # rows inserted ahead of the cursor during a prior page.
                    state.archive_cursor = None
                break
            last_job = jobs[-1]
            if archived:
                archive_cursor = _archive_cursor_from_job(last_job)
                if archive_cursor is not None:
                    (
                        created_before,
                        before_id,
                        before_uuid,
                        before_archive_locator,
                    ) = archive_cursor
            else:
                created_before = _created_at_cursor(last_job.get("created_at"))
                try:
                    before_id = int(last_job["id"])
                except (KeyError, TypeError, ValueError):
                    before_id = None
            if (
                created_before is None
                or before_id is None
                or (archived and archive_cursor is None)
            ):
                logger.warning(
                    "Prompt Studio cancellation reconciliation pagination stopped early"
                )
                if archived and state is not None:
                    state.archive_cursor = None
                break
            if archived and state is not None:
                state.archive_cursor = (
                    created_before,
                    before_id,
                    before_uuid or "",
                    before_archive_locator,
                )
                if archive_pages >= archive_page_budget:
                    break
            await asyncio.sleep(0)

    return reconciled


async def _cancelled_job_reconciliation_loop(
    job_manager: JobManager,
    *,
    stop_event: threading.Event | None = None,
) -> None:
    """Continuously repair cross-store cancellation state for queued jobs."""

    interval_seconds = min(
        60,
        max(
            1,
            _coerce_int(
                os.getenv("PROMPT_STUDIO_CANCEL_RECONCILE_SECONDS"),
                30,
            ),
        ),
    )
    state = _CancellationReconciliationState(stop_event=stop_event)
    while not state.stop_event.is_set():
        try:
            reconciled = await _reconcile_cancelled_optimization_jobs(
                job_manager,
                include_archived=True,
                state=state,
            )
            if reconciled:
                logger.info(
                    "Reconciled {} terminal Prompt Studio optimization job(s)",
                    reconciled,
                )
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - keep worker alive across store outages
            logger.bind(error_type=type(exc).__name__).warning(
                "Prompt Studio cancellation reconciliation failed"
            )
        if not state.stop_event.is_set():
            await asyncio.sleep(interval_seconds)


async def _handle_job(
    job: dict[str, Any],
    *,
    job_manager: JobManager | None = None,
) -> dict[str, Any]:
    job_type = str(job.get("job_type") or "").strip().lower()
    if job_type not in {"optimization", "evaluation", "generation"}:
        raise PromptStudioJobError(f"Unsupported prompt studio job type: {job_type}", retryable=False)

    payload = _normalize_payload(job.get("payload"))
    payload["job_id"] = str(job.get("uuid") or job.get("id"))
    payload.setdefault("request_id", job.get("request_id"))

    if job_type == "optimization":
        payload = _scrub_optimization_job_payload_snapshot(
            job=job,
            payload=payload,
            job_manager=job_manager,
        )

    entity_id = _resolve_entity_id(job_type, payload)
    if job_type == "optimization":
        payload.setdefault("optimization_id", entity_id)
    elif job_type == "evaluation":
        payload.setdefault("evaluation_id", entity_id)
    elif job_type == "generation":
        payload.setdefault("project_id", entity_id)

    owner_user_id = job.get("owner_user_id")
    owner_missing = owner_user_id is None or str(owner_user_id).strip() == ""
    if owner_missing and payload.get("user_id") is not None:
        raise PromptStudioJobError("Missing owner_user_id for prompt studio job", retryable=False)
    auth_mode = _auth_mode()
    user_id = _normalize_user_id(
        owner_user_id,
        allow_default=auth_mode == "single_user",
    )
    with _active_user_cache_scope(user_id):
        processor = _get_processor(user_id)

        if job_type == "optimization":
            runtime: ProviderCredentialRuntime | None = None
            identity_row = processor.db.get_optimization(
                entity_id,
                include_deleted=True,
            ) or {}
            if not _optimization_identity_matches(
                identity_row,
                job,
                require_identity=job_manager is not None,
            ):
                raise PromptStudioJobError(
                    "Prompt Studio job identity is invalid",
                    retryable=False,
                    failure_code="job_identity_invalid",
                )
            expected_optimization_uuid = (
                _required_optimization_uuid(job)
                if job_manager is not None
                else str(payload.get("optimization_uuid") or "").strip() or None
            )
            try:
                payload, durable_config = _secure_optimization_durable_state(
                    processor=processor,
                    optimization_id=entity_id,
                    job=job,
                    payload=payload,
                    job_manager=job_manager,
                    require_valid_config=True,
                )
                if job_manager is not None and _sync_live_job_cancellation(
                    processor=processor,
                    optimization_id=entity_id,
                    job=job,
                    job_manager=job_manager,
                ):
                    raise _cancelled_job_error()
                await _ensure_job_owner_active(user_id, auth_mode=auth_mode)
                current = processor.db.get_optimization(
                    entity_id,
                    include_deleted=True,
                ) or {}
                current_status = str(current.get("status") or "").lower()
                if current_status == "cancelled":
                    if job_manager is not None:
                        _cancel_live_job_for_prompt_outcome(
                            processor=processor,
                            optimization_id=entity_id,
                            job=job,
                            job_manager=job_manager,
                        )
                        raise _cancelled_job_error()
                    return await processor.process_optimization_job(
                        payload,
                        entity_id,
                    )
                if current_status == "completed":
                    return processor._completed_optimization_result(
                        entity_id,
                        current,
                    )
                model_selection = durable_config["model_config"]
                owner_id = int(user_id)
                try:
                    team_ids, org_ids = await _owner_membership_scope(owner_id)
                except Exception:  # noqa: BLE001 - map repository detail to one code
                    raise ByokResolutionError(
                        "credential_store_unavailable",
                        model_selection["provider"],
                    ) from None
                runtime = ProviderCredentialRuntime(
                    user_id=owner_id,
                    team_ids=team_ids,
                    org_ids=org_ids,
                    trusted_base_url_override=False,
                    override_snapshot_resolver=capture_provider_override_call_snapshot,
                )
                credentials = await runtime.resolve(
                    model_selection["provider"],
                    model=model_selection["model"],
                )
                app_config = credentials.app_config or {}
                if provider_requires_api_key(model_selection["provider"]) and not provider_auth_is_resolved(
                    model_selection["provider"],
                    api_key=credentials.api_key,
                    app_config=app_config,
                    credentials_resolved=credentials.credentials_resolved,
                ):
                    raise PromptStudioJobError(
                        "Provider credentials are unavailable",
                        retryable=False,
                        failure_code="missing_provider_credentials",
                    )

                execution_config = runtime_model_config(
                    model_selection,
                    api_key=credentials.api_key,
                    app_config=app_config,
                    credentials_resolved=credentials.credentials_resolved,
                )
                scorer_credentials = credentials
                scorer_execution_config: dict[str, Any] | None = None
                distinct_scorer_success_required = False
                raw_strategy_params = durable_config.get("strategy_params")
                strategy_params = (
                    raw_strategy_params
                    if isinstance(raw_strategy_params, dict)
                    else {}
                )
                scorer_model = strategy_params.get("scorer_model")
                strategy = str(
                    durable_config.get("optimizer_type")
                    or payload.get("optimizer_type")
                    or ""
                ).strip().lower()
                if strategy == "mcts" and scorer_model is not None:
                    if not isinstance(scorer_model, str) or not scorer_model.strip():
                        raise ValueError("MCTS scorer model is invalid")
                    scorer_model = scorer_model.strip()
                    if scorer_model != model_selection["model"]:
                        distinct_scorer_success_required = True
                        scorer_credentials = await runtime.resolve(
                            model_selection["provider"],
                            model=scorer_model,
                        )
                        scorer_app_config = scorer_credentials.app_config or {}
                        if provider_requires_api_key(
                            model_selection["provider"]
                        ) and not provider_auth_is_resolved(
                            model_selection["provider"],
                            api_key=scorer_credentials.api_key,
                            app_config=scorer_app_config,
                            credentials_resolved=(
                                scorer_credentials.credentials_resolved
                            ),
                        ):
                            raise PromptStudioJobError(
                                "Provider credentials are unavailable",
                                retryable=False,
                                failure_code="missing_provider_credentials",
                            )
                    else:
                        scorer_app_config = app_config
                    scorer_execution_config = runtime_model_config(
                        {**model_selection, "model": scorer_model},
                        api_key=scorer_credentials.api_key,
                        app_config=scorer_app_config,
                        credentials_resolved=(
                            scorer_credentials.credentials_resolved
                        ),
                    )

                marked = False
                mark_lock = asyncio.Lock()
                cancellation_sync_lock = asyncio.Lock()

                async def _sync_cancellation_after_provider_success() -> None:
                    if job_manager is None:
                        return
                    async with cancellation_sync_lock:
                        if _sync_live_job_cancellation(
                            processor=processor,
                            optimization_id=entity_id,
                            job=job,
                            job_manager=job_manager,
                        ):
                            raise _cancelled_job_error()

                async def _mark_provider_success() -> None:
                    nonlocal marked
                    async with mark_lock:
                        if not marked:
                            persisted = await mark_provider_credential_used(
                                runtime,
                                credentials,
                            )
                            if not persisted:
                                raise PromptStudioJobError(
                                    "Optimization provider execution was not validated",
                                    retryable=False,
                                    failure_code="provider_success_not_observed",
                                )
                            marked = True
                    await _sync_cancellation_after_provider_success()

                scorer_marked = False
                scorer_mark_lock = asyncio.Lock()

                async def _mark_scorer_provider_success() -> None:
                    nonlocal scorer_marked
                    if scorer_credentials is credentials:
                        await _mark_provider_success()
                        scorer_marked = marked
                        return
                    async with scorer_mark_lock:
                        if not scorer_marked:
                            persisted = await mark_provider_credential_used(
                                runtime,
                                scorer_credentials,
                            )
                            if not persisted:
                                raise PromptStudioJobError(
                                    "Optimization provider execution was not validated",
                                    retryable=False,
                                    failure_code="provider_success_not_observed",
                                )
                            scorer_marked = True
                    await _sync_cancellation_after_provider_success()

                async def _before_finalize() -> bool:
                    if job_manager is None:
                        return False
                    return _sync_live_job_cancellation(
                        processor=processor,
                        optimization_id=entity_id,
                        job=job,
                        job_manager=job_manager,
                    )

                def _require_provider_success(result: dict[str, Any]) -> None:
                    scorer_dispatched = result.get("_scorer_provider_dispatched")
                    if type(scorer_dispatched) is not bool:
                        scorer_dispatched = result.get("scorer_provider_dispatched")
                    provider_dispatches = result.get("_provider_dispatches")
                    if not isinstance(provider_dispatches, dict):
                        provider_dispatches = result.get("provider_dispatches")
                    if type(scorer_dispatched) is not bool:
                        scorer_count = (
                            provider_dispatches.get("scorer")
                            if isinstance(provider_dispatches, dict)
                            else None
                        )
                        scorer_dispatched = (
                            scorer_count > 0
                            if type(scorer_count) is int and scorer_count >= 0
                            else None
                        )
                    if not marked or (
                        distinct_scorer_success_required
                        and scorer_dispatched is not False
                        and not scorer_marked
                    ):
                        raise PromptStudioJobError(
                            "Optimization provider execution was not validated",
                            retryable=False,
                            failure_code="provider_success_not_observed",
                        )

                async def _before_completion(result: dict[str, Any]) -> None:
                    _require_provider_success(result)

                processor_kwargs: dict[str, Any] = {
                    "runtime_model_config": execution_config,
                    "provider_credentials": credentials,
                    "on_provider_success": _mark_provider_success,
                    "runtime_scorer_model_config": scorer_execution_config,
                    "scorer_provider_credentials": (
                        scorer_credentials
                        if scorer_execution_config is not None
                        else None
                    ),
                    "on_scorer_provider_success": (
                        _mark_scorer_provider_success
                        if scorer_execution_config is not None
                        else None
                    ),
                    "manage_failure_status": False,
                    "before_completion": _before_completion,
                }
                if job_manager is not None:
                    processor_kwargs["before_finalize"] = _before_finalize
                    if _sync_live_job_cancellation(
                        processor=processor,
                        optimization_id=entity_id,
                        job=job,
                        job_manager=job_manager,
                    ):
                        raise _cancelled_job_error()
                result = await processor.process_optimization_job(
                    payload,
                    entity_id,
                    **processor_kwargs,
                )
                latest = processor.db.get_optimization(
                    entity_id,
                    include_deleted=True,
                ) or {}
                latest_status = str(latest.get("status") or "").strip().lower()
                if latest_status == "cancelled":
                    if job_manager is not None:
                        _cancel_live_job_for_prompt_outcome(
                            processor=processor,
                            optimization_id=entity_id,
                            job=job,
                            job_manager=job_manager,
                        )
                        raise _cancelled_job_error()
                    cancelled_result = dict(result)
                    cancelled_result["status"] = "cancelled"
                    for key in (
                        "_provider_dispatches",
                        "provider_dispatches",
                        "_scorer_provider_dispatched",
                        "scorer_provider_dispatched",
                    ):
                        cancelled_result.pop(key, None)
                    return cancelled_result
                if job_manager is not None and _sync_live_job_cancellation(
                    processor=processor,
                    optimization_id=entity_id,
                    job=job,
                    job_manager=job_manager,
                ):
                    raise _cancelled_job_error()
                if latest_status != "completed":
                    if latest_status == "failed":
                        raise PromptStudioJobError(
                            "Optimization failed",
                            retryable=False,
                            failure_code="prompt_studio_job_failed",
                        )
                    raise PromptStudioJobError(
                        "Optimization state is temporarily unavailable",
                        retryable=True,
                        failure_code="job_state_unavailable",
                    )
                result_status = str(
                    result.get("status") or latest.get("status") or ""
                ).strip().lower()
                if result_status == "completed":
                    _require_provider_success(result)
                for key in (
                    "_provider_dispatches",
                    "provider_dispatches",
                    "_scorer_provider_dispatched",
                    "scorer_provider_dispatched",
                ):
                    result.pop(key, None)
                return result
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - mapped to bounded Jobs metadata
                bounded = _bounded_optimization_error(exc)
                if bounded.failure_code in {"job_cancelled", "job_identity_invalid"}:
                    raise_detached_error(bounded)
                if job_manager is not None:
                    try:
                        cancelled = _sync_live_job_cancellation(
                            processor=processor,
                            optimization_id=entity_id,
                            job=job,
                            job_manager=job_manager,
                        )
                    except PromptStudioJobError as state_error:
                        if state_error.retryable:
                            _mark_retry_pending_safely(
                                processor,
                                entity_id,
                                state_error,
                                expected_uuid=expected_optimization_uuid,
                            )
                        elif state_error.failure_code != "job_identity_invalid":
                            _mark_failed_safely(
                                processor,
                                entity_id,
                                state_error,
                                expected_uuid=expected_optimization_uuid,
                            )
                        raise_detached_error(state_error)
                    if cancelled:
                        raise_detached_error(_cancelled_job_error())
                if _retry_attempt_remains(job, bounded):
                    _mark_retry_pending_safely(
                        processor,
                        entity_id,
                        bounded,
                        expected_uuid=expected_optimization_uuid,
                    )
                else:
                    _mark_failed_safely(
                        processor,
                        entity_id,
                        bounded,
                        expected_uuid=expected_optimization_uuid,
                    )
                raise_detached_error(bounded)
            finally:
                if runtime is not None:
                    await await_owned_worker(runtime.close())
        await _ensure_job_owner_active(user_id, auth_mode=auth_mode)
        if job_type == "evaluation":
            return await processor.process_evaluation_job(payload, entity_id)
        return await processor.process_generation_job(payload, entity_id)


async def _inflight_quota_guard(job: dict[str, Any], jm: JobManager) -> bool:
    owner = job.get("owner_user_id")
    if owner is None or str(owner).strip() == "":
        return True
    owner_id = str(owner)
    try:
        await apply_prompt_studio_quota_policy(owner_id)
    except Exception as exc:
        logger.debug("Prompt Studio quota policy lookup failed for {}: {}", owner_id, exc)
    try:
        max_inflight = jm._quota_get("JOBS_QUOTA_MAX_INFLIGHT", _PROMPT_STUDIO_DOMAIN, owner_id)
    except Exception:
        max_inflight = 0
    if not max_inflight:
        return True
    current = jm.count_processing_for_owner(domain=_PROMPT_STUDIO_DOMAIN, owner_user_id=owner_id)
    if current > int(max_inflight):
        logger.info("Prompt Studio inflight quota reached for user {}; requeueing job {}", owner_id, job.get("id"))
        return False
    return True


async def _broadcast_completed_optimization(
    job: dict[str, Any],
    result: dict[str, Any],
) -> None:
    """Publish MCTS completion after core Jobs accepted completion."""

    if str(job.get("job_type") or "").strip().lower() != "optimization":
        return
    if str(result.get("status") or "").strip().lower() != "completed":
        return

    payload = _normalize_payload(job.get("payload"))
    config = _normalize_payload(payload.get("optimization_config"))
    strategy = str(
        payload.get("optimizer_type")
        or config.get("optimizer_type")
        or config.get("strategy")
        or ""
    ).strip().lower()
    if strategy != "mcts":
        return

    try:
        optimization_id = _resolve_entity_id("optimization", payload)
        expected_uuid = _required_optimization_uuid(job)
        auth_mode = _auth_mode()
        user_id = _normalize_user_id(
            job.get("owner_user_id"),
            allow_default=auth_mode == "single_user",
        )
        with _active_user_cache_scope(user_id):
            processor = _get_processor(user_id)
            current = processor.db.get_optimization(
                optimization_id,
                include_deleted=True,
            ) or {}
            if not _optimization_identity_matches(
                current,
                job,
                require_identity=True,
            ):
                return
            from ..optimization_engine import OptimizationEngine

            await OptimizationEngine(processor.db)._broadcast_mcts_completion(
                optimization_id,
                result,
                expected_optimization_uuid=expected_uuid,
            )
    except asyncio.CancelledError:
        raise
    except Exception as exc:  # noqa: BLE001 - completion is already durable
        logger.bind(error_type=type(exc).__name__).warning(
            "Prompt Studio post-completion event publication failed"
        )


async def _reconcile_rejected_optimization_completion(
    job: dict[str, Any],
    result: dict[str, Any],
    job_manager: JobManager,
) -> None:
    """Let an authoritative late Jobs cancellation override Prompt completion."""

    if str(job.get("job_type") or "").strip().lower() != "optimization":
        return
    if str(result.get("status") or "").strip().lower() != "completed":
        return
    try:
        live_job = _read_exact_job_state(job, job_manager)
        live_status = str(live_job.get("status") or "").strip().lower()
        if (
            live_status != "cancelled"
            and live_job.get("cancel_requested_at") is None
        ):
            return
        payload = _normalize_payload(job.get("payload"))
        optimization_id = _resolve_entity_id("optimization", payload)
        auth_mode = _auth_mode()
        user_id = _normalize_user_id(
            job.get("owner_user_id"),
            allow_default=auth_mode == "single_user",
        )
        with _active_user_cache_scope(user_id):
            processor = _get_processor(user_id)
            _sync_live_job_cancellation(
                processor=processor,
                optimization_id=optimization_id,
                job=job,
                job_manager=job_manager,
            )
    except asyncio.CancelledError:
        raise
    except Exception as exc:  # noqa: BLE001 - reconciliation loop remains fallback
        logger.bind(error_type=type(exc).__name__).warning(
            "Prompt Studio rejected completion reconciliation deferred"
        )


async def run_prompt_studio_jobs_worker(stop_event: asyncio.Event | None = None) -> None:
    worker_id = (os.getenv("PROMPT_STUDIO_JOBS_WORKER_ID") or f"prompt-studio-jobs-{os.getpid()}").strip()
    queue = (os.getenv("PROMPT_STUDIO_JOBS_QUEUE") or "default").strip() or "default"
    cfg = _build_worker_config(worker_id=worker_id, queue=queue)

    jm = _jobs_manager()
    sdk = WorkerSDK(jm, cfg)
    stop_task: asyncio.Task[None] | None = None
    reconcile_stop = threading.Event()
    reconcile_task = asyncio.create_task(
        _cancelled_job_reconciliation_loop(jm, stop_event=reconcile_stop)
    )
    if stop_event is not None:
        async def _watch_stop() -> None:
            await stop_event.wait()
            sdk.stop()

        stop_task = asyncio.create_task(_watch_stop())
    logger.info(f"Prompt Studio Jobs worker starting (queue={queue}, worker_id={worker_id})")
    try:
        async def _handle_with_manager(job: dict[str, Any]) -> dict[str, Any]:
            return await _handle_job(job, job_manager=jm)

        async def _reconcile_rejected_completion(
            job: dict[str, Any],
            result: dict[str, Any],
        ) -> None:
            await _reconcile_rejected_optimization_completion(job, result, jm)

        await sdk.run(
            handler=_handle_with_manager,
            acquire_guard=lambda job: _inflight_quota_guard(job, jm),
            on_completed=_broadcast_completed_optimization,
            on_completion_rejected=_reconcile_rejected_completion,
        )
    finally:
        # Prevent another page/row from starting before cancellation reaches an
        # owned executor call, then join any call already using shared backends.
        caller_cancelled = False
        reconcile_stop.set()
        reconcile_task.cancel()
        try:
            await await_owned_worker(reconcile_task)
        except asyncio.CancelledError:
            current_task = asyncio.current_task()
            caller_cancelled = bool(
                current_task is not None and current_task.cancelling()
            )
        if stop_task is not None and not stop_task.done():
            stop_task.cancel()
            try:
                await await_owned_worker(stop_task)
            except asyncio.CancelledError:
                current_task = asyncio.current_task()
                caller_cancelled = caller_cancelled or bool(
                    current_task is not None and current_task.cancelling()
                )
        # Prompt Studio database handles are thread-local. Drain on the worker's
        # event-loop thread so handles opened by normal job execution are
        # actually returned instead of attempting cleanup from an unrelated
        # executor thread.
        _drain_tenant_db_cache()
        if caller_cancelled:
            raise asyncio.CancelledError


async def main() -> None:
    await refresh_llm_provider_overrides(force=True)
    try:
        start_llm_provider_override_refresh_service()
        await run_prompt_studio_jobs_worker()
    finally:
        await shutdown_llm_provider_override_recovery()


if __name__ == "__main__":
    asyncio.run(main())
