"""
TTS Jobs worker for long-form speech generation.

- domain = "audio"
- queue = os.getenv("TTS_JOBS_QUEUE", "default")
- job_type = "tts_longform"
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
from types import SimpleNamespace
from typing import Any

from loguru import logger

from tldw_Server_API.app.api.v1.schemas.audio_schemas import OpenAISpeechRequest
from tldw_Server_API.app.core.Audio.tts_service import (
    _infer_tts_provider_from_model,
    _resolve_tts_byok,
)
from tldw_Server_API.app.core.AuthNZ.byok_helpers import is_trusted_base_url_principal
from tldw_Server_API.app.core.AuthNZ.byok_runtime import (
    ByokResolutionError,
    resolve_byok_credentials,
)
from tldw_Server_API.app.core.AuthNZ.llm_provider_overrides import (
    refresh_llm_provider_overrides,
    shutdown_llm_provider_override_recovery,
    start_llm_provider_override_refresh_service,
)
from tldw_Server_API.app.core.AuthNZ.orgs_teams import (
    list_active_team_memberships_for_user,
    list_org_memberships_for_user,
)
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal
from tldw_Server_API.app.core.AuthNZ.repos.rbac_repo import AuthnzRbacRepo
from tldw_Server_API.app.core.AuthNZ.repos.users_repo import AuthnzUsersRepo
from tldw_Server_API.app.core.AuthNZ.settings import get_settings as get_auth_settings
from tldw_Server_API.app.core.Chat.bounded_daemon import (
    BoundedDaemonPool,
    await_bounded_daemon_with_timeout,
)
from tldw_Server_API.app.core.Chat.streaming_utils import (
    await_stream_operation_bounded,
    invoke_stream_close_bounded,
)
from tldw_Server_API.app.core.config import settings
from tldw_Server_API.app.core.DB_Management.Collections_DB import CollectionsDatabase
from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths
from tldw_Server_API.app.core.DB_Management.media_db.api import create_media_database
from tldw_Server_API.app.core.Jobs.event_stream import emit_job_event
from tldw_Server_API.app.core.Jobs.manager import JobManager
from tldw_Server_API.app.core.Jobs.worker_sdk import WorkerConfig, WorkerSDK
from tldw_Server_API.app.core.Jobs.worker_utils import coerce_int as _coerce_int
from tldw_Server_API.app.core.Jobs.worker_utils import jobs_manager_from_env as _jobs_manager
from tldw_Server_API.app.core.Metrics.metrics_logger import log_counter, log_histogram
from tldw_Server_API.app.core.TTS.gateway_preflight import gateway_route_provenance
from tldw_Server_API.app.core.TTS.tts_exceptions import TTSError, is_retryable_error
from tldw_Server_API.app.core.TTS.tts_service_v2 import get_tts_service_v2
from tldw_Server_API.app.core.TTS.utils import (
    build_tts_segments_payload,
    compute_tts_history_text_hash,
    contains_tts_credential_fields,
    parse_bool,
    tts_history_text_length,
)

TTS_DOMAIN = "audio"
TTS_JOB_TYPE = "tts_longform"
TTS_PROVIDER_DISPATCH_STARTED = "tts_provider_dispatch_started"
TTS_JOB_USAGE_TOUCH_TIMEOUT_SECONDS = 0.25
TTS_JOB_RBAC_TIMEOUT_SECONDS = 5.0
TTS_JOB_RBAC_DAEMON_POOL = BoundedDaemonPool(4)
_TTS_JOB_CREDENTIAL_SCOPE_FIELDS = frozenset(
    {
        "owner_user_id",
        "team_ids",
        "org_ids",
        "credential_source",
        "trusted_base_url_requested",
    }
)
_TTS_JOB_CREDENTIAL_SOURCES = frozenset(
    {"user", "team", "org", "server_default", "none"}
)
_TTS_JOBS_NONCRITICAL_EXCEPTIONS: tuple[type[BaseException], ...] = (
    AttributeError,
    ConnectionError,
    LookupError,
    OSError,
    OverflowError,
    RuntimeError,
    TimeoutError,
    TypeError,
    ValueError,
    json.JSONDecodeError,
)


class TTSJobError(Exception):
    def __init__(
        self,
        message: str,
        *,
        retryable: bool = True,
        backoff_seconds: int | None = None,
        failure_code: str | None = None,
    ):
        super().__init__(message)
        self.retryable = retryable
        self.failure_code = failure_code
        if backoff_seconds is not None:
            self.backoff_seconds = int(backoff_seconds)


def _bounded_tts_job_failure(exc: Exception) -> TTSJobError:
    """Detach and terminally classify failures after provider dispatch."""
    if isinstance(exc, TTSError):
        provider_unavailable = is_retryable_error(exc)
        return TTSJobError(
            "TTS provider request failed" if provider_unavailable else "TTS request failed",
            retryable=False,
            failure_code=(
                "provider_unavailable" if provider_unavailable else "tts_request_failed"
            ),
        )
    return TTSJobError(
        "TTS generation failed",
        retryable=False,
        failure_code="tts_generation_failed",
    )


def _bounded_tts_credential_failure(exc: Exception) -> TTSJobError:
    """Keep credential retry policy typed without persisting raw details."""
    code = "invalid_provider_credentials"
    if isinstance(exc, ByokResolutionError):
        code = str(getattr(exc, "policy_code", None) or exc.code or code)
    else:
        detail = getattr(exc, "detail", None)
        if isinstance(detail, dict):
            code = str(detail.get("error_code") or code)
    if code not in {
        "credential_scope_revoked",
        "credential_store_unavailable",
        "invalid_provider_credentials",
        "missing_provider_credentials",
        "model_not_allowed",
        "provider_disabled",
    }:
        code = "invalid_provider_credentials"
    retryable = code == "credential_store_unavailable"
    return TTSJobError(
        (
            "provider credentials are temporarily unavailable"
            if retryable
            else "provider credentials are unavailable"
        ),
        retryable=retryable,
        failure_code=code,
    )


def _tts_replay_blocked() -> TTSJobError:
    """Return the terminal error for a job whose provider call already started."""
    return TTSJobError(
        "TTS provider dispatch was already started",
        retryable=False,
        failure_code="tts_replay_blocked",
    )


def _persist_tts_dispatch_marker(
    jm: JobManager,
    job: dict[str, Any],
    *,
    job_id: int,
) -> bool:
    """Fence and persist the monotonic provider-dispatch marker."""
    lease_seconds = _coerce_int(
        os.getenv("TTS_JOBS_LEASE_SECONDS") or os.getenv("JOBS_LEASE_SECONDS"),
        60,
    )
    try:
        return bool(
            jm.renew_job_lease(
                job_id,
                seconds=lease_seconds,
                worker_id=job.get("worker_id"),
                lease_id=job.get("lease_id"),
                progress_message=TTS_PROVIDER_DISPATCH_STARTED,
                enforce=True,
            )
        )
    except Exception as exc:  # noqa: BLE001 - durable dispatch boundary fails closed
        logger.bind(error_type=type(exc).__name__).warning(
            "TTS job dispatch marker persistence failed"
        )
        return False


async def _mark_tts_credentials_used(resolution: Any) -> None:
    """Bound best-effort first-byte usage accounting without masking cancellation."""
    try:
        await await_stream_operation_bounded(
            resolution.touch_last_used(),
            timeout=TTS_JOB_USAGE_TOUCH_TIMEOUT_SECONDS,
            cleanup=False,
        )
    except asyncio.CancelledError:
        current_task = asyncio.current_task()
        if current_task is not None and current_task.cancelling():
            raise
        logger.bind(error_type="CancelledError").debug(
            "TTS job credential usage update was cancelled"
        )
    except Exception as exc:  # noqa: BLE001 - usage accounting is best effort
        logger.bind(error_type=type(exc).__name__).debug(
            "TTS job credential usage update failed"
        )


async def _close_tts_speech_iter(speech_iter: Any) -> None:
    """Bound one owned speech-iterator close without masking caller cancellation."""
    close = getattr(speech_iter, "aclose", None)
    if not callable(close):
        return
    try:
        await invoke_stream_close_bounded(close)
    except asyncio.CancelledError:
        current_task = asyncio.current_task()
        if current_task is not None and current_task.cancelling():
            raise
        logger.bind(error_type="CancelledError").warning(
            "TTS job speech iterator cleanup was cancelled"
        )
    except Exception as exc:  # noqa: BLE001 - cleanup failure is logged by type only
        logger.bind(error_type=type(exc).__name__).warning(
            "TTS job speech iterator cleanup failed"
        )


def _resolve_user_id(job: dict[str, Any], payload: dict[str, Any]) -> str:
    owner = job.get("owner_user_id")
    if owner is None or str(owner).strip() == "":
        raise TTSJobError("missing user_id", retryable=False)
    owner_id = str(owner).strip()
    payload_user = payload.get("user_id")
    if payload_user is not None and str(payload_user).strip() != owner_id:
        raise TTSJobError("payload user_id does not match job owner", retryable=False)
    return owner_id


def _credential_scope_failure(*, store_unavailable: bool = False) -> TTSJobError:
    """Return a detached, bounded failure for queued authorization context."""
    return TTSJobError(
        (
            "provider credentials are temporarily unavailable"
            if store_unavailable
            else "provider credentials are unavailable"
        ),
        retryable=store_unavailable,
        failure_code=(
            "credential_store_unavailable"
            if store_unavailable
            else "credential_scope_revoked"
        ),
    )


def _parse_tts_job_credential_scope(
    payload: dict[str, Any],
    *,
    owner_user_id: int,
) -> dict[str, Any] | None:
    """Strictly parse the non-secret authority snapshot persisted at enqueue."""
    if "credential_scope" not in payload:
        return None
    raw_scope = payload.get("credential_scope")
    if not isinstance(raw_scope, dict):
        raise _credential_scope_failure()
    if set(raw_scope) != _TTS_JOB_CREDENTIAL_SCOPE_FIELDS:
        raise _credential_scope_failure()

    scope_owner = raw_scope.get("owner_user_id")
    team_ids = raw_scope.get("team_ids")
    org_ids = raw_scope.get("org_ids")
    source = raw_scope.get("credential_source")
    trusted_base_url_requested = raw_scope.get("trusted_base_url_requested")

    if type(scope_owner) is not int or scope_owner <= 0 or scope_owner != owner_user_id:
        raise _credential_scope_failure()
    if not isinstance(team_ids, list) or not isinstance(org_ids, list):
        raise _credential_scope_failure()
    if len(team_ids) > 1 or len(org_ids) > 1:
        raise _credential_scope_failure()
    if any(type(scope_id) is not int or scope_id <= 0 for scope_id in team_ids + org_ids):
        raise _credential_scope_failure()
    if type(source) is not str or source not in _TTS_JOB_CREDENTIAL_SOURCES:
        raise _credential_scope_failure()
    if type(trusted_base_url_requested) is not bool:
        raise _credential_scope_failure()

    if source == "team":
        if len(team_ids) != 1 or org_ids:
            raise _credential_scope_failure()
    elif source == "org":
        if len(org_ids) != 1 or team_ids:
            raise _credential_scope_failure()
    elif team_ids or org_ids:
        raise _credential_scope_failure()
    if trusted_base_url_requested and source not in {"user", "team", "org"}:
        raise _credential_scope_failure()

    return {
        "owner_user_id": scope_owner,
        "team_ids": list(team_ids),
        "org_ids": list(org_ids),
        "credential_source": source,
        "trusted_base_url_requested": trusted_base_url_requested,
    }


async def _revalidate_tts_job_credential_scope(
    scope: dict[str, Any],
) -> tuple[list[int], list[int]]:
    """Revalidate the one persisted shared scope against current memberships."""
    source = scope["credential_source"]
    if source not in {"team", "org"}:
        return [], []

    owner_user_id = scope["owner_user_id"]
    lookup_failure: TTSJobError | None = None
    memberships: Any = None
    try:
        if source == "team":
            memberships = await list_active_team_memberships_for_user(owner_user_id)
        else:
            memberships = await list_org_memberships_for_user(owner_user_id)
    except asyncio.CancelledError:
        raise
    except Exception as exc:  # noqa: BLE001 - store failures map to a safe retry
        logger.bind(error_type=type(exc).__name__).warning(
            "TTS job credential scope revalidation failed"
        )
        lookup_failure = _credential_scope_failure(store_unavailable=True)
    if lookup_failure is not None:
        raise lookup_failure
    if not isinstance(memberships, list):
        raise _credential_scope_failure(store_unavailable=True)

    id_field = "team_id" if source == "team" else "org_id"
    current_ids: set[int] = set()
    for membership in memberships:
        if not isinstance(membership, dict):
            raise _credential_scope_failure(store_unavailable=True)
        scope_id = membership.get(id_field)
        if type(scope_id) is not int or scope_id <= 0:
            raise _credential_scope_failure(store_unavailable=True)
        if source == "org" and str(membership.get("status") or "").strip().lower() != "active":
            continue
        current_ids.add(scope_id)

    expected_id = scope[f"{source}_ids"][0]
    if expected_id not in current_ids:
        raise _credential_scope_failure()
    return ([expected_id], []) if source == "team" else ([], [expected_id])


async def _ensure_tts_job_owner_active(owner_user_id: int) -> None:
    """Revalidate the queued owner account for every new scoped job."""
    lookup_failure: TTSJobError | None = None
    user: Any = None
    try:
        auth_settings = get_auth_settings()
        if str(auth_settings.AUTH_MODE).strip().lower() == "single_user":
            return
        users_repo = await AuthnzUsersRepo.from_pool()
        user = await users_repo.get_user_by_id(owner_user_id)
    except asyncio.CancelledError:
        raise
    except Exception as exc:  # noqa: BLE001 - owner store failures are retryable
        logger.bind(error_type=type(exc).__name__).warning(
            "TTS job owner revalidation failed"
        )
        lookup_failure = _credential_scope_failure(store_unavailable=True)
    if lookup_failure is not None:
        raise lookup_failure
    if not isinstance(user, dict) or user.get("is_active") is not True:
        raise _credential_scope_failure()


def _load_current_tts_job_rbac(
    owner_user_id: int,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Read current roles and permissions sequentially on one bounded worker."""

    rbac_repo = AuthnzRbacRepo(client_id="tts_jobs_worker")
    return (
        rbac_repo.get_user_roles(owner_user_id),
        rbac_repo.get_effective_permissions(owner_user_id),
    )


async def _current_tts_job_base_url_trust(owner_user_id: int) -> bool:
    """Reconstruct only the current claims needed to authorize custom endpoints."""
    lookup_failure: TTSJobError | None = None
    try:
        auth_settings = get_auth_settings()
        if str(auth_settings.AUTH_MODE).strip().lower() == "single_user":
            return owner_user_id == int(auth_settings.SINGLE_USER_FIXED_ID)

        users_repo = await AuthnzUsersRepo.from_pool()
        user = await users_repo.get_user_by_id(owner_user_id)
        if not isinstance(user, dict) or user.get("is_active") is not True:
            return False

        role_rows, permissions = await await_bounded_daemon_with_timeout(
            lambda: _load_current_tts_job_rbac(owner_user_id),
            pool=TTS_JOB_RBAC_DAEMON_POOL,
            name="tts-job-rbac-revalidation",
            timeout_seconds=TTS_JOB_RBAC_TIMEOUT_SECONDS,
            timeout_message="TTS job RBAC revalidation timed out",
        )
        if not isinstance(role_rows, list) or not isinstance(permissions, list):
            raise TypeError("invalid RBAC lookup result")
        roles = [
            str(row.get("name")).strip()
            for row in role_rows
            if isinstance(row, dict) and str(row.get("name") or "").strip()
        ]
        normalized_permissions = [str(permission) for permission in permissions]
        if user.get("is_superuser") is True or user.get("is_admin") is True:
            normalized_permissions.append("system.configure")
        principal = AuthPrincipal(
            kind="user",
            user_id=owner_user_id,
            roles=roles,
            permissions=normalized_permissions,
        )
        return is_trusted_base_url_principal(principal)
    except asyncio.CancelledError:
        raise
    except Exception as exc:  # noqa: BLE001 - current authority store must fail closed
        logger.bind(error_type=type(exc).__name__).warning(
            "TTS job endpoint authority revalidation failed"
        )
        lookup_failure = _credential_scope_failure(store_unavailable=True)
    if lookup_failure is not None:
        raise lookup_failure
    return False


def _tts_history_config() -> dict[str, Any]:
    return {
        "enabled": parse_bool(getattr(settings, "TTS_HISTORY_ENABLED", False), default=False),
        "store_text": parse_bool(getattr(settings, "TTS_HISTORY_STORE_TEXT", True), default=True),
        "store_failed": parse_bool(getattr(settings, "TTS_HISTORY_STORE_FAILED", True), default=True),
        "hash_key": getattr(settings, "TTS_HISTORY_HASH_KEY", None),
    }


def _open_media_db_for_history(user_id: str) -> Any | None:
    try:
        db_path = DatabasePaths.get_media_db_path(user_id)
        return create_media_database(
            client_id="tts_jobs_worker",
            db_path=str(db_path),
        )
    except Exception as exc:  # noqa: BLE001 - history is explicitly noncritical
        logger.bind(error_type=type(exc).__name__).debug(
            "TTS jobs worker failed to open media db for history"
        )
        return None


def _sanitize_params_json(
    request: OpenAISpeechRequest,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    params_json: dict[str, Any] = {"speed": request.speed}
    if request.extra_params:
        try:
            extra_params = dict(request.extra_params)
        except _TTS_JOBS_NONCRITICAL_EXCEPTIONS:
            extra_params = None
        if extra_params:
            extra_params.pop("voice_reference", None)
            params_json["extra_params"] = extra_params
    if request.lang_code:
        params_json["lang_code"] = request.lang_code
    route = gateway_route_provenance(
        requested_backend=request.backend,
        requested_model=request.model,
        metadata=metadata,
    )
    if route:
        if route["requested_backend"] != route["actual_backend"]:
            params_json["requested_backend"] = route["requested_backend"]
        params_json["fallback_used"] = route["fallback_used"]
        params_json["conversion_used"] = route["conversion_used"]
    return params_json


def _build_voice_info(request: OpenAISpeechRequest, metadata: dict[str, Any]) -> dict[str, Any] | None:
    voice_info: dict[str, Any] = {}
    meta_voice_info = metadata.get("voice_info")
    if isinstance(meta_voice_info, dict):
        voice_info.update(meta_voice_info)
    voice_info.pop("voice_reference", None)
    if request.voice_reference:
        voice_info["has_voice_reference"] = True
    if request.reference_duration_min is not None:
        voice_info["reference_duration_min"] = request.reference_duration_min
    return voice_info or None


async def _handle_tts_job(job: dict[str, Any]) -> dict[str, Any]:
    if job.get("progress_message") == TTS_PROVIDER_DISPATCH_STARTED:
        raise _tts_replay_blocked()

    raw_payload = job.get("payload")
    if raw_payload is None:
        payload: dict[str, Any] = {}
    elif isinstance(raw_payload, dict):
        payload = raw_payload
    else:
        raise TTSJobError(
            "invalid job payload",
            retryable=False,
            failure_code="invalid_job_payload",
        )
    job_type = str(job.get("job_type") or payload.get("job_type") or "").strip().lower()
    if job_type and job_type != TTS_JOB_TYPE:
        raise TTSJobError(f"unsupported job_type: {job_type}", retryable=False)

    raw_speech_payload = payload.get("speech_request")
    if not isinstance(raw_speech_payload, dict):
        raise TTSJobError("missing speech_request payload", retryable=False)
    if contains_tts_credential_fields(raw_speech_payload):
        raise TTSJobError(
            "credential fields are not allowed in speech_request",
            retryable=False,
        )

    # Keep the durable payload immutable while applying worker-only safety policy.
    speech_payload = dict(raw_speech_payload)
    speech_payload["stream"] = False
    request_failure: TTSJobError | None = None
    try:
        request = OpenAISpeechRequest(**speech_payload)
    except Exception:  # noqa: BLE001 - validation detail can contain rejected input
        request_failure = TTSJobError(
            "invalid speech_request",
            retryable=False,
            failure_code="invalid_speech_request",
        )
    if request_failure is not None:
        raise request_failure
    worker_extra_params = (
        dict(request.extra_params)
        if isinstance(request.extra_params, dict)
        else {}
    )
    worker_extra_params["segment_retry_max"] = 1
    request.extra_params = worker_extra_params

    explicit_backend = request.backend is not None
    user_id = _resolve_user_id(job, {} if explicit_backend else payload)
    provider_hint = (
        None
        if explicit_backend
        else _infer_tts_provider_from_model(request.model) or payload.get("provider_hint")
    )
    try:
        owner_user_id = int(user_id)
    except (TypeError, ValueError):
        raise TTSJobError("invalid user_id", retryable=False) from None
    if owner_user_id <= 0:
        raise TTSJobError("invalid user_id", retryable=False)

    credential_scope = (
        None
        if explicit_backend
        else _parse_tts_job_credential_scope(
            payload,
            owner_user_id=owner_user_id,
        )
    )
    scoped_team_ids: list[int] = []
    scoped_org_ids: list[int] = []
    trusted_base_url = False
    credential_resolver = None
    if credential_scope is not None:
        await _ensure_tts_job_owner_active(owner_user_id)
        scoped_team_ids, scoped_org_ids = await _revalidate_tts_job_credential_scope(
            credential_scope
        )
        if credential_scope["trusted_base_url_requested"]:
            trusted_base_url = await _current_tts_job_base_url_trust(owner_user_id)
            if not trusted_base_url:
                raise _credential_scope_failure()

        async def _scoped_credential_resolver(provider: str, **kwargs: Any) -> Any:
            resolver_kwargs = dict(kwargs)
            resolver_kwargs.pop("team_ids", None)
            resolver_kwargs.pop("org_ids", None)
            resolver_kwargs.pop("trusted_base_url_override", None)
            return await resolve_byok_credentials(
                provider,
                **resolver_kwargs,
                team_ids=scoped_team_ids,
                org_ids=scoped_org_ids,
                trusted_base_url_override=trusted_base_url,
                required_source=credential_scope["credential_source"],
            )

        credential_resolver = _scoped_credential_resolver

    if explicit_backend:
        await _ensure_tts_job_owner_active(owner_user_id)
        user_id_int = owner_user_id
        provider_overrides = None
        credential_resolution = None
    else:
        credential_failure: TTSJobError | None = None
        try:
            resolution_kwargs: dict[str, Any] = {
                "provider_hint": provider_hint,
                "model": request.model,
                "current_user": SimpleNamespace(id=owner_user_id),
                "request": SimpleNamespace(state=SimpleNamespace()),
            }
            if credential_resolver is not None:
                resolution_kwargs["credential_resolver"] = credential_resolver
            user_id_int, provider_overrides, credential_resolution = await _resolve_tts_byok(
                **resolution_kwargs,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - credential boundary must fail closed
            logger.bind(error_type=type(exc).__name__).warning(
                "TTS job credential resolution failed"
            )
            credential_failure = _bounded_tts_credential_failure(exc)
        if credential_failure is not None:
            raise credential_failure
        if user_id_int != owner_user_id:
            raise TTSJobError("resolved user_id does not match job owner", retryable=False)
    if not explicit_backend and credential_scope is not None:
        resolved_source = (
            getattr(credential_resolution, "source", None)
            if credential_resolution is not None
            else "none"
        )
        if resolved_source != credential_scope["credential_source"]:
            raise _credential_scope_failure()
        if resolved_source in {"user", "team", "org"}:
            credential_fields = getattr(
                credential_resolution,
                "credential_fields",
                None,
            )
            if not isinstance(credential_fields, dict):
                raise _credential_scope_failure()
            raw_base_url = credential_fields.get("base_url")
            has_base_url = isinstance(raw_base_url, str) and bool(raw_base_url.strip())
            if has_base_url != credential_scope["trusted_base_url_requested"]:
                raise _credential_scope_failure()

    jm = JobManager()
    tts_service = await get_tts_service_v2()
    job_id = int(job.get("id") or 0)
    request_id = str(job.get("request_id") or payload.get("request_id") or "")
    start_ts = asyncio.get_event_loop().time()

    history_cfg = _tts_history_config()
    history_enabled = history_cfg.get("enabled", False)
    history_db: Any | None = _open_media_db_for_history(user_id) if history_enabled else None
    history_written = False

    def _record_history(
        status: str,
        *,
        error_message: str | None = None,
        output_id: int | None = None,
        output_incarnation: str | None = None,
        artifact_ids: list[Any] | None = None,
    ) -> None:
        nonlocal history_written
        if history_written or history_db is None:
            return
        if status == "failed" and not history_cfg.get("store_failed", True):
            return
        try:
            text_hash = compute_tts_history_text_hash(request.input, history_cfg.get("hash_key"))
        except Exception as exc:  # noqa: BLE001 - history is explicitly noncritical
            logger.bind(error_type=type(exc).__name__).debug(
                "TTS jobs worker failed to compute text hash (job_id={}, request_id={})",
                job_id,
                request_id or "unknown",
            )
            return
        metadata = getattr(request, "_tts_metadata", None)
        if not isinstance(metadata, dict):
            metadata = {}
        provider = (
            metadata.get("actual_backend")
            or metadata.get("actual_provider")
            or metadata.get("provider")
            or provider_hint
        )
        model = metadata.get("model") or request.model
        voice_name = metadata.get("voice") or request.voice
        voice_id = metadata.get("voice_id")
        fmt = metadata.get("format") or request.response_format

        duration_ms = None
        duration_val = metadata.get("duration_ms")
        if isinstance(duration_val, (int, float)):
            duration_ms = int(duration_val)
        else:
            duration_seconds = metadata.get("duration_seconds") or metadata.get("duration")
            if isinstance(duration_seconds, (int, float)):
                duration_ms = int(float(duration_seconds) * 1000)

        segments_json = build_tts_segments_payload(metadata.get("segments"))
        generation_time_ms = int(max(0.0, (asyncio.get_event_loop().time() - start_ts) * 1000))
        text_length = tts_history_text_length(request.input)
        text_value = request.input if history_cfg.get("store_text", True) else None
        params_json = _sanitize_params_json(request, metadata)
        voice_info = _build_voice_info(request, metadata)

        final_status = status
        if final_status == "success" and parse_bool(metadata.get("partial"), default=False):
            final_status = "partial"

        try:
            insert_start = asyncio.get_event_loop().time()
            history_db.create_tts_history_entry(
                user_id=str(user_id),
                text_hash=text_hash,
                text=text_value,
                text_length=text_length,
                provider=str(provider) if provider is not None else None,
                model=str(model) if model is not None else None,
                voice_id=str(voice_id) if voice_id is not None else None,
                voice_name=str(voice_name) if voice_name is not None else None,
                voice_info=voice_info,
                format=str(fmt) if fmt is not None else None,
                duration_ms=duration_ms,
                generation_time_ms=generation_time_ms,
                params_json=params_json if params_json else None,
                status=final_status,
                segments_json=segments_json,
                job_id=job_id if job_id > 0 else None,
                output_id=output_id,
                output_incarnation=output_incarnation,
                artifact_ids=artifact_ids,
                error_message=error_message,
            )
            try:
                log_counter(
                    "tts_history_writes_total",
                    labels={
                        "status": str(final_status or "unknown"),
                        "provider": str(provider or "unknown"),
                    },
                )
                log_histogram(
                    "tts_history_write_latency_ms",
                    value=max(0.0, (asyncio.get_event_loop().time() - insert_start) * 1000),
                    labels={"status": str(final_status or "unknown")},
                )
            except Exception as exc:  # noqa: BLE001 - metrics cannot affect job outcome
                logger.bind(error_type=type(exc).__name__).debug(
                    "TTS jobs worker failed to record history metrics"
                )
            history_written = True
        except Exception as exc:  # noqa: BLE001 - history is explicitly noncritical
            logger.bind(error_type=type(exc).__name__).debug(
                "TTS jobs worker failed to write history record (job_id={}, request_id={})",
                job_id,
                request_id or "unknown",
            )

    def _emit_progress(percent: float, message: str, eta_seconds: float | None = None) -> None:
        if job_id <= 0:
            return
        with contextlib.suppress(_TTS_JOBS_NONCRITICAL_EXCEPTIONS):
            # Phase labels remain event-only so no stale worker can erase the
            # durable provider-dispatch marker with an unfenced progress write.
            jm.update_job_progress(job_id, progress_percent=percent)
        try:
            attrs = {"progress_percent": percent, "progress_message": message}
            if eta_seconds is not None:
                attrs["eta_seconds"] = max(0.0, float(eta_seconds))
            emit_job_event("job.progress", job={"id": job_id}, attrs=attrs)
        except _TTS_JOBS_NONCRITICAL_EXCEPTIONS:
            pass

    try:
        _emit_progress(5.0, "tts_started")
        if not _persist_tts_dispatch_marker(jm, job, job_id=job_id):
            raise TTSJobError(
                "TTS dispatch lease is no longer active",
                retryable=False,
                failure_code="tts_dispatch_lease_lost",
            )
        try:
            speech_iter = tts_service.generate_speech(
                request,
                provider=provider_hint,
                fallback=False,
                provider_overrides=provider_overrides,
                user_id=user_id_int,
            )
            audio_bytes = bytearray()
            credentials_marked_used = False
            last_update = start_ts
            expected_chars_per_sec = 15.0
            try:
                expected_chars_per_sec = float(
                    (request.extra_params or {}).get("audio_expected_chars_per_sec", expected_chars_per_sec)
                )
            except _TTS_JOBS_NONCRITICAL_EXCEPTIONS:
                expected_chars_per_sec = 15.0
            expected_sec = max(1.0, len(request.input or "") / max(1.0, expected_chars_per_sec))
            try:
                async for chunk in speech_iter:
                    if chunk:
                        audio_bytes.extend(chunk)
                        if credential_resolution is not None and not credentials_marked_used:
                            credentials_marked_used = True
                            await _mark_tts_credentials_used(credential_resolution)
                    now = asyncio.get_event_loop().time()
                    if now - last_update >= 1.0:
                        elapsed = now - start_ts
                        percent = min(80.0, (elapsed / expected_sec) * 80.0)
                        eta = max(0.0, expected_sec - elapsed)
                        _emit_progress(percent, "tts_synthesizing", eta_seconds=eta)
                        last_update = now
            finally:
                await _close_tts_speech_iter(speech_iter)
        except TTSError as exc:
            safe_failure = _bounded_tts_job_failure(exc)
        except Exception as exc:  # noqa: BLE001 - adapter boundary is sanitized below
            safe_failure = _bounded_tts_job_failure(exc)
        else:
            safe_failure = None
        if safe_failure is not None:
            if history_enabled:
                _record_history("failed", error_message=str(safe_failure))
            raise safe_failure

        if not audio_bytes:
            if history_enabled:
                _record_history("failed", error_message="empty_audio")
            raise TTSJobError("empty_audio", retryable=False)

        _emit_progress(85.0, "tts_synthesis_complete")
        filename = f"tts_job_{job.get('id')}.{request.response_format}"

        _emit_progress(92.0, "tts_writing_output")
        output_failure: TTSJobError | None = None
        try:
            outputs_dir = DatabasePaths.get_user_outputs_dir(user_id)
            with CollectionsDatabase.for_user(user_id=user_id) as cdb:
                safe_name = cdb.resolve_output_storage_path(filename)
                output_path = outputs_dir / safe_name
                output_path.write_bytes(bytes(audio_bytes))
                request_metadata = getattr(request, "_tts_metadata", None)
                if not isinstance(request_metadata, dict):
                    request_metadata = {}
                route = gateway_route_provenance(
                    requested_backend=request.backend,
                    requested_model=request.model,
                    metadata=request_metadata,
                )
                metadata = {
                    "artifact_type": "tts_audio",
                    "provider": route.get("actual_backend", provider_hint),
                    "model": route.get("actual_model", request.model),
                    "voice": request_metadata.get("voice") or request.voice,
                    "format": request.response_format,
                    **route,
                }
                row, output_incarnation = cdb.create_output_artifact_with_history_identity(
                    type_="tts_audio",
                    title=f"TTS Job {job.get('id')}",
                    format_=request.response_format,
                    storage_path=safe_name,
                    metadata_json=json.dumps(metadata),
                    job_id=int(job.get("id")),
                )
        except Exception as exc:  # noqa: BLE001 - persistence boundary is sanitized below
            logger.bind(error_type=type(exc).__name__).error(
                "TTS jobs worker output persistence failed"
            )
            if history_enabled:
                _record_history("failed", error_message="write_failed")
            output_failure = TTSJobError(
                "write_failed",
                retryable=False,
                failure_code="tts_output_persistence_failed",
            )
        if output_failure is not None:
            raise output_failure

        if history_enabled:
            artifact_ids: list[Any] | None = None
            if getattr(row, "id", None) is not None:
                artifact_ids = [f"output:{int(row.id)}"]
            _record_history(
                "success", output_id=row.id, output_incarnation=output_incarnation, artifact_ids=artifact_ids
            )

        _emit_progress(100.0, "tts_completed", eta_seconds=0.0)
        return {
            "output_id": row.id,
            "storage_path": row.storage_path,
            "format": row.format,
            "bytes": len(audio_bytes),
            **route,
        }
    finally:
        if history_db is not None:
            try:
                history_db.close_connection()
            except Exception as exc:  # noqa: BLE001 - history is explicitly noncritical
                logger.bind(error_type=type(exc).__name__).debug(
                    "TTS jobs worker failed to close history database"
                )


async def run_tts_jobs_worker(stop_event: asyncio.Event | None = None) -> None:
    worker_id = (os.getenv("TTS_JOBS_WORKER_ID") or f"tts-jobs-{os.getpid()}").strip()
    queue = (os.getenv("TTS_JOBS_QUEUE") or "default").strip() or "default"
    lease_seconds = _coerce_int(os.getenv("TTS_JOBS_LEASE_SECONDS") or os.getenv("JOBS_LEASE_SECONDS"), 60)
    renew_jitter = _coerce_int(
        os.getenv("TTS_JOBS_RENEW_JITTER_SECONDS")
        or os.getenv("JOBS_LEASE_RENEW_JITTER_SECONDS"),
        5,
    )
    renew_threshold = _coerce_int(
        os.getenv("TTS_JOBS_RENEW_THRESHOLD_SECONDS")
        or os.getenv("JOBS_LEASE_RENEW_THRESHOLD_SECONDS"),
        10,
    )
    cfg = WorkerConfig(
        domain=TTS_DOMAIN,
        queue=queue,
        worker_id=worker_id,
        lease_seconds=lease_seconds,
        renew_jitter_seconds=renew_jitter,
        renew_threshold_seconds=renew_threshold,
    )
    sdk = WorkerSDK(_jobs_manager(), cfg)
    _stop_watcher_task: asyncio.Task[None] | None = None

    if stop_event is not None:
        async def _watch_stop() -> None:
            await stop_event.wait()
            sdk.stop()

        _stop_watcher_task = asyncio.create_task(_watch_stop())

    logger.info("TTS Jobs worker starting (queue={}, worker_id={})", queue, worker_id)
    try:
        await sdk.run(handler=_handle_tts_job)
    finally:
        if _stop_watcher_task is not None and not _stop_watcher_task.done():
            _stop_watcher_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await _stop_watcher_task


async def main() -> None:
    """Bootstrap provider policy before accepting standalone TTS jobs."""
    await refresh_llm_provider_overrides(force=True)
    try:
        start_llm_provider_override_refresh_service()
        await run_tts_jobs_worker()
    finally:
        await shutdown_llm_provider_override_recovery()


if __name__ == "__main__":
    asyncio.run(main())
