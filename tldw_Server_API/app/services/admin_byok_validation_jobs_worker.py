from __future__ import annotations

import asyncio
import contextlib
import os
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from json import JSONDecodeError
from typing import Any, Protocol, TypedDict

from loguru import logger

from tldw_Server_API.app.core.AuthNZ.byok_helpers import load_server_config_snapshot
from tldw_Server_API.app.core.AuthNZ.byok_runtime import (
    ByokResolutionError,
    _extract_runtime_api_key,
    _extract_runtime_auth_source,
    merge_server_fallback_snapshot,
    resolve_static_server_fallback_from_snapshot,
)
from tldw_Server_API.app.core.AuthNZ.provider_credential_runtime import (
    configured_provider_model_from_snapshot,
)
from tldw_Server_API.app.core.AuthNZ.repos.users_repo import AuthnzUsersRepo
from tldw_Server_API.app.core.AuthNZ.user_provider_secrets import (
    decrypt_byok_payload,
    loads_envelope,
)
from tldw_Server_API.app.core.Chat.Chat_Deps import (
    ChatAuthenticationError,
    ChatBadRequestError,
    ChatProviderError,
    SanitizedProviderStreamError,
)
from tldw_Server_API.app.core.Chat.streaming_utils import (
    sanitized_provider_stream_exception,
)
from tldw_Server_API.app.core.exceptions import raise_detached_error
from tldw_Server_API.app.core.Jobs.manager import JobManager
from tldw_Server_API.app.core.Jobs.worker_sdk import WorkerConfig, WorkerSDK
from tldw_Server_API.app.core.testing import env_flag_enabled
from tldw_Server_API.app.services import admin_byok_service, admin_orgs_service

BYOK_VALIDATION_DOMAIN = "byok"
BYOK_VALIDATION_JOB_TYPE = "validation_sweep"


class ByokValidationCandidate(TypedDict, total=False):
    """One concrete BYOK credential candidate to validate."""

    provider: str
    api_key: str
    credential_fields: dict[str, Any] | None
    auth_source: str | None
    source: str
    scope_type: str
    scope_id: int
    user_id: int


@dataclass(frozen=True)
class CandidateLoadResult:
    """Candidates plus the number of skipped records encountered while loading."""

    candidates: list[ByokValidationCandidate]
    error_count: int = 0


def _validation_candidate_from_payload(
    payload: dict[str, Any],
    *,
    provider: str,
    source: str,
    scope_type: str | None = None,
    scope_id: int | None = None,
    user_id: int | None = None,
) -> ByokValidationCandidate:
    """Build one candidate using the credential runtime's source precedence."""
    if not isinstance(payload, dict):
        raise ValueError("Invalid BYOK credential payload")
    api_key = _extract_runtime_api_key(payload)
    auth_source = _extract_runtime_auth_source(
        payload,
        require_access_for_oauth=True,
    )
    if not api_key or not auth_source:
        raise ValueError("Invalid BYOK credential payload")
    credential_fields = payload.get("credential_fields")
    if credential_fields is not None and not isinstance(credential_fields, dict):
        raise ValueError("Invalid BYOK credential payload")

    candidate: ByokValidationCandidate = {
        "provider": provider,
        "api_key": api_key,
        "credential_fields": (
            dict(credential_fields) if credential_fields is not None else None
        ),
        "auth_source": auth_source,
        "source": source,
    }
    if scope_type is not None:
        candidate["scope_type"] = scope_type
    if scope_id is not None:
        candidate["scope_id"] = scope_id
    if user_id is not None:
        candidate["user_id"] = user_id
    return candidate


class SharedByokRepoProtocol(Protocol):
    """Subset of shared BYOK repo behavior used by validation loading."""

    async def list_secrets(
        self,
        *,
        scope_type: str | None = None,
        scope_id: int | None = None,
        provider: str | None = None,
    ) -> list[dict[str, Any]]:
        ...

    async def fetch_secret(
        self,
        scope_type: str,
        scope_id: int,
        provider: str,
    ) -> dict[str, Any] | None:
        ...


class UserByokRepoProtocol(Protocol):
    """Subset of per-user BYOK repo behavior used by validation loading."""

    async def list_secrets_for_user(self, user_id: int) -> list[dict[str, Any]]:
        ...

    async def fetch_secret_for_user(self, user_id: int, provider: str) -> dict[str, Any] | None:
        ...


class ValidationRunsRepoProtocol(Protocol):
    """Subset of run-repo behavior required by the Jobs worker."""

    async def get_run(self, run_id: str) -> dict[str, Any] | None:
        ...

    async def mark_running(self, run_id: str, *, job_id: str | None) -> dict[str, Any]:
        ...

    async def mark_complete(
        self,
        run_id: str,
        *,
        keys_checked: int,
        valid_count: int,
        invalid_count: int,
        error_count: int,
    ) -> dict[str, Any]:
        ...

    async def mark_failed(self, run_id: str, *, error_message: str) -> dict[str, Any]:
        ...


def byok_validation_queue() -> str:
    """Return the Jobs queue used for authoritative BYOK validation runs."""
    return (os.getenv("ADMIN_BYOK_VALIDATION_JOBS_QUEUE") or "default").strip() or "default"


def byok_validation_worker_enabled() -> bool:
    """Return True when the authoritative BYOK validation worker is enabled."""
    return env_flag_enabled("ADMIN_BYOK_VALIDATION_JOBS_WORKER_ENABLED")


def build_byok_validation_job_payload(*, run_id: str) -> dict[str, Any]:
    """Build the opaque Jobs payload for one BYOK validation run."""
    return {"run_id": str(run_id)}


def build_byok_validation_idempotency_key(*, run_id: str) -> str:
    """Return the Jobs idempotency key for one BYOK validation run enqueue."""
    return f"byok-validation:{run_id}"


async def _get_repo() -> ValidationRunsRepoProtocol:
    """Build the BYOK validation run repository for worker execution."""
    from tldw_Server_API.app.core.AuthNZ.database import get_db_pool
    from tldw_Server_API.app.core.AuthNZ.repos.byok_validation_runs_repo import (
        AuthnzByokValidationRunsRepo,
    )

    pool = await get_db_pool()
    repo = AuthnzByokValidationRunsRepo(pool)
    await repo.ensure_schema()
    return repo


def _per_provider_limit() -> int:
    """Return the max concurrent validation calls per provider."""
    from tldw_Server_API.app.core.AuthNZ.byok_testing import (
        provider_credential_validation_per_provider_capacity,
    )

    return provider_credential_validation_per_provider_capacity()


def _redact_validation_failure(exc: Exception) -> str:
    """Return a bounded redacted summary for a failed validation run."""
    if isinstance(exc, ChatAuthenticationError | ChatBadRequestError):
        return "invalid_credentials"
    if isinstance(exc, ChatProviderError):
        return "provider_validation_failed"
    return "provider_validation_failed"


def _validation_job_public_failure(exc: Exception) -> Exception:
    """Return a detached-safe failure for the Jobs finalizer."""
    if isinstance(exc, SanitizedProviderStreamError):
        return sanitized_provider_stream_exception(exc.code)
    if isinstance(exc, ByokResolutionError):
        return ByokResolutionError(exc.code, exc.provider)
    return sanitized_provider_stream_exception("provider_unavailable")


async def _mark_validation_failed_best_effort(
    repo: ValidationRunsRepoProtocol,
    run_id: str,
    exc: Exception,
) -> None:
    """Record a bounded run failure without replacing its public exception."""
    try:
        await repo.mark_failed(
            run_id,
            error_message=_redact_validation_failure(exc),
        )
    except Exception as mark_exc:  # noqa: BLE001
        logger.warning(
            "BYOK validation failure status update failed: run_id={} error_type={}",
            run_id,
            type(mark_exc).__name__,
        )


async def enqueue_byok_validation_run(
    run: dict[str, Any],
    *,
    job_manager: JobManager | None = None,
) -> str:
    """Enqueue one authoritative BYOK validation run into Jobs."""
    jobs = job_manager or JobManager()
    job = jobs.create_job(
        domain=BYOK_VALIDATION_DOMAIN,
        queue=byok_validation_queue(),
        job_type=BYOK_VALIDATION_JOB_TYPE,
        payload=build_byok_validation_job_payload(run_id=str(run["id"])),
        owner_user_id=(
            str(run["requested_by_user_id"]) if run.get("requested_by_user_id") is not None else None
        ),
        idempotency_key=build_byok_validation_idempotency_key(run_id=str(run["id"])),
    )
    return str(job.get("id"))


async def _load_team_scoped_shared_candidates(
    *,
    org_id: int,
    provider: str | None,
    shared_repo: SharedByokRepoProtocol,
) -> CandidateLoadResult:
    """Load team-scoped shared key candidates for one organization."""
    items: list[ByokValidationCandidate] = []
    error_count = 0
    offset = 0
    limit = 200
    while True:
        teams = await admin_orgs_service.list_teams_by_org(org_id, limit=limit, offset=offset)
        if not teams:
            break
        for team in teams:
            team_id = int(team["id"])
            team_rows = await shared_repo.list_secrets(
                scope_type="team",
                scope_id=team_id,
                provider=provider,
            )
            for row in team_rows:
                full_row = await shared_repo.fetch_secret("team", team_id, str(row["provider"]))
                if not full_row or not full_row.get("encrypted_blob"):
                    continue
                try:
                    payload = decrypt_byok_payload(loads_envelope(str(full_row["encrypted_blob"])))
                    candidate = _validation_candidate_from_payload(
                        payload,
                        provider=str(row["provider"]),
                        source="shared",
                        scope_type="team",
                        scope_id=team_id,
                    )
                except (JSONDecodeError, TypeError, ValueError) as exc:
                    error_count += 1
                    logger.warning(
                        "Skipping unreadable BYOK validation candidate: provider={} source=shared scope_type=team scope_id={} error_type={}",
                        row["provider"],
                        team_id,
                        type(exc).__name__,
                    )
                    continue
                items.append(candidate)
        if len(teams) < limit:
            break
        offset += limit
    return CandidateLoadResult(candidates=items, error_count=error_count)


async def load_default_validation_candidates(run: dict[str, Any]) -> CandidateLoadResult:
    """Load shared and per-user BYOK validation candidates for one run scope."""
    provider = str(run.get("provider") or "").strip() or None
    org_id = int(run["org_id"]) if run.get("org_id") is not None else None

    shared_repo: SharedByokRepoProtocol = await admin_byok_service.get_shared_byok_repo()
    user_repo: UserByokRepoProtocol = await admin_byok_service.get_user_byok_repo()
    users_repo = await AuthnzUsersRepo.from_pool()

    candidates: list[ByokValidationCandidate] = []
    error_count = 0

    if org_id is not None:
        shared_rows = await shared_repo.list_secrets(
            scope_type="org",
            scope_id=org_id,
            provider=provider,
        )
        team_load_result = await _load_team_scoped_shared_candidates(
            org_id=org_id,
            provider=provider,
            shared_repo=shared_repo,
        )
    else:
        shared_rows = await shared_repo.list_secrets(provider=provider)
        team_load_result = CandidateLoadResult(candidates=[], error_count=0)

    for row in shared_rows:
        scope_type = str(row["scope_type"])
        scope_id = int(row["scope_id"])
        full_row = await shared_repo.fetch_secret(scope_type, scope_id, str(row["provider"]))
        if not full_row or not full_row.get("encrypted_blob"):
            continue
        try:
            payload = decrypt_byok_payload(loads_envelope(str(full_row["encrypted_blob"])))
            candidate = _validation_candidate_from_payload(
                payload,
                provider=str(row["provider"]),
                source="shared",
                scope_type=scope_type,
                scope_id=scope_id,
            )
        except (JSONDecodeError, TypeError, ValueError) as exc:
            error_count += 1
            logger.warning(
                "Skipping unreadable BYOK validation candidate: provider={} source=shared scope_type={} scope_id={} error_type={}",
                row["provider"],
                scope_type,
                scope_id,
                type(exc).__name__,
            )
            continue
        candidates.append(candidate)
    candidates.extend(team_load_result.candidates)
    error_count += team_load_result.error_count

    offset = 0
    limit = 200
    while True:
        users, total = await users_repo.list_users(
            offset=offset,
            limit=limit,
            org_ids=[org_id] if org_id is not None else None,
        )
        if not users:
            break
        for user in users:
            user_id = int(user["id"])
            user_rows = await user_repo.list_secrets_for_user(user_id)
            for row in user_rows:
                row_provider = str(row["provider"])
                if provider is not None and row_provider != provider:
                    continue
                full_row = await user_repo.fetch_secret_for_user(user_id, row_provider)
                if not full_row or not full_row.get("encrypted_blob"):
                    continue
                try:
                    payload = decrypt_byok_payload(loads_envelope(str(full_row["encrypted_blob"])))
                    candidate = _validation_candidate_from_payload(
                        payload,
                        provider=row_provider,
                        source="user",
                        user_id=user_id,
                    )
                except (JSONDecodeError, TypeError, ValueError) as exc:
                    error_count += 1
                    logger.warning(
                        "Skipping unreadable BYOK validation candidate: provider={} source=user user_id={} error_type={}",
                        row_provider,
                        user_id,
                        type(exc).__name__,
                    )
                    continue
                candidates.append(candidate)
        offset += limit
        if offset >= total:
            break

    return CandidateLoadResult(candidates=candidates, error_count=error_count)


def _normalize_candidate_load_result(
    load_result: CandidateLoadResult | list[ByokValidationCandidate],
) -> CandidateLoadResult:
    """Normalize legacy loader outputs into the worker's structured load result."""
    if isinstance(load_result, CandidateLoadResult):
        return load_result
    return CandidateLoadResult(candidates=load_result, error_count=0)


async def _run_validation_scan(
    candidates: list[ByokValidationCandidate],
    *,
    test_provider_credentials_fn: Callable[..., Awaitable[Any]],
    initial_error_count: int = 0,
    max_workers: int | None = None,
    per_provider_limit: int | None = None,
    server_config_snapshot: Mapping[str, Any] | None = None,
) -> dict[str, int]:
    """Validate candidate credentials with bounded concurrency per provider."""
    if not candidates:
        return {
            "keys_checked": 0,
            "valid_count": 0,
            "invalid_count": 0,
            "error_count": initial_error_count,
        }

    if per_provider_limit is None:
        provider_limit = _per_provider_limit()
    else:
        try:
            provider_limit = min(8, max(1, int(per_provider_limit)))
        except (TypeError, ValueError):
            provider_limit = _per_provider_limit()
    semaphores: dict[str, asyncio.Semaphore] = {}
    counts = {
        "keys_checked": len(candidates),
        "valid_count": 0,
        "invalid_count": 0,
        "error_count": initial_error_count,
    }

    async def _validate_candidate(candidate: ByokValidationCandidate) -> str:
        provider = str(candidate["provider"])
        semaphore = semaphores.setdefault(provider, asyncio.Semaphore(provider_limit))
        async with semaphore:
            try:
                validation_kwargs: dict[str, Any] = {
                    "provider": provider,
                    "api_key": str(candidate["api_key"]),
                    "credential_fields": candidate.get("credential_fields"),
                    "model": None,
                }
                if server_config_snapshot is not None:
                    base_fallback = resolve_static_server_fallback_from_snapshot(
                        provider,
                        server_config_snapshot,
                    )
                    candidate_fields = candidate.get("credential_fields")
                    if candidate_fields is None:
                        candidate_fields = {}
                    candidate_fallback = merge_server_fallback_snapshot(
                        provider,
                        base_fallback,
                        api_key=str(candidate["api_key"]),
                        credential_fields=candidate_fields,
                        auth_source=candidate.get("auth_source"),
                        provider_config={},
                    )
                    app_config = dict(candidate_fallback.app_config or {})
                    model = configured_provider_model_from_snapshot(
                        provider,
                        app_config,
                    )
                    if not model:
                        raise ChatProviderError(
                            provider=provider,
                            message="Provider validation configuration is unavailable",
                            status_code=503,
                        )
                    validation_kwargs.update(
                        {
                            "api_key": candidate_fallback.api_key,
                            "credential_fields": dict(
                                candidate_fallback.credential_fields
                            ),
                            "app_config": app_config,
                            "model": model,
                            "include_override_model": False,
                        }
                    )
                await test_provider_credentials_fn(**validation_kwargs)
                return "valid"
            except (ChatAuthenticationError, ChatBadRequestError):
                return "invalid"
            except SanitizedProviderStreamError as exc:
                if exc.code in {
                    "provider_authentication_failed",
                    "provider_configuration_invalid",
                }:
                    return "invalid"
                raise

    queue: asyncio.Queue[ByokValidationCandidate | None] = asyncio.Queue()
    for candidate in candidates:
        queue.put_nowait(candidate)

    provider_count = len({str(candidate["provider"]) for candidate in candidates})
    worker_count = min(
        len(candidates),
        max(1, max_workers or (provider_limit * max(1, provider_count))),
    )
    for _ in range(worker_count):
        queue.put_nowait(None)

    async def _worker() -> None:
        while True:
            candidate = await queue.get()
            try:
                if candidate is None:
                    return
                status = await _validate_candidate(candidate)
                if status == "valid":
                    counts["valid_count"] += 1
                else:
                    counts["invalid_count"] += 1
            finally:
                queue.task_done()

    tasks = [asyncio.create_task(_worker()) for _ in range(worker_count)]
    try:
        await asyncio.gather(*tasks)
    except Exception:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise

    return counts


async def handle_byok_validation_job(
    job: dict[str, Any],
    *,
    repo: ValidationRunsRepoProtocol | None = None,
    candidate_loader: Callable[
        [dict[str, Any]],
        Awaitable[CandidateLoadResult | list[ByokValidationCandidate]],
    ]
    | None = None,
    test_provider_credentials_fn: Callable[..., Awaitable[Any]] | None = None,
) -> dict[str, Any]:
    """Execute one authoritative BYOK validation run from the Jobs queue."""
    from tldw_Server_API.app.core.AuthNZ.byok_testing import test_provider_credentials

    payload = job.get("payload") or {}
    run_id = str(payload.get("run_id") or "").strip()
    if not run_id:
        raise ValueError("missing_run_id")

    try:
        repo = repo or await _get_repo()
    except Exception as exc:  # noqa: BLE001 - sanitize repository initialization failures
        raise_detached_error(_validation_job_public_failure(exc))

    try:
        run = await repo.get_run(run_id)
    except Exception as exc:  # noqa: BLE001 - persist bounded failure across the Jobs boundary
        await _mark_validation_failed_best_effort(repo, run_id, exc)
        raise_detached_error(_validation_job_public_failure(exc))
    if not run:
        raise ValueError("missing_run")

    job_id = str(job.get("id")) if job.get("id") is not None else None
    loader = candidate_loader or load_default_validation_candidates
    validator = test_provider_credentials_fn or test_provider_credentials

    try:
        await repo.mark_running(run_id, job_id=job_id)
        load_result = _normalize_candidate_load_result(await loader(run))
        server_config_snapshot = (
            load_server_config_snapshot()
            if validator is test_provider_credentials
            else None
        )
        summary = await _run_validation_scan(
            load_result.candidates,
            test_provider_credentials_fn=validator,
            initial_error_count=load_result.error_count,
            server_config_snapshot=server_config_snapshot,
        )
        await repo.mark_complete(
            run_id,
            keys_checked=int(summary["keys_checked"]),
            valid_count=int(summary["valid_count"]),
            invalid_count=int(summary["invalid_count"]),
            error_count=int(summary["error_count"]),
        )
    except Exception as exc:
        await _mark_validation_failed_best_effort(repo, run_id, exc)
        raise_detached_error(_validation_job_public_failure(exc))

    logger.info(
        "BYOK validation job completed: run_id={} job_id={} keys_checked={} valid={} invalid={} errors={}",
        run_id,
        job_id,
        summary["keys_checked"],
        summary["valid_count"],
        summary["invalid_count"],
        summary["error_count"],
    )
    return {
        "status": "complete",
        "run_id": run_id,
        "job_id": job_id,
        "keys_checked": int(summary["keys_checked"]),
        "valid_count": int(summary["valid_count"]),
        "invalid_count": int(summary["invalid_count"]),
        "error_count": int(summary["error_count"]),
    }


async def run_admin_byok_validation_jobs_worker(stop_event: asyncio.Event | None = None) -> None:
    """Run the WorkerSDK loop for authoritative BYOK validation jobs."""
    worker_id = (
        os.getenv("ADMIN_BYOK_VALIDATION_JOBS_WORKER_ID") or f"admin-byok-validation-{os.getpid()}"
    ).strip()
    cfg = WorkerConfig(
        domain=BYOK_VALIDATION_DOMAIN,
        queue=byok_validation_queue(),
        worker_id=worker_id,
    )
    jm = JobManager()
    sdk = WorkerSDK(jm, cfg)
    stop_task: asyncio.Task[None] | None = None
    if stop_event is not None:
        async def _watch_stop() -> None:
            await stop_event.wait()
            sdk.stop()

        stop_task = asyncio.create_task(
            _watch_stop(),
            name="admin_byok_validation_jobs_worker_stop_watch",
        )
    logger.info(
        "Admin BYOK validation Jobs worker starting: queue={} worker_id={}",
        cfg.queue,
        worker_id,
    )
    try:
        await sdk.run(handler=handle_byok_validation_job)
    finally:
        if stop_task is not None:
            stop_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await stop_task


async def start_admin_byok_validation_jobs_worker(
    stop_event: asyncio.Event | None = None,
) -> asyncio.Task | None:
    """Start the BYOK validation Jobs worker when explicitly enabled."""
    if not byok_validation_worker_enabled():
        return None
    return asyncio.create_task(
        run_admin_byok_validation_jobs_worker(stop_event),
        name="admin_byok_validation_jobs_worker",
    )


__all__ = [
    "BYOK_VALIDATION_DOMAIN",
    "BYOK_VALIDATION_JOB_TYPE",
    "_run_validation_scan",
    "build_byok_validation_idempotency_key",
    "build_byok_validation_job_payload",
    "byok_validation_queue",
    "byok_validation_worker_enabled",
    "enqueue_byok_validation_run",
    "handle_byok_validation_job",
    "load_default_validation_candidates",
    "run_admin_byok_validation_jobs_worker",
    "start_admin_byok_validation_jobs_worker",
]
