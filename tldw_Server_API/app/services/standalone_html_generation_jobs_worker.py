"""Jobs handler for receipt-backed standalone HTML presentation generation."""

from __future__ import annotations

import asyncio
import hmac
import inspect
import json
import os
import re
from collections.abc import Awaitable, Callable, Mapping
from datetime import datetime, timedelta, timezone
from typing import Any

from loguru import logger

from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths
from tldw_Server_API.app.core.Jobs.manager import JobManager
from tldw_Server_API.app.core.Jobs.worker_sdk import (
    WorkerConfig,
    WorkerSDK,
    WorkerTerminalOutcome,
)
from tldw_Server_API.app.core.Slides.slides_db import (
    ConflictError,
    SlidesDatabase,
    SlidesGenerationReceiptRow,
)
from tldw_Server_API.app.core.Slides.standalone_html_config import (
    ResolvedExecutionTarget,
    SlidesStandaloneHtmlConfig,
)
from tldw_Server_API.app.core.Slides.standalone_html_contracts import (
    StandaloneHtmlValidationError,
)
from tldw_Server_API.app.core.Slides.standalone_html_provider import (
    StandaloneHtmlProviderError,
    generate_standalone_html,
)
from tldw_Server_API.app.core.Slides.standalone_html_registry import (
    DigestKeySnapshot,
    DigestKeyUnavailableError,
    StandaloneHtmlHmacKeyring,
)
from tldw_Server_API.app.core.Slides.standalone_html_service import (
    JOB_DOMAIN,
    JOB_QUEUE,
    JOB_TYPE,
    StandaloneHtmlGenerationError,
    StandaloneHtmlGenerationService,
    build_generation_user_content,
    expected_job_payload,
)
from tldw_Server_API.app.core.Slides.standalone_html_validation_pool import (
    StandaloneHtmlValidationPool,
)

_JOBS_IDEMPOTENCY_KEY_RE = re.compile(r"slides:v1:[0-9a-f]{64}\Z")


class StandaloneHtmlGenerationRetry(RuntimeError):
    """Typed retryable precommit failure consumed by WorkerSDK."""

    retryable = True

    def __init__(
        self,
        code: str,
        *,
        backoff_seconds: int = 1,
    ) -> None:
        self.failure_code = code
        self.backoff_seconds = backoff_seconds
        super().__init__(code)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


def _safe_now(now: Callable[[], datetime]) -> datetime:
    value = now().replace(microsecond=0)
    if value.tzinfo is None or value.utcoffset() != timedelta(0):
        raise RuntimeError("generation clock must return aware UTC")
    return value


async def _load_digest_snapshot(
    loader: Callable[[], Awaitable[DigestKeySnapshot]],
) -> DigestKeySnapshot:
    try:
        snapshot = await loader()
        if not isinstance(snapshot, DigestKeySnapshot):
            raise DigestKeyUnavailableError("invalid digest registry snapshot")
        snapshot.require_generation_ready()
    except asyncio.CancelledError:
        raise
    except Exception:  # noqa: BLE001 - collapse registry implementation detail
        raise DigestKeyUnavailableError("generation digest key unavailable") from None
    return snapshot


def _parse_utc(value: object) -> datetime | None:
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    else:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _normalized_payload(
    job_manager: JobManager,
    job: Mapping[str, Any],
) -> dict[str, Any]:
    try:
        payload = job_manager._maybe_decrypt_json(job_manager._parse_json_value(job.get("payload")))
    except Exception:  # noqa: BLE001 - reduce the Jobs boundary to one code
        raise StandaloneHtmlGenerationError(
            "generation_correlation_mismatch",
            status_code=409,
        ) from None
    if not isinstance(payload, dict):
        raise StandaloneHtmlGenerationError(
            "generation_correlation_mismatch",
            status_code=409,
        )
    return payload


def _normalized_job(
    job_manager: JobManager,
    job: Mapping[str, Any],
) -> dict[str, Any]:
    return {**dict(job), "payload": _normalized_payload(job_manager, job)}


def _receipt_id(job: Mapping[str, Any]) -> str:
    payload = job.get("payload")
    if not isinstance(payload, dict) or set(payload) != {"receipt_id"}:
        raise StandaloneHtmlGenerationError(
            "generation_correlation_mismatch",
            status_code=409,
        )
    receipt_id = payload.get("receipt_id")
    if not isinstance(receipt_id, str) or payload != expected_job_payload(receipt_id):
        raise StandaloneHtmlGenerationError(
            "generation_correlation_mismatch",
            status_code=409,
        )
    return receipt_id


def _owner(job: Mapping[str, Any]) -> str:
    owner = job.get("owner_user_id")
    if not isinstance(owner, str) or not owner.strip() or owner != owner.strip():
        raise StandaloneHtmlGenerationError(
            "generation_correlation_mismatch",
            status_code=409,
        )
    return owner


def _job_scope_is_exact(job: Mapping[str, Any]) -> bool:
    return job.get("domain") == JOB_DOMAIN and job.get("queue") == JOB_QUEUE and job.get("job_type") == JOB_TYPE


def _target_from_input(generation_input: Any) -> ResolvedExecutionTarget:
    return ResolvedExecutionTarget(
        provider=generation_input.provider,
        model=generation_input.model,
        adapter_id=generation_input.adapter_id,
        endpoint_identity=generation_input.endpoint_identity,
    )


def _target_failure_code(
    target: ResolvedExecutionTarget,
    current_config_loader: Callable[[], SlidesStandaloneHtmlConfig],
) -> str | None:
    try:
        current = current_config_loader()
    except Exception:  # noqa: BLE001 - configuration errors are source-free
        return "standalone_html_egress_disabled"
    if not isinstance(current, SlidesStandaloneHtmlConfig) or not current.feature_enabled or not current.egress_enabled:
        return "standalone_html_egress_disabled"
    if target in current.allowed_targets:
        return None
    if any(
        candidate.provider == target.provider
        and candidate.adapter_id == target.adapter_id
        and candidate.endpoint_identity == target.endpoint_identity
        for candidate in current.allowed_targets
    ):
        return "standalone_html_model_not_allowed"
    return "standalone_html_endpoint_not_allowed"


def _completed_metadata(
    service: StandaloneHtmlGenerationService,
    receipt: SlidesGenerationReceiptRow,
) -> dict[str, Any]:
    if not receipt.presentation_id:
        raise StandaloneHtmlGenerationError(
            "generation_correlation_mismatch",
            status_code=409,
        )
    try:
        presentation = service.slides_db.get_presentation_by_id(
            receipt.presentation_id,
            include_deleted=True,
        )
    except KeyError:
        raise StandaloneHtmlGenerationError(
            "generation_correlation_mismatch",
            status_code=409,
        ) from None
    if presentation.generation_job_uuid != receipt.job_uuid:
        raise StandaloneHtmlGenerationError(
            "generation_correlation_mismatch",
            status_code=409,
        )
    return {
        "presentation_id": presentation.id,
        "content_kind": "standalone_html",
        "html_bytes": presentation.html_bytes,
        "html_slide_count": presentation.html_slide_count,
        "validation_status": "accepted",
    }


def _terminal_outcome(
    service: StandaloneHtmlGenerationService,
    receipt: SlidesGenerationReceiptRow,
) -> WorkerTerminalOutcome:
    stored_code = receipt.error_code
    stored_code_is_safe = (
        isinstance(stored_code, str) and re.fullmatch(r"[a-z][a-z0-9_.-]{0,127}", stored_code) is not None
    )
    safe_code = stored_code if stored_code_is_safe else "generation_failed"
    if receipt.receipt_status == "cancelled":
        return WorkerTerminalOutcome(
            status="cancelled",
            error_code=stored_code if stored_code_is_safe else "generation_cancelled",
            message="Generation was cancelled.",
        )
    return WorkerTerminalOutcome(
        status="failed",
        error_code=safe_code,
        message="Generation failed.",
    )


def _terminalize(
    service: StandaloneHtmlGenerationService,
    receipt: SlidesGenerationReceiptRow,
    *,
    status: str,
    code: str,
    message: str,
    now: datetime,
) -> dict[str, Any] | WorkerTerminalOutcome:
    try:
        applied = service.terminalize(
            receipt=receipt,
            status=status,
            error_code=code,
            error_message=message,
            terminal_at=now,
        )
    except Exception:  # noqa: BLE001 - source-free retry boundary
        raise StandaloneHtmlGenerationRetry("generation_store_unavailable") from None
    if applied:
        return WorkerTerminalOutcome(
            status=status,
            error_code=code,
            message=message,
        )
    try:
        winner = service.slides_db.get_generation_receipt(
            receipt.id,
            owner_user_id=receipt.owner_user_id,
        )
    except KeyError:
        raise StandaloneHtmlGenerationError(
            "generation_correlation_mismatch",
            status_code=409,
        ) from None
    if winner.job_uuid != receipt.job_uuid:
        raise StandaloneHtmlGenerationError(
            "generation_correlation_mismatch",
            status_code=409,
        )
    if winner.receipt_status == "completed":
        return _completed_metadata(service, winner)
    if winner.receipt_status in {"failed", "cancelled"}:
        return _terminal_outcome(service, winner)
    raise StandaloneHtmlGenerationError(
        "generation_correlation_mismatch",
        status_code=409,
    )


def _retry_budget_exhausted(job: Mapping[str, Any]) -> bool:
    retry_count = job.get("retry_count")
    max_retries = job.get("max_retries")
    return (
        isinstance(retry_count, int)
        and not isinstance(retry_count, bool)
        and isinstance(max_retries, int)
        and not isinstance(max_retries, bool)
        and retry_count >= max_retries
    )


def _would_quarantine(job: Mapping[str, Any], code: str) -> bool:
    try:
        threshold = int(os.getenv("JOBS_QUARANTINE_THRESHOLD", "2") or "2")
    except (TypeError, ValueError):
        threshold = 2
    threshold = max(1, threshold)
    streak_code = job.get("failure_streak_code")
    streak_count = job.get("failure_streak_count")
    next_count = (
        int(streak_count) + 1
        if streak_code == code
        and isinstance(streak_count, int)
        and not isinstance(streak_count, bool)
        and streak_count >= 0
        else 1
    )
    return next_count >= threshold


def _retry(
    service: StandaloneHtmlGenerationService,
    receipt: SlidesGenerationReceiptRow,
    job: Mapping[str, Any],
    *,
    code: str,
    now: datetime,
) -> dict[str, Any] | WorkerTerminalOutcome:
    if _retry_budget_exhausted(job):
        return _terminalize(
            service,
            receipt,
            status="failed",
            code="generation_retry_exhausted",
            message="Generation retry budget was exhausted.",
            now=now,
        )
    if _would_quarantine(job, code):
        return _terminalize(
            service,
            receipt,
            status="failed",
            code="generation_quarantined",
            message="Generation was quarantined.",
            now=now,
        )
    try:
        reset = receipt.job_uuid is not None and service.slides_db.reset_generation_receipt_queued(
            receipt_id=receipt.id,
            owner_user_id=receipt.owner_user_id,
            job_uuid=receipt.job_uuid,
            error_code=code,
            error_message="Generation will retry.",
            updated_at=now.isoformat(),
        )
    except Exception:  # noqa: BLE001 - source-free retry boundary
        raise StandaloneHtmlGenerationRetry("generation_store_unavailable") from None
    if not reset:
        try:
            winner = service.slides_db.get_generation_receipt(
                receipt.id,
                owner_user_id=receipt.owner_user_id,
            )
        except KeyError:
            raise StandaloneHtmlGenerationError(
                "generation_correlation_mismatch",
                status_code=409,
            ) from None
        if winner.job_uuid != receipt.job_uuid:
            raise StandaloneHtmlGenerationError(
                "generation_correlation_mismatch",
                status_code=409,
            )
        if winner.receipt_status == "completed":
            return _completed_metadata(service, winner)
        if winner.receipt_status in {"failed", "cancelled"}:
            return _terminal_outcome(service, winner)
        raise StandaloneHtmlGenerationError(
            "generation_correlation_mismatch",
            status_code=409,
        )
    raise StandaloneHtmlGenerationRetry(code)


def _job_identity_is_exact(
    candidate: Mapping[str, Any],
    acquired_job: Mapping[str, Any],
) -> bool:
    candidate_jobs_key = candidate.get("idempotency_key")
    acquired_jobs_key = acquired_job.get("idempotency_key")
    jobs_key_matches = bool(
        isinstance(candidate_jobs_key, str)
        and _JOBS_IDEMPOTENCY_KEY_RE.fullmatch(candidate_jobs_key) is not None
        and isinstance(acquired_jobs_key, str)
        and _JOBS_IDEMPOTENCY_KEY_RE.fullmatch(acquired_jobs_key) is not None
        and hmac.compare_digest(candidate_jobs_key, acquired_jobs_key)
    )
    return (
        _job_scope_is_exact(candidate)
        and candidate.get("id") == acquired_job.get("id")
        and candidate.get("uuid") == acquired_job.get("uuid")
        and candidate.get("owner_user_id") == acquired_job.get("owner_user_id")
        and jobs_key_matches
        and candidate.get("payload") == acquired_job.get("payload")
    )


def _final_job_is_live(
    final_job: Mapping[str, Any],
    acquired_job: Mapping[str, Any],
    *,
    now: datetime,
) -> bool:
    lease_deadline = _parse_utc(final_job.get("leased_until"))
    return (
        _job_identity_is_exact(final_job, acquired_job)
        and final_job.get("status") == "processing"
        and final_job.get("worker_id") == acquired_job.get("worker_id")
        and final_job.get("lease_id") == acquired_job.get("lease_id")
        and lease_deadline is not None
        and lease_deadline > now
        and final_job.get("cancel_requested_at") is None
        and final_job.get("cancelled_at") is None
    )


def make_generation_acquire_guard(
    digest_snapshot_loader: Callable[[], Awaitable[DigestKeySnapshot]],
) -> Callable[[dict[str, Any]], Any]:
    """Build the source-free WorkerSDK gate used while key material is absent."""

    async def guard(_job: dict[str, Any]) -> bool:
        try:
            await _load_digest_snapshot(digest_snapshot_loader)
        except DigestKeyUnavailableError:
            return False
        return True

    return guard


def _release_for_missing_key(
    job_manager: JobManager,
    job: Mapping[str, Any],
) -> None:
    job_id = job.get("id")
    worker_id = job.get("worker_id")
    lease_id = job.get("lease_id")
    if (
        isinstance(job_id, bool)
        or not isinstance(job_id, int)
        or not isinstance(worker_id, str)
        or not isinstance(lease_id, str)
    ):
        raise StandaloneHtmlGenerationError(
            "generation_correlation_mismatch",
            status_code=409,
        )
    try:
        job_manager.release_job(
            job_id,
            worker_id=worker_id,
            lease_id=lease_id,
            reason="generation_digest_key_unavailable",
            enforce=True,
        )
    except Exception:  # noqa: BLE001 - bounded control-plane failure
        raise StandaloneHtmlGenerationRetry("generation_digest_key_unavailable") from None
    raise StandaloneHtmlGenerationRetry("generation_digest_key_unavailable")


def _reset_and_release_for_missing_key(
    service: StandaloneHtmlGenerationService,
    receipt: SlidesGenerationReceiptRow,
    job_manager: JobManager,
    job: Mapping[str, Any],
    *,
    now: datetime,
) -> dict[str, Any] | WorkerTerminalOutcome:
    if receipt.job_uuid is None:
        raise StandaloneHtmlGenerationError(
            "generation_correlation_mismatch",
            status_code=409,
        )
    try:
        reset = service.slides_db.reset_generation_receipt_queued(
            receipt_id=receipt.id,
            owner_user_id=receipt.owner_user_id,
            job_uuid=receipt.job_uuid,
            error_code="generation_digest_key_unavailable",
            error_message="Generation will retry.",
            updated_at=now.isoformat(),
        )
    except Exception:  # noqa: BLE001 - source-free retry boundary
        raise StandaloneHtmlGenerationRetry("generation_store_unavailable") from None
    if not reset:
        try:
            winner = service.slides_db.get_generation_receipt(
                receipt.id,
                owner_user_id=receipt.owner_user_id,
            )
        except KeyError:
            raise StandaloneHtmlGenerationError(
                "generation_correlation_mismatch",
                status_code=409,
            ) from None
        if winner.job_uuid != receipt.job_uuid:
            raise StandaloneHtmlGenerationError(
                "generation_correlation_mismatch",
                status_code=409,
            )
        if winner.receipt_status == "completed":
            return _completed_metadata(service, winner)
        if winner.receipt_status in {"failed", "cancelled"}:
            return _terminal_outcome(service, winner)
        raise StandaloneHtmlGenerationError(
            "generation_correlation_mismatch",
            status_code=409,
        )
    _release_for_missing_key(job_manager, job)


def _cancel_requested(job: Mapping[str, Any]) -> bool:
    return (
        job.get("status") == "cancelled"
        or job.get("cancel_requested_at") is not None
        or job.get("cancelled_at") is not None
    )


def _terminal_jobs_outcome(
    service: StandaloneHtmlGenerationService,
    receipt: SlidesGenerationReceiptRow,
    job: Mapping[str, Any],
    *,
    now: datetime,
) -> dict[str, Any] | WorkerTerminalOutcome | None:
    if _cancel_requested(job):
        return _terminalize(
            service,
            receipt,
            status="cancelled",
            code="generation_cancelled",
            message="Generation was cancelled.",
            now=now,
        )
    status = job.get("status")
    if status == "quarantined":
        return _terminalize(
            service,
            receipt,
            status="failed",
            code="generation_quarantined",
            message="Generation was quarantined.",
            now=now,
        )
    if status in {"failed", "completed"}:
        return _terminalize(
            service,
            receipt,
            status="failed",
            code="generation_job_terminal",
            message="Generation job became terminal.",
            now=now,
        )
    return None


def _fence_job(
    service: StandaloneHtmlGenerationService,
    receipt: SlidesGenerationReceiptRow,
    job_manager: JobManager,
    acquired_job: Mapping[str, Any],
    *,
    now: datetime,
) -> dict[str, Any] | WorkerTerminalOutcome | None:
    """Require exact immutable identity plus one live, uncancelled lease."""
    try:
        candidate = job_manager.get_job_by_uuid(str(acquired_job.get("uuid") or ""))
    except Exception:  # noqa: BLE001 - source-free Jobs retry boundary
        return _retry(
            service,
            receipt,
            acquired_job,
            code="generation_jobs_unavailable",
            now=now,
        )
    if candidate is None:
        return _retry(
            service,
            receipt,
            acquired_job,
            code="generation_job_state_changed",
            now=now,
        )
    try:
        candidate = _normalized_job(job_manager, candidate)
    except StandaloneHtmlGenerationError:
        return _terminalize(
            service,
            receipt,
            status="failed",
            code="generation_correlation_mismatch",
            message="Generation correlation failed.",
            now=now,
        )
    if not _job_identity_is_exact(candidate, acquired_job):
        return _terminalize(
            service,
            receipt,
            status="failed",
            code="generation_correlation_mismatch",
            message="Generation correlation failed.",
            now=now,
        )
    terminal = _terminal_jobs_outcome(service, receipt, candidate, now=now)
    if terminal is not None:
        return terminal
    if not _final_job_is_live(candidate, acquired_job, now=now):
        return _retry(
            service,
            receipt,
            acquired_job,
            code="generation_job_state_changed",
            now=now,
        )
    return None


async def process_standalone_html_generation_job(
    job: dict[str, Any],
    *,
    job_manager: JobManager,
    slides_db_factory: Callable[[str], SlidesDatabase],
    keyring: StandaloneHtmlHmacKeyring,
    digest_snapshot_loader: Callable[[], Awaitable[DigestKeySnapshot]],
    validation_pool: StandaloneHtmlValidationPool,
    current_config_loader: Callable[[], SlidesStandaloneHtmlConfig],
    provider_api_key_loader: Callable[[ResolvedExecutionTarget], str | None],
    provider_generate: Callable[..., Any] = generate_standalone_html,
    now: Callable[[], datetime] = _utc_now,
) -> dict[str, Any] | WorkerTerminalOutcome:
    """Correlate, generate, validate, fence, and atomically commit one job."""
    current_time = _safe_now(now)
    if not isinstance(job, dict) or not _job_scope_is_exact(job):
        raise StandaloneHtmlGenerationError(
            "generation_correlation_mismatch",
            status_code=409,
        )
    job = _normalized_job(job_manager, job)
    try:
        start_snapshot = await _load_digest_snapshot(digest_snapshot_loader)
    except DigestKeyUnavailableError:
        _release_for_missing_key(job_manager, job)
    receipt_id = _receipt_id(job)
    owner = _owner(job)
    try:
        slides_db = slides_db_factory(owner)
    except Exception:  # noqa: BLE001 - reduce storage failures to one code
        raise StandaloneHtmlGenerationRetry("generation_store_unavailable") from None
    if not isinstance(slides_db, SlidesDatabase):
        raise StandaloneHtmlGenerationError(
            "generation_correlation_mismatch",
            status_code=409,
        )
    service = StandaloneHtmlGenerationService(
        slides_db=slides_db,
        job_manager=job_manager,
        keyring=keyring,
        digest_snapshot_loader=digest_snapshot_loader,
        now=now,
    )
    try:
        receipt = slides_db.get_generation_receipt(
            receipt_id,
            owner_user_id=owner,
        )
    except KeyError:
        raise StandaloneHtmlGenerationError(
            "generation_correlation_mismatch",
            status_code=409,
        ) from None
    try:
        receipt = service.correlate_job(
            job,
            owner_user_id=owner,
            receipt_id=receipt_id,
            digest_snapshot=start_snapshot,
        )
    except StandaloneHtmlGenerationError as exc:
        if exc.code == "generation_correlation_mismatch":
            return _terminalize(
                service,
                receipt,
                status="failed",
                code="generation_correlation_mismatch",
                message="Generation correlation failed.",
                now=current_time,
            )
        raise
    except Exception:  # noqa: BLE001 - reduce storage failures to one code
        raise StandaloneHtmlGenerationRetry("generation_store_unavailable") from None
    if receipt.receipt_status == "completed":
        return _completed_metadata(service, receipt)
    if receipt.receipt_status in {"failed", "cancelled"}:
        return _terminal_outcome(service, receipt)
    if _retry_budget_exhausted(job):
        return _terminalize(
            service,
            receipt,
            status="failed",
            code="generation_retry_exhausted",
            message="Generation retry budget was exhausted.",
            now=current_time,
        )

    try:
        generation_input = service.verified_input(
            receipt,
            digest_snapshot=start_snapshot,
        )
    except StandaloneHtmlGenerationError as exc:
        if exc.code != "generation_correlation_mismatch":
            raise
        return _terminalize(
            service,
            receipt,
            status="failed",
            code="generation_correlation_mismatch",
            message="Generation correlation failed.",
            now=current_time,
        )
    input_deadline = _parse_utc(generation_input.input_expires_at)
    if input_deadline is None or current_time >= input_deadline:
        return _terminalize(
            service,
            receipt,
            status="failed",
            code="generation_expired",
            message="Generation input expired.",
            now=input_deadline or current_time,
        )
    try:
        target = _target_from_input(generation_input)
    except Exception:  # noqa: BLE001 - malformed persisted target is correlation failure
        return _terminalize(
            service,
            receipt,
            status="failed",
            code="generation_correlation_mismatch",
            message="Generation correlation failed.",
            now=current_time,
        )
    target_failure = _target_failure_code(target, current_config_loader)
    if target_failure is not None:
        return _terminalize(
            service,
            receipt,
            status="failed",
            code=target_failure,
            message="Standalone HTML generation is unavailable.",
            now=current_time,
        )
    if receipt.job_uuid is None:
        raise StandaloneHtmlGenerationError(
            "generation_correlation_mismatch",
            status_code=409,
        )
    try:
        receipt = slides_db.set_generation_receipt_running(
            receipt_id=receipt.id,
            owner_user_id=receipt.owner_user_id,
            job_uuid=receipt.job_uuid,
            updated_at=current_time.isoformat(),
        )
    except ConflictError:
        try:
            winner = slides_db.get_generation_receipt(
                receipt.id,
                owner_user_id=receipt.owner_user_id,
            )
        except KeyError:
            raise StandaloneHtmlGenerationError(
                "generation_correlation_mismatch",
                status_code=409,
            ) from None
        if winner.receipt_status == "completed":
            return _completed_metadata(service, winner)
        if winner.receipt_status in {"failed", "cancelled"}:
            return _terminal_outcome(service, winner)
        raise StandaloneHtmlGenerationError(
            "generation_correlation_mismatch",
            status_code=409,
        ) from None
    except Exception:  # noqa: BLE001 - source-free retry boundary
        raise StandaloneHtmlGenerationRetry("generation_store_unavailable") from None

    reservation = None
    try:
        try:
            reservation = await validation_pool.acquire_generation_reservation()
        except StandaloneHtmlValidationError as exc:
            return _retry(service, receipt, job, code=exc.code, now=current_time)
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 - validator internals are source-bearing
            return _retry(
                service,
                receipt,
                job,
                code="standalone_html_validator_unavailable",
                now=current_time,
            )

        pre_provider_time = _safe_now(now)
        if input_deadline is None or pre_provider_time >= input_deadline:
            return _terminalize(
                service,
                receipt,
                status="failed",
                code="generation_expired",
                message="Generation input expired.",
                now=input_deadline or pre_provider_time,
            )
        try:
            await _load_digest_snapshot(digest_snapshot_loader)
        except DigestKeyUnavailableError:
            return _reset_and_release_for_missing_key(
                service,
                receipt,
                job_manager,
                job,
                now=pre_provider_time,
            )
        target_failure = _target_failure_code(target, current_config_loader)
        if target_failure is not None:
            return _terminalize(
                service,
                receipt,
                status="failed",
                code=target_failure,
                message="Standalone HTML generation is unavailable.",
                now=pre_provider_time,
            )
        fenced = _fence_job(
            service,
            receipt,
            job_manager,
            job,
            now=pre_provider_time,
        )
        if fenced is not None:
            return fenced

        try:
            provider_api_key = provider_api_key_loader(target)
        except Exception:  # noqa: BLE001 - credentials must never escape
            return _retry(
                service,
                receipt,
                job,
                code="standalone_html_provider_credentials_unavailable",
                now=pre_provider_time,
            )
        try:
            generated = provider_generate(
                stored_target=target,
                system_prompt=generation_input.system_prompt,
                user_content=build_generation_user_content(generation_input),
                provider_api_key=provider_api_key,
                current_config_loader=current_config_loader,
            )
            html_document = await generated if inspect.isawaitable(generated) else generated
        except asyncio.CancelledError:
            raise
        except StandaloneHtmlProviderError as exc:
            provider_failure_time = _safe_now(now)
            if exc.code in {
                "standalone_html_provider_timeout",
                "standalone_html_provider_unavailable",
                "standalone_html_provider_http_error",
            }:
                return _retry(service, receipt, job, code=exc.code, now=provider_failure_time)
            return _terminalize(
                service,
                receipt,
                status="failed",
                code=exc.code,
                message="Standalone HTML generation failed.",
                now=provider_failure_time,
            )
        except Exception:  # noqa: BLE001 - provider/source bytes must never escape
            return _retry(
                service,
                receipt,
                job,
                code="standalone_html_provider_unavailable",
                now=_safe_now(now),
            )
        post_provider_time = _safe_now(now)
        try:
            await _load_digest_snapshot(digest_snapshot_loader)
        except DigestKeyUnavailableError:
            return _reset_and_release_for_missing_key(
                service,
                receipt,
                job_manager,
                job,
                now=post_provider_time,
            )
        try:
            options = json.loads(generation_input.html_options_json)
            validation = await reservation.validate(
                html_document,
                delivery_style=options["delivery_style"],
            )
        except asyncio.CancelledError:
            raise
        except StandaloneHtmlValidationError as exc:
            return _terminalize(
                service,
                receipt,
                status="failed",
                code=exc.code,
                message="Generated HTML did not pass validation.",
                now=_safe_now(now),
            )
        except Exception:  # noqa: BLE001 - validator/source bytes must never escape
            return _terminalize(
                service,
                receipt,
                status="failed",
                code="standalone_html_validator_failed",
                message="Generated HTML did not pass validation.",
                now=_safe_now(now),
            )

        final_time = _safe_now(now)
        if input_deadline is None or final_time >= input_deadline:
            return _terminalize(
                service,
                receipt,
                status="failed",
                code="generation_expired",
                message="Generation input expired.",
                now=input_deadline or final_time,
            )
        fenced = _fence_job(
            service,
            receipt,
            job_manager,
            job,
            now=final_time,
        )
        if fenced is not None:
            return fenced
        try:
            commit_snapshot = await _load_digest_snapshot(digest_snapshot_loader)
        except DigestKeyUnavailableError:
            return _reset_and_release_for_missing_key(
                service,
                receipt,
                job_manager,
                job,
                now=_safe_now(now),
            )
        try:
            presentation = service.commit(
                receipt=receipt,
                html_document=html_document,
                validation_result=validation,
                digest_snapshot=commit_snapshot,
            )
        except (ConflictError, StandaloneHtmlGenerationError) as exc:
            try:
                winner = slides_db.get_generation_receipt(
                    receipt.id,
                    owner_user_id=receipt.owner_user_id,
                )
            except KeyError:
                raise StandaloneHtmlGenerationError(
                    "generation_correlation_mismatch",
                    status_code=409,
                ) from None
            if winner.receipt_status == "completed":
                return _completed_metadata(service, winner)
            if winner.receipt_status in {"failed", "cancelled"}:
                return _terminal_outcome(service, winner)
            failure_code = exc.code if isinstance(exc, StandaloneHtmlGenerationError) else str(exc)
            if failure_code == "generation_expired":
                return _terminalize(
                    service,
                    winner,
                    status="failed",
                    code="generation_expired",
                    message="Generation input expired.",
                    now=input_deadline or final_time,
                )
            if failure_code == "generation_correlation_mismatch":
                return _terminalize(
                    service,
                    winner,
                    status="failed",
                    code="generation_correlation_mismatch",
                    message="Generation correlation failed.",
                    now=final_time,
                )
            return _retry(
                service,
                winner,
                job,
                code="generation_commit_conflict",
                now=final_time,
            )
        except Exception:  # noqa: BLE001 - source-free storage retry boundary
            return _retry(
                service,
                receipt,
                job,
                code="generation_store_unavailable",
                now=final_time,
            )
        return {
            "presentation_id": presentation.id,
            "content_kind": "standalone_html",
            "html_bytes": presentation.html_bytes,
            "html_slide_count": presentation.html_slide_count,
            "validation_status": "accepted",
        }
    finally:
        if reservation is not None:
            try:
                await reservation.release()
            except Exception:  # noqa: BLE001 - never expose validator/source state
                logger.warning("Standalone HTML validation reservation release failed")


def _jobs_manager() -> JobManager:
    db_url = (os.getenv("JOBS_DB_URL") or "").strip()
    if not db_url:
        return JobManager()
    return JobManager(
        backend="postgres" if db_url.startswith("postgres") else None,
        db_url=db_url,
    )


def _slides_db(owner_user_id: str) -> SlidesDatabase:
    return SlidesDatabase(
        db_path=DatabasePaths.get_slides_db_path(owner_user_id),
        client_id=owner_user_id,
    )


async def run_standalone_html_generation_jobs_worker(
    *,
    keyring: StandaloneHtmlHmacKeyring,
    digest_snapshot_loader: Callable[[], Awaitable[DigestKeySnapshot]],
    validation_pool: StandaloneHtmlValidationPool,
    current_config_loader: Callable[[], SlidesStandaloneHtmlConfig],
    provider_api_key_loader: Callable[[ResolvedExecutionTarget], str | None],
    stop_event: asyncio.Event | None = None,
) -> None:
    """Run the Task 8 handler until the shared worker is stopped."""
    worker_id = (os.getenv("SLIDES_STANDALONE_HTML_WORKER_ID") or "standalone-html-generation-worker").strip()
    manager = _jobs_manager()
    sdk = WorkerSDK(
        manager,
        WorkerConfig(
            domain=JOB_DOMAIN,
            queue=JOB_QUEUE,
            worker_id=worker_id,
            lease_seconds=180,
            retry_on_exception=True,
            retry_backoff_seconds=1,
        ),
    )

    async def handler(job: dict[str, Any]):
        return await process_standalone_html_generation_job(
            job,
            job_manager=manager,
            slides_db_factory=_slides_db,
            keyring=keyring,
            digest_snapshot_loader=digest_snapshot_loader,
            validation_pool=validation_pool,
            current_config_loader=current_config_loader,
            provider_api_key_loader=provider_api_key_loader,
        )

    async def watch_stop() -> None:
        if stop_event is not None:
            await stop_event.wait()
            sdk.stop()

    stop_task = asyncio.create_task(watch_stop())
    try:
        await sdk.run(
            handler=handler,
            job_type=JOB_TYPE,
            acquire_guard=make_generation_acquire_guard(digest_snapshot_loader),
        )
    finally:
        stop_task.cancel()


__all__ = [
    "StandaloneHtmlGenerationRetry",
    "make_generation_acquire_guard",
    "process_standalone_html_generation_job",
    "run_standalone_html_generation_jobs_worker",
]
