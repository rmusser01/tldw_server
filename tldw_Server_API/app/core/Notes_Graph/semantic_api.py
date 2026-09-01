"""Application service and derived status projection for Notes semantic indexing."""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import UUID

from tldw_Server_API.app.core.DB_Management.chacha.note_semantic_models import (
    SemanticDesiredState,
    SemanticGenerationState,
    SemanticIndexConfig,
    SemanticIndexingError,
    SemanticOperationReceipt,
)
from tldw_Server_API.app.core.Embeddings.simplified_config import get_config

from .semantic_capabilities import (
    SemanticCapabilities,
    SemanticCapabilityContract,
    build_semantic_capabilities,
)
from .semantic_content import (
    SEMANTIC_CHUNKER_VERSION,
    SEMANTIC_NORMALIZATION_VERSION,
)
from .semantic_jobs import (
    JOB_DOMAIN,
    JOB_QUEUE,
    JOB_TYPE,
    SemanticJobAdmission,
    SemanticJobCommand,
    SemanticJobCoordinator,
    SemanticJobsError,
)
from .semantic_settings import DEFAULT_SEMANTIC_INDEX_SETTINGS, SemanticIndexSettings

_ACTIVE_JOB_STATUSES = frozenset({"queued", "processing"})
_SAFE_ERROR_CODE = re.compile(r"^notes_semantic_[a-z0-9_]{1,96}$")
_KNOWN_DIMENSIONS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}
_PROVIDER_ENDPOINTS = {
    "cohere": "https://api.cohere.com",
    "mistral": "https://api.mistral.ai",
    "openai": "https://api.openai.com",
    "openrouter": "https://openrouter.ai",
    "voyage": "https://api.voyageai.com",
}


class SemanticAPIError(RuntimeError):
    """Stable HTTP-facing semantic application error."""

    def __init__(self, status_code: int, code: str) -> None:
        self.status_code = status_code
        self.code = code
        self.failure_code = code
        super().__init__(code)


@dataclass(frozen=True, slots=True)
class SemanticStatusFacts:
    desired_state: str
    has_active_generation: bool
    has_active_job: bool
    active_job_failed: bool
    pending_notes: int
    failed_notes: int
    cleanup_pending: bool
    indexing_available: bool
    configuration_stale: bool
    cleanup_stalled: bool = False


def derive_semantic_state(facts: SemanticStatusFacts) -> tuple[str, str | None]:
    """Derive the UI state from durable authority facts."""

    if facts.desired_state == "disabled":
        if facts.cleanup_stalled:
            return "needs_attention", "cleanup_stalled"
        return "off", "cleanup_pending" if facts.cleanup_pending else None
    if not facts.indexing_available:
        return "needs_attention", "unavailable"
    if facts.configuration_stale:
        return "needs_attention", "stale_configuration"
    if not facts.has_active_generation:
        if facts.active_job_failed:
            return "needs_attention", "unavailable"
        return "preparing", "building"
    if facts.failed_notes:
        return "needs_attention", "degraded"
    if facts.has_active_job or facts.pending_notes:
        return "updating", "building"
    return "ready", None


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def load_semantic_settings() -> SemanticIndexSettings:
    """Load the operator kill switch while retaining hard-bounded defaults."""

    raw = os.getenv("NOTES_SEMANTIC_INDEXING_ENABLED")
    if raw is None:
        return DEFAULT_SEMANTIC_INDEX_SETTINGS
    return SemanticIndexSettings(indexing_enabled=_truthy(raw))


def _active_note_count(note_db: Any) -> int:
    counter = getattr(note_db, "count_chatbook_scope_category", None)
    if not callable(counter):
        return 0
    try:
        return max(0, int(counter("notes")))
    except (OSError, RuntimeError, TypeError, ValueError):
        return 0


def resolve_semantic_capabilities(
    note_db: Any,
    *,
    settings: SemanticIndexSettings,
) -> SemanticCapabilities:
    """Resolve non-secret production capability facts without provider I/O."""

    config = get_config()
    provider = str(config.default_provider or "").strip().lower()
    model = str(config.default_model or "").strip()
    provider_config = config.get_provider(provider)
    endpoint = None
    credential_available = False
    if provider_config is not None:
        endpoint = provider_config.api_url
        credential_available = bool(provider_config.api_key)
    endpoint = endpoint or _PROVIDER_ENDPOINTS.get(provider)
    execution_boundary = "local" if provider in {"local", "local_api"} else "external"
    backend = (os.getenv("NOTES_SEMANTIC_VECTOR_BACKEND") or "chromadb").strip().lower()
    dimensions = _KNOWN_DIMENSIONS.get(model)
    if dimensions is None:
        raw_dimensions = os.getenv("NOTES_SEMANTIC_EMBEDDING_DIMENSIONS", "").strip()
        try:
            dimensions = int(raw_dimensions) if raw_dimensions else None
        except ValueError:
            dimensions = None
    contract = SemanticCapabilityContract(
        provider=provider,
        model=model,
        endpoint_url=endpoint,
        execution_boundary=execution_boundary,
        vector_backend=backend,
        storage_boundary="local" if backend in {"chromadb", "pgvector"} else "unavailable",
        resolved_dimensions=dimensions,
        normalization_version=SEMANTIC_NORMALIZATION_VERSION,
        chunker_version=SEMANTIC_CHUNKER_VERSION,
        credential_source="durable" if credential_available else "none",
        provider_healthy=provider_config is not None and provider_config.enabled,
        vector_storage_available=backend in {"chromadb", "pgvector"},
        active_note_count=_active_note_count(note_db),
    )
    return build_semantic_capabilities(contract, settings=settings)


def _capability_response(capabilities: SemanticCapabilities) -> dict[str, Any]:
    return {
        "active_note_count": capabilities.active_note_count,
        "estimated_chunk_count": capabilities.estimated_chunk_count,
        "estimated_run_count": capabilities.estimated_run_count,
        "provider_label": capabilities.provider_label,
        "model": capabilities.model,
        "execution_boundary": capabilities.execution_boundary,
        "storage_boundary": capabilities.storage_boundary,
        "storage_label": capabilities.storage_label,
        "outbound_data_categories": capabilities.outbound_data_categories,
        "capability_revision": capabilities.capability_revision,
        "indexing_available": capabilities.indexing_available,
        "unavailable_reason": capabilities.unavailable_reason,
        "metric": capabilities.metric,
        "resolved_dimensions": capabilities.resolved_dimensions,
    }


def _safe_job_error(job: Mapping[str, Any]) -> str | None:
    result = job.get("result")
    candidates = (
        result.get("error_code") if isinstance(result, dict) else None,
        job.get("error_code"),
        job.get("last_error_code"),
    )
    for candidate in candidates:
        if isinstance(candidate, str) and _SAFE_ERROR_CODE.fullmatch(candidate):
            return candidate
    return "notes_semantic_run_failed" if job.get("status") == "failed" else None


def _canonical_digest(value: Mapping[str, object]) -> str:
    encoded = json.dumps(
        dict(value),
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _key_digest(value: str) -> str:
    normalized = str(value).strip()
    if not normalized or normalized != value or len(normalized.encode("utf-8")) > 256:
        raise SemanticAPIError(422, "notes_semantic_invalid_request")
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


class SemanticIndexAPI:
    """Owner/dataset application service backed by Notes authority and Jobs."""

    def __init__(
        self,
        *,
        note_db: Any,
        jobs: Any,
        owner_user_id: str,
        dataset_id: str,
        settings: SemanticIndexSettings = DEFAULT_SEMANTIC_INDEX_SETTINGS,
        capability_resolver: Callable[[], SemanticCapabilities],
        clock: Callable[[], datetime] = _utc_now,
    ) -> None:
        self._note_db = note_db
        self._store = note_db.note_semantic_store
        self._jobs = jobs
        self._owner_user_id = str(owner_user_id)
        self._dataset_id = str(dataset_id)
        self._settings = settings
        self._capability_resolver = capability_resolver
        self._clock = clock

    def _coordinator(self) -> SemanticJobCoordinator:
        if self._jobs is None:
            raise SemanticAPIError(503, "notes_semantic_jobs_unavailable")
        return SemanticJobCoordinator(
            jobs=self._jobs,
            owner_user_id=self._owner_user_id,
            clock=self._clock,
        )

    def _capabilities(self) -> SemanticCapabilities:
        try:
            return self._capability_resolver()
        except SemanticAPIError:
            raise
        except Exception as exc:  # noqa: BLE001 - capability details stay internal
            raise SemanticAPIError(503, "notes_semantic_provider_unavailable") from exc

    def capabilities(self) -> dict[str, Any]:
        return _capability_response(self._capabilities())

    def _begin_operation(
        self,
        *,
        action: str,
        request_identity: Mapping[str, object],
        idempotency_key: str,
        run_id: str | None,
        expected_revision: int,
    ) -> tuple[SemanticOperationReceipt, str, str, bool]:
        identity = {
            "owner_user_id": self._owner_user_id,
            **dict(request_identity),
        }
        key = _key_digest(idempotency_key)
        fingerprint = _canonical_digest(identity)
        try:
            receipt, replayed = self._store.begin_operation_receipt(
                dataset_id=self._dataset_id,
                key_digest=key,
                action=action,
                request_fingerprint=fingerprint,
                run_id=run_id,
                expected_revision=expected_revision,
                expires_at=self._clock() + timedelta(days=31),
                now=self._clock(),
            )
        except SemanticIndexingError as exc:
            status_code = 409 if exc.code.endswith("conflict") else 503
            raise SemanticAPIError(status_code, exc.code) from None
        return receipt, key, fingerprint, replayed

    @staticmethod
    def _receipt_response(receipt: SemanticOperationReceipt) -> dict[str, Any] | None:
        if receipt.state != "completed" or receipt.response_json is None:
            return None
        try:
            response = json.loads(receipt.response_json)
        except (TypeError, ValueError):
            raise SemanticAPIError(503, "notes_semantic_jobs_unavailable") from None
        if not isinstance(response, dict):
            raise SemanticAPIError(503, "notes_semantic_jobs_unavailable")
        return response

    def _complete_operation(
        self,
        *,
        key_digest: str,
        request_fingerprint: str,
        run_id: str | None,
        response: dict[str, Any],
    ) -> dict[str, Any]:
        try:
            self._store.complete_operation_receipt(
                dataset_id=self._dataset_id,
                key_digest=key_digest,
                request_fingerprint=request_fingerprint,
                run_id=run_id,
                response=response,
                now=self._clock(),
            )
        except (SemanticIndexingError, TypeError, ValueError) as exc:
            code = getattr(exc, "code", "notes_semantic_jobs_unavailable")
            status_code = 409 if str(code).endswith("conflict") else 503
            raise SemanticAPIError(status_code, str(code)) from None
        return response

    def _jobs_for_dataset(self, *, limit: int = 10) -> tuple[dict[str, Any], ...]:
        if self._jobs is None or not callable(getattr(self._jobs, "list_jobs", None)):
            return ()
        rows = self._jobs.list_jobs(
            domain=JOB_DOMAIN,
            queue=JOB_QUEUE,
            owner_user_id=self._owner_user_id,
            job_type=JOB_TYPE,
            limit=limit,
        )
        matching: list[dict[str, Any]] = []
        for row in rows:
            payload = row.get("payload")
            if isinstance(payload, dict) and payload.get("dataset_id") == self._dataset_id:
                matching.append(row)
        return tuple(matching)

    def _active_job(self) -> dict[str, Any] | None:
        for job in self._jobs_for_dataset():
            if str(job.get("status")) in _ACTIVE_JOB_STATUSES:
                return job
        return None

    def _latest_failed_job(self) -> dict[str, Any] | None:
        for job in self._jobs_for_dataset():
            if str(job.get("status")) in {"failed", "quarantined"}:
                return job
        return None

    def _current_active_note_count(self) -> int:
        try:
            return self._capabilities().active_note_count
        except SemanticAPIError as exc:
            if exc.code != "notes_semantic_provider_unavailable":
                raise
            return _active_note_count(self._note_db)

    def _integrity(self, config: SemanticIndexConfig) -> dict[str, int]:
        generation_id = config.active_generation_id
        if generation_id is None:
            return {
                "indexed_notes": 0,
                "excluded_notes": 0,
                "failed_notes": 0,
                "pending_notes": self._current_active_note_count(),
                "published_chunks": 0,
            }
        try:
            integrity = self._store.get_generation_integrity(
                self._dataset_id,
                generation_id,
            )
        except (OSError, RuntimeError, ValueError):
            return {
                "indexed_notes": 0,
                "excluded_notes": 0,
                "failed_notes": 0,
                "pending_notes": self._current_active_note_count(),
                "published_chunks": 0,
            }
        return {
            "indexed_notes": integrity.indexed_note_count,
            "excluded_notes": integrity.excluded_note_count,
            "failed_notes": integrity.failed_note_count,
            "pending_notes": integrity.pending_note_count,
            "published_chunks": integrity.published_chunk_count,
        }

    def _run_response(self, job: Mapping[str, Any]) -> dict[str, Any]:
        payload = job.get("payload") if isinstance(job.get("payload"), dict) else {}
        result = job.get("result") if isinstance(job.get("result"), dict) else {}
        indexed = int(result.get("indexed_notes") or 0)
        excluded = int(result.get("excluded_notes") or 0)
        failed = int(result.get("failed_notes") or 0)
        pending = max(
            0,
            self._current_active_note_count() - indexed - excluded - failed,
        )
        run_id = str(job["uuid"])
        return {
            "run_id": run_id,
            "mode": str(payload.get("mode") or "unknown"),
            "status": str(job.get("status") or "unknown"),
            "revision": int(payload.get("configuration_revision") or 0),
            "indexed_notes": indexed,
            "excluded_notes": excluded,
            "failed_notes": failed,
            "pending_notes": pending,
            "published_chunks": int(result.get("published_chunks") or 0),
            "cleanup_complete": bool(result.get("cleanup_complete", False)),
            "error_code": _safe_job_error(job),
            "link": f"/api/v1/notes/graph/semantic-index/runs/{run_id}",
        }

    def status(self) -> dict[str, Any]:
        config = self._store.get_configuration(self._dataset_id)
        cleanup_pending = bool(self._store.has_pending_cleanup(self._dataset_id))
        cleanup_stalled = bool(
            self._store.has_stalled_cleanup(
                self._dataset_id,
                expired_before=self._clock()
                - timedelta(
                    seconds=max(
                        int(os.getenv("NOTES_SEMANTIC_LEASE_SECONDS", "180") or "180"),
                        1,
                    )
                ),
            )
        )
        active_job = self._active_job()
        failed_job = self._latest_failed_job()
        if config is None:
            return {
                "state": "needs_attention" if cleanup_stalled else "off",
                "detail_reason": (
                    "cleanup_stalled"
                    if cleanup_stalled
                    else "cleanup_pending" if cleanup_pending else None
                ),
                "desired_state": "disabled",
                "configuration_revision": 0,
                "semantic_index_revision": 0,
                "active_generation_id": None,
                "indexed_notes": 0,
                "excluded_notes": 0,
                "failed_notes": 0,
                "pending_notes": 0,
                "published_chunks": 0,
                "cleanup_pending": cleanup_pending,
                "active_run": self._run_response(active_job) if active_job else None,
            }
        counts = self._integrity(config)
        try:
            capabilities = self._capabilities()
        except SemanticAPIError as exc:
            if exc.code != "notes_semantic_provider_unavailable":
                raise
            capabilities = None
        state, reason = derive_semantic_state(
            SemanticStatusFacts(
                desired_state=config.desired_state.value,
                has_active_generation=config.active_generation_id is not None,
                has_active_job=active_job is not None,
                active_job_failed=failed_job is not None,
                pending_notes=counts["pending_notes"],
                failed_notes=counts["failed_notes"],
                cleanup_pending=cleanup_pending,
                indexing_available=(
                    capabilities.indexing_available
                    if capabilities is not None
                    else False
                ),
                configuration_stale=(
                    capabilities is not None
                    and config.capability_revision is not None
                    and config.capability_revision != capabilities.capability_revision
                ),
                cleanup_stalled=cleanup_stalled,
            )
        )
        return {
            "state": state,
            "detail_reason": reason,
            "desired_state": config.desired_state.value,
            "configuration_revision": config.configuration_revision,
            "semantic_index_revision": config.semantic_index_revision,
            "active_generation_id": config.active_generation_id,
            **counts,
            "cleanup_pending": cleanup_pending,
            "active_run": self._run_response(active_job) if active_job else None,
        }

    def _require_capability(self, revision: str) -> SemanticCapabilities:
        capabilities = self._capabilities()
        if capabilities.capability_revision != revision:
            raise SemanticAPIError(409, "notes_semantic_capability_revision_conflict")
        if not capabilities.indexing_available:
            raise SemanticAPIError(503, "notes_semantic_provider_unavailable")
        return capabilities

    def _admit(
        self,
        command: SemanticJobCommand,
        *,
        idempotency_key: str,
        request_identity: Mapping[str, object] | None = None,
    ) -> SemanticJobAdmission:
        try:
            return self._coordinator().admit(
                command,
                idempotency_key=idempotency_key,
                request_identity=request_identity,
            )
        except SemanticJobsError as exc:
            if exc.code == "notes_semantic_quota_exceeded":
                status_code = 429
            elif exc.code.endswith("conflict"):
                status_code = 409
            else:
                status_code = 503
            raise SemanticAPIError(status_code, exc.code) from exc

    def _replay(
        self,
        command: SemanticJobCommand,
        *,
        idempotency_key: str,
        request_identity: Mapping[str, object],
    ) -> SemanticJobAdmission | None:
        try:
            return self._coordinator().replay(
                command,
                idempotency_key=idempotency_key,
                request_identity=request_identity,
            )
        except SemanticJobsError as exc:
            status_code = 409 if exc.code.endswith("conflict") else 503
            raise SemanticAPIError(status_code, exc.code) from exc

    def enable(
        self,
        *,
        expected_revision: int,
        capability_revision: str,
        idempotency_key: str,
    ) -> dict[str, Any]:
        config = self._store.get_configuration(self._dataset_id)
        command_revision = 2 if expected_revision == 0 else expected_revision + 1
        command_mode = (
            "rebuild"
            if expected_revision > 0
            and config is not None
            and config.desired_state is SemanticDesiredState.ENABLED
            else "build"
        )
        command = SemanticJobCommand(
            dataset_id=self._dataset_id,
            configuration_revision=command_revision,
            mode=command_mode,
        )
        request_identity = {
            "action": "enable",
            "dataset_id": self._dataset_id,
            "expected_revision": expected_revision,
            "capability_revision": capability_revision,
        }
        receipt, key_digest, request_fingerprint, receipt_replayed = (
            self._begin_operation(
                action="enable",
                request_identity=request_identity,
                idempotency_key=idempotency_key,
                run_id=None,
                expected_revision=expected_revision,
            )
        )
        stored = self._receipt_response(receipt)
        if stored is not None:
            return stored
        replay = self._replay(
            command,
            idempotency_key=idempotency_key,
            request_identity=request_identity,
        )
        if replay is not None:
            response = {
                "resource": self.status(),
                "run": self._run_response(replay.job),
            }
            return self._complete_operation(
                key_digest=key_digest,
                request_fingerprint=request_fingerprint,
                run_id=replay.run_id,
                response=response,
            )

        config = self._store.get_configuration(self._dataset_id)
        committed_retry = (
            receipt_replayed
            and receipt.state == "pending"
            and config is not None
            and config.desired_state is SemanticDesiredState.ENABLED
            and config.configuration_revision == command_revision
            and config.capability_revision == capability_revision
        )
        capabilities = (
            None
            if committed_retry
            else self._require_capability(capability_revision)
        )
        if config is None:
            if expected_revision != 0:
                raise SemanticAPIError(409, "notes_semantic_configuration_revision_conflict")
            if capabilities is None:
                raise SemanticAPIError(409, "notes_semantic_configuration_revision_conflict")
            if capabilities.endpoint_display is None:
                raise SemanticAPIError(503, "notes_semantic_provider_unavailable")
            config = self._store.create_configuration(
                dataset_id=self._dataset_id,
                capability_revision=capabilities.capability_revision,
                disclosure_hash=capabilities.disclosure_hash,
                provider=capabilities.provider_label.lower(),
                model=capabilities.model,
                model_revision=capabilities.model_revision,
                endpoint_origin_revision=capabilities.endpoint_origin_revision,
                endpoint_origin_display=capabilities.endpoint_display,
                data_boundary=capabilities.execution_boundary,
                vector_backend=(
                    os.getenv("NOTES_SEMANTIC_VECTOR_BACKEND") or "chromadb"
                ).strip().lower(),
                storage_boundary=capabilities.storage_boundary,
                storage_label=capabilities.storage_label,
                normalization_version=SEMANTIC_NORMALIZATION_VERSION,
                chunker_version=SEMANTIC_CHUNKER_VERSION,
                now=self._clock(),
            )
        if config.desired_state is SemanticDesiredState.DISABLED:
            allowed_revisions = {config.configuration_revision}
            if config.configuration_revision == 1:
                allowed_revisions.add(0)
            if expected_revision not in allowed_revisions:
                raise SemanticAPIError(409, "notes_semantic_configuration_revision_conflict")
            enabled = self._store.enable_configuration(
                dataset_id=self._dataset_id,
                expected_configuration_revision=config.configuration_revision,
                capability_revision=capability_revision,
                now=self._clock(),
            )
            if enabled is None:
                raise SemanticAPIError(409, "notes_semantic_configuration_revision_conflict")
            config = enabled
        elif not committed_retry:
            if (
                capabilities is None
                or config.configuration_revision != expected_revision
                or config.capability_revision == capabilities.capability_revision
                or capabilities.compatibility_hash is None
                or capabilities.resolved_dimensions is None
                or capabilities.endpoint_display is None
            ):
                raise SemanticAPIError(
                    409,
                    "notes_semantic_configuration_revision_conflict",
                )
            renewed = self._store.renew_configuration_consent(
                dataset_id=self._dataset_id,
                expected_configuration_revision=expected_revision,
                capability_revision=capabilities.capability_revision,
                disclosure_hash=capabilities.disclosure_hash,
                compatibility_hash=capabilities.compatibility_hash,
                provider=capabilities.provider_label.lower(),
                model=capabilities.model,
                model_revision=capabilities.model_revision,
                endpoint_origin_revision=capabilities.endpoint_origin_revision,
                endpoint_origin_display=capabilities.endpoint_display,
                data_boundary=capabilities.execution_boundary,
                vector_backend=(
                    os.getenv("NOTES_SEMANTIC_VECTOR_BACKEND") or "chromadb"
                ).strip().lower(),
                storage_boundary=capabilities.storage_boundary,
                storage_label=capabilities.storage_label,
                resolved_dimensions=capabilities.resolved_dimensions,
                normalization_version=SEMANTIC_NORMALIZATION_VERSION,
                chunker_version=SEMANTIC_CHUNKER_VERSION,
                now=self._clock(),
            )
            if renewed is None:
                raise SemanticAPIError(
                    409,
                    "notes_semantic_configuration_revision_conflict",
                )
            config = renewed

        if config.configuration_revision != command_revision:
            raise SemanticAPIError(409, "notes_semantic_configuration_revision_conflict")
        admission = self._admit(
            command,
            idempotency_key=idempotency_key,
            request_identity=request_identity,
        )
        response = {
            "resource": self.status(),
            "run": self._run_response(admission.job),
        }
        return self._complete_operation(
            key_digest=key_digest,
            request_fingerprint=request_fingerprint,
            run_id=admission.run_id,
            response=response,
        )

    def disable(
        self,
        *,
        expected_revision: int,
        idempotency_key: str,
    ) -> dict[str, Any]:
        config = self._store.get_configuration(self._dataset_id)
        if config is None:
            raise SemanticAPIError(409, "notes_semantic_configuration_revision_conflict")
        if config.desired_state is SemanticDesiredState.ENABLED:
            if config.configuration_revision != expected_revision:
                raise SemanticAPIError(409, "notes_semantic_configuration_revision_conflict")
            command_revision = expected_revision + 1
        elif expected_revision in {
            config.configuration_revision,
            config.configuration_revision - 1,
        }:
            command_revision = config.configuration_revision
        else:
            raise SemanticAPIError(409, "notes_semantic_configuration_revision_conflict")
        command = SemanticJobCommand(
            dataset_id=self._dataset_id,
            configuration_revision=command_revision,
            mode="delete",
        )
        request_identity = {
            "action": "disable",
            "dataset_id": self._dataset_id,
            "expected_revision": expected_revision,
        }
        replay = self._replay(
            command,
            idempotency_key=idempotency_key,
            request_identity=request_identity,
        )
        if replay is not None:
            return {"resource": self.status(), "run": self._run_response(replay.job)}

        active_job = self._active_job()
        if config.desired_state is SemanticDesiredState.ENABLED:
            disabled = self._store.disable_and_schedule_cleanup(
                dataset_id=self._dataset_id,
                expected_configuration_revision=config.configuration_revision,
                now=self._clock(),
            )
            if disabled is None:
                raise SemanticAPIError(409, "notes_semantic_configuration_revision_conflict")
            config = disabled
        if active_job is not None:
            payload = active_job.get("payload")
            if isinstance(payload, dict) and payload.get("mode") != "delete":
                try:
                    self._coordinator().cancel(
                        str(active_job["uuid"]),
                        expected_revision=int(payload["configuration_revision"]),
                    )
                except (KeyError, TypeError, ValueError) as exc:
                    raise SemanticAPIError(
                        409,
                        "notes_semantic_run_revision_conflict",
                    ) from exc
                except SemanticJobsError as exc:
                    status_code = 409 if exc.code.endswith("conflict") else 503
                    raise SemanticAPIError(status_code, exc.code) from exc
        admission = self._admit(
            command,
            idempotency_key=idempotency_key,
            request_identity=request_identity,
        )
        return {"resource": self.status(), "run": self._run_response(admission.job)}

    def create_run(
        self,
        *,
        mode: str,
        expected_revision: int,
        idempotency_key: str,
    ) -> dict[str, Any]:
        if mode not in {"rebuild", "retry_failed"}:
            raise SemanticAPIError(422, "notes_semantic_invalid_request")
        config = self._store.get_configuration(self._dataset_id)
        generation_id = (
            config.active_generation_id
            if config is not None and mode == "retry_failed"
            else None
        )
        command = SemanticJobCommand(
            dataset_id=self._dataset_id,
            configuration_revision=expected_revision,
            generation_id=generation_id,
            mode=mode,
        )
        request_identity = {
            "action": "create_run",
            "dataset_id": self._dataset_id,
            "mode": mode,
            "expected_revision": expected_revision,
        }
        replay = self._replay(
            command,
            idempotency_key=idempotency_key,
            request_identity=request_identity,
        )
        if replay is not None:
            return self._run_response(replay.job)
        if (
            config is None
            or config.desired_state is not SemanticDesiredState.ENABLED
            or config.configuration_revision != expected_revision
        ):
            raise SemanticAPIError(409, "notes_semantic_configuration_revision_conflict")
        self._require_capability(config.capability_revision or "")
        if mode == "retry_failed" and generation_id is None:
            raise SemanticAPIError(409, "notes_semantic_active_generation_required")
        admission = self._admit(
            command,
            idempotency_key=idempotency_key,
            request_identity=request_identity,
        )
        return self._run_response(admission.job)

    def get_run(self, *, run_id: UUID) -> dict[str, Any]:
        job = self._coordinator().get_job_for_run(str(run_id))
        payload = job.get("payload") if isinstance(job, dict) else None
        if not isinstance(payload, dict) or payload.get("dataset_id") != self._dataset_id:
            raise SemanticAPIError(404, "notes_semantic_run_not_found")
        return self._run_response(job)

    def cancel_run(
        self,
        *,
        run_id: UUID,
        expected_revision: int,
        idempotency_key: str,
    ) -> dict[str, Any]:
        if not idempotency_key:
            raise SemanticAPIError(422, "notes_semantic_invalid_request")
        job = self._coordinator().get_job_for_run(str(run_id))
        payload = job.get("payload") if isinstance(job, dict) else None
        if not isinstance(payload, dict) or payload.get("dataset_id") != self._dataset_id:
            raise SemanticAPIError(404, "notes_semantic_run_not_found")
        if payload.get("mode") not in {"build", "rebuild", "retry_failed"}:
            raise SemanticAPIError(422, "notes_semantic_invalid_request")
        if payload.get("configuration_revision") != expected_revision:
            raise SemanticAPIError(409, "notes_semantic_run_revision_conflict")
        request_identity = {
            "action": "cancel",
            "dataset_id": self._dataset_id,
            "run_id": str(run_id),
            "expected_revision": expected_revision,
        }
        receipt, key_digest, request_fingerprint, _receipt_replayed = (
            self._begin_operation(
                action="cancel",
                request_identity=request_identity,
                idempotency_key=idempotency_key,
                run_id=str(run_id),
                expected_revision=expected_revision,
            )
        )
        stored = self._receipt_response(receipt)
        if stored is not None:
            return stored
        generation = self._store.get_generation_by_root_job_id(
            self._dataset_id,
            str(run_id),
        )
        if generation is not None:
            fenced = (
                generation.state is SemanticGenerationState.FAILED
                and generation.terminal_error_code == "notes_semantic_run_cancelled"
            )
            if generation.state is SemanticGenerationState.STAGING:
                fenced = self._store.fail_generation(
                    dataset_id=self._dataset_id,
                    generation_id=generation.id,
                    generation_fencing_token=str(generation.root_job_id or ""),
                    expected_configuration_revision=generation.configuration_revision,
                    error_code="notes_semantic_run_cancelled",
                    now=self._clock(),
                )
                if not fenced:
                    current = self._store.get_generation(
                        self._dataset_id,
                        generation.id,
                    )
                    fenced = (
                        current is not None
                        and current.state is SemanticGenerationState.FAILED
                        and current.terminal_error_code
                        == "notes_semantic_run_cancelled"
                    )
            if not fenced:
                raise SemanticAPIError(409, "notes_semantic_run_revision_conflict")
        try:
            cancelled = self._coordinator().cancel(
                str(run_id),
                expected_revision=expected_revision,
            )
        except SemanticJobsError as exc:
            if exc.code.endswith("not_found"):
                status_code = 404
            elif exc.code.endswith("conflict"):
                status_code = 409
            else:
                status_code = 503
            raise SemanticAPIError(status_code, exc.code) from exc
        response = {
            "resource": self.status(),
            "run": self._run_response(cancelled),
        }
        return self._complete_operation(
            key_digest=key_digest,
            request_fingerprint=request_fingerprint,
            run_id=str(run_id),
            response=response,
        )


def build_notes_semantic_api(
    *,
    note_db: Any,
    jobs: Any,
    owner_user_id: str,
    dataset_id: str,
    settings: SemanticIndexSettings | None = None,
) -> SemanticIndexAPI:
    """Build the production semantic application service for one request scope."""

    effective_settings = settings or load_semantic_settings()
    return SemanticIndexAPI(
        note_db=note_db,
        jobs=jobs,
        owner_user_id=owner_user_id,
        dataset_id=dataset_id,
        settings=effective_settings,
        capability_resolver=lambda: resolve_semantic_capabilities(
            note_db,
            settings=effective_settings,
        ),
    )


__all__ = [
    "SemanticAPIError",
    "SemanticIndexAPI",
    "SemanticStatusFacts",
    "build_notes_semantic_api",
    "derive_semantic_state",
    "load_semantic_settings",
    "resolve_semantic_capabilities",
]
