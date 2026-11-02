"""Jobs-owned, allowlisted read projections under the actual owner's RLS scope."""

import math
from uuid import UUID

from tldw_Server_API.app.api.v1.schemas.jobs_owned import OwnedJobStatus
from tldw_Server_API.app.core.claims_analytics_export_contract import (
    is_routable_claims_owner_id_text,
)
from tldw_Server_API.app.core.claims_cluster_rebuild_contract import (
    CLAIMS_REBUILD_CLUSTERS_JOB_TYPE,
)
from tldw_Server_API.app.core.Jobs.manager import JobManager
from tldw_Server_API.app.core.Jobs.models import TERMINAL_JOB_STATUSES

_FAILURE_CODES = frozenset(
    {
        "claims_cluster_owner_database_missing",
        "claims_cluster_source_changed",
        "claims_cluster_foreign_reference",
        "claims_cluster_corruption",
        "claims_cluster_resource_limit",
        "claims_cluster_configuration",
        "claims_cluster_attempt_inactive",
        "claims_cluster_storage_unavailable",
        "claims_cluster_failed",
        "claims_cluster_vector_access",
        "claims_cluster_vector_busy",
        "claims_cluster_vector_closed",
        "claims_cluster_vector_configuration",
        "claims_cluster_vector_corruption",
        "claims_cluster_vector_forked_handle",
        "claims_cluster_vector_invalid_request",
        "claims_cluster_vector_lock_unsupported",
        "claims_cluster_vector_log_coverage",
        "claims_cluster_vector_mixed_access",
        "claims_cluster_vector_path_alias",
        "claims_cluster_vector_read_only",
        "claims_cluster_vector_shutdown",
        "claims_cluster_vector_source_changed",
        "claims_cluster_vector_unsupported_sdk",
        "claims_invalid_payload",
        "claims_missing_owner",
        "claims_unsupported_payload_version",
    }
)
_STATUSES = TERMINAL_JOB_STATUSES | {"queued", "processing"}


def _bounded_number(value: object, maximum: float) -> bool:
    """Reject coercible strings, bools, nonfinite numbers and out-of-range values."""
    return type(value) in (int, float) and 0 <= value <= maximum and math.isfinite(value)


def _cluster_result(value: object) -> dict[str, int | float | str | None] | None:
    """Copy only recognized scalar results, never free text or nested data."""
    if not isinstance(value, dict):
        return None
    result: dict[str, int | float | str | None] = {}
    for field in ("clusters_created", "claims_assigned", "claims_skipped"):
        count = value.get(field)
        if type(count) is int and 0 <= count <= 50_000:
            result[field] = count
    for field, allowed in (
        ("method", ("exact", "embeddings")),
        ("outcome", ("ok", "skipped")),
        ("reason", ("no_embeddings", "superseded")),
    ):
        item = value.get(field)
        if isinstance(item, str) and item in allowed:
            result[field] = item
    threshold = value.get("similarity_threshold")
    if _bounded_number(threshold, 1):
        result["similarity_threshold"] = threshold
    return result


def get_owned_job_status(manager: JobManager, *, job_uuid: str, owner_user_id: str) -> OwnedJobStatus | None:
    """Read an exact owner UUID with tokens scoped/restored in this executing thread."""
    if not is_routable_claims_owner_id_text(owner_user_id):
        return None
    try:
        canonical_uuid = str(UUID(job_uuid))
    except (ValueError, TypeError, AttributeError):
        return None
    with manager.scoped_rls_context(is_admin=False, domain_allowlist="claims", owner_user_id=owner_user_id):
        job = manager.get_job_or_archived_by_uuid(canonical_uuid, domain="claims", owner_user_id=owner_user_id)
        if (
            not job
            or job.get("uuid") != canonical_uuid
            or job.get("owner_user_id") != owner_user_id
            or job.get("domain") != "claims"
            or job.get("job_type") != CLAIMS_REBUILD_CLUSTERS_JOB_TYPE
            or job.get("status") not in _STATUSES
        ):
            return None
        progress = job.get("progress_percent")
        failure = job.get("error_code")
        return OwnedJobStatus(
            id=job["id"],
            uuid=canonical_uuid,
            status=job["status"],
            progress_percent=progress if _bounded_number(progress, 100) else None,
            result=_cluster_result(job.get("result")),
            failure_code=failure if isinstance(failure, str) and failure in _FAILURE_CODES else None,
        )
