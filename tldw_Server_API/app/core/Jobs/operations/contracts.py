"""Compatible Jobs import surface for the canonical, inert shared contracts."""

from __future__ import annotations

from ...Utils.jobs_operation_contracts import (
    _ADMIN_WEBHOOK_MARKER_MAX_BYTES as _ADMIN_WEBHOOK_MARKER_MAX_BYTES,
)
from ...Utils.jobs_operation_contracts import (
    _ADMIN_WEBHOOK_NO_ATTEMPT_FAIL_REASONS as _ADMIN_WEBHOOK_NO_ATTEMPT_FAIL_REASONS,
)
from ...Utils.jobs_operation_contracts import (
    _OPAQUE_TOKEN_RE as _OPAQUE_TOKEN_RE,
)
from ...Utils.jobs_operation_contracts import (
    _REASON_CODE_RE as _REASON_CODE_RE,
)
from ...Utils.jobs_operation_contracts import (
    ADMIN_WEBHOOK_DELIVERY_DOMAIN as ADMIN_WEBHOOK_DELIVERY_DOMAIN,
)
from ...Utils.jobs_operation_contracts import (
    ADMIN_WEBHOOK_DELIVERY_IDEMPOTENCY_PREFIX as ADMIN_WEBHOOK_DELIVERY_IDEMPOTENCY_PREFIX,
)
from ...Utils.jobs_operation_contracts import (
    ADMIN_WEBHOOK_DELIVERY_JOB_TYPE as ADMIN_WEBHOOK_DELIVERY_JOB_TYPE,
)
from ...Utils.jobs_operation_contracts import (
    ADMIN_WEBHOOK_DELIVERY_MAX_RETRIES as ADMIN_WEBHOOK_DELIVERY_MAX_RETRIES,
)
from ...Utils.jobs_operation_contracts import (
    ADMIN_WEBHOOK_DELIVERY_PRIORITY as ADMIN_WEBHOOK_DELIVERY_PRIORITY,
)
from ...Utils.jobs_operation_contracts import (
    ADMIN_WEBHOOK_DELIVERY_QUARANTINE_THRESHOLD as ADMIN_WEBHOOK_DELIVERY_QUARANTINE_THRESHOLD,
)
from ...Utils.jobs_operation_contracts import (
    ADMIN_WEBHOOK_DELIVERY_QUEUE as ADMIN_WEBHOOK_DELIVERY_QUEUE,
)
from ...Utils.jobs_operation_contracts import (
    UUID as UUID,
)
from ...Utils.jobs_operation_contracts import (
    AcquireJobCommand as AcquireJobCommand,
)
from ...Utils.jobs_operation_contracts import (
    AdminWebhookDispositionMarker as AdminWebhookDispositionMarker,
)
from ...Utils.jobs_operation_contracts import (
    AdmissionRejectionReason as AdmissionRejectionReason,
)
from ...Utils.jobs_operation_contracts import (
    AdmissionResult as AdmissionResult,
)
from ...Utils.jobs_operation_contracts import (
    Any as Any,
)
from ...Utils.jobs_operation_contracts import (
    ApplyPreparedDispositionCommand as ApplyPreparedDispositionCommand,
)
from ...Utils.jobs_operation_contracts import (
    BatchRenewLeaseItem as BatchRenewLeaseItem,
)
from ...Utils.jobs_operation_contracts import (
    BatchRenewLeasesCommand as BatchRenewLeasesCommand,
)
from ...Utils.jobs_operation_contracts import (
    BatchRenewLeasesResult as BatchRenewLeasesResult,
)
from ...Utils.jobs_operation_contracts import (
    CreateJobCommand as CreateJobCommand,
)
from ...Utils.jobs_operation_contracts import (
    EnsureLeaseHorizonCommand as EnsureLeaseHorizonCommand,
)
from ...Utils.jobs_operation_contracts import (
    Enum as Enum,
)
from ...Utils.jobs_operation_contracts import (
    ExpiredLeasePolicy as ExpiredLeasePolicy,
)
from ...Utils.jobs_operation_contracts import (
    FindJobByIdentityCommand as FindJobByIdentityCommand,
)
from ...Utils.jobs_operation_contracts import (
    IdempotentOperationAdmission as IdempotentOperationAdmission,
)
from ...Utils.jobs_operation_contracts import (
    IdempotentOperationCommand as IdempotentOperationCommand,
)
from ...Utils.jobs_operation_contracts import (
    IdempotentOperationConflict as IdempotentOperationConflict,
)
from ...Utils.jobs_operation_contracts import (
    IdempotentOperationConflictReason as IdempotentOperationConflictReason,
)
from ...Utils.jobs_operation_contracts import (
    IdempotentOperationDisposition as IdempotentOperationDisposition,
)
from ...Utils.jobs_operation_contracts import (
    IdempotentOperationUnavailableError as IdempotentOperationUnavailableError,
)
from ...Utils.jobs_operation_contracts import (
    JobIdentityLookupResult as JobIdentityLookupResult,
)
from ...Utils.jobs_operation_contracts import (
    JobIdentityLookupState as JobIdentityLookupState,
)
from ...Utils.jobs_operation_contracts import (
    LeaseHorizonResult as LeaseHorizonResult,
)
from ...Utils.jobs_operation_contracts import (
    LifecycleResult as LifecycleResult,
)
from ...Utils.jobs_operation_contracts import (
    NoTransitionReason as NoTransitionReason,
)
from ...Utils.jobs_operation_contracts import (
    OperationOutcome as OperationOutcome,
)
from ...Utils.jobs_operation_contracts import (
    PreparedDispositionKind as PreparedDispositionKind,
)
from ...Utils.jobs_operation_contracts import (
    PreparedDispositionOrigin as PreparedDispositionOrigin,
)
from ...Utils.jobs_operation_contracts import (
    PreparedDispositionResult as PreparedDispositionResult,
)
from ...Utils.jobs_operation_contracts import (
    PreparedJobDisposition as PreparedJobDisposition,
)
from ...Utils.jobs_operation_contracts import (
    ReleaseJobCommand as ReleaseJobCommand,
)
from ...Utils.jobs_operation_contracts import (
    RenewLeaseCommand as RenewLeaseCommand,
)
from ...Utils.jobs_operation_contracts import (
    Sequence as Sequence,
)
from ...Utils.jobs_operation_contracts import (
    SlidesArchiveNormalizationError as SlidesArchiveNormalizationError,
)
from ...Utils.jobs_operation_contracts import (
    TerminalOperationResultPatchCommand as TerminalOperationResultPatchCommand,
)
from ...Utils.jobs_operation_contracts import (
    TerminalOperationResultPatchOutcome as TerminalOperationResultPatchOutcome,
)
from ...Utils.jobs_operation_contracts import (
    __all__ as __all__,
)
from ...Utils.jobs_operation_contracts import (
    _aware_utc as _aware_utc,
)
from ...Utils.jobs_operation_contracts import (
    _bounded_identity as _bounded_identity,
)
from ...Utils.jobs_operation_contracts import (
    _canonical_admin_webhook_marker as _canonical_admin_webhook_marker,
)
from ...Utils.jobs_operation_contracts import (
    _canonical_schedule_matches as _canonical_schedule_matches,
)
from ...Utils.jobs_operation_contracts import (
    _canonical_uuid4 as _canonical_uuid4,
)
from ...Utils.jobs_operation_contracts import (
    _stored_utc as _stored_utc,
)
from ...Utils.jobs_operation_contracts import (
    admin_webhook_disposition_marker_matches as admin_webhook_disposition_marker_matches,
)
from ...Utils.jobs_operation_contracts import (
    canonical_admin_webhook_delivery_id as canonical_admin_webhook_delivery_id,
)
from ...Utils.jobs_operation_contracts import (
    canonical_admin_webhook_idempotency_key as canonical_admin_webhook_idempotency_key,
)
from ...Utils.jobs_operation_contracts import (
    canonical_admin_webhook_row_matches as canonical_admin_webhook_row_matches,
)
from ...Utils.jobs_operation_contracts import (
    copy as copy,
)
from ...Utils.jobs_operation_contracts import (
    dataclass as dataclass,
)
from ...Utils.jobs_operation_contracts import (
    datetime as datetime,
)
from ...Utils.jobs_operation_contracts import (
    field as field,
)
from ...Utils.jobs_operation_contracts import (
    hashlib as hashlib,
)
from ...Utils.jobs_operation_contracts import (
    is_admin_webhook_delivery_queue as is_admin_webhook_delivery_queue,
)
from ...Utils.jobs_operation_contracts import (
    json as json,
)
from ...Utils.jobs_operation_contracts import (
    normalize_acquire_job_types as normalize_acquire_job_types,
)
from ...Utils.jobs_operation_contracts import (
    prepared_disposition_fingerprint as prepared_disposition_fingerprint,
)
from ...Utils.jobs_operation_contracts import (
    project_admin_webhook_disposition_marker as project_admin_webhook_disposition_marker,
)
from ...Utils.jobs_operation_contracts import (
    re as re,
)
from ...Utils.jobs_operation_contracts import (
    reconstruct_legacy_admin_webhook_archive_fingerprint as reconstruct_legacy_admin_webhook_archive_fingerprint,
)
from ...Utils.jobs_operation_contracts import (
    terminal_operation_result_fingerprint as terminal_operation_result_fingerprint,
)
from ...Utils.jobs_operation_contracts import (
    timezone as timezone,
)
