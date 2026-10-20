"""Immutable, closed facts for explicitly enrolled Jobs workflows.

Decoding facts is not authorization. A trusted capability policy and a live,
owner-scoped transaction are required before any future lifecycle operation.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from uuid import UUID

from .contracts import CreateJobCommand, ExpiredLeasePolicy, is_admin_webhook_delivery_queue

_CODE = re.compile(r"[a-z][a-z0-9_.-]{0,63}", re.ASCII)
_DIGEST = re.compile(r"[0-9a-f]{64}", re.ASCII)
NO_PROGRESS_EXHAUSTED = "no_progress_exhausted"
WORKFLOW_FACTS_MAX_BYTES = 4096


class WorkflowAction(str, Enum):
    """Closed Jobs lifecycle intents, not arbitrary callbacks."""

    CONTINUE = "continue"
    RETRY = "retry"
    COMPLETE = "complete"
    FAIL = "fail"
    CANCEL = "cancel"


class WorkflowProgressKind(str, Enum):
    """Durable facts that can demonstrate useful progress."""

    ROOT_BINDING = "root_binding"
    OUTPUT_RECEIPT = "output_receipt"
    CURSOR = "cursor"
    ACCOUNTING = "accounting"
    SINK_LIFECYCLE = "sink_lifecycle"


class WorkflowOutcome(str, Enum):
    """Closed capability outcomes, shared by both storage backends."""

    APPLIED = "applied"
    ALREADY_APPLIED = "already_applied"
    ACTIVE = "active"
    ARCHIVED = "archived"
    MISSING = "missing"
    CONFLICT = "conflict"
    STALE_LEASE = "stale_lease"
    TERMINAL = "terminal"
    INSUFFICIENT_HORIZON = "insufficient_horizon"
    REJECTED = "rejected"
    UNAVAILABLE = "unavailable"
    EXHAUSTED = "exhausted"


class WorkflowFenceError(RuntimeError):
    """A closed rejection of transaction or live-root authority."""

    def __init__(self, outcome: WorkflowOutcome) -> None:
        self.outcome = outcome
        super().__init__(outcome.value)


@dataclass(frozen=True)
class WorkflowAdmissionInputs:
    """Policy-prepared admission facts, without a connection or commit."""

    job: CreateJobCommand
    max_queued_quota: int
    submits_per_minute_quota: int
    counters_enabled: bool
    outbox_enabled: bool


def _code(value: object) -> None:
    if type(value) is not str or _CODE.fullmatch(value) is None:
        raise ValueError("workflow fixed code must be 1-64 ASCII characters")


def _text(value: object, maximum: int) -> None:
    try:
        valid = type(value) is str and bool(value.strip()) and len(value.encode("utf-8")) <= maximum
    except UnicodeError:
        valid = False
    if not valid:
        raise ValueError(f"workflow identity must be nonblank and at most {maximum} UTF-8 bytes")


def _uuid(value: object) -> None:
    try:
        parsed = UUID(value) if type(value) is str else None
    except ValueError:
        parsed = None
    if parsed is None or parsed.version != 4 or str(parsed) != value:
        raise ValueError("workflow identity must be a canonical UUIDv4")


def _digest(value: object) -> None:
    if type(value) is not str or _DIGEST.fullmatch(value) is None:
        raise ValueError("workflow key must be 64 lowercase hexadecimal characters")


def _integer(value: object, minimum: int) -> None:
    if type(value) is not int or value < minimum:
        raise ValueError("workflow control must be an exact bounded integer")


def _utc(value: object) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("workflow datetime must be timezone-aware")
    return value.astimezone(timezone.utc)


def _exact(value: object, expected: type) -> None:
    if type(value) is not expected:
        raise ValueError("workflow fact has an unsupported type")


@dataclass(frozen=True)
class WorkflowScope:
    """Trusted, domain-neutral location of a workflow root."""

    domain: str
    queue: str
    job_type: str

    def __post_init__(self) -> None:
        for value in (self.domain, self.queue, self.job_type):
            _code(value)
        if is_admin_webhook_delivery_queue(self.domain, self.queue):
            raise ValueError("Admin Webhook scope is reserved")


@dataclass(frozen=True)
class WorkflowPolicy:
    """Trusted payload-field and reason vocabulary; never inferred from input."""

    scope: WorkflowScope
    payload_field: str
    reason_codes: frozenset[str]

    def __post_init__(self) -> None:
        _exact(self.scope, WorkflowScope)
        _code(self.payload_field)
        if not isinstance(self.reason_codes, (set, frozenset, list, tuple)):
            raise ValueError("workflow reasons must be a collection of fixed codes")
        for reason in self.reason_codes:
            _code(reason)
        reasons = frozenset(self.reason_codes)
        if NO_PROGRESS_EXHAUSTED not in reasons:
            raise ValueError("workflow reasons must include no_progress_exhausted")
        object.__setattr__(self, "reason_codes", reasons)

    def validate_effect(self, effect: WorkflowEffect) -> None:
        """Reject reasons outside this trusted policy's allowlist."""
        _exact(effect, WorkflowEffect)
        if effect.reason_code is not None and effect.reason_code not in self.reason_codes:
            raise ValueError("workflow reason is not allowed by policy")


@dataclass(frozen=True)
class WorkflowControls:
    """Persisted retry-neutral expiry and bounded no-progress policy."""

    max_retries: int
    quarantine_threshold: int | None
    expired_lease_policy: ExpiredLeasePolicy
    no_progress_limit: int = 5

    def __post_init__(self) -> None:
        _integer(self.max_retries, 0)
        if self.quarantine_threshold is not None:
            _integer(self.quarantine_threshold, 1)
        if self.expired_lease_policy is not ExpiredLeasePolicy.REQUEUE_NO_ATTEMPT:
            raise ValueError("workflow expiry must be REQUEUE_NO_ATTEMPT")
        _integer(self.no_progress_limit, 1)
        if self.no_progress_limit > 5:
            raise ValueError("workflow no-progress limit must be 1-5")


@dataclass(frozen=True)
class WorkflowAdmissionCommand:
    """Exact owner/reference/idempotency identity and persisted controls."""

    owner_user_id: str
    reference_id: str
    idempotency_key: str
    controls: WorkflowControls

    def __post_init__(self) -> None:
        _text(self.owner_user_id, 512)
        _uuid(self.reference_id)
        _text(self.idempotency_key, 200)
        _exact(self.controls, WorkflowControls)


@dataclass(frozen=True)
class WorkflowRoot:
    """Durable authority; the numeric Jobs ID is only a lookup hint."""

    job_uuid: str
    owner_user_id: str
    scope: WorkflowScope
    reference_id: str
    idempotency_key: str
    controls: WorkflowControls
    job_id_hint: int | None = None

    def __post_init__(self) -> None:
        _uuid(self.job_uuid)
        _exact(self.scope, WorkflowScope)
        WorkflowAdmissionCommand(self.owner_user_id, self.reference_id, self.idempotency_key, self.controls)
        if self.job_id_hint is not None:
            _integer(self.job_id_hint, 1)


@dataclass(frozen=True)
class WorkflowLease:
    """Opaque current-delivery claims, excluded from logical effect identity."""

    worker_id: str
    lease_id: str

    def __post_init__(self) -> None:
        _text(self.worker_id, 512)
        _text(self.lease_id, 512)


@dataclass(frozen=True)
class WorkflowEffect:
    """One scheduled or terminal intent with no unrelated retry facts."""

    effect_key: str
    action: WorkflowAction
    guard_fingerprint: str
    reason_code: str | None = None
    not_before_at: datetime | None = None

    def __post_init__(self) -> None:
        _digest(self.effect_key)
        _exact(self.action, WorkflowAction)
        _digest(self.guard_fingerprint)
        needs_reason = self.action in {WorkflowAction.RETRY, WorkflowAction.FAIL}
        if needs_reason != (self.reason_code is not None):
            raise ValueError("workflow reason is required only for retry/fail")
        if self.reason_code is not None:
            _code(self.reason_code)
        scheduled = self.action in {WorkflowAction.CONTINUE, WorkflowAction.RETRY}
        if scheduled != (self.not_before_at is not None):
            raise ValueError("workflow schedule is required only for continue/retry")
        if self.not_before_at is not None:
            object.__setattr__(self, "not_before_at", _utc(self.not_before_at))


@dataclass(frozen=True)
class WorkflowProgress:
    """A bounded, typed durable-progress proof identity."""

    proof_key: str
    kind: WorkflowProgressKind
    facts_fingerprint: str

    def __post_init__(self) -> None:
        _digest(self.proof_key)
        _exact(self.kind, WorkflowProgressKind)
        _digest(self.facts_fingerprint)


_ACTION_STATUSES = {
    WorkflowAction.CONTINUE: frozenset({"queued"}),
    WorkflowAction.RETRY: frozenset({"queued", "failed", "quarantined"}),
    WorkflowAction.COMPLETE: frozenset({"completed"}),
    WorkflowAction.FAIL: frozenset({"failed", "quarantined"}),
    WorkflowAction.CANCEL: frozenset({"cancelled"}),
}


@dataclass(frozen=True)
class WorkflowReceipt:
    """Immutable acknowledgment facts; release does not create another slot."""

    effect_key: str
    facts_fingerprint: str
    job_uuid: str
    action: WorkflowAction
    resulting_status: str
    applied_at: datetime
    released: bool = False

    def __post_init__(self) -> None:
        _digest(self.effect_key)
        _digest(self.facts_fingerprint)
        _uuid(self.job_uuid)
        _exact(self.action, WorkflowAction)
        _exact(self.resulting_status, str)
        if self.resulting_status not in _ACTION_STATUSES[self.action]:
            raise ValueError("workflow receipt status does not match action")
        object.__setattr__(self, "applied_at", _utc(self.applied_at))
        _exact(self.released, bool)


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def workflow_effect_fingerprint(root: WorkflowRoot, effect: WorkflowEffect) -> str:
    """Hash logical authority and original UTC schedule, never delivery claims."""
    _exact(root, WorkflowRoot)
    _exact(effect, WorkflowEffect)
    facts = {
        "version": "jobs_workflow_effect_v1",
        "owner_user_id": root.owner_user_id,
        "job_uuid": root.job_uuid,
        "scope": [root.scope.domain, root.scope.queue, root.scope.job_type],
        "reference_id": root.reference_id,
        "idempotency_key": root.idempotency_key,
        "controls": [
            root.controls.max_retries,
            root.controls.quarantine_threshold,
            root.controls.expired_lease_policy.value,
            root.controls.no_progress_limit,
        ],
        "effect_key": effect.effect_key,
        "action": effect.action.value,
        "guard_fingerprint": effect.guard_fingerprint,
        "reason_code": effect.reason_code,
        "not_before_at": effect.not_before_at.isoformat() if effect.not_before_at else None,
    }
    return hashlib.sha256(_canonical(facts)).hexdigest()


def bounded_workflow_json(raw: bytes) -> object:
    """Parse at most 4096 raw bytes; never echo untrusted input on failure."""
    if type(raw) is not bytes or len(raw) > WORKFLOW_FACTS_MAX_BYTES:
        raise ValueError("workflow JSON must be bytes of at most 4096 bytes")

    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate workflow JSON key")
            result[key] = value
        return result

    def invalid_constant(_value):
        raise ValueError("nonfinite workflow JSON value")

    try:
        decoded = json.loads(raw.decode("utf-8"), object_pairs_hook=pairs, parse_constant=invalid_constant)
        _canonical(decoded)
        # A shallow wire format keeps deeply nested facts out of typed decoders.
        stack = [(decoded, 0)]
        while stack:
            value, depth = stack.pop()
            if depth > 32:
                raise ValueError("workflow JSON nesting exceeds bound")
            if isinstance(value, dict):
                stack.extend((item, depth + 1) for item in value.values())
            elif isinstance(value, list):
                stack.extend((item, depth + 1) for item in value)
        return decoded
    except (ValueError, UnicodeError, RecursionError, OverflowError):
        raise ValueError("invalid bounded workflow JSON") from None


def encode_workflow_receipt(receipt: WorkflowReceipt) -> bytes:
    """Serialize canonical, byte-bounded receipt facts without mutable aliases."""
    _exact(receipt, WorkflowReceipt)
    raw = _canonical(
        {
            "effect_key": receipt.effect_key,
            "facts_fingerprint": receipt.facts_fingerprint,
            "job_uuid": receipt.job_uuid,
            "action": receipt.action.value,
            "resulting_status": receipt.resulting_status,
            "applied_at": receipt.applied_at.isoformat(),
            "released": receipt.released,
        }
    )
    bounded_workflow_json(raw)
    return raw


def decode_workflow_receipt(raw: bytes) -> WorkflowReceipt:
    """Decode only the closed scalar receipt schema, not execution authority."""
    facts = bounded_workflow_json(raw)
    fields = {"effect_key", "facts_fingerprint", "job_uuid", "action", "resulting_status", "applied_at", "released"}
    if type(facts) is not dict or set(facts) != fields:
        raise ValueError("invalid workflow receipt fields")
    try:
        if any(type(facts[key]) is not str for key in fields - {"released"}):
            raise ValueError("invalid workflow receipt scalar")
        return WorkflowReceipt(
            facts["effect_key"],
            facts["facts_fingerprint"],
            facts["job_uuid"],
            WorkflowAction(facts["action"]),
            facts["resulting_status"],
            datetime.fromisoformat(facts["applied_at"]),
            facts["released"],
        )
    except (ValueError, TypeError):
        raise ValueError("invalid workflow receipt facts") from None


def _outcome(outcome: WorkflowOutcome, allowed: set[WorkflowOutcome]) -> None:
    _exact(outcome, WorkflowOutcome)
    if outcome not in allowed:
        raise ValueError("unsupported workflow result outcome")


_REJECTIONS = {WorkflowOutcome.MISSING, WorkflowOutcome.CONFLICT, WorkflowOutcome.REJECTED, WorkflowOutcome.UNAVAILABLE}


@dataclass(frozen=True)
class WorkflowAdmissionResult:
    """Admission either supplies exact root authority or no root at all."""

    outcome: WorkflowOutcome
    root: WorkflowRoot | None = None

    def __post_init__(self) -> None:
        success = {WorkflowOutcome.APPLIED, WorkflowOutcome.ALREADY_APPLIED}
        _outcome(self.outcome, success | _REJECTIONS | {WorkflowOutcome.TERMINAL})
        if self.outcome in success:
            _exact(self.root, WorkflowRoot)
        elif self.root is not None:
            raise ValueError("unsuccessful workflow admission cannot carry root authority")


@dataclass(frozen=True)
class WorkflowLookupResult:
    """Only active/archive lookups carry a root and corresponding Jobs status."""

    outcome: WorkflowOutcome
    root: WorkflowRoot | None = None
    status: str | None = None

    def __post_init__(self) -> None:
        success = {WorkflowOutcome.ACTIVE, WorkflowOutcome.ARCHIVED}
        _outcome(self.outcome, success | _REJECTIONS)
        if self.outcome in success:
            _exact(self.root, WorkflowRoot)
            allowed = {"completed", "failed", "cancelled", "quarantined"}
            if self.outcome is WorkflowOutcome.ACTIVE:
                allowed |= {"queued", "processing"}
            if type(self.status) is not str or self.status not in allowed:
                raise ValueError("workflow lookup needs a valid Jobs status")
        elif self.root is not None or self.status is not None:
            raise ValueError("unsuccessful workflow lookup cannot carry facts")


@dataclass(frozen=True)
class WorkflowEffectResult:
    """Only an applied/replayed action acknowledges a receipt."""

    outcome: WorkflowOutcome
    receipt: WorkflowReceipt | None = None

    def __post_init__(self) -> None:
        success = {WorkflowOutcome.APPLIED, WorkflowOutcome.ALREADY_APPLIED}
        _outcome(
            self.outcome,
            success | _REJECTIONS | {WorkflowOutcome.STALE_LEASE, WorkflowOutcome.TERMINAL, WorkflowOutcome.EXHAUSTED},
        )
        if self.outcome in success:
            _exact(self.receipt, WorkflowReceipt)
        elif self.receipt is not None:
            raise ValueError("unapplied workflow action cannot acknowledge a receipt")


@dataclass(frozen=True)
class WorkflowHorizonResult:
    """A positive guarantee exists only with an aware lease deadline."""

    outcome: WorkflowOutcome
    leased_until: datetime | None = None
    guaranteed_seconds: int = 0

    def __post_init__(self) -> None:
        _outcome(
            self.outcome,
            _REJECTIONS
            | {
                WorkflowOutcome.APPLIED,
                WorkflowOutcome.INSUFFICIENT_HORIZON,
                WorkflowOutcome.STALE_LEASE,
                WorkflowOutcome.TERMINAL,
            },
        )
        _integer(self.guaranteed_seconds, 0)
        if self.outcome is WorkflowOutcome.APPLIED:
            _integer(self.guaranteed_seconds, 1)
            object.__setattr__(self, "leased_until", _utc(self.leased_until))
        elif self.leased_until is not None or self.guaranteed_seconds != 0:
            raise ValueError("unsuccessful horizon cannot promise a lease")


@dataclass(frozen=True)
class WorkflowProgressResult:
    """A reset can only accompany newly applied durable progress."""

    outcome: WorkflowOutcome
    reset: bool = False

    def __post_init__(self) -> None:
        _outcome(
            self.outcome,
            _REJECTIONS
            | {
                WorkflowOutcome.APPLIED,
                WorkflowOutcome.ALREADY_APPLIED,
                WorkflowOutcome.STALE_LEASE,
                WorkflowOutcome.TERMINAL,
                WorkflowOutcome.EXHAUSTED,
            },
        )
        _exact(self.reset, bool)
        if self.reset and self.outcome is not WorkflowOutcome.APPLIED:
            raise ValueError("only newly applied progress can reset a streak")
