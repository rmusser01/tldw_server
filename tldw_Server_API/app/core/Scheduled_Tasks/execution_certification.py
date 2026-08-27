"""Fail-closed feasibility certification for Scheduled Tasks Agent execution.

Certification is a prerequisite, not execution authority. The current resolver
only emits repository characterization and therefore cannot certify a runtime.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Literal

CertificationOutcome = Literal["certified", "draft_only", "unsupported"]
EvidenceState = Literal["passed", "failed", "missing", "stale"]
EvidenceVerification = Literal[
    "server_verified",
    "host_gated_unverified",
    "repository_characterization",
    "mock",
    "self_asserted",
]
EvidenceSource = Literal["server_verified", "repository_characterization", "none"]
RequirementId = Literal[
    "isolation_attestation",
    "hostile_boundary",
    "scheduled_transcript_non_disclosure",
    "adapter_dispatch_recovery",
    "monotonic_execution_evidence",
    "brokered_credentials_and_mediation",
    "operational_fail_closed",
]

REQUIRED_EVIDENCE_DOMAINS: tuple[RequirementId, ...] = (
    "isolation_attestation",
    "hostile_boundary",
    "scheduled_transcript_non_disclosure",
    "adapter_dispatch_recovery",
    "monotonic_execution_evidence",
    "brokered_credentials_and_mediation",
    "operational_fail_closed",
)

MAX_REASON_CODES = 32
_EVIDENCE_VALIDITY = timedelta(hours=24)
_SHA256_ID_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")
_BUILD_SHA_PATTERN = re.compile(r"^[0-9a-fA-F]{40}(?:[0-9a-fA-F]{24})?$")

_BASE_REASON_CODES = {
    "agent_execution_stack_unimplemented",
    "authoritative_receipt_mismatch",
    "authoritative_receipt_missing",
    "authoritative_receipt_stale",
    "deployment_identity_unverified",
    "isolation_profile_identity_unverified",
    "runtime_not_untrusted_eligible",
    "runtime_strict_deny_all_unavailable",
    "safety_boundary_breached",
    "server_build_identity_unverified",
}
_REQUIREMENT_REASON_CODES = {
    f"{requirement_id}_{suffix}"
    for requirement_id in REQUIRED_EVIDENCE_DOMAINS
    for suffix in ("failed", "missing", "stale", "subject_mismatch", "unverified")
}
CERTIFICATION_REASON_CODES = frozenset(
    _BASE_REASON_CODES | _REQUIREMENT_REASON_CODES
)


def _canonical_digest(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _normalized_identity(value: object, *, lowercase: bool = True) -> str:
    normalized = str(value or "").strip()
    if not normalized:
        return "unverified"
    return normalized.lower() if lowercase else normalized


def _aware_utc(value: datetime | None) -> datetime | None:
    if value is None or value.tzinfo is None or value.utcoffset() is None:
        return None
    return value.astimezone(timezone.utc)


def _isoformat(value: datetime | None) -> str | None:
    normalized = _aware_utc(value)
    return normalized.isoformat() if normalized is not None else None


def _valid_sha256_id(value: str | None) -> bool:
    return bool(value and _SHA256_ID_PATTERN.fullmatch(value))


@dataclass(frozen=True)
class IsolationProfile:
    """Exact isolation inputs collapsed into one opaque profile fingerprint."""

    runtime_image_digest: str
    mount_policy_hash: str
    egress_policy_hash: str
    credential_policy_hash: str
    tenant_boundary_policy_hash: str
    mediation_policy_hash: str
    isolation_profile_version: str

    def canonical_payload(self) -> dict[str, str]:
        """Return deterministic private identity inputs for hashing."""

        values = {
            "credential_policy_hash": _normalized_identity(
                self.credential_policy_hash
            ),
            "egress_policy_hash": _normalized_identity(self.egress_policy_hash),
            "isolation_profile_version": _normalized_identity(
                self.isolation_profile_version,
                lowercase=False,
            ),
            "mediation_policy_hash": _normalized_identity(
                self.mediation_policy_hash
            ),
            "mount_policy_hash": _normalized_identity(self.mount_policy_hash),
            "runtime_image_digest": _normalized_identity(
                self.runtime_image_digest
            ),
            "tenant_boundary_policy_hash": _normalized_identity(
                self.tenant_boundary_policy_hash
            ),
        }
        return dict(sorted(values.items()))

    @property
    def isolation_profile_id(self) -> str:
        """Return the opaque digest exposed through deployment identity."""

        return _canonical_digest(self.canonical_payload())

    @property
    def is_verified_identity(self) -> bool:
        """Return whether every profile component has an explicit identity."""

        return all(value != "unverified" for value in self.canonical_payload().values())


@dataclass(frozen=True)
class DeploymentClass:
    """Exact subject to which Scheduled Tasks execution evidence is bound."""

    host_os_family: str
    host_architecture: str
    auth_mode: str
    sandbox_runtime: str
    adapter_id: str
    adapter_version: str
    server_build_sha: str
    isolation_profile: IsolationProfile

    def canonical_payload(self) -> dict[str, str]:
        """Return stable subject fields without exposing raw profile inputs."""

        values = {
            "adapter_id": _normalized_identity(self.adapter_id),
            "adapter_version": _normalized_identity(
                self.adapter_version,
                lowercase=False,
            ),
            "auth_mode": _normalized_identity(self.auth_mode),
            "host_architecture": _normalized_identity(self.host_architecture),
            "host_os_family": _normalized_identity(self.host_os_family),
            "isolation_profile_id": self.isolation_profile.isolation_profile_id,
            "sandbox_runtime": _normalized_identity(self.sandbox_runtime),
            "server_build_sha": _normalized_identity(self.server_build_sha),
        }
        return dict(sorted(values.items()))

    @property
    def deployment_class_id(self) -> str:
        """Return an opaque digest for the exact deployment class."""

        return _canonical_digest(self.canonical_payload())

    @property
    def has_verified_identity(self) -> bool:
        """Return whether all top-level and isolation identities are explicit."""

        return (
            all(value != "unverified" for value in self.canonical_payload().values())
            and self.isolation_profile.is_verified_identity
        )


@dataclass(frozen=True)
class RuntimeEligibility:
    """Static isolation and deny-all eligibility used before evidence."""

    untrusted_eligible: bool
    strict_deny_all: bool


@dataclass(frozen=True)
class RequirementEvidence:
    """One bounded evidence-domain result for an exact deployment class."""

    requirement_id: str
    state: EvidenceState
    verification: EvidenceVerification
    subject_id: str
    observed_at: datetime | None
    valid_until: datetime | None
    evidence_sha256: str | None
    safety_boundary_breached: bool = False

    def canonical_payload(self) -> dict[str, object]:
        """Return the sanitized fields covered by the bundle receipt."""

        values: dict[str, object] = {
            "evidence_sha256": self.evidence_sha256,
            "observed_at": _isoformat(self.observed_at),
            "requirement_id": self.requirement_id,
            "safety_boundary_breached": self.safety_boundary_breached,
            "state": self.state,
            "subject_id": self.subject_id,
            "valid_until": _isoformat(self.valid_until),
            "verification": self.verification,
        }
        return dict(sorted(values.items()))


def canonical_evidence_bundle_digest(
    evidence: Sequence[RequirementEvidence],
) -> str:
    """Return the digest an authoritative verifier must cover."""

    payload = [
        item.canonical_payload()
        for item in sorted(evidence, key=lambda candidate: candidate.requirement_id)
    ]
    return _canonical_digest(payload)


class _ServerVerifierAuthority:
    """Unserializable constructor authority reserved for a later verifier."""


_SERVER_VERIFIER_AUTHORITY = _ServerVerifierAuthority()


@dataclass(frozen=True)
class _AuthoritativeBundleReceipt:
    """Internal receipt proving that the server verified one exact bundle."""

    deployment_class_id: str
    evidence_id: str
    bundle_digest: str
    observed_at: datetime | None
    valid_until: datetime | None
    _authority: object = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._authority is not _SERVER_VERIFIER_AUTHORITY:
            raise ValueError("authoritative receipt requires server verifier authority")


@dataclass(frozen=True)
class ExecutionCertification:
    """Immutable feasibility outcome projected to capability and admission gates."""

    outcome: CertificationOutcome
    deployment_class_id: str
    evidence_id: str | None
    evidence_source: EvidenceSource
    observed_at: datetime | None
    expires_at: datetime | None
    reason_codes: tuple[str, ...]


@dataclass(frozen=True)
class AgentExecutionDispatchReadiness:
    """Conjunction of certification and the independently delivered stack."""

    ready: bool
    reason: str | None


def _validate_requirement_set(evidence: Sequence[RequirementEvidence]) -> None:
    seen: set[str] = set()
    allowed = set(REQUIRED_EVIDENCE_DOMAINS)
    for item in evidence:
        if item.requirement_id not in allowed:
            raise ValueError(f"unknown certification requirement: {item.requirement_id!r}")
        if item.requirement_id in seen:
            raise ValueError(
                f"duplicate certification requirement: {item.requirement_id!r}"
            )
        seen.add(item.requirement_id)


def _record_reason_codes(
    item: RequirementEvidence,
    *,
    subject_id: str,
    now: datetime,
) -> set[str]:
    prefix = item.requirement_id
    reasons: set[str] = set()
    if item.state == "missing":
        reasons.add(f"{prefix}_missing")
    elif item.state == "failed":
        reasons.add(f"{prefix}_failed")
    elif item.state == "stale":
        reasons.add(f"{prefix}_stale")
    if item.subject_id != subject_id:
        reasons.add(f"{prefix}_subject_mismatch")
    if item.verification != "server_verified":
        reasons.add(f"{prefix}_unverified")

    observed_at = _aware_utc(item.observed_at)
    valid_until = _aware_utc(item.valid_until)
    if (
        observed_at is None
        or valid_until is None
        or observed_at > now
        or valid_until <= now
        or valid_until <= observed_at
    ):
        reasons.add(f"{prefix}_stale")
    if not _valid_sha256_id(item.evidence_sha256):
        reasons.add(f"{prefix}_unverified")
    return reasons


def _receipt_reason_codes(
    receipt: _AuthoritativeBundleReceipt | None,
    *,
    subject_id: str,
    evidence: Sequence[RequirementEvidence],
    now: datetime,
) -> set[str]:
    if receipt is None:
        return {"authoritative_receipt_missing"}

    reasons: set[str] = set()
    if (
        receipt.deployment_class_id != subject_id
        or receipt.bundle_digest != canonical_evidence_bundle_digest(evidence)
        or not _valid_sha256_id(receipt.evidence_id)
        or not _valid_sha256_id(receipt.bundle_digest)
    ):
        reasons.add("authoritative_receipt_mismatch")

    observed_at = _aware_utc(receipt.observed_at)
    valid_until = _aware_utc(receipt.valid_until)
    if (
        observed_at is None
        or valid_until is None
        or observed_at > now
        or valid_until <= now
        or valid_until <= observed_at
    ):
        reasons.add("authoritative_receipt_stale")
    return reasons


def _bounded_reason_codes(reasons: set[str]) -> tuple[str, ...]:
    unknown = reasons - CERTIFICATION_REASON_CODES
    if unknown:
        raise ValueError("unregistered certification reason code")
    return tuple(sorted(reasons)[:MAX_REASON_CODES])


def evaluate_execution_certification(
    subject: DeploymentClass,
    evidence: Sequence[RequirementEvidence],
    verification_receipt: _AuthoritativeBundleReceipt | None,
    *,
    runtime_eligibility: RuntimeEligibility,
    now: datetime,
) -> ExecutionCertification:
    """Evaluate an exact evidence bundle without I/O or mutable configuration."""

    evaluated_at = _aware_utc(now)
    if evaluated_at is None:
        raise ValueError("certification evaluation requires an aware UTC clock")
    _validate_requirement_set(evidence)

    reasons: set[str] = set()
    if not runtime_eligibility.untrusted_eligible:
        reasons.add("runtime_not_untrusted_eligible")
    if not runtime_eligibility.strict_deny_all:
        reasons.add("runtime_strict_deny_all_unavailable")
    if not subject.has_verified_identity:
        reasons.add("deployment_identity_unverified")
    if subject.server_build_sha == "unverified":
        reasons.add("server_build_identity_unverified")
    if not subject.isolation_profile.is_verified_identity:
        reasons.add("isolation_profile_identity_unverified")

    by_requirement = {item.requirement_id: item for item in evidence}
    for requirement_id in REQUIRED_EVIDENCE_DOMAINS:
        item = by_requirement.get(requirement_id)
        if item is None:
            reasons.add(f"{requirement_id}_missing")
            continue
        reasons.update(
            _record_reason_codes(
                item,
                subject_id=subject.deployment_class_id,
                now=evaluated_at,
            )
        )
        if item.safety_boundary_breached:
            reasons.add("safety_boundary_breached")

    reasons.update(
        _receipt_reason_codes(
            verification_receipt,
            subject_id=subject.deployment_class_id,
            evidence=evidence,
            now=evaluated_at,
        )
    )

    unsupported = (
        not runtime_eligibility.untrusted_eligible
        or not runtime_eligibility.strict_deny_all
        or "safety_boundary_breached" in reasons
    )
    if unsupported:
        outcome: CertificationOutcome = "unsupported"
    elif reasons:
        outcome = "draft_only"
    else:
        outcome = "certified"

    repository_characterization = any(
        item.verification == "repository_characterization" for item in evidence
    )
    if outcome == "certified":
        evidence_source: EvidenceSource = "server_verified"
    elif repository_characterization:
        evidence_source = "repository_characterization"
    else:
        evidence_source = "none"

    observed_values = [
        value
        for value in (_aware_utc(item.observed_at) for item in evidence)
        if value is not None
    ]
    expiry_values = [
        value
        for value in (_aware_utc(item.valid_until) for item in evidence)
        if value is not None
    ]
    return ExecutionCertification(
        outcome=outcome,
        deployment_class_id=subject.deployment_class_id,
        evidence_id=(
            verification_receipt.evidence_id
            if outcome == "certified" and verification_receipt is not None
            else None
        ),
        evidence_source=evidence_source,
        observed_at=max(observed_values) if observed_values else None,
        expires_at=min(expiry_values) if expiry_values else None,
        reason_codes=_bounded_reason_codes(reasons),
    )


def agent_execution_dispatch_readiness(
    certification: ExecutionCertification,
    *,
    execution_stack_ready: bool,
) -> AgentExecutionDispatchReadiness:
    """Require both certified evidence and a separately implemented stack."""

    if certification.outcome != "certified":
        return AgentExecutionDispatchReadiness(
            ready=False,
            reason=f"execution_certification_{certification.outcome}",
        )
    if not execution_stack_ready:
        return AgentExecutionDispatchReadiness(
            ready=False,
            reason="agent_execution_stack_unimplemented",
        )
    return AgentExecutionDispatchReadiness(ready=True, reason=None)


def current_agent_execution_stack_ready() -> bool:
    """Return the source-defined Phase 4D execution-stack readiness state."""

    return False


def _normalize_build_sha(value: str | None) -> str:
    normalized = str(value or "").strip()
    if not _BUILD_SHA_PATTERN.fullmatch(normalized):
        return "unverified"
    return normalized.lower()


def _default_isolation_profile() -> IsolationProfile:
    return IsolationProfile(
        runtime_image_digest="unverified",
        mount_policy_hash="unverified",
        egress_policy_hash="unverified",
        credential_policy_hash="unverified",
        tenant_boundary_policy_hash="unverified",
        mediation_policy_hash="unverified",
        isolation_profile_version="phase4d0f-current",
    )


def _runtime_eligibility(runtime: str) -> RuntimeEligibility:
    from tldw_Server_API.app.core.Sandbox.runtime_capabilities import (
        runtime_isolation_metadata,
        runtime_network_policy_metadata,
    )

    try:
        isolation = runtime_isolation_metadata(runtime)
        deny_all = runtime_network_policy_metadata(runtime).deny_all
    except ValueError:
        return RuntimeEligibility(untrusted_eligible=False, strict_deny_all=False)
    return RuntimeEligibility(
        untrusted_eligible=isolation.untrusted_eligible,
        strict_deny_all=(
            deny_all.strict_enforcement
            and deny_all.support_state in {"supported", "host_gated"}
        ),
    )


def _current_identity_defaults() -> tuple[str, str]:
    from tldw_Server_API.app.core.Agent_Client_Protocol.config import (
        load_acp_sandbox_config,
    )
    from tldw_Server_API.app.core.AuthNZ.settings import get_settings

    return load_acp_sandbox_config().runtime, get_settings().AUTH_MODE


def resolve_current_agent_execution_certification(
    *,
    now: datetime | None = None,
    host_os_family: str | None = None,
    host_architecture: str | None = None,
    auth_mode: str | None = None,
    sandbox_runtime: str | None = None,
    adapter_id: str = "acp",
    adapter_version: str = "1",
    isolation_profile: IsolationProfile | None = None,
) -> ExecutionCertification:
    """Project current static metadata without granting certification authority."""

    if sandbox_runtime is None or auth_mode is None:
        default_runtime, default_auth_mode = _current_identity_defaults()
        sandbox_runtime = sandbox_runtime or default_runtime
        auth_mode = auth_mode or default_auth_mode
    evaluated_at = _aware_utc(now or datetime.now(timezone.utc))
    if evaluated_at is None:
        raise ValueError("certification resolution requires an aware UTC clock")

    subject = DeploymentClass(
        host_os_family=host_os_family or platform.system(),
        host_architecture=host_architecture or platform.machine(),
        auth_mode=auth_mode,
        sandbox_runtime=sandbox_runtime,
        adapter_id=adapter_id,
        adapter_version=adapter_version,
        server_build_sha=_normalize_build_sha(os.getenv("TLDW_BUILD_SHA")),
        isolation_profile=isolation_profile or _default_isolation_profile(),
    )
    valid_until = evaluated_at + _EVIDENCE_VALIDITY
    evidence = tuple(
        RequirementEvidence(
            requirement_id=requirement_id,
            state="missing",
            verification="repository_characterization",
            subject_id=subject.deployment_class_id,
            observed_at=evaluated_at,
            valid_until=valid_until,
            evidence_sha256=_canonical_digest(
                {
                    "requirement_id": requirement_id,
                    "source": "repository_characterization",
                    "subject_id": subject.deployment_class_id,
                }
            ),
        )
        for requirement_id in REQUIRED_EVIDENCE_DOMAINS
    )
    return evaluate_execution_certification(
        subject,
        evidence,
        None,
        runtime_eligibility=_runtime_eligibility(sandbox_runtime),
        now=evaluated_at,
    )
