"""Fail-closed Scheduled Tasks Agent execution certification tests."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest

from tldw_Server_API.app.core.Scheduled_Tasks import execution_certification as cert

pytestmark = pytest.mark.unit

NOW = datetime(2026, 8, 26, 19, 30, tzinfo=timezone.utc)
BUILD_SHA = "a" * 40
REQUIREMENT_IDS = (
    "isolation_attestation",
    "hostile_boundary",
    "scheduled_transcript_non_disclosure",
    "adapter_dispatch_recovery",
    "monotonic_execution_evidence",
    "brokered_credentials_and_mediation",
    "operational_fail_closed",
)


def _profile() -> cert.IsolationProfile:
    return cert.IsolationProfile(
        runtime_image_digest="sha256:" + "1" * 64,
        mount_policy_hash="sha256:" + "2" * 64,
        egress_policy_hash="sha256:" + "3" * 64,
        credential_policy_hash="sha256:" + "4" * 64,
        tenant_boundary_policy_hash="sha256:" + "5" * 64,
        mediation_policy_hash="sha256:" + "6" * 64,
        isolation_profile_version="phase4d0f-v1",
    )


def _deployment(**changes: object) -> cert.DeploymentClass:
    values: dict[str, object] = {
        "host_os_family": "darwin",
        "host_architecture": "arm64",
        "auth_mode": "single_user",
        "sandbox_runtime": "docker",
        "adapter_id": "acp",
        "adapter_version": "1",
        "server_build_sha": BUILD_SHA,
        "isolation_profile": _profile(),
    }
    values.update(changes)
    return cert.DeploymentClass(**values)


def _evidence(
    subject: cert.DeploymentClass,
    *,
    verification: cert.EvidenceVerification = "server_verified",
    state: cert.EvidenceState = "passed",
) -> tuple[cert.RequirementEvidence, ...]:
    return tuple(
        cert.RequirementEvidence(
            requirement_id=requirement_id,
            state=state,
            verification=verification,
            subject_id=subject.deployment_class_id,
            observed_at=NOW - timedelta(minutes=5),
            valid_until=NOW + timedelta(hours=1),
            evidence_sha256="sha256:" + f"{index:x}" * 64,
        )
        for index, requirement_id in enumerate(REQUIREMENT_IDS, start=1)
    )


def _authoritative_receipt(
    subject: cert.DeploymentClass,
    evidence: tuple[cert.RequirementEvidence, ...],
    *,
    valid_until: datetime | None = None,
    bundle_digest: str | None = None,
) -> cert._AuthoritativeBundleReceipt:
    """Construct the otherwise unavailable verifier receipt for pure rule tests."""

    return cert._AuthoritativeBundleReceipt(
        deployment_class_id=subject.deployment_class_id,
        evidence_id="sha256:" + "e" * 64,
        bundle_digest=bundle_digest or cert.canonical_evidence_bundle_digest(evidence),
        observed_at=NOW - timedelta(minutes=1),
        valid_until=valid_until or NOW + timedelta(minutes=30),
        _authority=cert._SERVER_VERIFIER_AUTHORITY,
    )


def _evaluate(
    subject: cert.DeploymentClass,
    evidence: tuple[cert.RequirementEvidence, ...],
    receipt: cert._AuthoritativeBundleReceipt | None,
    *,
    untrusted_eligible: bool = True,
    strict_deny_all: bool = True,
) -> cert.ExecutionCertification:
    return cert.evaluate_execution_certification(
        subject,
        evidence,
        receipt,
        runtime_eligibility=cert.RuntimeEligibility(
            untrusted_eligible=untrusted_eligible,
            strict_deny_all=strict_deny_all,
        ),
        now=NOW,
    )


def test_deployment_identity_is_canonical_and_opaque() -> None:
    """A field-order change must not alter an exact deployment-class identity."""

    deployment = _deployment()
    payload = deployment.canonical_payload()
    expected = "sha256:" + hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("ascii")
    ).hexdigest()

    assert list(payload) == sorted(payload)
    assert deployment.deployment_class_id == expected
    assert len(deployment.deployment_class_id) == 71


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("host_os_family", "linux"),
        ("host_architecture", "x86_64"),
        ("auth_mode", "multi_user"),
        ("sandbox_runtime", "lima"),
        ("adapter_id", "acp-v2"),
        ("adapter_version", "2"),
        ("server_build_sha", "b" * 40),
        (
            "isolation_profile",
            replace(_profile(), egress_policy_hash="sha256:" + "7" * 64),
        ),
    ],
)
def test_every_deployment_identity_field_changes_the_digest(
    field: str,
    replacement: object,
) -> None:
    """Changing any exact subject binding must invalidate its evidence identity."""

    baseline = _deployment()
    changed = replace(baseline, **{field: replacement})

    assert changed.deployment_class_id != baseline.deployment_class_id


def test_all_exact_server_verified_requirements_can_certify() -> None:
    """The closed outcome remains reachable only behind an authoritative receipt."""

    subject = _deployment()
    evidence = _evidence(subject)

    result = _evaluate(subject, evidence, _authoritative_receipt(subject, evidence))

    assert result.outcome == "certified"
    assert result.evidence_id == "sha256:" + "e" * 64
    assert result.evidence_source == "server_verified"
    assert result.reason_codes == ()


@pytest.mark.parametrize(
    ("mutation", "expected_reason"),
    [
        ("missing", "operational_fail_closed_missing"),
        ("stale_state", "operational_fail_closed_stale"),
        ("expired", "operational_fail_closed_stale"),
        ("wrong_subject", "operational_fail_closed_subject_mismatch"),
        ("self_asserted", "operational_fail_closed_unverified"),
        ("host_gated_unverified", "operational_fail_closed_unverified"),
        ("mock", "operational_fail_closed_unverified"),
        ("no_validity", "operational_fail_closed_stale"),
        ("no_digest", "operational_fail_closed_unverified"),
    ],
)
def test_one_untrusted_or_incomplete_requirement_stays_draft_only(
    mutation: str,
    expected_reason: str,
) -> None:
    """One bad domain must prevent a partial bundle from becoming certified."""

    subject = _deployment()
    evidence = list(_evidence(subject))
    target = evidence[-1]
    if mutation == "missing":
        target = replace(target, state="missing")
    elif mutation == "stale_state":
        target = replace(target, state="stale")
    elif mutation == "expired":
        target = replace(target, valid_until=NOW - timedelta(seconds=1))
    elif mutation == "wrong_subject":
        target = replace(target, subject_id="sha256:" + "f" * 64)
    elif mutation in {"self_asserted", "host_gated_unverified", "mock"}:
        target = replace(target, verification=mutation)
    elif mutation == "no_validity":
        target = replace(target, valid_until=None)
    elif mutation == "no_digest":
        target = replace(target, evidence_sha256=None)
    evidence[-1] = target
    evidence_tuple = tuple(evidence)

    result = _evaluate(
        subject,
        evidence_tuple,
        _authoritative_receipt(subject, evidence_tuple),
    )

    assert result.outcome == "draft_only"
    assert expected_reason in result.reason_codes
    assert result.evidence_source != "server_verified"


def test_receipt_is_required_and_must_match_the_exact_bundle() -> None:
    """Self-labeled records or a receipt for other bytes must never certify."""

    subject = _deployment()
    evidence = _evidence(subject)

    without_receipt = _evaluate(subject, evidence, None)
    mismatched_receipt = _evaluate(
        subject,
        evidence,
        _authoritative_receipt(
            subject,
            evidence,
            bundle_digest="sha256:" + "0" * 64,
        ),
    )

    assert without_receipt.outcome == "draft_only"
    assert "authoritative_receipt_missing" in without_receipt.reason_codes
    assert mismatched_receipt.outcome == "draft_only"
    assert "authoritative_receipt_mismatch" in mismatched_receipt.reason_codes


def test_expired_receipt_cannot_certify_fresh_records() -> None:
    """Bundle authority must expire even when every domain record is still fresh."""

    subject = _deployment()
    evidence = _evidence(subject)

    result = _evaluate(
        subject,
        evidence,
        _authoritative_receipt(
            subject,
            evidence,
            valid_until=NOW - timedelta(seconds=1),
        ),
    )

    assert result.outcome == "draft_only"
    assert "authoritative_receipt_stale" in result.reason_codes


@pytest.mark.parametrize(
    ("untrusted_eligible", "strict_deny_all", "expected_reason"),
    [
        (False, True, "runtime_not_untrusted_eligible"),
        (True, False, "runtime_strict_deny_all_unavailable"),
    ],
)
def test_static_runtime_ineligibility_is_unsupported(
    untrusted_eligible: bool,
    strict_deny_all: bool,
    expected_reason: str,
) -> None:
    """Host-local or weak-network runtimes must not be described as draft-ready."""

    subject = _deployment()
    evidence = _evidence(subject)

    result = _evaluate(
        subject,
        evidence,
        _authoritative_receipt(subject, evidence),
        untrusted_eligible=untrusted_eligible,
        strict_deny_all=strict_deny_all,
    )

    assert result.outcome == "unsupported"
    assert expected_reason in result.reason_codes


def test_boundary_breach_is_unsupported_but_missing_feature_is_not() -> None:
    """A demonstrated escape and an absent dependency must remain distinguishable."""

    subject = _deployment()
    evidence = list(_evidence(subject))
    evidence[1] = replace(
        evidence[1],
        state="failed",
        safety_boundary_breached=True,
    )
    breached = tuple(evidence)
    missing = tuple(
        replace(item, state="missing")
        if item.requirement_id == "scheduled_transcript_non_disclosure"
        else item
        for item in _evidence(subject)
    )

    breached_result = _evaluate(
        subject,
        breached,
        _authoritative_receipt(subject, breached),
    )
    missing_result = _evaluate(
        subject,
        missing,
        _authoritative_receipt(subject, missing),
    )

    assert breached_result.outcome == "unsupported"
    assert "safety_boundary_breached" in breached_result.reason_codes
    assert missing_result.outcome == "draft_only"
    assert "safety_boundary_breached" not in missing_result.reason_codes


@pytest.mark.parametrize(
    "invalid_ids",
    [
        ("unknown_requirement",),
        ("isolation_attestation", "isolation_attestation"),
    ],
)
def test_unknown_or_duplicate_requirements_are_rejected(
    invalid_ids: tuple[str, ...],
) -> None:
    """Malformed requirement sets must fail instead of being silently ignored."""

    subject = _deployment()
    evidence = list(_evidence(subject))
    for index, requirement_id in enumerate(invalid_ids):
        evidence[index] = replace(evidence[index], requirement_id=requirement_id)

    with pytest.raises(ValueError, match="requirement"):
        _evaluate(subject, tuple(evidence), None)


def test_reason_codes_are_closed_sorted_and_bounded() -> None:
    """Evidence-controlled text must never leak into public diagnostic reasons."""

    subject = _deployment()
    evidence = tuple(
        replace(
            item,
            state="missing",
            verification="self_asserted",
            subject_id="attacker-controlled free text",
            valid_until=None,
            evidence_sha256=None,
        )
        for item in _evidence(subject)
    )

    result = _evaluate(subject, evidence, None)

    assert result.reason_codes == tuple(sorted(set(result.reason_codes)))
    assert len(result.reason_codes) <= cert.MAX_REASON_CODES
    assert all(reason in cert.CERTIFICATION_REASON_CODES for reason in result.reason_codes)
    assert "attacker-controlled free text" not in " ".join(result.reason_codes)


def test_certification_is_not_sufficient_for_dispatch() -> None:
    """A certified fixture must remain blocked until the execution stack exists."""

    subject = _deployment()
    evidence = _evidence(subject)
    certification = _evaluate(
        subject,
        evidence,
        _authoritative_receipt(subject, evidence),
    )

    blocked = cert.agent_execution_dispatch_readiness(
        certification,
        execution_stack_ready=False,
    )
    ready = cert.agent_execution_dispatch_readiness(
        certification,
        execution_stack_ready=True,
    )

    assert blocked.ready is False
    assert blocked.reason == "agent_execution_stack_unimplemented"
    assert ready.ready is True
    assert ready.reason is None


def test_current_resolver_cannot_certify_repository_characterization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Current local metadata must remain a draft even with a verified build SHA."""

    monkeypatch.setenv("TLDW_BUILD_SHA", BUILD_SHA.upper())

    result = cert.resolve_current_agent_execution_certification(
        now=NOW,
        host_os_family="darwin",
        host_architecture="arm64",
        auth_mode="single_user",
        sandbox_runtime="docker",
        adapter_id="acp",
        adapter_version="1",
        isolation_profile=_profile(),
    )

    assert result.outcome == "draft_only"
    assert result.evidence_id is None
    assert result.evidence_source == "repository_characterization"
    assert "authoritative_receipt_missing" in result.reason_codes


def test_current_resolver_fails_closed_on_unverified_build_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A missing or malformed configured build SHA must never become an identity claim."""

    monkeypatch.setenv("TLDW_BUILD_SHA", "not-a-commit")

    result = cert.resolve_current_agent_execution_certification(
        now=NOW,
        host_os_family="darwin",
        host_architecture="arm64",
        auth_mode="single_user",
        sandbox_runtime="docker",
        adapter_id="acp",
        adapter_version="1",
        isolation_profile=_profile(),
    )

    assert result.outcome == "draft_only"
    assert "server_build_identity_unverified" in result.reason_codes


def test_current_ineligible_runtime_is_unsupported() -> None:
    """The production resolver must project host-local worktrees as unsupported."""

    result = cert.resolve_current_agent_execution_certification(
        now=NOW,
        host_os_family="darwin",
        host_architecture="arm64",
        auth_mode="single_user",
        sandbox_runtime="worktree",
        adapter_id="acp",
        adapter_version="1",
        isolation_profile=_profile(),
    )

    assert result.outcome == "unsupported"
    assert "runtime_not_untrusted_eligible" in result.reason_codes
    assert "runtime_strict_deny_all_unavailable" in result.reason_codes
