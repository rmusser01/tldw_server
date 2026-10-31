"""Test-only authority boundary for Scheduled Agent certification fixtures.

Production code intentionally exposes no receipt-minting API. Pure evaluator tests
use this module as the single explicit boundary for constructing authoritative
receipts without spreading private verifier authority through the test suite.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from tldw_Server_API.app.core.Scheduled_Tasks import execution_certification as cert


def issue_test_authoritative_receipt(
    subject: cert.DeploymentClass,
    evidence: tuple[cert.RequirementEvidence, ...],
    *,
    observed_at: datetime,
    valid_until: datetime,
    evidence_id: str,
    bundle_digest: str | None = None,
) -> Any:
    """Issue a verifier receipt exclusively for isolated rule-evaluator tests."""

    return cert._AuthoritativeBundleReceipt(
        deployment_class_id=subject.deployment_class_id,
        evidence_id=evidence_id,
        bundle_digest=bundle_digest or cert.canonical_evidence_bundle_digest(evidence),
        observed_at=observed_at,
        valid_until=valid_until,
        _authority=cert._SERVER_VERIFIER_AUTHORITY,
    )
