"""Exercise exact checkpoint-advisory policy without importing any ML library."""

from __future__ import annotations

from dataclasses import replace
from datetime import date
from pathlib import Path

import pytest
from Helper_Scripts.Supply_Chain.dependency_review import derive_allow_ghsas, validate_allow_ghsas
from Helper_Scripts.Supply_Chain.exception_policy import PolicyError, load_policy

pytestmark = pytest.mark.unit
POLICY = Path(".github/supply-chain/vulnerability-exceptions.json")
TODAY = date(2026, 9, 10)
GHSA = "GHSA-qqmf-gpg7-g8gw"
CVE = "CVE-2026-58659"


def _change(**overrides: object) -> dict:
    """Supply the reviewed dependency API identity independently of alias lookup."""
    return {
        "change_type": "added",
        "manifest": "uv.lock",
        "ecosystem": "pip",
        "name": "lightning",
        "version": "2.6.5",
        "package_url": "pkg:pypi/lightning@2.6.5",
        "scope": "runtime",
        "vulnerabilities": [{"severity": "high", "advisory_ghsa_id": GHSA}],
        **overrides,
    }


@pytest.mark.parametrize("manifest", ["pyproject.toml", "uv.lock"])
@pytest.mark.parametrize("today", [TODAY, date(2026, 9, 17)])
def test_current_exact_source_approval_projects_at_both_date_boundaries(manifest: str, today: date) -> None:
    """Catch a missing or mistyped Lightning alias or canonical source identity."""
    policy = load_policy(POLICY, today=TODAY)
    assert derive_allow_ghsas([_change(manifest=manifest)], policy=policy, today=today) == (GHSA,)


@pytest.mark.parametrize("today", [date(2026, 9, 9), date(2026, 9, 18)])
def test_lightning_allowance_is_inactive_outside_approval_dates(today: date) -> None:
    """Catch ignoring the start date or expiry during action-output revalidation."""
    policy = load_policy(POLICY, today=TODAY)
    with pytest.raises(PolicyError, match="no longer fully covers"):
        validate_allow_ghsas(GHSA, [_change()], policy=policy, today=today)


@pytest.mark.parametrize(
    "overrides",
    [
        {"version": "2.6.6"},
        {"version": ""},
        {"package_url": "pkg:pypi/lightning@2.6.5?extra=custom"},
        {"name": "pytorch-lightning"},
        {"manifest": "nested/uv.lock"},
        {"ecosystem": "npm"},
        {"vulnerabilities": [{"severity": "critical", "advisory_ghsa_id": GHSA}]},
    ],
)
def test_one_uncovered_lightning_occurrence_prevents_global_advisory_allowance(overrides: dict) -> None:
    """Catch widening the reviewed identity or authorizing only the first occurrence."""
    policy = load_policy(POLICY, today=TODAY)
    changes = [_change(), _change(**overrides)]
    assert derive_allow_ghsas(changes, policy=policy, today=TODAY) == ()
    with pytest.raises(PolicyError, match="no longer fully covers"):
        validate_allow_ghsas(GHSA, changes, policy=policy, today=TODAY)


def test_lightning_alias_without_canonical_approval_does_not_authorize() -> None:
    """Catch treating an advisory identity alias as a standalone exception."""
    policy = load_policy(POLICY, today=TODAY)
    without_lightning = replace(policy, exceptions=tuple(r for r in policy.exceptions if r.vulnerability_id != CVE))
    assert derive_allow_ghsas([_change()], policy=without_lightning, today=TODAY) == ()


def test_optional_advisories_remain_gated_alongside_lightning() -> None:
    """Catch extending the Lightning decision to unrelated Hydra or NeMo advisories."""
    policy = load_policy(POLICY, today=TODAY)
    hydra = _change(
        name="hydra-core",
        version="1.3.2",
        package_url="pkg:pypi/hydra-core@1.3.2",
        vulnerabilities=[{"severity": "high", "advisory_ghsa_id": "GHSA-2cp2-2r3c-7p7r"}],
    )
    nemo = _change(
        name="nemo-toolkit",
        version="2.0.0",
        package_url="pkg:pypi/nemo-toolkit@2.0.0",
        vulnerabilities=[
            {"severity": "high", "advisory_ghsa_id": ghsa}
            for ghsa in (
                "GHSA-9379-mwvr-7wxx",
                "GHSA-hvjw-vp7g-39h5",
                "GHSA-m4jw-wgmf-889x",
                "GHSA-v7v2-m736-cf3c",
            )
        ],
    )
    assert derive_allow_ghsas([_change(), hydra, nemo], policy=policy, today=TODAY) == (GHSA,)
