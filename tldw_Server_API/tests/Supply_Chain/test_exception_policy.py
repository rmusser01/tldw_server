"""Behavior tests for the fail-closed vulnerability exception policy."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest
from Helper_Scripts.Supply_Chain.exception_policy import (
    ExceptionPolicy,
    PolicyError,
    evaluate_trivy_report,
    load_policy,
    write_trivy_ignore,
)

pytestmark = pytest.mark.unit

TODAY = date(2026, 8, 30)


def _valid_record(**overrides: object) -> dict[str, object]:
    """Return hand-authored policy data for one exact known finding."""
    record: dict[str, object] = {
        "id": "VEX-2026-1000",
        "vulnerability_id": "CVE-2026-1000",
        "component": "image-app",
        "purl": "pkg:pypi/example@1.0.0",
        "installed_version": "1.0.0",
        "severity": "CRITICAL",
        "rationale": "Upstream fix is not yet released for the pinned dependency.",
        "mitigation": "The service is isolated behind authenticated API access.",
        "owner": "security@example.invalid",
        "approval": "https://github.com/rmusser01/tldw_server/issues/13013",
        "created_on": "2026-08-30",
        "expires_on": "2026-09-06",
        "supersedes": None,
    }
    record.update(overrides)
    return record


def _write_policy(tmp_path: Path, exceptions: list[dict[str, object]]) -> Path:
    path = tmp_path / "exceptions.json"
    path.write_text(
        json.dumps({"schema_version": 1, "exceptions": exceptions}),
        encoding="utf-8",
    )
    return path


def test_empty_policy_is_valid(tmp_path: Path) -> None:
    """Catches a loader that rejects the approved empty default policy."""
    path = tmp_path / "exceptions.json"
    path.write_text('{"schema_version":1,"exceptions":[]}\n', encoding="utf-8")

    assert load_policy(path, today=TODAY).exceptions == ()


@pytest.mark.parametrize(
    "component",
    ("source-python-root", "source-apps-workspace", "source-admin-ui"),
)
def test_source_sbom_component_classes_are_supported(
    tmp_path: Path,
    component: str,
) -> None:
    """Catches the source workflow using a component outside the reviewed policy enum."""
    policy = load_policy(_write_policy(tmp_path, []), today=TODAY)

    decision = evaluate_trivy_report(
        {"Results": []},
        component=component,
        policy=policy,
        today=TODAY,
    )

    assert decision.blocking == ()
    assert decision.excepted == ()
    assert decision.unmatched_exception_ids == ()


def test_lower_severity_findings_remain_in_raw_report_without_blocking(
    tmp_path: Path,
) -> None:
    """Catches source admission mutating complete evidence to select its blockers."""
    policy = load_policy(_write_policy(tmp_path, []), today=TODAY)
    report = {
        "Results": [
            {
                "Target": "sbom-source-python-root.cdx.json",
                "Vulnerabilities": [
                    {
                        "VulnerabilityID": "CVE-2026-LOW",
                        "PkgIdentifier": {"PURL": "pkg:pypi/example@1.0.0"},
                        "InstalledVersion": "1.0.0",
                        "Severity": "LOW",
                    }
                ],
            }
        ]
    }

    decision = evaluate_trivy_report(
        report,
        component="source-python-root",
        policy=policy,
        today=TODAY,
    )

    assert decision.blocking == ()
    assert report["Results"][0]["Vulnerabilities"][0]["Severity"] == "LOW"


def test_unmatched_exceptions_are_scoped_to_the_evaluated_component(
    tmp_path: Path,
    trivy_report: dict[str, object],
) -> None:
    """Catches one component scan rejecting an exception owned by another component."""
    policy = load_policy(
        _write_policy(
            tmp_path,
            [
                _valid_record(component="source-python-root"),
                _valid_record(
                    id="VEX-2026-1001",
                    vulnerability_id="CVE-2026-2000",
                    component="source-admin-ui",
                    purl="pkg:npm/example@2.0.0",
                    installed_version="2.0.0",
                ),
            ],
        ),
        today=TODAY,
    )

    decision = evaluate_trivy_report(
        trivy_report,
        component="source-python-root",
        policy=policy,
        today=TODAY,
    )

    assert decision.blocking == ()
    assert decision.unmatched_exception_ids == ()


@pytest.mark.parametrize(
    "payload",
    (
        {},
        {"schema_version": 2, "exceptions": []},
        {"schema_version": 1, "exceptions": "all"},
        {"schema_version": 1, "exceptions": [{"id": "incomplete"}]},
    ),
)
def test_invalid_policy_fails_closed(tmp_path: Path, payload: object) -> None:
    """Catches a loader that admits malformed exception policy structures."""
    path = tmp_path / "exceptions.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(PolicyError):
        load_policy(path, today=TODAY)


@pytest.mark.parametrize(
    "changes",
    (
        {"expires_on": "2026-09-07"},
        {"severity": "HIGH", "expires_on": "2026-09-30"},
        {"purl": "example@1.0.0"},
        {"owner": " "},
        {"rationale": ""},
        {"mitigation": "\t"},
        {"approval": "https://example.com/issues/13013"},
        {"expires_on": "2026-08-29"},
        {"created_on": "2026-08-01", "expires_on": "2026-08-02"},
        {"component": "unknown"},
        {"supersedes": "VEX-2026-1000"},
        {"created_on": "2026-8-30"},
    ),
    ids=(
        "critical_eight_days",
        "high_thirty_one_days",
        "invalid_purl",
        "blank_owner",
        "blank_rationale",
        "blank_mitigation",
        "non_repository_approval",
        "expiry_before_creation",
        "expiry_before_today",
        "unknown_component",
        "self_supersedes",
        "noncanonical_date",
    ),
)
def test_invalid_record_fails_closed(
    tmp_path: Path,
    changes: dict[str, object],
) -> None:
    """Catches policy validation changes that widen an exception's scope or duration."""
    path = _write_policy(tmp_path, [_valid_record(**changes)])

    with pytest.raises(PolicyError):
        load_policy(path, today=TODAY)


def test_high_exception_may_last_thirty_days(tmp_path: Path) -> None:
    """Catches a duration check that rejects the approved 30-day High maximum."""
    path = _write_policy(
        tmp_path,
        [_valid_record(severity="HIGH", expires_on="2026-09-29")],
    )

    assert load_policy(path, today=TODAY).exceptions[0].expires_on == date(2026, 9, 29)


def test_duplicate_exception_id_fails_closed(tmp_path: Path) -> None:
    """Catches a loader that permits two approvals with the same stable ID."""
    path = _write_policy(tmp_path, [_valid_record(), _valid_record()])

    with pytest.raises(PolicyError):
        load_policy(path, today=TODAY)


@pytest.mark.parametrize(
    "payload",
    (
        {"schema_version": True, "exceptions": []},
        {"schema_version": 1, "exceptions": [], "unexpected": "value"},
        {
            "schema_version": 1,
            "exceptions": [_valid_record(owner=7)],
        },
        {
            "schema_version": 1,
            "exceptions": [_valid_record(supersedes=False)],
        },
        {
            "schema_version": 1,
            "exceptions": [{**_valid_record(), "unexpected": "value"}],
        },
    ),
    ids=(
        "boolean_schema_version",
        "unknown_top_level",
        "nonstring_owner",
        "boolean_supersedes",
        "unknown_record_key",
    ),
)
def test_policy_rejects_unknown_keys_and_wrong_scalar_types(
    tmp_path: Path,
    payload: dict[str, object],
) -> None:
    """Catches permissive JSON coercion or undeclared exception-policy fields."""
    path = tmp_path / "exceptions.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(PolicyError):
        load_policy(path, today=TODAY)


@pytest.mark.parametrize(
    "renewal",
    (
        _valid_record(
            id="VEX-2026-1001",
            supersedes=None,
            approval="https://github.com/rmusser01/tldw_server/pull/13014",
            rationale="Re-assessed risk remains pending upstream release.",
        ),
        _valid_record(
            id="VEX-2026-1000",
            supersedes="VEX-2026-1000",
            approval="https://github.com/rmusser01/tldw_server/pull/13014",
            rationale="Re-assessed risk remains pending upstream release.",
        ),
        _valid_record(
            id="VEX-2026-1001",
            supersedes="VEX-2026-1000",
            approval="https://github.com/rmusser01/tldw_server/issues/13013",
            rationale="Re-assessed risk remains pending upstream release.",
        ),
        _valid_record(
            id="VEX-2026-1001",
            supersedes="VEX-2026-1000",
            approval="https://github.com/rmusser01/tldw_server/pull/13014",
            rationale="Upstream fix is not yet released for the pinned dependency.",
        ),
    ),
    ids=("missing_link", "same_id", "same_approval", "same_rationale"),
)
def test_present_renewal_requires_new_approval_and_rationale(
    tmp_path: Path,
    renewal: dict[str, object],
) -> None:
    """Catches renewal validation that accepts only an expiry edit for an active record."""
    path = _write_policy(tmp_path, [_valid_record(), renewal])

    with pytest.raises(PolicyError):
        load_policy(path, today=TODAY)


@pytest.mark.parametrize(
    "approval",
    (
        "https://github.com/rmusser01/tldw_server/issues/13013/",
        "https://github.com/rmusser01/tldw_server/issues/13013?revision=1",
        "https://github.com/rmusser01/tldw_server/issues/13013#approval",
    ),
    ids=("trailing_slash", "query", "fragment"),
)
def test_policy_rejects_noncanonical_approval_references(
    tmp_path: Path,
    approval: str,
) -> None:
    """Catches approvals that make a prior approval appear newly reviewed."""
    path = _write_policy(tmp_path, [_valid_record(approval=approval)])

    with pytest.raises(PolicyError):
        load_policy(path, today=TODAY)


def test_renewal_rejects_rationale_changed_only_by_outer_whitespace(
    tmp_path: Path,
) -> None:
    """Catches a renewal that disguises an unchanged rationale with whitespace."""
    path = _write_policy(
        tmp_path,
        [
            _valid_record(),
            _valid_record(
                id="VEX-2026-1001",
                supersedes="VEX-2026-1000",
                approval="https://github.com/rmusser01/tldw_server/pull/13014",
                rationale="Upstream fix is not yet released for the pinned dependency. ",
            ),
        ],
    )

    with pytest.raises(PolicyError):
        load_policy(path, today=TODAY)


@pytest.mark.parametrize(
    "raw_policy",
    (
        ('{"schema_version":1,"exceptions":[],"exceptions":[]}'),
        (
            '{"schema_version":1,"exceptions":[{'
            '"id":"VEX-2026-1000",'
            '"vulnerability_id":"CVE-2026-1000",'
            '"component":"image-app",'
            '"purl":"pkg:pypi/example@1.0.0",'
            '"installed_version":"1.0.0",'
            '"severity":"CRITICAL",'
            '"severity":"HIGH",'
            '"rationale":"Upstream fix is not yet released for the pinned dependency.",'
            '"mitigation":"The service is isolated behind authenticated API access.",'
            '"owner":"security@example.invalid",'
            '"approval":"https://github.com/rmusser01/tldw_server/issues/13013",'
            '"created_on":"2026-08-30",'
            '"expires_on":"2026-09-06",'
            '"supersedes":null}]}'
        ),
        (
            '{"schema_version":1,"exceptions":[{'
            '"id":"VEX-2026-1000",'
            '"vulnerability_id":"CVE-2026-1000",'
            '"component":"image-app",'
            '"purl":"pkg:pypi/example@1.0.0",'
            '"installed_version":"1.0.0",'
            '"severity":"CRITICAL",'
            '"rationale":"Upstream fix is not yet released for the pinned dependency.",'
            '"mitigation":"The service is isolated behind authenticated API access.",'
            '"owner":"security@example.invalid",'
            '"approval":"https://github.com/rmusser01/tldw_server/issues/13013",'
            '"approval":"https://github.com/rmusser01/tldw_server/pull/13014",'
            '"created_on":"2026-08-30",'
            '"expires_on":"2026-09-06",'
            '"supersedes":null}]}'
        ),
    ),
    ids=("top_level_exceptions", "record_severity", "record_approval"),
)
def test_policy_rejects_duplicate_json_object_members(
    tmp_path: Path,
    raw_policy: str,
) -> None:
    """Catches JSON decoding that silently chooses a duplicate member's last value."""
    path = tmp_path / "exceptions.json"
    path.write_text(raw_policy, encoding="utf-8")

    with pytest.raises(PolicyError):
        load_policy(path, today=TODAY)


def test_present_renewal_with_distinct_approval_and_rationale_is_valid(
    tmp_path: Path,
) -> None:
    """Catches renewal validation that rejects a fully re-reviewed active replacement."""
    path = _write_policy(
        tmp_path,
        [
            _valid_record(),
            _valid_record(
                id="VEX-2026-1001",
                supersedes="VEX-2026-1000",
                approval="https://github.com/rmusser01/tldw_server/pull/13014",
                rationale="Re-assessed risk remains pending upstream release.",
            ),
        ],
    )

    assert [item.id for item in load_policy(path, today=TODAY).exceptions] == [
        "VEX-2026-1000",
        "VEX-2026-1001",
    ]


def test_retained_renewal_can_reference_removed_prior_id(tmp_path: Path) -> None:
    """Catches a loader that demands archived approvals remain in the active policy."""
    path = _write_policy(
        tmp_path,
        [
            _valid_record(
                id="VEX-2026-1001",
                supersedes="VEX-2026-1000",
                approval="https://github.com/rmusser01/tldw_server/pull/13014",
                rationale="Re-assessed risk remains pending upstream release.",
            )
        ],
    )

    assert load_policy(path, today=TODAY).exceptions[0].supersedes == "VEX-2026-1000"


@pytest.fixture
def valid_policy(tmp_path: Path) -> ExceptionPolicy:
    """Load the policy that independently describes the Trivy fixture finding."""
    return load_policy(_write_policy(tmp_path, [_valid_record()]), today=TODAY)


@pytest.fixture
def trivy_report() -> dict[str, object]:
    """Load a complete Trivy-shaped report without mocking scanner behavior."""
    fixture = Path(__file__).parent / "fixtures" / "trivy-critical.json"
    return json.loads(fixture.read_text(encoding="utf-8"))


def test_exception_matches_only_exact_component_package_and_version(
    valid_policy: ExceptionPolicy,
    trivy_report: dict[str, object],
) -> None:
    """Catches an evaluator that suppresses a finding beyond its exact approval."""
    decision = evaluate_trivy_report(
        trivy_report,
        component="image-app",
        policy=valid_policy,
        today=TODAY,
    )

    assert decision.blocking == ()
    assert [item.vulnerability_id for item in decision.excepted] == ["CVE-2026-1000"]
    assert decision.unmatched_exception_ids == ()


@pytest.mark.parametrize(
    ("component", "changes", "expected_unmatched"),
    (
        ("image-worker", {}, ()),
        ("image-app", {"VulnerabilityID": "CVE-2026-9999"}, ("VEX-2026-1000",)),
        ("image-app", {"purl": "pkg:pypi/other@1.0.0"}, ("VEX-2026-1000",)),
        ("image-app", {"InstalledVersion": "2.0.0"}, ("VEX-2026-1000",)),
        ("image-app", {"Severity": "HIGH"}, ("VEX-2026-1000",)),
    ),
    ids=("component", "vulnerability", "purl", "version", "severity"),
)
def test_mismatched_finding_remains_blocking_and_marks_exception_stale(
    valid_policy: ExceptionPolicy,
    trivy_report: dict[str, object],
    component: str,
    changes: dict[str, str],
    expected_unmatched: tuple[str, ...],
) -> None:
    """Catches any matching rule that omits one approved finding identity field."""
    report = json.loads(json.dumps(trivy_report))
    finding = report["Results"][0]["Vulnerabilities"][0]
    if "purl" in changes:
        finding["PkgIdentifier"]["PURL"] = changes["purl"]
    else:
        finding.update(changes)

    decision = evaluate_trivy_report(
        report,
        component=component,
        policy=valid_policy,
        today=TODAY,
    )

    assert [item.vulnerability_id for item in decision.blocking] == [changes.get("VulnerabilityID", "CVE-2026-1000")]
    assert decision.excepted == ()
    assert decision.unmatched_exception_ids == expected_unmatched


def test_ignore_projection_contains_only_the_selected_component(
    tmp_path: Path,
    valid_policy: ExceptionPolicy,
) -> None:
    """Catches an ignore writer that leaks one component's exception into another scan."""
    output = tmp_path / ".trivyignore.yaml"

    write_trivy_ignore(valid_policy, component="image-worker", output=output)

    assert json.loads(output.read_text(encoding="utf-8")) == {"vulnerabilities": []}


def test_ignore_projection_preserves_exact_approved_finding_identity(
    tmp_path: Path,
    valid_policy: ExceptionPolicy,
) -> None:
    """Catches an ignore projection that drops PURL or expiry identity fields."""
    output = tmp_path / ".trivyignore.yaml"

    write_trivy_ignore(valid_policy, component="image-app", output=output)

    assert json.loads(output.read_text(encoding="utf-8")) == {
        "vulnerabilities": [
            {
                "expired_at": "2026-09-06",
                "id": "CVE-2026-1000",
                "purls": ["pkg:pypi/example@1.0.0"],
                "statement": ("VEX-2026-1000: " "Upstream fix is not yet released for the pinned dependency."),
            }
        ]
    }
