"""Fail-closed parsing and evaluation for reviewed vulnerability exceptions."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from urllib.parse import urlparse

_COMPONENTS = frozenset(
    {
        "source-python-root",
        "source-apps-workspace",
        "source-admin-ui",
        "image-app",
        "image-worker",
        "image-audio-worker",
        "image-webui",
        "image-admin-ui",
        "reference-caddy",
        "reference-postgresql",
        "reference-redis",
        "reference-prometheus",
        "reference-alertmanager",
        "reference-grafana",
    }
)
_SEVERITIES = frozenset({"CRITICAL", "HIGH"})
_RECORD_FIELDS = frozenset(
    {
        "id",
        "vulnerability_id",
        "component",
        "purl",
        "installed_version",
        "severity",
        "rationale",
        "mitigation",
        "owner",
        "approval",
        "created_on",
        "expires_on",
        "supersedes",
    }
)
_POLICY_FIELDS = frozenset({"schema_version", "exceptions"})
_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_PURL_PATTERN = re.compile(r"^pkg:[A-Za-z0-9.+-]+/[^\s@]+@[^\s@]+$")
_DATE_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_APPROVAL_PATH_PATTERN = re.compile(r"^/rmusser01/tldw_server/(?:issues|pull)/[1-9][0-9]*$")


class PolicyError(ValueError):
    """Raised when a policy or scanner report cannot safely be evaluated."""


@dataclass(frozen=True)
class VulnerabilityException:
    """A reviewed, component-scoped vulnerability exception."""

    id: str
    vulnerability_id: str
    component: str
    purl: str
    installed_version: str
    severity: str
    rationale: str
    mitigation: str
    owner: str
    approval: str
    created_on: date
    expires_on: date
    supersedes: str | None


@dataclass(frozen=True)
class ExceptionPolicy:
    """The canonical versioned vulnerability exception policy."""

    schema_version: int
    exceptions: tuple[VulnerabilityException, ...]


@dataclass(frozen=True)
class Finding:
    """The security-relevant identity of one Trivy finding."""

    vulnerability_id: str
    purl: str
    installed_version: str
    severity: str
    target: str


@dataclass(frozen=True)
class ScanDecision:
    """The policy-adjusted result for one component's complete Trivy report."""

    component: str
    blocking: tuple[Finding, ...]
    excepted: tuple[Finding, ...]
    unmatched_exception_ids: tuple[str, ...]


def _policy_field(field: str) -> PolicyError:
    return PolicyError(f"policy field {field}")


def _exception_field(exception_id: str, field: str) -> PolicyError:
    return PolicyError(f"exception {exception_id} field {field}")


def _record_id(value: object) -> str:
    if type(value) is not str or not _ID_PATTERN.fullmatch(value):
        raise _exception_field("invalid", "id")
    return value


def _string_field(record: Mapping[str, object], exception_id: str, field: str) -> str:
    value = record[field]
    if type(value) is not str or not value.strip():
        raise _exception_field(exception_id, field)
    return value


def _parse_date(value: str, exception_id: str, field: str) -> date:
    if not _DATE_PATTERN.fullmatch(value):
        raise _exception_field(exception_id, field)
    try:
        parsed = date.fromisoformat(value)
    except ValueError as error:
        raise _exception_field(exception_id, field) from error
    if parsed.isoformat() != value:
        raise _exception_field(exception_id, field)
    return parsed


def _validate_approval(value: str, exception_id: str) -> None:
    try:
        parsed = urlparse(value)
    except ValueError as error:
        raise _exception_field(exception_id, "approval") from error
    if (
        parsed.scheme != "https"
        or parsed.netloc != "github.com"
        or parsed.params
        or parsed.query
        or parsed.fragment
        or not _APPROVAL_PATH_PATTERN.fullmatch(parsed.path)
    ):
        raise _exception_field(exception_id, "approval")


def _validate_component(component: object, exception_id: str | None = None) -> str:
    if type(component) is not str or component not in _COMPONENTS:
        if exception_id is None:
            raise _policy_field("component")
        raise _exception_field(exception_id, "component")
    return component


def _parse_exception(value: object, today: date) -> VulnerabilityException:
    if type(value) is not dict:
        raise _exception_field("invalid", "record")
    record: dict[str, object] = value
    if set(record) != _RECORD_FIELDS:
        exception_id = _record_id(record.get("id")) if "id" in record else "invalid"
        raise _exception_field(exception_id, "fields")

    exception_id = _record_id(record["id"])
    vulnerability_id = _string_field(record, exception_id, "vulnerability_id")
    component = _validate_component(record["component"], exception_id)
    purl = _string_field(record, exception_id, "purl")
    if not _PURL_PATTERN.fullmatch(purl):
        raise _exception_field(exception_id, "purl")
    installed_version = _string_field(record, exception_id, "installed_version")
    severity = _string_field(record, exception_id, "severity")
    if severity not in _SEVERITIES:
        raise _exception_field(exception_id, "severity")
    rationale = _string_field(record, exception_id, "rationale").strip()
    mitigation = _string_field(record, exception_id, "mitigation")
    owner = _string_field(record, exception_id, "owner")
    approval = _string_field(record, exception_id, "approval")
    _validate_approval(approval, exception_id)
    created_on = _parse_date(_string_field(record, exception_id, "created_on"), exception_id, "created_on")
    expires_on = _parse_date(_string_field(record, exception_id, "expires_on"), exception_id, "expires_on")
    if expires_on < created_on or expires_on < today:
        raise _exception_field(exception_id, "expires_on")
    maximum_days = 7 if severity == "CRITICAL" else 30
    if (expires_on - created_on).days > maximum_days:
        raise _exception_field(exception_id, "expires_on")

    supersedes_value = record["supersedes"]
    if supersedes_value is not None:
        if type(supersedes_value) is not str or not _ID_PATTERN.fullmatch(supersedes_value):
            raise _exception_field(exception_id, "supersedes")
        if supersedes_value == exception_id:
            raise _exception_field(exception_id, "supersedes")

    return VulnerabilityException(
        id=exception_id,
        vulnerability_id=vulnerability_id,
        component=component,
        purl=purl,
        installed_version=installed_version,
        severity=severity,
        rationale=rationale,
        mitigation=mitigation,
        owner=owner,
        approval=approval,
        created_on=created_on,
        expires_on=expires_on,
        supersedes=supersedes_value,
    )


def _validate_renewals(exceptions: tuple[VulnerabilityException, ...]) -> None:
    by_id = {item.id: item for item in exceptions}
    superseded_ids = {item.supersedes for item in exceptions if item.supersedes is not None}
    for item in exceptions:
        matching_active_ids = {
            other.id
            for other in exceptions
            if other.id != item.id
            and other.component == item.component
            and other.vulnerability_id == item.vulnerability_id
            and other.purl == item.purl
            and other.installed_version == item.installed_version
            and other.severity == item.severity
        }
        if matching_active_ids and item.id not in superseded_ids and item.supersedes not in matching_active_ids:
            raise _exception_field(item.id, "supersedes")
        if item.supersedes is None or item.supersedes not in by_id:
            continue
        prior = by_id[item.supersedes]
        if item.approval == prior.approval:
            raise _exception_field(item.id, "approval")
        if item.rationale == prior.rationale:
            raise _exception_field(item.id, "rationale")


def _duplicate_rejecting_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    """Build a JSON object only when every member name is unique."""
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise _policy_field("json")
        result[key] = value
    return result


def load_policy(path: Path, *, today: date) -> ExceptionPolicy:
    """Load an unexpired canonical policy, rejecting every malformed record."""
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_duplicate_rejecting_object,
        )
    except OSError as error:
        raise _policy_field("file") from error
    except json.JSONDecodeError as error:
        raise _policy_field("json") from error
    if type(payload) is not dict or set(payload) != _POLICY_FIELDS:
        raise _policy_field("fields")
    schema_version = payload["schema_version"]
    if type(schema_version) is not int or schema_version != 1:
        raise _policy_field("schema_version")
    records = payload["exceptions"]
    if type(records) is not list:
        raise _policy_field("exceptions")

    exceptions = tuple(_parse_exception(record, today) for record in records)
    if len({item.id for item in exceptions}) != len(exceptions):
        raise _policy_field("exception_ids")
    _validate_renewals(exceptions)
    return ExceptionPolicy(schema_version=schema_version, exceptions=exceptions)


def write_trivy_ignore(policy: ExceptionPolicy, *, component: str, output: Path) -> None:
    """Write an ephemeral Trivy ignore policy for one explicit component."""
    _validate_component(component)
    payload = {
        "vulnerabilities": [
            {
                "id": item.vulnerability_id,
                "purls": [item.purl],
                "statement": f"{item.id}: {item.rationale}",
                "expired_at": item.expires_on.isoformat(),
            }
            for item in policy.exceptions
            if item.component == component
        ]
    }
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _report_field(field: str) -> PolicyError:
    return PolicyError(f"report field {field}")


def _report_string(value: object, field: str) -> str:
    if type(value) is not str or not value.strip():
        raise _report_field(field)
    return value


def _parse_findings(report: Mapping[str, object]) -> tuple[Finding, ...]:
    if not isinstance(report, Mapping):
        raise _report_field("root")
    results = report.get("Results")
    if type(results) is not list:
        raise _report_field("Results")

    findings: list[Finding] = []
    for result in results:
        if not isinstance(result, Mapping):
            raise _report_field("Results")
        target = _report_string(result.get("Target"), "Target")
        vulnerabilities = result.get("Vulnerabilities")
        if vulnerabilities is None:
            continue
        if type(vulnerabilities) is not list:
            raise _report_field("Vulnerabilities")
        for vulnerability in vulnerabilities:
            if not isinstance(vulnerability, Mapping):
                raise _report_field("Vulnerabilities")
            vulnerability_id = _report_string(vulnerability.get("VulnerabilityID"), "VulnerabilityID")
            package = vulnerability.get("PkgIdentifier")
            if not isinstance(package, Mapping):
                raise _report_field("PkgIdentifier")
            purl = _report_string(package.get("PURL"), "PURL")
            if not _PURL_PATTERN.fullmatch(purl):
                raise _report_field("PURL")
            installed_version = _report_string(vulnerability.get("InstalledVersion"), "InstalledVersion")
            severity = _report_string(vulnerability.get("Severity"), "Severity")
            if severity in _SEVERITIES:
                findings.append(
                    Finding(
                        vulnerability_id=vulnerability_id,
                        purl=purl,
                        installed_version=installed_version,
                        severity=severity,
                        target=target,
                    )
                )
    return tuple(findings)


def evaluate_trivy_report(
    report: Mapping[str, object],
    *,
    component: str,
    policy: ExceptionPolicy,
    today: date,
) -> ScanDecision:
    """Independently evaluate complete Trivy JSON against exact policy entries."""
    _validate_component(component)
    scoped_exceptions = tuple(item for item in policy.exceptions if item.component == component)
    matched_ids: set[str] = set()
    blocking: list[Finding] = []
    excepted: list[Finding] = []

    for finding in _parse_findings(report):
        matches = tuple(
            item
            for item in scoped_exceptions
            if item.expires_on >= today
            and item.vulnerability_id == finding.vulnerability_id
            and item.purl == finding.purl
            and item.installed_version == finding.installed_version
            and item.severity == finding.severity
        )
        if len(matches) == 1:
            excepted.append(finding)
            matched_ids.add(matches[0].id)
        else:
            blocking.append(finding)

    unmatched_exception_ids = tuple(item.id for item in scoped_exceptions if item.id not in matched_ids)
    return ScanDecision(
        component=component,
        blocking=tuple(blocking),
        excepted=tuple(excepted),
        unmatched_exception_ids=unmatched_exception_ids,
    )


__all__ = [
    "ExceptionPolicy",
    "Finding",
    "PolicyError",
    "ScanDecision",
    "VulnerabilityException",
    "evaluate_trivy_report",
    "load_policy",
    "write_trivy_ignore",
]
