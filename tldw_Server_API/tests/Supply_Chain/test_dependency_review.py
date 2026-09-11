"""Behavior checks for canonical dependency-review exception projection."""

from __future__ import annotations

import importlib
import json
import os
import subprocess
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pytest
import yaml
from Helper_Scripts.Supply_Chain.exception_policy import PolicyError, load_policy

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[3]
TODAY = date(2026, 9, 10)
GHSA = "GHSA-f4j7-r4q5-qw2c"


def _adapter():
    """Load the implementation inside tests so a missing adapter fails a check."""
    assert (ROOT / "Helper_Scripts/Supply_Chain/dependency_review.py").is_file()
    return importlib.import_module("Helper_Scripts.Supply_Chain.dependency_review")


def _record(**overrides: object) -> dict[str, object]:
    """Describe one exact approval independently of adapter identity tables."""
    record = {
        "id": "TEST-CHROMA-SOURCE",
        "vulnerability_id": "CVE-2026-45829",
        "component": "source-python-root",
        "purl": "pkg:pypi/chromadb@1.5.9",
        "installed_version": "1.5.9",
        "severity": "CRITICAL",
        "rationale": "The reviewed embedded deployment does not expose this server path.",
        "mitigation": "Retain the reviewed embedded deployment profile.",
        "owner": "rmusser01",
        "approval": "https://github.com/rmusser01/tldw_server/pull/2869",
        "created_on": "2026-09-10",
        "expires_on": "2026-09-17",
        "supersedes": None,
    }
    record.update(overrides)
    return record


def _write_policy(tmp_path: Path, records: list[dict[str, object]]) -> Path:
    """Write the canonical policy format used by the real policy loader."""
    path = tmp_path / "policy.json"
    path.write_text(json.dumps({"schema_version": 1, "exceptions": records}), encoding="utf-8")
    return path


def _change(**overrides: object) -> dict[str, object]:
    """Mirror the dependency comparison API's complete record shape."""
    change = {
        "change_type": "added",
        "manifest": "uv.lock",
        "ecosystem": "pip",
        "name": "chromadb",
        "version": "1.5.9",
        "package_url": "pkg:pypi/chromadb@1.5.9",
        "license": "Apache-2.0",
        "source_repository_url": "https://github.com/chroma-core/chroma",
        "scope": "runtime",
        "vulnerabilities": [
            {
                "severity": "critical",
                "advisory_ghsa_id": GHSA,
                "advisory_summary": "Server vulnerability excluded for embedded usage",
                "advisory_url": f"https://github.com/advisories/{GHSA}",
            }
        ],
    }
    change.update(overrides)
    return change


@pytest.mark.parametrize("manifest", ["pyproject.toml", "uv.lock"])
def test_exact_root_dependency_is_allowed(tmp_path: Path, manifest: str) -> None:
    """Catches failure to project a current exact canonical source approval."""
    policy = load_policy(_write_policy(tmp_path, [_record()]), today=TODAY)
    assert _adapter().derive_allow_ghsas([_change(manifest=manifest)], policy=policy, today=TODAY) == (GHSA,)


@pytest.mark.parametrize(
    "overrides",
    [
        {"component": "image-app"},
        {"vulnerability_id": "CVE-2026-99999"},
        {"purl": "pkg:pypi/chromadb@1.5.10"},
        {"installed_version": "1.5.10"},
        {"severity": "HIGH"},
        {"created_on": "2026-09-11", "expires_on": "2026-09-18"},
    ],
)
def test_alias_does_not_authorize_without_exact_active_policy(tmp_path: Path, overrides: dict[str, object]) -> None:
    """Catches treating alias identity or a different policy scope as authorization."""
    policy = load_policy(_write_policy(tmp_path, [_record(**overrides)]), today=TODAY)
    assert _adapter().derive_allow_ghsas([_change()], policy=policy, today=TODAY) == ()


def test_empty_policy_does_not_authorize_known_advisories(tmp_path: Path) -> None:
    """Catches a static advisory-wide allowlist hidden in the alias mapping."""
    policy = load_policy(_write_policy(tmp_path, []), today=TODAY)
    assert _adapter().derive_allow_ghsas([_change()], policy=policy, today=TODAY) == ()


@pytest.mark.parametrize(
    ("ghsa", "cve", "severity"),
    [
        ("GHSA-f4j7-r4q5-qw2c", "CVE-2026-45829", "critical"),
        ("GHSA-36p7-vc44-83pf", "CVE-2026-45833", "critical"),
        ("GHSA-2wm9-hf6c-p5cr", "CVE-2026-45830", "high"),
        ("GHSA-xph7-9rjv-w5fr", "CVE-2026-45831", "high"),
    ],
)
def test_each_alias_resolves_only_its_own_canonical_record(tmp_path: Path, ghsa: str, cve: str, severity: str) -> None:
    """Catches a mistyped alias silently widening or losing the canonical CVE identity."""
    policy = load_policy(
        _write_policy(tmp_path, [_record(vulnerability_id=cve, severity=severity.upper())]), today=TODAY
    )
    change = _change(vulnerabilities=[{"severity": severity, "advisory_ghsa_id": ghsa}])
    assert _adapter().derive_allow_ghsas([change], policy=policy, today=TODAY) == (ghsa,)


def test_unresolved_version_does_not_break_review_or_receive_an_allowance(tmp_path: Path) -> None:
    """Catches rejecting valid unresolved manifest declarations or matching them to a pinned approval."""
    policy = load_policy(_write_policy(tmp_path, [_record()]), today=TODAY)
    unresolved = _change(name="PyJWT", version="", package_url="pkg:pypi/pyjwt", vulnerabilities=[])
    adapter = _adapter()
    assert adapter.derive_allow_ghsas([_change(), unresolved], policy=policy, today=TODAY) == (GHSA,)
    assert adapter.derive_allow_ghsas([_change(), _change(version="")], policy=policy, today=TODAY) == ()


@pytest.mark.parametrize(
    "overrides",
    [
        {"version": "1.5.10", "package_url": "pkg:pypi/chromadb@1.5.10"},
        {"manifest": "nested/uv.lock"},
        {"manifest": "requirements.txt"},
        {"ecosystem": "npm"},
        {"name": "other"},
        {"package_url": "pkg:pypi/other@1.5.9"},
        {"package_url": "pkg:pypi/chromadb@1.5.9?extra=server"},
        {"vulnerabilities": [{"severity": "high", "advisory_ghsa_id": GHSA}]},
    ],
)
def test_one_uncovered_occurrence_prevents_advisory_wide_allowance(
    tmp_path: Path, overrides: dict[str, object]
) -> None:
    """Catches allowing a GHSA after seeing only its first approved occurrence."""
    policy = load_policy(_write_policy(tmp_path, [_record()]), today=TODAY)
    changes = [_change(), _change(**overrides), _change()]
    assert _adapter().derive_allow_ghsas(changes, policy=policy, today=TODAY) == ()


def test_removed_occurrence_cannot_authorize_or_poison_an_added_approval(tmp_path: Path) -> None:
    """Catches treating removed packages as vulnerabilities introduced by this diff."""
    policy = load_policy(_write_policy(tmp_path, [_record()]), today=TODAY)
    removed = _change(change_type="removed", version="1.4.0", package_url="pkg:pypi/chromadb@1.4.0")
    adapter = _adapter()
    assert adapter.derive_allow_ghsas([removed], policy=policy, today=TODAY) == ()
    assert adapter.derive_allow_ghsas([removed, _change()], policy=policy, today=TODAY) == (GHSA,)


def test_separate_advisory_is_not_hidden_by_one_valid_allowance(tmp_path: Path) -> None:
    """Catches hiding unrelated advisories on an otherwise approved package."""
    policy = load_policy(_write_policy(tmp_path, [_record()]), today=TODAY)
    change = _change()
    change["vulnerabilities"].append({"severity": "high", "advisory_ghsa_id": "GHSA-aaaa-bbbb-cccc"})
    assert _adapter().derive_allow_ghsas([change], policy=policy, today=TODAY) == (GHSA,)
    assert len(change["vulnerabilities"]) == 2


@pytest.mark.parametrize("scope", ["runtime", "development", "unknown"])
def test_all_scopes_are_checked_for_uncovered_manifest(tmp_path: Path, scope: str) -> None:
    """Catches filtering scopes before checking advisory-wide exposure."""
    policy = load_policy(_write_policy(tmp_path, [_record()]), today=TODAY)
    changes = [_change(), _change(scope=scope, manifest="other/pyproject.toml")]
    assert _adapter().derive_allow_ghsas(changes, policy=policy, today=TODAY) == ()


@pytest.mark.parametrize(
    "changes",
    [
        None,
        {},
        [None],
        [_change(change_type="updated")],
        [_change(manifest=None)],
        [_change(version=1.59)],
        [_change(package_url="")],
        [_change(vulnerabilities=None)],
        [_change(vulnerabilities={})],
        [_change(vulnerabilities=[None])],
        [_change(vulnerabilities=[{"severity": "high"}])],
        [_change(vulnerabilities=[{"advisory_ghsa_id": GHSA}])],
        [_change(vulnerabilities=[{"severity": "medium", "advisory_ghsa_id": GHSA}])],
        [_change(vulnerabilities=[{"severity": "high", "advisory_ghsa_id": "GHSA-bad\nallow-ghsas=x"}])],
    ],
)
def test_malformed_complete_diff_fails_closed(tmp_path: Path, changes: object) -> None:
    """Catches permissive parser defaults hiding malformed report identities."""
    policy = load_policy(_write_policy(tmp_path, [_record()]), today=TODAY)
    with pytest.raises(PolicyError):
        _adapter().derive_allow_ghsas(changes, policy=policy, today=TODAY)


@pytest.mark.parametrize("field", ["manifest", "name", "ecosystem", "version", "package_url", "vulnerabilities"])
def test_missing_report_fields_fail_closed(tmp_path: Path, field: str) -> None:
    """Catches silently converting incomplete API records to clean reports."""
    policy = load_policy(_write_policy(tmp_path, [_record()]), today=TODAY)
    change = _change()
    del change[field]
    with pytest.raises(PolicyError):
        _adapter().derive_allow_ghsas([change], policy=policy, today=TODAY)


def test_empty_and_duplicate_approved_changes_remain_valid(tmp_path: Path) -> None:
    """Catches rejecting a clean diff or counting duplicate findings as extra approvals."""
    policy = load_policy(_write_policy(tmp_path, [_record()]), today=TODAY)
    adapter = _adapter()
    assert adapter.derive_allow_ghsas([], policy=policy, today=TODAY) == ()
    assert adapter.derive_allow_ghsas([_change(extra="tolerated"), _change()], policy=policy, today=TODAY) == (GHSA,)


@pytest.mark.parametrize("payload", [[], {}, [None], [[_change()], None], [_change()]])
def test_paginated_report_requires_at_least_one_complete_page(tmp_path: Path, payload: object) -> None:
    """Catches accepting an absent or incorrectly flattened gh --slurp response."""
    path = tmp_path / "changes.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(PolicyError):
        _adapter().load_changes(path, paginated=True)


def test_every_api_page_is_evaluated(tmp_path: Path) -> None:
    """Catches a first-page-only decision missing an uncovered version on a later page."""
    path = tmp_path / "changes.json"
    path.write_text(json.dumps([[_change()], [_change(manifest="other/uv.lock")]]), encoding="utf-8")
    policy = load_policy(_write_policy(tmp_path, [_record()]), today=TODAY)
    adapter = _adapter()
    assert adapter.derive_allow_ghsas(adapter.load_changes(path, paginated=True), policy=policy, today=TODAY) == ()


@pytest.mark.parametrize("raw", ["", "[", '{"changes":[]}', '[{"vulnerabilities": [], "vulnerabilities": []}]'])
def test_missing_or_ambiguous_json_fails_closed(tmp_path: Path, raw: str) -> None:
    """Catches accepting truncated, missing, wrapped or duplicate-key JSON output."""
    path = tmp_path / "changes.json"
    path.write_text(raw, encoding="utf-8")
    with pytest.raises(PolicyError):
        _adapter().load_changes(path)


@pytest.mark.parametrize("changes", [[], [_change(), _change(manifest="other/uv.lock")], [_change(version="1.5.10")]])
def test_action_output_drift_rejects_previously_allowed_advisory(tmp_path: Path, changes: object) -> None:
    """Catches validating the action's filtered output or trusting stale prefetched scope."""
    policy = load_policy(_write_policy(tmp_path, [_record()]), today=TODAY)
    with pytest.raises(PolicyError):
        _adapter().validate_allow_ghsas(GHSA, changes, policy=policy, today=TODAY)


def test_expiry_between_prepare_and_validate_fails_closed(tmp_path: Path) -> None:
    """Catches reusing an allowance after its canonical approval expires."""
    policy = load_policy(_write_policy(tmp_path, [_record()]), today=TODAY)
    with pytest.raises(PolicyError):
        _adapter().validate_allow_ghsas(GHSA, [_change()], policy=policy, today=date(2026, 9, 18))


@pytest.mark.parametrize("allowed", ["", GHSA])
def test_action_output_revalidates_only_passed_allowances(tmp_path: Path, allowed: str) -> None:
    """Catches treating newly available approvals as allowances actually used by the action."""
    policy = load_policy(_write_policy(tmp_path, [_record()]), today=TODAY)
    _adapter().validate_allow_ghsas(allowed, [_change()], policy=policy, today=TODAY)


@pytest.mark.parametrize("allowed", [f"{GHSA},{GHSA}", "CVE-2026-45829", ",", f"{GHSA}\nextra=x"])
def test_invalid_allowance_output_fails_closed(tmp_path: Path, allowed: str) -> None:
    """Catches malformed or injected action allowance values."""
    policy = load_policy(_write_policy(tmp_path, [_record()]), today=TODAY)
    with pytest.raises(PolicyError):
        _adapter().validate_allow_ghsas(allowed, [_change()], policy=policy, today=TODAY)


def test_cli_preserves_raw_evidence_and_writes_action_output(tmp_path: Path) -> None:
    """Catches CLI wiring dropping policy scope, JSON pages or GitHub output."""
    now = datetime.now(timezone.utc).date()
    policy_path = _write_policy(
        tmp_path, [_record(created_on=now.isoformat(), expires_on=(now + timedelta(days=7)).isoformat())]
    )
    report = tmp_path / "changes.json"
    raw = json.dumps([[_change()]])
    report.write_text(raw, encoding="utf-8")
    output = tmp_path / "github-output"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "Helper_Scripts.Supply_Chain.dependency_review",
            "--policy",
            str(policy_path),
            "--changes",
            str(report),
            "--paginated",
            "--github-output",
            str(output),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert output.read_text(encoding="utf-8") == f"allow-ghsas={GHSA}\n"
    assert report.read_text(encoding="utf-8") == raw


def _run_comparison_script(
    tmp_path: Path, step: dict, responses: list[object], env: dict[str, str]
) -> subprocess.CompletedProcess[str]:
    """Execute the trusted workflow script with only GitHub's HTTP boundary replaced."""
    fixture = tmp_path / "api-responses.json"
    fixture.write_text(json.dumps(responses), encoding="utf-8")
    script = tmp_path / "compare.cjs"
    script.write_text(
        "const fs = require('node:fs');\n"
        "const assert = require('node:assert/strict');\n"
        "const context = {repo: {owner: 'test', repo: 'repo'}};\n"
        "const core = {setFailed(message) {console.error(message); process.exitCode = 1;}};\n"
        "const github = {paginate: {async *iterator(route, parameters) {\n"
        "  assert.equal(route, 'GET /repos/{owner}/{repo}/dependency-graph/compare/{basehead}');\n"
        "  assert.deepEqual(parameters, {owner: 'test', repo: 'repo', "
        "basehead: 'a'.repeat(40) + '...' + 'b'.repeat(40), per_page: 100});\n"
        "  for (const response of JSON.parse(fs.readFileSync(process.env.API_FIXTURE, 'utf8'))) yield response;\n"
        "}}};\n"
        "(async () => {\n" + step["with"]["script"] + "\n})().catch(error => core.setFailed(error.message));\n",
        encoding="utf-8",
    )
    return subprocess.run(
        ["node", str(script)],
        cwd=tmp_path,
        env={**env, "API_FIXTURE": str(fixture)},
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.mark.parametrize("responses", [[], [{}], [{"data": None}], [{"data": {}}], [{"data": []}, {}]])
def test_trusted_comparison_rejects_missing_or_malformed_pages(tmp_path: Path, responses: list[object]) -> None:
    """Catches the credential-holding fetch step converting missing API data into a clean diff."""
    workflow = yaml.safe_load((ROOT / ".github/workflows/security-required.yml").read_text(encoding="utf-8"))
    by_id = {step.get("id"): step for step in workflow["jobs"]["security-required"]["steps"]}
    assert "dependency_review_compare" in by_id
    result = _run_comparison_script(
        tmp_path,
        by_id["dependency_review_compare"],
        responses,
        {**os.environ, "BASE_SHA": "a" * 40, "HEAD_SHA": "b" * 40, "RUNNER_TEMP": str(tmp_path)},
    )
    assert result.returncode == 1, result.stderr


@pytest.mark.parametrize(
    "post_report",
    [
        [_change(extra="x" * 200_000 + "\nDEPENDENCY_REVIEW_OUTPUT_JSON\n$(exit 19)\n'\n")],
        [_change(), _change(manifest="nested/uv.lock")],
    ],
)
def test_workflow_runs_complete_compare_and_revalidates_raw_action_output(tmp_path: Path, post_report: list) -> None:
    """Catches workflow integration using different refs, partial pages or filtered action output."""
    workflow = yaml.safe_load((ROOT / ".github/workflows/security-required.yml").read_text(encoding="utf-8"))
    steps = workflow["jobs"]["security-required"]["steps"]
    by_id = {step.get("id"): step for step in steps}
    assert "dependency_review_compare" in by_id
    assert "dependency_review_prepare" in by_id
    assert "dependency_review_validate" in by_id
    prepare = by_id["dependency_review_prepare"]
    compare = by_id["dependency_review_compare"]
    review = by_id["dependency_review"]
    validate = by_id["dependency_review_validate"]
    assert review["uses"] == "actions/dependency-review-action@2031cfc080254a8a887f58cffee85186f0e49e48"
    assert review["with"]["fail-on-severity"] == "high"
    assert review["with"]["allow-ghsas"] == "${{ steps.dependency_review_prepare.outputs.allow-ghsas }}"
    assert review.get("continue-on-error", False) is False
    assert not review["with"].get("warn-only", False)
    assert "DEPENDENCY_CHANGES" not in validate.get("env", {})
    assert "always()" in validate["if"]
    assert compare["env"]["BASE_SHA"] == review["with"]["base-ref"]
    assert compare["env"]["HEAD_SHA"] == review["with"]["head-ref"]
    assert compare["uses"].startswith("actions/github-script@")
    upload = next(step for step in steps if step.get("name") == "Upload dependency review evidence")
    assert "always()" in upload["if"]

    # Run the real shell and adapter; replace only the external GitHub API boundary.
    workspace = tmp_path / "workspace"
    policy_dir = workspace / ".github/supply-chain"
    policy_dir.mkdir(parents=True)
    now = datetime.now(timezone.utc).date()
    policy = _write_policy(
        tmp_path, [_record(created_on=now.isoformat(), expires_on=(now + timedelta(days=7)).isoformat())]
    )
    (policy_dir / "vulnerability-exceptions.json").write_bytes(policy.read_bytes())
    env = {
        **os.environ,
        "PYTHONPATH": str(ROOT),
        "BASE_SHA": "a" * 40,
        "HEAD_SHA": "b" * 40,
        "RUNNER_TEMP": str(tmp_path),
        "GITHUB_OUTPUT": str(tmp_path / "output"),
    }
    pages = [[_change()], [_change(vulnerabilities=[])]]
    result = _run_comparison_script(tmp_path, compare, [{"data": page} for page in pages], env)
    assert result.returncode == 0, result.stderr
    assert json.loads((tmp_path / "dependency-review/compare-pages.json").read_text()) == pages
    result = subprocess.run(
        ["bash", "-euo", "pipefail", "-c", prepare["run"]],
        cwd=workspace,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "output").read_text(encoding="utf-8") == f"allow-ghsas={GHSA}\n"
    env.update(ALLOW_GHSAS=GHSA)
    # Actions renders toJSON as a JSON string before executing the generated script.
    validate_script = tmp_path / "validate.sh"
    validate_script.write_text(
        validate["run"].replace(
            "${{ toJSON(steps.dependency_review.outputs.dependency-changes) }}", json.dumps(json.dumps(post_report))
        ),
        encoding="utf-8",
    )
    result = subprocess.run(
        ["bash", "-euo", "pipefail", str(validate_script)],
        cwd=workspace,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == (0 if len(post_report) == 1 else 1), result.stderr
    assert json.loads((tmp_path / "dependency-review/dependency-changes.json").read_text()) == post_report
