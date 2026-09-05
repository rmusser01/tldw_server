"""Contracts for the pinned, fail-closed source SBOM workflow."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[3]
WORKFLOW_PATH = ROOT / ".github/workflows/sbom.yml"
UV_IMAGE = (
    "ghcr.io/astral-sh/uv:0.12.7@"
    "sha256:95f2aa1fe59274951cfe9b0cbc7972e879ff1004bc8945d130a32eb0dbd85945"
)
CDXGEN_IMAGE = (
    "ghcr.io/cdxgen/cdxgen:v13@"
    "sha256:0be75639a833b59d1ba29b3c8ac00dfd2e41e7568d56b6c039007caadebebc0d"
)
CYCLONEDX_IMAGE = (
    "docker.io/cyclonedx/cyclonedx-cli:0.33.1@"
    "sha256:252c2e26f468c25fea1e63ecde1bc3198ad6e9dbb57f5ed3236bddcb2281b3a7"
)
TRIVY_IMAGE = (
    "ghcr.io/aquasecurity/trivy:0.74.0@"
    "sha256:62b1e65e8869bc4b4c6aa4fa2b21595256c7c2f6018a9d9ad61caf87187c1969"
)
PRODUCER_JOBS = {
    "generate-python",
    "generate-apps-workspace",
    "generate-admin-ui",
    "merge-source",
    "scan-source",
}


def _load() -> dict[str, Any]:
    return yaml.safe_load(WORKFLOW_PATH.read_text(encoding="utf-8"))


def _text() -> str:
    return WORKFLOW_PATH.read_text(encoding="utf-8")


def _triggers(workflow: dict[str, Any]) -> dict[str, Any]:
    # PyYAML 1.1 parses the unquoted key `on` as boolean true.
    return workflow.get("on", workflow.get(True, {}))


def _steps(workflow: dict[str, Any], job: str) -> list[dict[str, Any]]:
    return workflow["jobs"][job]["steps"]


def _run_text(workflow: dict[str, Any], job: str) -> str:
    return "\n".join(str(step.get("run", "")) for step in _steps(workflow, job))


def _uses_steps(workflow: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        step
        for job in workflow["jobs"].values()
        for step in job.get("steps", [])
        if "uses" in step
    ]


def test_sbom_workflow_has_read_only_reusable_entry_points() -> None:
    """Catches a source gate that cannot be reused or gains write authority."""
    workflow = _load()
    triggers = _triggers(workflow)

    assert {"workflow_call", "workflow_dispatch", "pull_request", "push"} <= set(triggers)
    assert "main" in triggers["push"]["branches"]
    assert workflow["permissions"] == {"contents": "read"}
    assert "packages: write" not in _text()
    assert "id-token: write" not in _text()


def test_sbom_workflow_upload_root_is_visible_and_concurrency_is_caller_scoped() -> None:
    """Catches hidden evidence omission or unrelated reusable runs cancelling each other."""
    workflow = _load()
    evidence_dir = str(workflow["env"]["EVIDENCE_DIR"])

    assert all(not part.startswith(".") for part in Path(evidence_dir).parts)
    assert "github.workflow" in str(workflow["concurrency"]["group"])


def test_sbom_workflow_has_exact_component_outputs() -> None:
    """Catches missing roots, mutable fallbacks, or soft validation."""
    text = _text()
    for name in (
        "sbom-python-root.cdx.json",
        "sbom-apps-workspace.cdx.json",
        "sbom-admin-ui.cdx.json",
        "sbom-source-aggregate.cdx.json",
    ):
        assert name in text

    for forbidden in (
        "package-lock.json",
        "continue-on-error: true",
        "npx -y",
        "pip install",
        "Resolve CycloneDX CLI digest",
    ):
        assert forbidden not in text


def test_sbom_workflow_pins_every_tool_and_external_action() -> None:
    """Catches a tag-only tool image or mutable third-party Action."""
    workflow = _load()
    text = _text()

    for image in (UV_IMAGE, CDXGEN_IMAGE, CYCLONEDX_IMAGE, TRIVY_IMAGE):
        assert image in text

    for step in _uses_steps(workflow):
        reference = str(step["uses"])
        if reference.startswith("./"):
            continue
        assert re.fullmatch(r"[^@]+@[0-9a-f]{40}", reference), reference


def test_python_sbom_comes_from_the_locked_production_resolution() -> None:
    """Catches SBOM generation from a hand-built or development dependency list."""
    workflow = _load()
    script = _run_text(workflow, "generate-python")

    assert '"$RUNNER_TEMP/uv" export' in script
    assert "uv 0\\.12\\.7" in script
    assert "--locked" in script
    assert "--no-dev" in script
    assert "--no-editable" in script
    assert "--format cyclonedx1.5" in script
    assert "sbom-python-root.cdx.json" in script
    for contract in ("bomFormat", "specVersion", "serialNumber", "components", "tldw-server"):
        assert contract in script
    assert "SENSITIVE_ENV_MARKERS" in script


def test_bun_sboms_use_pinned_required_only_cdxgen_profiles() -> None:
    """Catches npm-only or development-inclusive JavaScript inventories."""
    workflow = _load()
    apps = _run_text(workflow, "generate-apps-workspace")
    admin = _run_text(workflow, "generate-admin-ui")

    assert "cdxgen --version" in apps
    assert r"13\.0\.1([^0-9]|$)" in apps
    assert "--required-only" in apps
    assert "--recurse" in apps
    assert "/workspace/apps:ro" in apps
    assert '"**/.next/**"' in apps
    assert '"**/node_modules/**"' in apps
    assert "sbom-frontend-root.cdx.json" in apps
    assert "sbom-extension-root.cdx.json" in apps
    assert "sbom-ui-root.cdx.json" in apps
    assert "metadata_root_ids" in apps
    assert '("@tldw", "ui")' in apps
    for root in ("tldw-frontend", "tldw-assistant", "@tldw/ui"):
        assert root in apps
    assert 'assert "@playwright/test" not in names' in apps
    assert "SENSITIVE_ENV_MARKERS" in apps

    assert "cdxgen --version" in admin
    assert r"13\.0\.1([^0-9]|$)" in admin
    assert "--required-only" in admin
    assert "/workspace/admin-ui:ro" in admin
    assert '"**/.next/**"' in admin
    assert '"**/node_modules/**"' in admin
    assert "tldw-admin" in admin
    assert 'assert "@playwright/test" not in names' in admin
    assert "SENSITIVE_ENV_MARKERS" in admin


def test_containerized_source_tools_run_with_minimal_runtime_authority() -> None:
    """Catches source-reading tools retaining avoidable network or Linux authority."""
    workflow = _load()

    for job in ("generate-apps-workspace", "generate-admin-ui"):
        script = _run_text(workflow, job)
        assert "--network none" in script
        assert "--cap-drop ALL" in script
        assert "--security-opt no-new-privileges:true" in script
        assert "--read-only" in script

    merge = _run_text(workflow, "merge-source")
    assert "--network none" in merge
    assert "--cap-drop ALL" in merge
    assert "--security-opt no-new-privileges:true" in merge

    scan = _run_text(workflow, "scan-source")
    assert "trivy_scan()" in scan
    assert "--network none" in scan


def test_merge_validates_all_components_and_enforces_count_bounds() -> None:
    """Catches a partial merge or an implausibly small/large dependency graph."""
    workflow = _load()
    script = _run_text(workflow, "merge-source")

    for name in (
        "sbom-python-root.cdx.json",
        "sbom-apps-workspace.cdx.json",
        "sbom-admin-ui.cdx.json",
        "sbom-source-aggregate.cdx.json",
    ):
        assert f"validate --input-file {name}" in script
    assert "cyclonedx merge" in script
    assert "--input-files" in script
    assert "COMPONENT_BOUNDS" in script
    assert (
        "COMPONENT_BOUNDS='"
        '{"python-root":[276,461],"apps-workspace":[937,1563],'
        '"admin-ui":[253,423]}\''
    ) in script
    assert "tldw-server" in script
    assert "tldw-monorepo" in script
    assert "tldw-admin" in script


def test_source_scans_preserve_raw_reports_and_apply_exception_policy() -> None:
    """Catches incomplete Trivy evidence or policy applied before raw capture."""
    workflow = _load()
    script = _run_text(workflow, "scan-source")

    assert "image --download-db-only" in script
    assert "trivy --version" in script
    assert "metadata.json" in script
    assert "timedelta(hours=24)" in script
    assert "timedelta(minutes=-5)" in script
    assert "--scanners vuln" in script
    assert "--ignore-unfixed=false" in script
    assert "--format json" in script
    assert "--skip-db-update" in script
    assert "vulnerability-exceptions.json" in script
    assert "evaluate_trivy_report" in script
    for component in ("python-root", "apps-workspace", "admin-ui"):
        assert f"trivy-source-{component}.json" in script
        assert f"scan-decision-source-{component}.json" in script
    assert "policy_status=$?" in script
    assert 'exit "$policy_status"' in script


def test_artifact_uploads_fail_when_any_evidence_is_missing() -> None:
    """Catches successful jobs that silently upload an empty evidence set."""
    workflow = _load()
    upload_steps = [
        step
        for step in _uses_steps(workflow)
        if str(step["uses"]).startswith("actions/upload-artifact@")
    ]

    assert len(upload_steps) >= 5
    assert all(step.get("with", {}).get("if-no-files-found") == "error" for step in upload_steps)
    scan_upload = next(
        step for step in upload_steps if step.get("name") == "Upload complete source scan evidence"
    )
    assert "always()" in str(scan_upload["if"])


def test_final_gate_needs_every_producer_and_verifies_named_evidence() -> None:
    """Catches skipped, cancelled, or missing producers being reported as success."""
    workflow = _load()
    gate = workflow["jobs"]["source-gate"]
    script = _run_text(workflow, "source-gate")

    assert set(gate["needs"]) == PRODUCER_JOBS
    assert "always()" in str(gate["if"])
    for job in PRODUCER_JOBS:
        assert f"needs.{job}.result" in script
    for name in (
        "sbom-python-root.cdx.json",
        "sbom-apps-workspace.cdx.json",
        "sbom-admin-ui.cdx.json",
        "sbom-source-aggregate.cdx.json",
        "trivy-source-python-root.json",
        "trivy-source-apps-workspace.json",
        "trivy-source-admin-ui.json",
        "scan-decision-source-python-root.json",
        "scan-decision-source-apps-workspace.json",
        "scan-decision-source-admin-ui.json",
    ):
        assert name in script
    assert "sha256sum -c" in script
    assert "cmp gate/python/sbom-python-root.cdx.json" in script
    assert "cmp gate/apps/sbom-apps-workspace.cdx.json" in script
    assert "cmp gate/admin/sbom-admin-ui.cdx.json" in script
