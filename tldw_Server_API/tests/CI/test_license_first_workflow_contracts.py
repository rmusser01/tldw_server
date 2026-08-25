from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
WORKFLOW_PATH = REPO_ROOT / ".github/workflows/license-first-admission.yml"
CHANGE_DETECTOR_PATH = REPO_ROOT / ".github/actions/detect-required-gate-changes/action.yml"
CHECKOUT_ACTION = "actions/checkout@de0fac2e4500dabe0009e67214ff5f5447ce83dd"
OUTPUT_NAMES = ("pr_number", "head_sha", "base_sha", "base_ref", "should_run")
EXPECTED_PERMISSIONS = {
    "actions": "read",
    "contents": "read",
    "pull-requests": "read",
    "statuses": "read",
}
ORDINARY_WORKFLOW_NAMES = (
    "actionlint.yml",
    "backend-required.yml",
    "ci.yml",
    "codeql.yml",
    "container-build-check.yml",
    "coverage-required.yml",
    "e2e-required.yml",
    "e2e-smoke.yml",
    "frontend-e2e-tiers.yml",
    "frontend-required.yml",
    "frontend-ux-gates.yml",
    "jobs-suite.yml",
    "mcp-unified-rc.yml",
    "notes-remediation-targeted.yml",
    "onboarding-docs-gate.yml",
    "pre-commit.yml",
    "pypi-package.yml",
    "sbom.yml",
    "security-required.yml",
    "ui-characters-harness-tests.yml",
    "ui-dictionaries-tests.yml",
    "ui-playground-quality-gates.yml",
    "ui-research-workspace-parity.yml",
    "ui-watchlists-a11y-gates.yml",
    "ui-watchlists-extension-e2e.yml",
    "ui-watchlists-help-tests.yml",
    "ui-watchlists-scale-gates.yml",
    "ui-worldbooks-tests.yml",
)
DIRECT_TRIGGER_DIGESTS = {
    "actionlint.yml": "d31daa2c3e010b3a70dcbd640ef24f573bea18619228e7548c50992c54df0e99",
    "backend-required.yml": "4b65b09e5b40faee5abc68a5e88fd114d8fd4b5b0ae84eb977474336ffdd6653",
    "ci.yml": "a2ef246f468b2b946f6fa5f9805497d9c8a1c4ca7f3ef395172a290a17a3b3a6",
    "codeql.yml": "8592a8565d97d233d6b27fb13d9a563f673550f3d8220d0b262487346a70e25f",
    "container-build-check.yml": "4b65b09e5b40faee5abc68a5e88fd114d8fd4b5b0ae84eb977474336ffdd6653",
    "coverage-required.yml": "4b65b09e5b40faee5abc68a5e88fd114d8fd4b5b0ae84eb977474336ffdd6653",
    "e2e-required.yml": "4b65b09e5b40faee5abc68a5e88fd114d8fd4b5b0ae84eb977474336ffdd6653",
    "e2e-smoke.yml": "47fa5c3b099aeee7d744da19e5c8db6f403d82963b9d19b9f60292f0377d7180",
    "frontend-e2e-tiers.yml": "c157a675e397e879a8b394b26a99e472dfbe87f5f3f6236c63907d6d21897a79",
    "frontend-required.yml": "4b65b09e5b40faee5abc68a5e88fd114d8fd4b5b0ae84eb977474336ffdd6653",
    "frontend-ux-gates.yml": "4b65b09e5b40faee5abc68a5e88fd114d8fd4b5b0ae84eb977474336ffdd6653",
    "jobs-suite.yml": "023309eacbc53b533c94c56e00e710422e6e5885ba06a0921524964db7cf4476",
    "mcp-unified-rc.yml": "8b1a9335ecab966807fcf0a8c81560b264942decc458cd9bc98275d6062310aa",
    "notes-remediation-targeted.yml": "090c7ff3c4b668cd6c76fc3b9fc9a1b3be128b21b3b2b236cdfe33dd6f5edb5a",
    "onboarding-docs-gate.yml": "38fc27642d68b7b198e20a9e910ddfa2e1f626b32be5911df63d64c89cb60afa",
    "pre-commit.yml": "d3d381eead20326078cb0268da7340f8217d95336af86c8d9701419c5e67d3be",
    "pypi-package.yml": "1faf356cfe858d94100ea61d578f0b9d132c73533277c552fd03f1e3ac141823",
    "sbom.yml": "6ba4774ab2129a605bb160c5ee6f48ebc51544881592ad60ad5268b910fcfbde",
    "security-required.yml": "4b65b09e5b40faee5abc68a5e88fd114d8fd4b5b0ae84eb977474336ffdd6653",
    "ui-characters-harness-tests.yml": "8977899d9903b59c454c686e1fda272a7b919f63782d84ec8c24973449f2eee6",
    "ui-dictionaries-tests.yml": "f191b9920abca964265ff7ff4510cc9ac554770d4e4a0b8b172854147c0c0dbe",
    "ui-playground-quality-gates.yml": "af4b03b1d48dd12b93e1bd39a8036109b26a7890ee5cd8ef8ed57b7d2da3008a",
    "ui-research-workspace-parity.yml": "aa73478a8917cbf0eb7be01e1f4212f1a0692200919d9a09982ab6525af49503",
    "ui-watchlists-a11y-gates.yml": "3c66abf5ccddb422faad37764afec0e3c9d0ec66e74c36844f99425eb0214526",
    "ui-watchlists-extension-e2e.yml": "4252eda6295ace0621dc9b3bc1a98e157753a6ba984218103942fbf6bbcdda42",
    "ui-watchlists-help-tests.yml": "fe0ad30818f046c2ae0c54a10a1bb9f481968d8b73545ebe934135b2fed88907",
    "ui-watchlists-scale-gates.yml": "9001042a9f6b85245d7f80dc4d13e16b214df17c2629e1c733ee691e99a4c884",
    "ui-worldbooks-tests.yml": "c09cbf04ae823361ff2fb13da6b9c4c2b55773a2e58e88bf8bea635f30221ed2",
}
ORIGINAL_JOB_NAMES = {
    "actionlint.yml": ("actionlint",),
    "backend-required.yml": ("changes", "backend-required"),
    "ci.yml": (
        "http-client-patch-guard",
        "syntax-check",
        "preflight-python-310",
        "shard-coverage",
        "quickstart-dry-run",
        "lint",
        "frontend-lint",
        "wizard-tests",
        "changes",
        "full-suite-linux-311-smoke",
        "full-suite-linux-312-shards",
        "full-suite-linux-312-summary",
        "full-suite-linux-313-shards",
        "full-suite-linux-313-summary",
        "full-suite-macos-312-shards",
        "full-suite-macos-312-summary",
        "full-suite-windows-312-shards",
        "full-suite-windows-312-summary",
        "full-suite-os-313-release-shards",
        "character-chat-rate-limits",
    ),
    "codeql.yml": ("analyze",),
    "container-build-check.yml": ("build", "container-build-check"),
    "coverage-required.yml": ("changes", "coverage-required"),
    "e2e-required.yml": ("changes", "e2e-required"),
    "e2e-smoke.yml": ("e2e-smoke",),
    "frontend-e2e-tiers.yml": ("critical", "features", "admin"),
    "frontend-required.yml": ("changes", "frontend-unit-tests", "frontend-required"),
    "frontend-ux-gates.yml": ("onboarding-gate", "smoke-gate"),
    "jobs-suite.yml": ("jobs-sqlite", "jobs-postgres"),
    "mcp-unified-rc.yml": ("internal-rc", "portable-stdio"),
    "notes-remediation-targeted.yml": ("notes-ui-remediation", "notes-backend-remediation"),
    "onboarding-docs-gate.yml": ("onboarding-docs-gate",),
    "pre-commit.yml": ("run-pre-commit",),
    "pypi-package.yml": ("build-and-check",),
    "sbom.yml": ("build-sbom",),
    "security-required.yml": ("changes", "security-required"),
    "ui-characters-harness-tests.yml": ("characters-harness",),
    "ui-dictionaries-tests.yml": ("dictionaries-vitest",),
    "ui-playground-quality-gates.yml": ("playground-quality",),
    "ui-research-workspace-parity.yml": (
        "webui-research-workspace-parity",
        "extension-research-workspace-parity",
    ),
    "ui-watchlists-a11y-gates.yml": ("watchlists-a11y-gate",),
    "ui-watchlists-extension-e2e.yml": ("watchlists-extension-e2e",),
    "ui-watchlists-help-tests.yml": ("watchlists-help-vitest",),
    "ui-watchlists-scale-gates.yml": ("watchlists-scale-gate",),
    "ui-worldbooks-tests.yml": ("worldbooks-vitest",),
}
ORIGINAL_DEPENDENCIES = {
    ("backend-required.yml", "backend-required"): ("changes",),
    ("ci.yml", "full-suite-linux-311-smoke"): ("lint", "syntax-check", "changes"),
    ("ci.yml", "full-suite-linux-312-shards"): ("lint", "syntax-check", "changes"),
    ("ci.yml", "full-suite-linux-312-summary"): ("full-suite-linux-312-shards", "changes"),
    ("ci.yml", "full-suite-linux-313-shards"): ("lint", "syntax-check", "changes"),
    ("ci.yml", "full-suite-linux-313-summary"): ("full-suite-linux-313-shards", "changes"),
    ("ci.yml", "full-suite-macos-312-shards"): ("lint", "syntax-check", "changes"),
    ("ci.yml", "full-suite-macos-312-summary"): ("full-suite-macos-312-shards", "changes"),
    ("ci.yml", "full-suite-windows-312-shards"): ("lint", "syntax-check", "changes"),
    ("ci.yml", "full-suite-windows-312-summary"): ("full-suite-windows-312-shards", "changes"),
    ("ci.yml", "full-suite-os-313-release-shards"): ("lint", "syntax-check"),
    ("ci.yml", "character-chat-rate-limits"): (
        "full-suite-linux-312-summary",
        "full-suite-linux-313-summary",
        "changes",
    ),
    ("container-build-check.yml", "container-build-check"): ("build",),
    ("coverage-required.yml", "coverage-required"): ("changes",),
    ("e2e-required.yml", "e2e-required"): ("changes",),
    ("frontend-required.yml", "frontend-unit-tests"): ("changes",),
    ("frontend-required.yml", "frontend-required"): (
        "changes",
        "frontend-unit-tests",
    ),
    ("jobs-suite.yml", "jobs-postgres"): ("jobs-sqlite",),
    ("security-required.yml", "security-required"): ("changes",),
}
ALWAYS_ROLLUPS = {
    ("container-build-check.yml", "container-build-check"),
    ("ci.yml", "full-suite-linux-312-summary"),
    ("ci.yml", "full-suite-linux-313-summary"),
    ("ci.yml", "full-suite-macos-312-summary"),
    ("ci.yml", "full-suite-windows-312-summary"),
}
DIRECT_ADMISSION_JOBS = ALWAYS_ROLLUPS | {
    ("backend-required.yml", "backend-required"),
    ("frontend-required.yml", "frontend-unit-tests"),
    ("frontend-required.yml", "frontend-required"),
    ("security-required.yml", "security-required"),
}
NON_ADMITTED_ROOT_JOBS = {
    ("ci.yml", "preflight-python-310"),
}
BACKEND_CHANGED_JOBS = {
    "full-suite-linux-311-smoke",
    "full-suite-linux-312-shards",
    "full-suite-linux-312-summary",
    "full-suite-linux-313-shards",
    "full-suite-linux-313-summary",
    "full-suite-macos-312-shards",
    "full-suite-macos-312-summary",
    "full-suite-windows-312-shards",
    "full-suite-windows-312-summary",
    "character-chat-rate-limits",
}
FETCH_DEPTH_CHECKOUTS = {
    ("backend-required.yml", "changes"),
    ("backend-required.yml", "backend-required"),
    ("ci.yml", "changes"),
    ("coverage-required.yml", "changes"),
    ("e2e-required.yml", "changes"),
    ("frontend-required.yml", "changes"),
    ("frontend-required.yml", "frontend-unit-tests"),
    ("frontend-required.yml", "frontend-required"),
    ("onboarding-docs-gate.yml", "onboarding-docs-gate"),
    ("pre-commit.yml", "run-pre-commit"),
    ("security-required.yml", "changes"),
}
CHANGE_CLASSIFIER_OUTPUTS = {
    "backend-required.yml": {
        "backend_changed",
        "frontend_changed",
        "e2e_changed",
        "security_relevant_changed",
        "coverage_required",
    },
    "ci.yml": {"backend_changed"},
    "coverage-required.yml": {
        "backend_changed",
        "frontend_changed",
        "e2e_changed",
        "security_relevant_changed",
        "coverage_required",
    },
    "e2e-required.yml": {
        "backend_changed",
        "frontend_changed",
        "e2e_changed",
        "security_relevant_changed",
        "coverage_required",
    },
    "frontend-required.yml": {
        "backend_changed",
        "frontend_changed",
        "tldw_frontend_changed",
        "family_guardrails_changed",
        "admin_ui_changed",
        "e2e_changed",
        "security_relevant_changed",
        "coverage_required",
    },
    "security-required.yml": {
        "backend_changed",
        "frontend_changed",
        "e2e_changed",
        "security_relevant_changed",
        "coverage_required",
    },
}
CHANGE_CLASSIFIER_JOBS = {(name, "changes") for name in CHANGE_CLASSIFIER_OUTPUTS}


def _load_workflow() -> tuple[dict[str, Any], str]:
    assert WORKFLOW_PATH.exists(), "the reusable admission workflow is missing"
    text = WORKFLOW_PATH.read_text(encoding="utf-8")
    data = yaml.safe_load(text)
    assert isinstance(data, dict)
    return data, text


def _trigger(data: dict[str, Any]) -> dict[str, Any]:
    trigger_key = "on" if "on" in data else True
    trigger = data[trigger_key]
    assert isinstance(trigger, dict)
    return trigger


def _load_ordinary_workflows() -> dict[str, tuple[dict[str, Any], str]]:
    workflows: dict[str, tuple[dict[str, Any], str]] = {}
    for path in sorted((REPO_ROOT / ".github/workflows").glob("*.yml")):
        text = path.read_text(encoding="utf-8")
        data = yaml.safe_load(text)
        if not isinstance(data, dict):
            continue
        trigger = _trigger(data)
        if "pull_request" in trigger:
            workflows[path.name] = (data, text)
    return workflows


def _needs(job: dict[str, Any]) -> list[str]:
    needs = job.get("needs", [])
    return [needs] if isinstance(needs, str) else list(needs)


def _normalized(expression: object) -> str:
    text = str(expression).strip()
    if text.startswith("${{") and text.endswith("}}"):
        text = text[3:-2]
    return re.sub(r"\s+", "", text)


def _admission_job(data: dict[str, Any]) -> dict[str, Any]:
    assert set(data["jobs"]) == {"admission"}
    return data["jobs"]["admission"]


def _unquoted_shell_expansions(script: str) -> list[str]:
    expansions: list[str] = []
    pattern = re.compile(r"\$(?:\{[^}]+\}|\([^)]*\)|[A-Za-z_][A-Za-z0-9_]*)")
    in_single_quote = False
    for line in script.splitlines():
        visible = []
        for character in line:
            if character == "'":
                in_single_quote = not in_single_quote
                visible.append(" ")
            else:
                visible.append(" " if in_single_quote else character)
        visible_line = "".join(visible)
        for match in pattern.finditer(visible_line):
            prefix = visible_line[: match.start()]
            unescaped_double_quotes = len(re.findall(r'(?<!\\)"', prefix))
            if unescaped_double_quotes % 2 == 0:
                expansions.append(match.group())
    return expansions


def test_ordinary_pr_workflow_inventory_and_direct_triggers_are_frozen() -> None:
    workflows = _load_ordinary_workflows()
    assert tuple(workflows) == ORDINARY_WORKFLOW_NAMES
    assert set(DIRECT_TRIGGER_DIGESTS) == set(ORDINARY_WORKFLOW_NAMES)

    for name, (data, _) in workflows.items():
        direct_trigger = dict(_trigger(data))
        direct_trigger.pop("workflow_run", None)
        encoded = json.dumps(direct_trigger, sort_keys=True, separators=(",", ":")).encode()
        assert hashlib.sha256(encoded).hexdigest() == DIRECT_TRIGGER_DIGESTS[name], name


def test_all_ordinary_workflows_call_exact_inert_admission_gate() -> None:
    for name, (data, _) in _load_ordinary_workflows().items():
        assert _trigger(data)["workflow_run"] == {
            "workflows": ["Frontend License Gate Audit"],
            "types": ["completed"],
        }, name

        admission = data["jobs"]["admission"]
        assert admission == {
            "if": (
                "vars.LICENSE_FIRST_CI_ENABLED == 'true' && "
                "github.event_name == 'workflow_run' && "
                "github.event.workflow_run.conclusion == 'success'"
            ),
            "uses": "./.github/workflows/license-first-admission.yml",
            "with": {"workflow_file": name},
            "permissions": EXPECTED_PERMISSIONS,
        }, name


def test_runner_roots_cannot_bypass_admission_and_checkouts_are_immutable() -> None:
    checkout_count = 0
    admission_clause = (
        """
        always() && !cancelled() &&
        (
          (
            github.event_name == 'workflow_run' &&
            needs.admission.result == 'success' &&
            needs.admission.outputs.should_run == 'true'
          ) ||
          github.event_name != 'workflow_run'
        )
        """
    )
    event_ref = (
        "${{ github.event.workflow_run.pull_requests[0].head.sha || "
        "github.event.pull_request.head.sha || github.sha }}"
    )
    admission_ref = (
        "${{ needs.admission.outputs.head_sha || "
        "github.event.workflow_run.pull_requests[0].head.sha || "
        "github.event.pull_request.head.sha || github.sha }}"
    )
    classifier_ref = (
        "${{ needs.admission.outputs.head_sha || "
        "github.event.workflow_run.pull_requests[0].head.sha || github.sha }}"
    )
    backend_changed = (
        "(github.event_name != 'pull_request' && "
        "github.event_name != 'workflow_run') || "
        "needs.changes.outputs.backend_changed == 'true'"
    )
    frontend_conditions = {
        "critical": (
            "(github.event_name == 'pull_request' || "
            "github.event_name == 'workflow_run') || "
            "(github.event_name == 'workflow_dispatch' && "
            "contains(fromJSON('[\"critical\",\"all-tiers\"]'), github.event.inputs.tier))"
        ),
        "features": (
            "github.event_name == 'workflow_dispatch' && "
            "contains(fromJSON('[\"features\",\"all-tiers\"]'), github.event.inputs.tier)"
        ),
        "admin": (
            "github.event_name == 'workflow_dispatch' && "
            "contains(fromJSON('[\"admin\",\"all-tiers\"]'), github.event.inputs.tier)"
        ),
    }

    for name, (data, _) in _load_ordinary_workflows().items():
        jobs = data["jobs"]
        assert tuple(job_name for job_name in jobs if job_name != "admission") == ORIGINAL_JOB_NAMES[name]
        for job_name in ORIGINAL_JOB_NAMES[name]:
            job = jobs[job_name]
            needs = _needs(job)
            original_needs = ORIGINAL_DEPENDENCIES.get((name, job_name), ())
            assert tuple(dependency for dependency in needs if dependency != "admission") == original_needs

            root = not original_needs
            directly_guarded = (name, job_name) in DIRECT_ADMISSION_JOBS
            non_admitted_root = (name, job_name) in NON_ADMITTED_ROOT_JOBS
            if root or directly_guarded:
                if non_admitted_root:
                    assert "admission" not in needs, (name, job_name)
                    assert job.get("if") is None, (name, job_name)
                    continue
                assert needs.count("admission") == 1, (name, job_name)
                extra_condition = None
                if name == "frontend-e2e-tiers.yml":
                    extra_condition = frontend_conditions[job_name]
                elif name == "ci.yml" and job_name in BACKEND_CHANGED_JOBS:
                    extra_condition = backend_changed
                elif (name, job_name) in {
                    ("backend-required.yml", "backend-required"),
                    ("frontend-required.yml", "frontend-unit-tests"),
                    ("frontend-required.yml", "frontend-required"),
                    ("security-required.yml", "security-required"),
                }:
                    extra_condition = "needs.changes.result == 'success'"
                    if (name, job_name) == (
                        "frontend-required.yml",
                        "frontend-unit-tests",
                    ):
                        extra_condition += (
                            " && needs.changes.outputs.tldw_frontend_changed == 'true'"
                        )
                expected_condition = admission_clause
                if extra_condition:
                    expected_condition += f" && ({extra_condition})"
                assert _normalized(job.get("if")) == _normalized(expected_condition), (name, job_name)
            else:
                assert "admission" not in needs, (name, job_name)
                if name == "ci.yml" and job_name in BACKEND_CHANGED_JOBS:
                    assert _normalized(job.get("if")) == _normalized(backend_changed), (name, job_name)
                elif (name, job_name) == ("ci.yml", "full-suite-os-313-release-shards"):
                    assert _normalized(job.get("if")) == _normalized(
                        "github.event_name != 'pull_request' && "
                        "github.event_name != 'workflow_run'"
                    )
                else:
                    assert job.get("if") is None, (name, job_name)

            for step in job.get("steps", []):
                if not str(step.get("uses", "")).startswith("actions/checkout@"):
                    continue
                checkout_count += 1
                checkout_inputs = step.get("with", {})
                assert checkout_inputs.get("persist-credentials") is False, (name, job_name)
                if (name, job_name) in CHANGE_CLASSIFIER_JOBS:
                    expected_ref = classifier_ref
                else:
                    expected_ref = admission_ref if "admission" in needs else event_ref
                assert checkout_inputs.get("ref") == expected_ref, (name, job_name)
                other_inputs = {
                    key: value
                    for key, value in checkout_inputs.items()
                    if key not in {"ref", "persist-credentials"}
                }
                expected_other_inputs = (
                    {"fetch-depth": 0} if (name, job_name) in FETCH_DEPTH_CHECKOUTS else {}
                )
                assert other_inputs == expected_other_inputs, (name, job_name)

    assert checkout_count == 55


def test_pr_context_and_base_diff_logic_are_workflow_run_safe() -> None:
    workflows = _load_ordinary_workflows()
    concurrency_suffix = (
        "${{ github.event.workflow_run.pull_requests[0].number || "
        "github.event.pull_request.number || github.ref || github.run_id }}"
    )
    for name, (data, text) in workflows.items():
        if name == "actionlint.yml":
            assert "concurrency" not in data
        else:
            prefix = "${{ github.workflow }}" if name == "jobs-suite.yml" else name.removesuffix(".yml")
            assert data["concurrency"]["group"] == f"{prefix}-{concurrency_suffix}", name
            expected_cancel: object = True
            if name == "jobs-suite.yml":
                expected_cancel = (
                    "${{ github.event_name == 'pull_request' || "
                    "github.event_name == 'workflow_run' }}"
                )
            assert data["concurrency"]["cancel-in-progress"] == expected_cancel, name

        assert "github.head_ref" not in text
        assert "github.base_ref" not in text
        assert "GITHUB_HEAD_REF" not in text
        assert "GITHUB_BASE_REF" not in text
        assert "github.event.workflow_run.head_sha" not in text

    combined_text = "\n".join(text for _, text in workflows.values())
    assert combined_text.count("github.event.workflow_run.pull_requests[0].number") == 27
    assert combined_text.count("github.event.pull_request.number") == 27
    assert combined_text.count("github.event.workflow_run.pull_requests[0].head.sha") == 55
    assert combined_text.count("github.event.pull_request.head.sha") == 52
    assert combined_text.count("github.event.pull_request.base.sha") == 4
    assert combined_text.count("needs.admission.outputs.base_sha") == 11

    for name, output_names in CHANGE_CLASSIFIER_OUTPUTS.items():
        changes_job = workflows[name][0]["jobs"]["changes"]
        assert set(changes_job["outputs"]) == output_names
        for output_name in output_names:
            assert changes_job["outputs"][output_name] == (
                f"${{{{ steps.detect.outputs.{output_name} || "
                f"steps.detect_admitted.outputs.{output_name} }}}}"
            )
        direct_step = next(step for step in changes_job["steps"] if step.get("id") == "detect")
        admitted_step = next(
            step for step in changes_job["steps"] if step.get("id") == "detect_admitted"
        )
        assert direct_step["if"] == "github.event_name != 'workflow_run'"
        assert direct_step["uses"] == "./.github/actions/detect-required-gate-changes"
        assert admitted_step["if"] == "github.event_name == 'workflow_run'"
        assert admitted_step["env"] == {
            "BASE_SHA": "${{ needs.admission.outputs.base_sha }}",
            "HEAD_SHA": "${{ needs.admission.outputs.head_sha }}",
        }
        admitted_script = admitted_step["run"]
        assert "< <(git diff" not in admitted_script
        assert 'git cat-file -e "${BASE_SHA}^{commit}"' in admitted_script
        assert 'git cat-file -e "${HEAD_SHA}^{commit}"' in admitted_script
        assert 'git fetch --no-tags --depth=1 origin "$BASE_SHA"' in admitted_script
        assert 'git fetch --no-tags --depth=1 origin "$HEAD_SHA"' in admitted_script
        assert (
            'git diff --name-only "$BASE_SHA" "$HEAD_SHA" > "$CHANGED_FILES_PATH"'
            in admitted_script
        )
        assert 'mapfile -t CHANGED_FILES < "$CHANGED_FILES_PATH"' in admitted_script
        assert (
            'python -m Helper_Scripts.ci.emit_ci_gate_flags "${CHANGED_FILES[@]}"'
            in admitted_script
        )

    backend_text = workflows["backend-required.yml"][1]
    assert (
        'BASE_SHA="${{ needs.admission.outputs.base_sha || '
        'github.event.pull_request.base.sha || github.event.before }}"'
    ) in backend_text
    assert (
        'HEAD_SHA="${{ needs.admission.outputs.head_sha || '
        'github.event.pull_request.head.sha || github.sha }}"'
    ) in backend_text
    assert 'git diff --name-only "$BASE_SHA" "$HEAD_SHA"' in backend_text
    assert "< <(git diff" not in backend_text
    backend_mypy_script = next(
        step["run"]
        for step in workflows["backend-required.yml"][0]["jobs"]["backend-required"]["steps"]
        if step.get("name") == "Type check changed backend modules"
    )
    assert 'git cat-file -e "${BASE_SHA}^{commit}"' in backend_mypy_script
    assert 'git cat-file -e "${HEAD_SHA}^{commit}"' in backend_mypy_script
    assert 'git fetch --no-tags --depth=1 origin "$BASE_SHA"' in backend_mypy_script
    assert 'git fetch --no-tags --depth=1 origin "$HEAD_SHA"' in backend_mypy_script

    frontend_text = workflows["frontend-required.yml"][1]
    assert (
        'if [[ "${{ github.event_name }}" == "pull_request" || '
        '"${{ github.event_name }}" == "workflow_run" ]]; then'
    ) in frontend_text
    assert (
        'BASE_SHA="${{ needs.admission.outputs.base_sha || '
        'github.event.pull_request.base.sha }}"'
    ) in frontend_text
    frontend_unit_job = workflows["frontend-required.yml"][0]["jobs"]["frontend-unit-tests"]
    frontend_checkout = next(
        step
        for step in frontend_unit_job["steps"]
        if step.get("name") == "Checkout"
    )
    assert frontend_checkout["with"]["fetch-depth"] == 0
    frontend_unit_script = next(
        step["run"]
        for step in frontend_unit_job["steps"]
        if step.get("name") == "Run package-owned frontend unit tests"
    )
    assert 'git cat-file -e "${BASE_SHA}^{commit}"' in frontend_unit_script
    assert 'git fetch --no-tags --depth=1 origin "$BASE_SHA"' in frontend_unit_script
    assert frontend_unit_script.index('git cat-file -e "${BASE_SHA}^{commit}"') < (
        frontend_unit_script.index('package_vitest_args=("--changed=${BASE_SHA}"')
    )

    pre_commit = workflows["pre-commit.yml"][0]
    pre_commit_script = next(
        step["run"]
        for step in pre_commit["jobs"]["run-pre-commit"]["steps"]
        if step.get("name") == "Run pre-commit"
    )
    workflow_run_start = pre_commit_script.index(
        'if [ "${{ github.event_name }}" = "workflow_run" ]; then'
    )
    pull_request_start = pre_commit_script.index(
        'elif [ "${{ github.event_name }}" = "pull_request" ]; then'
    )
    push_start = pre_commit_script.index(
        'elif [ "${{ github.event_name }}" = "push" ]; then'
    )
    workflow_run_branch = pre_commit_script[workflow_run_start:pull_request_start]
    pull_request_branch = pre_commit_script[pull_request_start:push_start]
    assert 'HEAD_SHA="${{ needs.admission.outputs.head_sha }}"' in workflow_run_branch
    assert 'FROM_REF="${{ needs.admission.outputs.base_sha }}"' in workflow_run_branch
    assert 'HEAD_SHA}^' not in workflow_run_branch
    assert 'FROM_REF="${{ github.event.pull_request.base.sha }}"' in pull_request_branch
    assert 'FROM_REF="${HEAD_SHA}^"' in pull_request_branch

    security_job = workflows["security-required.yml"][0]["jobs"]["security-required"]
    pull_request_only_steps = [
        step
        for step in security_job["steps"]
        if step.get("name") == "Dependency review (high/critical)"
    ]
    assert len(pull_request_only_steps) == 1
    assert pull_request_only_steps[0]["if"] == (
        "github.event_name == 'pull_request' || github.event_name == 'workflow_run'"
    )
    assert pull_request_only_steps[0]["with"] == {
        "base-ref": "${{ needs.admission.outputs.base_sha || github.event.pull_request.base.sha }}",
        "head-ref": (
            "${{ needs.admission.outputs.head_sha || "
            "github.event.pull_request.head.sha || github.sha }}"
        ),
        "fail-on-severity": "high",
    }

    jobs_suite_text = workflows["jobs-suite.yml"][1]
    assert "ports[5432]" not in jobs_suite_text
    assert jobs_suite_text.count("ports['5432']") == 4


def test_shared_change_detector_fails_closed_on_diff_errors() -> None:
    script = yaml.safe_load(CHANGE_DETECTOR_PATH.read_text(encoding="utf-8"))["runs"][
        "steps"
    ][0]["run"]

    assert "< <(git diff" not in script
    assert (
        'git diff --name-only "$BASE_SHA" "${{ github.sha }}" > "$CHANGED_FILES_PATH"'
        in script
    )
    assert 'mapfile -t CHANGED_FILES < "$CHANGED_FILES_PATH"' in script


def test_actionlint_scans_the_complete_workflow_directory() -> None:
    actionlint_job = _load_ordinary_workflows()["actionlint.yml"][0]["jobs"]["actionlint"]
    lint_step = next(
        step for step in actionlint_job["steps"] if step.get("name") == "Run actionlint"
    )
    assert lint_step["run"] == (
        "set -euo pipefail\n"
        "./actionlint -color -config-file .github/actionlint.yaml\n"
    )


def test_admitted_jobs_have_no_secrets_or_write_scoped_credentials() -> None:
    for name, (data, text) in _load_ordinary_workflows().items():
        assert "${{ secrets." not in text, name
        assert "secrets: inherit" not in text, name
        assert "write" not in data.get("permissions", {}).values(), name

        for job_name, job in data["jobs"].items():
            permissions = job.get("permissions", {})
            assert "write" not in permissions.values(), (name, job_name, permissions)

            for step in job.get("steps", []):
                serialized_step = json.dumps(step, sort_keys=True)
                if "${{ github.token }}" in serialized_step:
                    token_condition = _normalized(step.get("if"))
                    assert token_condition == "github.event_name!='workflow_run'" or (
                        token_condition.startswith("github.event_name!='workflow_run'&&")
                    ), (name, job_name, step.get("name"))
                for value in step.get("env", {}).values():
                    value_text = str(value)
                    assert "${{ secrets." not in value_text, (name, job_name, step.get("name"))
                    assert "${{ github.token }}" not in value_text, (
                        name,
                        job_name,
                        step.get("name"),
                    )

    codeql = _load_ordinary_workflows()["codeql.yml"][0]["jobs"]["analyze"]
    assert codeql["permissions"] == {
        "actions": "read",
        "contents": "read",
        "security-events": "read",
    }
    analyze = next(
        step for step in codeql["steps"] if step.get("name") == "Perform CodeQL Analysis"
    )
    assert analyze["with"]["upload"] is False


def test_admitted_jobs_restore_but_cannot_save_shared_caches() -> None:
    setup_helper_count = 0
    setup_python_cache_count = 0
    cache_save_count = 0
    cache_restore_count = 0

    for name, (data, _) in _load_ordinary_workflows().items():
        for job_name, job in data["jobs"].items():
            steps = job.get("steps", [])
            for step in steps:
                uses = str(step.get("uses", ""))
                inputs = step.get("with", {})

                if uses == "./.github/actions/setup-python-deps":
                    setup_helper_count += 1
                    cache_condition = inputs.get("enable-pip-cache")
                    assert cache_condition in {
                        "${{ github.event_name != 'workflow_run' }}",
                        (
                            "${{ github.event_name != 'workflow_run' && "
                            "runner.os != 'Windows' }}"
                        ),
                    }, (
                        name,
                        job_name,
                        step.get("name"),
                    )

                if uses.startswith("actions/setup-python@") and "cache" in inputs:
                    setup_python_cache_count += 1
                    assert inputs["cache"] == (
                        "${{ github.event_name != 'workflow_run' && 'pip' || '' }}"
                    ), (name, job_name, step.get("name"))

                if uses.startswith("actions/cache@"):
                    cache_save_count += 1
                    save_condition = _normalized(step.get("if"))
                    assert save_condition == "github.event_name!='workflow_run'" or (
                        save_condition.startswith("github.event_name!='workflow_run'&&")
                    ), (
                        name,
                        job_name,
                        step.get("name"),
                    )
                    matching_restores = [
                        candidate
                        for candidate in steps
                        if str(candidate.get("uses", "")).startswith("actions/cache/restore@")
                        and candidate.get("with", {}) == inputs
                    ]
                    assert len(matching_restores) == 1, (name, job_name, step.get("name"))
                    restore_condition = _normalized(matching_restores[0].get("if"))
                    assert restore_condition == "github.event_name=='workflow_run'" or (
                        restore_condition.startswith("github.event_name=='workflow_run'&&")
                    )

                if uses.startswith("actions/cache/restore@"):
                    cache_restore_count += 1

                if uses.startswith("actions/cache/save@"):
                    save_condition = _normalized(step.get("if"))
                    assert save_condition == "github.event_name!='workflow_run'" or (
                        save_condition.startswith("github.event_name!='workflow_run'&&")
                    )

    assert setup_helper_count == 24
    assert setup_python_cache_count == 1
    assert cache_save_count == 6
    assert cache_restore_count == 6


def test_reusable_workflow_has_exact_interface_and_read_permissions() -> None:
    data, text = _load_workflow()
    trigger_key = "on" if "on" in data else True

    assert set(data) == {"name", trigger_key, "permissions", "jobs"}
    trigger = _trigger(data)
    assert set(trigger) == {"workflow_call"}
    call = trigger["workflow_call"]
    assert set(call) == {"inputs", "outputs"}
    assert call["inputs"] == {
        "workflow_file": {
            "description": "Caller workflow filename used for conservative path routing",
            "required": True,
            "type": "string",
        }
    }
    assert "secrets" not in call
    assert set(call["outputs"]) == set(OUTPUT_NAMES)
    assert call["outputs"] == {
        name: {
            "description": f"Validated {name.replace('_', ' ')}",
            "value": f"${{{{ jobs.admission.outputs.{name} }}}}",
        }
        for name in OUTPUT_NAMES
    }
    assert data["permissions"] == EXPECTED_PERMISSIONS
    assert not re.search(r"\b(?:write|secrets?)\b", text, re.IGNORECASE)


def test_admission_job_is_one_short_runner_with_only_validated_outputs() -> None:
    data, _ = _load_workflow()
    job = _admission_job(data)

    assert set(job) == {
        "runs-on",
        "timeout-minutes",
        "outputs",
        "steps",
    }
    assert job["runs-on"] == "ubuntu-latest"
    assert job["timeout-minutes"] == 5
    assert job["outputs"] == {name: f"${{{{ steps.admit.outputs.{name} }}}}" for name in OUTPUT_NAMES}
    assert "strategy" not in job
    assert "environment" not in job
    assert "secrets" not in job
    assert "env" not in job

    steps = job["steps"]
    assert len(steps) == 2
    assert [(step.get("name"), step.get("id")) for step in steps] == [
        ("Checkout trusted admission source", None),
        ("Validate trusted run and route caller", "admit"),
    ]
    assert all("if" not in step for step in steps)
    assert all("continue-on-error" not in step for step in steps)


def test_checkout_is_pinned_to_trusted_workflow_source_without_credentials() -> None:
    data, text = _load_workflow()
    checkout, admission = _admission_job(data)["steps"]

    assert checkout == {
        "name": "Checkout trusted admission source",
        "uses": CHECKOUT_ACTION,
        "with": {
            "ref": "${{ github.workflow_sha }}",
            "persist-credentials": False,
        },
    }
    assert "uses" not in admission
    assert text.count("actions/checkout@") == 1
    assert "github.event.workflow_run.head_sha" not in checkout["with"]["ref"]
    assert "github.event.pull_request" not in text
    assert "actions/cache" not in text
    assert "cache:" not in text
    assert "save-always" not in text


def test_metadata_fetches_validate_api_path_components_and_fail_closed() -> None:
    data, _ = _load_workflow()
    step = _admission_job(data)["steps"][1]
    script = step["run"]

    assert step["shell"] == "bash"
    assert step["env"] == {
        "GH_TOKEN": "${{ github.token }}",
        "WORKFLOW_FILE": "${{ inputs.workflow_file }}",
    }
    assert script.startswith("set -euo pipefail\n")
    assert '[[ "${GITHUB_EVENT_NAME}" == "workflow_run" ]]' in script

    pr_extract = script.index('pr_number="$(jq -er')
    pr_api = script.index('"repos/${GITHUB_REPOSITORY}/pulls/${pr_number}"')
    assert pr_extract < pr_api
    assert 'select(type == "array" and length == 1)' in script
    assert 'select(type == "number" and . > 0 and . == floor)' in script
    assert 'select(test("^[1-9][0-9]*$"))' in script

    head_extract = script.index('head_sha="$(jq -er')
    status_api = script.index('"repos/${GITHUB_REPOSITORY}/commits/${head_sha}/status?per_page=100"')
    assert pr_api < head_extract < status_api
    assert 'select(type == "string" and test("^[0-9a-f]{40}$"))' in script

    assert (
        'gh api "repos/${GITHUB_REPOSITORY}/actions/workflows/' 'frontend-license-gate.yml" > "${workflow_json}"'
    ) in script
    assert ('gh api "repos/${GITHUB_REPOSITORY}/pulls/${pr_number}" ' '> "${pull_json}"') in script
    assert (
        "gh api --paginate --slurp "
        '"repos/${GITHUB_REPOSITORY}/commits/${head_sha}/status?per_page=100" '
        '> "${combined_status_pages_json}"'
    ) in script

    files_api = (
        "gh api --paginate --slurp "
        '"repos/${GITHUB_REPOSITORY}/pulls/${pr_number}/files?per_page=100" '
        '> "${file_pages_json}"'
    )
    assert f"if {files_api}; then" in script
    assert "files_complete=true" in script
    assert "files_complete=false" in script
    assert "printf '[]\\n' > \"${file_pages_json}\"" in script
    assert script.index(files_api) > status_api
    assert script.count("if gh api") == 1


def test_combined_status_pages_are_validated_and_merged_before_admission() -> None:
    data, _ = _load_workflow()
    script = _admission_job(data)["steps"][1]["run"]
    head_sha = "a" * 40

    assert ('combined_status_pages_json="${RUNNER_TEMP}/' 'license-first-combined-status-pages.json"') in script
    assert (
        "gh api --paginate --slurp "
        '"repos/${GITHUB_REPOSITORY}/commits/${head_sha}/status?per_page=100" '
        '> "${combined_status_pages_json}"'
    ) in script

    filter_prefix = 'jq -e --arg head_sha "${head_sha}" \''
    filter_suffix = '\' "${combined_status_pages_json}" > "${combined_status_json}"'
    filter_start = script.index(filter_prefix) + len(filter_prefix)
    filter_end = script.index(filter_suffix, filter_start)
    merge_filter = script[filter_start:filter_end]
    helper_start = script.index("python3 Helper_Scripts/ci/license_first_admission.py")
    status_fetch = "gh api --paginate --slurp " '"repos/${GITHUB_REPOSITORY}/commits/${head_sha}/status?per_page=100"'
    assert script.index(status_fetch) < filter_start
    assert filter_end < helper_start

    pages = [
        {
            "sha": head_sha,
            "statuses": [
                {
                    "context": "unrelated/check",
                    "state": "success",
                }
            ],
        },
        {
            "sha": head_sha,
            "statuses": [
                {
                    "context": "frontend-license-policy/trusted/main",
                    "state": "success",
                }
            ],
        },
    ]
    merged = subprocess.run(
        ["jq", "-e", "--arg", "head_sha", head_sha, merge_filter],
        input=json.dumps(pages),
        text=True,
        capture_output=True,
        check=True,
    )
    assert json.loads(merged.stdout) == {
        "sha": head_sha,
        "statuses": pages[0]["statuses"] + pages[1]["statuses"],
    }

    invalid_pages = (
        {},
        [],
        [pages[0], {"sha": "b" * 40, "statuses": []}],
        [{"sha": head_sha, "statuses": {}}],
    )
    for invalid in invalid_pages:
        rejected = subprocess.run(
            ["jq", "-e", "--arg", "head_sha", head_sha, merge_filter],
            input=json.dumps(invalid),
            text=True,
            capture_output=True,
            check=False,
        )
        assert rejected.returncode != 0


def test_helper_is_the_only_checked_out_program_and_owns_all_outputs() -> None:
    data, _ = _load_workflow()
    script = _admission_job(data)["steps"][1]["run"]

    response_files = (
        "workflow_json",
        "pull_json",
        "combined_status_pages_json",
        "combined_status_json",
        "file_pages_json",
    )
    for name in response_files:
        assert f'{name}="${{RUNNER_TEMP}}/license-first-{name.removesuffix("_json").replace("_", "-")}.json"' in script

    command = (
        "python3 Helper_Scripts/ci/license_first_admission.py "
        '--event "${GITHUB_EVENT_PATH}" '
        '--workflow "${workflow_json}" '
        '--pull "${pull_json}" '
        '--combined-status "${combined_status_json}" '
        '--file-pages "${file_pages_json}" '
        '--workflow-file "${WORKFLOW_FILE}" '
        '--routes ".github/license-first-paths.json" '
        '--files-complete "${files_complete}" '
        '--github-output "${GITHUB_OUTPUT}"'
    )
    assert command in " ".join(script.split())
    assert script.count("python3 ") == 1
    assert script.count('"${GITHUB_OUTPUT}"') == 1
    assert ">>" not in script
    assert _unquoted_shell_expansions(script) == []
