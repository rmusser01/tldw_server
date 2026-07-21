from __future__ import annotations

import re
from hashlib import sha256
from pathlib import Path
from typing import Any

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
WORKFLOW_PATH = REPO_ROOT / ".github/workflows/frontend-license-gate.yml"
ACTIONLINT_PATH = REPO_ROOT / ".github/workflows/actionlint.yml"
CHECKOUT_ACTION = "actions/checkout@de0fac2e4500dabe0009e67214ff5f5447ce83dd"
JOB_ID = "frontend-license-gate-audit"
SUPPORTED_BASE_REFS = ("main", "dev")
STATUS_CONTEXT_PREFIX = "frontend-license-policy/trusted"
STATUS_CONTEXT_EXPRESSION = "frontend-license-policy/trusted/${{ github.event.pull_request.base.ref }}"
EXPECTED_STATUS_CONTEXTS = {
    "main": "frontend-license-policy/trusted/main",
    "dev": "frontend-license-policy/trusted/dev",
}
CLASSIFIER = "Helper_Scripts/ci/check_frontend_license_gate.py"
EXPECTED_STEP_IDENTITIES = (
    ("Mark trusted policy pending", None, None),
    ("Checkout trusted policy", None, CHECKOUT_ACTION),
    ("Evaluate immutable pull request metadata", "evaluate", None),
    ("Publish trusted policy result", None, None),
)
TRUSTED_RUN_SHA256 = {
    "Mark trusted policy pending": "1d7e75459f8d398a821dd7987e2a2dcfa73a7a34d0535ef7cd786f93251074f0",
    "Evaluate immutable pull request metadata": "dd667ebf4091850b7d580e9febdec4e77f1e815532984b4f726326868d2126c4",
    "Publish trusted policy result": "314715f909e05c812a2d63ce6eb13afee2b69811809f0cc037d1fc3eb6c4f388",
}
COMMON_METADATA_VALIDATIONS = (
    '[[ "${PR_NUMBER}" =~ ^[1-9][0-9]*$ ]]',
    '[[ "${BASE_SHA}" =~ ^[0-9a-f]{40}$ ]]',
    '[[ "${HEAD_SHA}" =~ ^[0-9a-f]{40}$ ]]',
    '[[ "${BASE_REF}" == main || "${BASE_REF}" == dev ]]',
    '[[ -n "${PR_AUTHOR}" && -n "${REPOSITORY_OWNER}" ]]',
)
EXPECTED_JOB_ENV = {
    "GH_TOKEN": "${{ github.token }}",
    "STATUS_CONTEXT": STATUS_CONTEXT_EXPRESSION,
    "STATUS_REPOSITORY": "${{ github.repository }}",
    "HEAD_SHA": "${{ github.event.pull_request.head.sha }}",
    "BASE_SHA": "${{ github.event.pull_request.base.sha }}",
    "BASE_REF": "${{ github.event.pull_request.base.ref }}",
    "PR_NUMBER": "${{ github.event.pull_request.number }}",
    "PR_AUTHOR": "${{ github.event.pull_request.user.login }}",
    "REPOSITORY_OWNER": "${{ github.repository_owner }}",
}


def load_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(data, dict)
    return data


def assert_exact_step_structure(steps: list[dict[str, Any]]) -> None:
    identities = [(step.get("name"), step.get("id"), step.get("uses")) for step in steps]

    assert identities == list(EXPECTED_STEP_IDENTITIES)
    assert [set(step) for step in steps] == [
        {"name", "shell", "run"},
        {"name", "uses", "with"},
        {"id", "name", "shell", "run"},
        {"name", "if", "shell", "env", "run"},
    ]
    assert [step.get("shell") for step in steps] == ["bash", None, "bash", "bash"]
    assert steps[3]["if"] == "always()"
    assert steps[3]["env"] == {"VERDICT": "${{ steps.evaluate.outputs.verdict }}"}


def assert_trusted_run_bodies(steps: list[dict[str, Any]]) -> None:
    digests = {step["name"]: sha256(step["run"].encode()).hexdigest() for step in steps if "run" in step}

    assert digests == TRUSTED_RUN_SHA256


def assert_branch_specific_status_context(job: dict[str, Any], triggers: dict[str, Any]) -> None:
    base_refs = tuple(triggers["pull_request_target"]["branches"])
    selected_contexts = {base_ref: f"{STATUS_CONTEXT_PREFIX}/{base_ref}" for base_ref in base_refs}

    assert base_refs == SUPPORTED_BASE_REFS
    assert job["env"]["STATUS_CONTEXT"] == STATUS_CONTEXT_EXPRESSION
    assert selected_contexts == EXPECTED_STATUS_CONTEXTS
    assert len(set(selected_contexts.values())) == len(SUPPORTED_BASE_REFS)


def assert_exact_privileged_job_surface(data: dict[str, Any]) -> None:
    trigger_key = "on" if "on" in data else True
    job = data["jobs"][JOB_ID]

    assert set(data) == {"name", trigger_key, "permissions", "concurrency", "jobs"}
    assert data["name"] == "Frontend License Gate Audit"
    assert data["concurrency"] == {
        "group": "frontend-license-gate-${{ github.event.pull_request.number }}",
        "cancel-in-progress": True,
    }
    assert set(job) == {"runs-on", "timeout-minutes", "env", "steps"}
    assert job["runs-on"] == "ubuntu-latest"
    assert job["timeout-minutes"] == 5
    assert job["env"] == EXPECTED_JOB_ENV


def assert_common_validations_precede_owner(script: str) -> None:
    positions = []
    for validation in COMMON_METADATA_VALIDATIONS:
        assert script.count(validation) == 1
        positions.append(script.index(validation))

    owner_branch = script.index('if [[ "${PR_AUTHOR,,}" == "${REPOSITORY_OWNER,,}" ]]; then')
    external_fetch = script.index("git fetch --no-tags")
    external_diff = script.index("git --no-pager diff")
    assert max(positions) < owner_branch < external_fetch < external_diff


def assert_success_follows_prerequisites(script: str) -> None:
    success_positions = [
        match.start(1) for match in re.finditer(r"^[ \t]*(verdict=success)[ \t]*$", script, re.MULTILINE)
    ]
    classifier_positions = [match.start() for match in re.finditer(f"python3 {re.escape(CLASSIFIER)}", script)]

    assert len(success_positions) == 2
    assert len(classifier_positions) == 2
    assert classifier_positions[0] < success_positions[0] < script.index("exit 0")

    prerequisites = (
        '[[ "${fetched_base}" == "${BASE_SHA}" ]]',
        '[[ "${fetched_head}" == "${HEAD_SHA}" ]]',
        "git --no-pager diff",
        f"python3 {CLASSIFIER}",
        'readonly pipeline_status=("${PIPESTATUS[@]}")',
        '[[ "${pipeline_status[0]}" -eq 0 ]]',
        '[[ "${pipeline_status[1]}" -eq 0 ]]',
    )
    positions = [script.index(prerequisite, classifier_positions[0] + 1) for prerequisite in prerequisites]
    assert positions == sorted(positions)
    assert positions[-1] < success_positions[1]
    assert success_positions[1] == script.rfind("verdict=success")


def test_workflow_uses_the_base_controlled_trigger_and_minimum_permissions() -> None:
    data = load_yaml(WORKFLOW_PATH)
    triggers = data.get("on", data.get(True))

    assert set(triggers) == {"pull_request_target"}
    assert triggers["pull_request_target"]["branches"] == ["main", "dev"]
    assert triggers["pull_request_target"]["types"] == [
        "opened",
        "reopened",
        "synchronize",
        "ready_for_review",
        "edited",
    ]
    assert data["permissions"] == {"contents": "read", "statuses": "write"}
    assert set(data["jobs"]) == {JOB_ID}


def test_workflow_locks_the_complete_privileged_job_surface() -> None:
    assert_exact_privileged_job_surface(load_yaml(WORKFLOW_PATH))


def test_supported_bases_select_distinct_contexts_and_shared_context_is_rejected() -> None:
    data = load_yaml(WORKFLOW_PATH)
    triggers = data.get("on", data.get(True))
    job = data["jobs"][JOB_ID]

    assert_branch_specific_status_context(job, triggers)

    shared_context_job = {
        **job,
        "env": {**job["env"], "STATUS_CONTEXT": STATUS_CONTEXT_PREFIX},
    }
    with pytest.raises(AssertionError):
        assert_branch_specific_status_context(shared_context_job, triggers)


def test_workflow_checks_out_only_the_trusted_base_revision() -> None:
    job = load_yaml(WORKFLOW_PATH)["jobs"][JOB_ID]
    steps = job["steps"]
    action_steps = [step for step in steps if "uses" in step]

    assert_exact_step_structure(steps)
    assert_trusted_run_bodies(steps)
    assert action_steps == [
        {
            "name": "Checkout trusted policy",
            "uses": CHECKOUT_ACTION,
            "with": {
                "ref": "${{ github.sha }}",
                "fetch-depth": 0,
                "persist-credentials": False,
            },
        }
    ]


def test_step_contract_rejects_an_appended_privileged_run_step() -> None:
    steps = load_yaml(WORKFLOW_PATH)["jobs"][JOB_ID]["steps"]
    unsafe_step = {"name": "Execute untrusted head", "shell": "bash", "run": "./pr-head/payload"}

    with pytest.raises(AssertionError):
        assert_exact_step_structure([*steps, unsafe_step])


def test_step_contract_rejects_a_custom_evaluator_shell() -> None:
    steps = load_yaml(WORKFLOW_PATH)["jobs"][JOB_ID]["steps"]
    evaluate_index = next(index for index, step in enumerate(steps) if step.get("id") == "evaluate")
    mutated_steps = [*steps]
    mutated_steps[evaluate_index] = {**steps[evaluate_index], "shell": "./pr-head/shell {0}"}

    assert [(step.get("name"), step.get("id"), step.get("uses")) for step in mutated_steps] == list(
        EXPECTED_STEP_IDENTITIES
    )
    assert_trusted_run_bodies(mutated_steps)
    with pytest.raises(AssertionError):
        assert_exact_step_structure(mutated_steps)


def test_run_body_contract_rejects_an_unsafe_evaluator_command() -> None:
    steps = load_yaml(WORKFLOW_PATH)["jobs"][JOB_ID]["steps"]
    evaluate_index = next(index for index, step in enumerate(steps) if step.get("id") == "evaluate")
    mutated_steps = [*steps]
    evaluator = {**steps[evaluate_index], "run": f'{steps[evaluate_index]["run"]}\n./pr-head/payload\n'}
    mutated_steps[evaluate_index] = evaluator

    assert_exact_step_structure(mutated_steps)
    assert_success_follows_prerequisites(evaluator["run"])
    with pytest.raises(AssertionError):
        assert_trusted_run_bodies(mutated_steps)


def test_workflow_posts_pending_before_a_fail_closed_evaluation() -> None:
    job = load_yaml(WORKFLOW_PATH)["jobs"][JOB_ID]
    steps = job["steps"]
    pending_index = next(index for index, step in enumerate(steps) if step.get("name") == "Mark trusted policy pending")
    checkout_index = next(index for index, step in enumerate(steps) if step.get("name") == "Checkout trusted policy")
    evaluate_index = next(index for index, step in enumerate(steps) if step.get("id") == "evaluate")
    pending_script = steps[pending_index]["run"]

    assert pending_index < checkout_index < evaluate_index
    assert '"repos/${STATUS_REPOSITORY}/statuses/${HEAD_SHA}"' in pending_script
    assert "-f state=pending" in pending_script
    assert '-f context="${STATUS_CONTEXT}"' in pending_script
    assert "Trusted frontend license policy is evaluating this exact commit" in pending_script
    assert job["env"]["STATUS_CONTEXT"] == STATUS_CONTEXT_EXPRESSION
    assert job["env"]["HEAD_SHA"] == "${{ github.event.pull_request.head.sha }}"
    assert JOB_ID not in EXPECTED_STATUS_CONTEXTS.values()

    evaluate_script = steps[evaluate_index]["run"]
    normalized = " ".join(evaluate_script.replace("\\\n", " ").split())
    assert "verdict=failure" in evaluate_script
    assert "trap emit_verdict EXIT" in evaluate_script
    assert '[[ "${PR_NUMBER}" =~ ^[1-9][0-9]*$ ]]' in evaluate_script
    assert '[[ "${BASE_SHA}" =~ ^[0-9a-f]{40}$ ]]' in evaluate_script
    assert '[[ "${HEAD_SHA}" =~ ^[0-9a-f]{40}$ ]]' in evaluate_script
    assert '[[ "${BASE_REF}" == main || "${BASE_REF}" == dev ]]' in evaluate_script
    assert (
        'git fetch --no-tags --depth=1 "${public_remote}" ' '"+refs/heads/${BASE_REF}:refs/remotes/license-gate/base"'
    ) in normalized
    assert (
        'git fetch --no-tags --depth=1 "${public_remote}" '
        '"+refs/pull/${PR_NUMBER}/head:refs/remotes/license-gate/pr-head"'
    ) in normalized
    assert '[[ "${fetched_base}" == "${BASE_SHA}" ]]' in evaluate_script
    assert '[[ "${fetched_head}" == "${HEAD_SHA}" ]]' in evaluate_script

    diff_match = re.search(r"git --no-pager diff (?P<arguments>.*?) \| python3", normalized)
    assert diff_match is not None
    diff_arguments = diff_match.group("arguments")
    for argument in ("--name-only", "-z", "--no-renames", "--no-ext-diff", "--no-textconv"):
        assert argument in diff_arguments.split()
    assert diff_arguments.endswith('"${BASE_SHA}" "${HEAD_SHA}" --')
    assert normalized.count(f"python3 {CLASSIFIER}") == 2
    assert normalized.count("--null") == 2
    assert_common_validations_precede_owner(evaluate_script)
    assert_success_follows_prerequisites(evaluate_script)


def test_evaluator_contract_rejects_external_success_before_sha_checks() -> None:
    steps = load_yaml(WORKFLOW_PATH)["jobs"][JOB_ID]["steps"]
    script = next(step for step in steps if step.get("id") == "evaluate")["run"]
    external_success = script.rfind("verdict=success")
    assert external_success > 0
    mutated = script[:external_success] + script[external_success + len("verdict=success") :]
    first_sha_check = '[[ "${fetched_base}" == "${BASE_SHA}" ]]'
    mutated = mutated.replace(first_sha_check, f"verdict=success\n{first_sha_check}", 1)

    with pytest.raises(AssertionError):
        assert_success_follows_prerequisites(mutated)


def test_evaluator_contract_rejects_common_validation_after_owner_branch() -> None:
    steps = load_yaml(WORKFLOW_PATH)["jobs"][JOB_ID]["steps"]
    script = next(step for step in steps if step.get("id") == "evaluate")["run"]
    validation = COMMON_METADATA_VALIDATIONS[0]
    mutated = script.replace(f"{validation}\n", "", 1)
    external_anchor = "fi\n\nreadonly public_remote="
    mutated = mutated.replace(external_anchor, f"fi\n\n{validation}\n\nreadonly public_remote=", 1)

    assert mutated.count(validation) == 1
    assert_success_follows_prerequisites(mutated)
    with pytest.raises(AssertionError):
        assert_common_validations_precede_owner(mutated)


def test_workflow_publishes_success_only_for_an_explicit_success_verdict() -> None:
    steps = load_yaml(WORKFLOW_PATH)["jobs"][JOB_ID]["steps"]
    publisher = next(step for step in steps if step.get("name") == "Publish trusted policy result")
    script = publisher["run"]

    assert publisher["if"] == "always()"
    assert publisher["env"] == {"VERDICT": "${{ steps.evaluate.outputs.verdict }}"}
    assert "state=failure" in script
    assert script.count("state=success") == 1
    assert re.search(r'if \[\[ "\$\{VERDICT\}" == success \]\]; then\s+state=success', script)
    assert "Trusted frontend license policy authorized this exact commit" in script
    assert '"repos/${STATUS_REPOSITORY}/statuses/${HEAD_SHA}"' in script
    assert '-f context="${STATUS_CONTEXT}"' in script
    assert '[[ "${state}" == success ]]' in script


def test_actionlint_targets_the_trusted_workflow() -> None:
    steps = load_yaml(ACTIONLINT_PATH)["jobs"]["actionlint"]["steps"]
    invocation = next(step for step in steps if step.get("name") == "Run actionlint on targeted workflows")["run"]

    assert ".github/workflows/frontend-license-gate.yml" in invocation.split()
