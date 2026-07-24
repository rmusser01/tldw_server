from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
WORKFLOW_PATH = REPO_ROOT / ".github/workflows/license-first-admission.yml"
CHECKOUT_ACTION = "actions/checkout@de0fac2e4500dabe0009e67214ff5f5447ce83dd"
OUTPUT_NAMES = ("pr_number", "head_sha", "base_sha", "base_ref", "should_run")
EXPECTED_PERMISSIONS = {
    "actions": "read",
    "contents": "read",
    "pull-requests": "read",
    "statuses": "read",
}


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
