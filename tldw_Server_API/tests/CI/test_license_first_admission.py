from __future__ import annotations

import ast
import copy
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = REPO_ROOT / "Helper_Scripts/ci/license_first_admission.py"
ROUTES_PATH = REPO_ROOT / ".github/license-first-paths.json"
WORKFLOWS_PATH = REPO_ROOT / ".github/workflows"

SPEC = importlib.util.spec_from_file_location("license_first_admission", SCRIPT_PATH)
assert SPEC is not None
assert SPEC.loader is not None
admission = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(admission)

HEAD_SHA = "1" * 40
AUDITED_BASE_SHA = "2" * 40
CURRENT_BASE_SHA = "3" * 40
RUN_STARTED_AT = "2026-07-24T12:00:00Z"
STATUS_CREATED_AT = "2026-07-24T12:00:01Z"
WORKFLOW_NAME = "Frontend License Gate Audit"
WORKFLOW_PATH = ".github/workflows/frontend-license-gate.yml"

FILTERED_WORKFLOWS = {
    "actionlint.yml",
    "frontend-e2e-tiers.yml",
    "jobs-suite.yml",
    "mcp-unified-rc.yml",
    "notes-remediation-targeted.yml",
    "onboarding-docs-gate.yml",
    "pypi-package.yml",
    "ui-characters-harness-tests.yml",
    "ui-dictionaries-tests.yml",
    "ui-playground-quality-gates.yml",
    "ui-research-workspace-parity.yml",
    "ui-watchlists-a11y-gates.yml",
    "ui-watchlists-extension-e2e.yml",
    "ui-watchlists-help-tests.yml",
    "ui-watchlists-scale-gates.yml",
    "ui-worldbooks-tests.yml",
}


def valid_inputs(
    *,
    workflow_file: str = "ci.yml",
    routes: dict | None = None,
    file_pages: list[list[dict]] | None = None,
) -> dict:
    event = {
        "action": "completed",
        "workflow_run": {
            "name": WORKFLOW_NAME,
            "path": WORKFLOW_PATH,
            "workflow_id": 987,
            "event": "pull_request_target",
            "conclusion": "success",
            "run_started_at": RUN_STARTED_AT,
            "pull_requests": [
                {
                    "number": 17,
                    "head": {
                        "sha": HEAD_SHA,
                        "repo": {"id": 101},
                    },
                    "base": {
                        "sha": AUDITED_BASE_SHA,
                        "ref": "main",
                        "repo": {"id": 202},
                    },
                }
            ],
        },
    }
    workflow = {
        "id": 987,
        "name": WORKFLOW_NAME,
        "path": WORKFLOW_PATH,
    }
    pull = {
        "number": 17,
        "state": "open",
        "head": {
            "sha": HEAD_SHA,
            "repo": {"id": 101},
        },
        "base": {
            "sha": AUDITED_BASE_SHA,
            "ref": "main",
            "repo": {"id": 202},
        },
    }
    combined_status = {
        "sha": HEAD_SHA,
        "statuses": [
            {
                "context": "frontend-license-policy/trusted/main",
                "state": "success",
                "created_at": STATUS_CREATED_AT,
            }
        ],
    }
    return {
        "event": event,
        "workflow": workflow,
        "pull": pull,
        "combined_status": combined_status,
        "file_pages": ([[{"filename": "README.md", "status": "modified"}]] if file_pages is None else file_pages),
        "workflow_file": workflow_file,
        "routes": {} if routes is None else routes,
        "files_complete": True,
    }


def set_nested(value: dict, path: tuple[object, ...], replacement: object) -> None:
    target: object = value
    for part in path[:-1]:
        target = target[part]  # type: ignore[index]
    target[path[-1]] = replacement  # type: ignore[index]


def route(mode: str, *patterns: str) -> dict:
    return {"filtered.yml": {"mode": mode, "patterns": list(patterns)}}


def decision(
    mode: str,
    patterns: tuple[str, ...],
    *files: dict,
    files_complete: bool = True,
) -> str:
    inputs = valid_inputs(
        workflow_file="filtered.yml",
        routes=route(mode, *patterns),
        file_pages=[list(files)],
    )
    inputs["files_complete"] = files_complete
    return admission.admit(**inputs)["should_run"]


def test_valid_trusted_run_returns_only_validated_outputs() -> None:
    assert admission.admit(**valid_inputs()) == {
        "pr_number": "17",
        "head_sha": HEAD_SHA,
        "base_sha": AUDITED_BASE_SHA,
        "base_ref": "main",
        "should_run": "true",
    }


@pytest.mark.parametrize(
    ("path", "replacement"),
    [
        (("event", "action"), "requested"),
        (("event", "workflow_run", "name"), f"{WORKFLOW_NAME} "),
        (("event", "workflow_run", "path"), f"{WORKFLOW_PATH}.bak"),
        (("event", "workflow_run", "workflow_id"), 988),
        (("workflow", "id"), 988),
        (("workflow", "name"), "Lookalike License Gate"),
        (("workflow", "path"), f"{WORKFLOW_PATH}.bak"),
        (("event", "workflow_run", "event"), "pull_request"),
        (("event", "workflow_run", "conclusion"), "failure"),
        (("event", "workflow_run", "pull_requests"), []),
        (("event", "workflow_run", "pull_requests", 0, "number"), 0),
        (("event", "workflow_run", "pull_requests", 0, "number"), True),
        (("event", "workflow_run", "pull_requests", 0, "head", "sha"), "short"),
        (("event", "workflow_run", "pull_requests", 0, "base", "sha"), "short"),
        (("event", "workflow_run", "pull_requests", 0, "head", "repo", "id"), 0),
        (("event", "workflow_run", "pull_requests", 0, "base", "repo", "id"), -1),
        (("event", "workflow_run", "pull_requests", 0, "base", "ref"), "feature"),
        (("event", "workflow_run", "pull_requests", 0, "base", "ref"), ["main"]),
        (("pull", "number"), 18),
        (("pull", "state"), "closed"),
        (("pull", "head", "sha"), "4" * 40),
        (("pull", "head", "repo", "id"), 303),
        (("pull", "base", "sha"), "short"),
        (("pull", "base", "ref"), "dev"),
        (("pull", "base", "repo", "id"), 303),
        (("combined_status", "sha"), "4" * 40),
        (("combined_status", "statuses", 0, "context"), "frontend-license-policy/trusted/dev"),
        (("combined_status", "statuses", 0, "state"), "pending"),
        (("combined_status", "statuses", 0, "created_at"), "2026-07-24T11:59:59Z"),
        (("event", "workflow_run", "run_started_at"), "not-a-time"),
    ],
)
def test_invalid_or_stale_trust_metadata_fails_closed(
    path: tuple[object, ...],
    replacement: object,
) -> None:
    inputs = valid_inputs()
    set_nested(inputs, path, replacement)

    with pytest.raises(admission.AdmissionError):
        admission.admit(**inputs)


def test_missing_action_fails_closed() -> None:
    inputs = valid_inputs()
    del inputs["event"]["action"]

    with pytest.raises(admission.AdmissionError):
        admission.admit(**inputs)


def test_null_action_fails_closed() -> None:
    inputs = valid_inputs()
    inputs["event"]["action"] = None

    with pytest.raises(admission.AdmissionError):
        admission.admit(**inputs)


def test_exactly_one_associated_pull_request_is_required() -> None:
    inputs = valid_inputs()
    pull_request = copy.deepcopy(inputs["event"]["workflow_run"]["pull_requests"][0])
    inputs["event"]["workflow_run"]["pull_requests"].append(pull_request)

    with pytest.raises(admission.AdmissionError):
        admission.admit(**inputs)


def test_dev_uses_the_dev_qualified_status_context() -> None:
    inputs = valid_inputs()
    inputs["event"]["workflow_run"]["pull_requests"][0]["base"]["ref"] = "dev"
    inputs["pull"]["base"]["ref"] = "dev"
    inputs["combined_status"]["statuses"][0]["context"] = "frontend-license-policy/trusted/dev"

    assert admission.admit(**inputs)["base_ref"] == "dev"


def test_duplicate_exact_status_context_is_ambiguous() -> None:
    inputs = valid_inputs()
    inputs["combined_status"]["statuses"].append(copy.deepcopy(inputs["combined_status"]["statuses"][0]))

    with pytest.raises(admission.AdmissionError):
        admission.admit(**inputs)


@pytest.mark.parametrize(
    ("pattern", "filename", "expected"),
    [
        ("README.md", "README.md", "true"),
        ("README.md", "README.rst", "false"),
        ("src/*.py", "src/main.py", "true"),
        ("src/*.py", "src/pkg/main.py", "false"),
        ("**/*.py", "main.py", "true"),
        ("**/*.py", "src/pkg/main.py", "true"),
        ("src/file?.py", "src/fil.py", "true"),
        ("src/file?.py", "src/file.py", "true"),
        ("src/file?.py", "src/filex.py", "false"),
        ("src/[ab].py", "src/a.py", "true"),
        ("src/[!a-c].py", "src/z.py", "true"),
        ("src/[!a-c].py", "src/b.py", "false"),
    ],
)
def test_paths_support_github_style_globs(
    pattern: str,
    filename: str,
    expected: str,
) -> None:
    assert decision("paths", (pattern,), {"filename": filename, "status": "modified"}) == expected


@pytest.mark.parametrize(
    ("filename", "expected"),
    [
        ("src/main.py", "true"),
        ("src/generated/other.py", "false"),
        ("src/generated/keep.py", "true"),
    ],
)
def test_paths_apply_ordered_exclusion_and_reinclusion(
    filename: str,
    expected: str,
) -> None:
    assert (
        decision(
            "paths",
            ("src/**", "!src/generated/**", "src/generated/keep.py"),
            {"filename": filename, "status": "modified"},
        )
        == expected
    )


@pytest.mark.parametrize(
    ("patterns", "filename", "expected"),
    [
        (("docs/**",), "docs/guide.md", "false"),
        (("docs/**",), "src/main.py", "true"),
        (("docs/**", "!docs/required.md"), "docs/required.md", "true"),
    ],
)
def test_paths_ignore_runs_when_any_path_is_not_ignored(
    patterns: tuple[str, ...],
    filename: str,
    expected: str,
) -> None:
    assert (
        decision(
            "paths-ignore",
            patterns,
            {"filename": filename, "status": "modified"},
        )
        == expected
    )


@pytest.mark.parametrize(
    "file",
    [
        {
            "filename": "new/location.py",
            "previous_filename": "src/matched.py",
            "status": "renamed",
        },
        {
            "filename": "src/matched.py",
            "previous_filename": "old/location.py",
            "status": "renamed",
        },
        {"filename": "src/matched.py", "status": "removed"},
    ],
)
def test_renames_and_deletions_route_on_every_affected_path(file: dict) -> None:
    assert decision("paths", ("src/**",), file) == "true"


def test_all_file_pages_are_examined() -> None:
    inputs = valid_inputs(
        workflow_file="filtered.yml",
        routes=route("paths", "src/**"),
        file_pages=[
            [{"filename": "docs/guide.md", "status": "modified"}],
            [{"filename": "src/main.py", "status": "added"}],
        ],
    )

    assert admission.admit(**inputs)["should_run"] == "true"


@pytest.mark.parametrize("mode", ["paths", "paths-ignore"])
def test_empty_complete_diff_skips_path_filtered_workflow(mode: str) -> None:
    assert decision(mode, ("src/**",), files_complete=True) == "false"


@pytest.mark.parametrize(
    ("patterns", "file_pages", "files_complete"),
    [
        (("src/**",), [[{"filename": "docs/guide.md", "status": "modified"}]], False),
        (("src/{one,two}.py",), [[{"filename": "docs/guide.md", "status": "modified"}]], True),
        (("src/file+.py",), [[{"filename": "docs/guide.md", "status": "modified"}]], True),
        (("src/[abc.py",), [[{"filename": "docs/guide.md", "status": "modified"}]], True),
        (("src/**",), [[{"status": "modified"}]], True),
        (("src/**",), [[{"filename": "new.py", "status": "renamed"}]], True),
        (("src/**",), [{"filename": "docs/guide.md", "status": "modified"}], True),
    ],
)
def test_uncertain_path_data_always_runs(
    patterns: tuple[str, ...],
    file_pages: list,
    files_complete: bool,
) -> None:
    inputs = valid_inputs(
        workflow_file="filtered.yml",
        routes=route("paths", *patterns),
        file_pages=file_pages,
    )
    inputs["files_complete"] = files_complete

    assert admission.admit(**inputs)["should_run"] == "true"


def test_more_than_300_files_runs_conservatively() -> None:
    files = [{"filename": f"docs/file-{index}.md", "status": "modified"} for index in range(301)]

    assert decision("paths", ("src/**",), *files) == "true"


def test_exactly_300_complete_files_can_be_skipped() -> None:
    files = [{"filename": f"docs/file-{index}.md", "status": "modified"} for index in range(300)]

    assert decision("paths", ("src/**",), *files) == "false"


def test_current_base_drift_preserves_audited_base_and_disables_skipping() -> None:
    inputs = valid_inputs(
        workflow_file="filtered.yml",
        routes=route("paths", "src/**"),
        file_pages=[[{"filename": "docs/guide.md", "status": "modified"}]],
    )
    inputs["pull"]["base"]["sha"] = CURRENT_BASE_SHA

    assert admission.admit(**inputs) == {
        "pr_number": "17",
        "head_sha": HEAD_SHA,
        "base_sha": AUDITED_BASE_SHA,
        "base_ref": "main",
        "should_run": "true",
    }


@pytest.mark.parametrize(
    "routes",
    [
        {},
        {"filtered.yml": {"mode": "unknown", "patterns": ["src/**"]}},
        {"filtered.yml": {"mode": "paths", "patterns": "src/**"}},
    ],
)
def test_unfiltered_or_invalid_route_runs_conservatively(routes: dict) -> None:
    inputs = valid_inputs(
        workflow_file="filtered.yml",
        routes=routes,
        file_pages=[[{"filename": "docs/guide.md", "status": "modified"}]],
    )

    assert admission.admit(**inputs)["should_run"] == "true"


def load_workflow_trigger(filename: str) -> dict:
    workflow = yaml.load(
        (WORKFLOWS_PATH / filename).read_text(encoding="utf-8"),
        Loader=yaml.BaseLoader,
    )
    return workflow["on"]["pull_request"]


def test_route_manifest_exactly_copies_current_pull_request_filters() -> None:
    routes = json.loads(ROUTES_PATH.read_text(encoding="utf-8"))
    expected = {}
    for filename in sorted(FILTERED_WORKFLOWS):
        trigger = load_workflow_trigger(filename)
        modes = [mode for mode in ("paths", "paths-ignore") if mode in trigger]
        assert len(modes) == 1
        mode = modes[0]
        expected[filename] = {
            "mode": mode,
            "patterns": trigger[mode],
        }

    assert routes == expected
    assert "mcp-unified-publish.yml" not in routes
    assert "publish-pypi.yml" not in routes


def test_pypi_lock_change_is_admitted_for_package_validation() -> None:
    """A locked dependency change must trigger the actual package check."""
    inputs = valid_inputs(
        workflow_file="pypi-package.yml",
        routes=json.loads(ROUTES_PATH.read_text(encoding="utf-8")),
        file_pages=[[{"filename": "uv.lock", "status": "modified"}]],
    )
    assert admission.admit(**inputs)["should_run"] == "true"


@pytest.mark.parametrize("filename", [
    "apps/extension/tests/e2e/prompt-improvement.spec.ts",
    "apps/packages/ui/src/store/model.tsx",
    "apps/packages/ui/src/components/Chat/composer/ChatComposer.tsx",
    "apps/packages/ui/src/assets/locale/en/chat.json",
])
def test_watchlists_extension_routes_shared_journey_changes(filename: str) -> None:
    """Shared journey changes must reach the audited extension E2E workflow."""
    inputs = valid_inputs(
        workflow_file="ui-watchlists-extension-e2e.yml",
        routes=json.loads(ROUTES_PATH.read_text(encoding="utf-8")),
        file_pages=[[{"filename": filename, "status": "modified"}]],
    )
    assert admission.admit(**inputs)["should_run"] == "true"


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def test_cli_reads_json_inputs_and_writes_only_github_outputs(tmp_path: Path) -> None:
    inputs = valid_inputs()
    arguments = []
    for option, key in (
        ("--event", "event"),
        ("--workflow", "workflow"),
        ("--pull", "pull"),
        ("--combined-status", "combined_status"),
        ("--file-pages", "file_pages"),
        ("--routes", "routes"),
    ):
        path = tmp_path / f"{key}.json"
        write_json(path, inputs[key])
        arguments.extend((option, str(path)))
    output_path = tmp_path / "github-output"

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            *arguments,
            "--workflow-file",
            inputs["workflow_file"],
            "--files-complete",
            "true",
            "--github-output",
            str(output_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert result.stdout == ""
    assert result.stderr == ""
    assert output_path.read_text(encoding="utf-8").splitlines() == [
        "pr_number=17",
        f"head_sha={HEAD_SHA}",
        f"base_sha={AUDITED_BASE_SHA}",
        "base_ref=main",
        "should_run=true",
    ]


def test_helper_imports_only_the_approved_standard_library_modules() -> None:
    tree = ast.parse(SCRIPT_PATH.read_text(encoding="utf-8"))
    imported = {
        alias.name.split(".", 1)[0] for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names
    }
    imported.update(
        node.module.split(".", 1)[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module != "__future__"
    )

    assert imported <= {"argparse", "datetime", "json", "pathlib", "re"}
