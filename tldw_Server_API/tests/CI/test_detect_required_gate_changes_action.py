# Local Git fixtures and the repository's comparison script only.
import os
import shutil
import subprocess  # nosec B404
from pathlib import Path

import pytest
import yaml


def test_action_exposes_required_outputs() -> None:
    action_path = Path(".github/actions/detect-required-gate-changes/action.yml")
    data = yaml.safe_load(action_path.read_text(encoding="utf-8"))
    outputs = data["outputs"]

    for output_name in [
        "backend_changed",
        "frontend_changed",
        "tldw_frontend_changed",
        "family_guardrails_changed",
        "admin_ui_changed",
        "e2e_changed",
        "security_relevant_changed",
        "coverage_required",
    ]:
        assert output_name in outputs


def test_action_invokes_gate_emitter_as_module() -> None:
    action_path = Path(".github/actions/detect-required-gate-changes/action.yml")
    data = yaml.safe_load(action_path.read_text(encoding="utf-8"))
    run_script = data["runs"]["steps"][0]["run"]
    assert "python -m Helper_Scripts.ci.emit_ci_gate_flags" in run_script


@pytest.mark.parametrize(
    ("event", "dispatch_base", "expected"),
    [
        ("workflow_dispatch", "explicit", {"frontend.ts", "README.md"}),
        ("workflow_dispatch", "", {"README.md"}),
        ("pull_request", "", {"frontend.ts", "README.md"}),
        ("push", "", {"frontend.ts", "README.md"}),
        ("workflow_dispatch", "--invalid-base", None),
    ],
)
def test_action_compares_the_requested_commit_range(tmp_path, event, dispatch_base, expected):
    action = yaml.safe_load(Path(".github/actions/detect-required-gate-changes/action.yml").read_text())
    step = action["runs"]["steps"][0]
    git_binary = shutil.which("git")
    bash_binary = shutil.which("bash")
    assert git_binary is not None and bash_binary is not None

    def git(*args):
        # Arguments are fixed test commands in a disposable repository.
        return subprocess.check_output(  # nosec B603
            [git_binary, *args], cwd=tmp_path, text=True, stderr=subprocess.DEVNULL, timeout=30
        ).strip()

    git("init")
    git("config", "user.name", "CI comparison fixture")
    git("config", "user.email", "ci-fixture@example.invalid")
    for filename, content in [("README.md", "base"), ("frontend.ts", "change"), ("README.md", "head")]:
        (tmp_path / filename).write_text(content)
        git("add", filename)
        git("commit", "-m", content)
    base = git("rev-parse", "HEAD~2")
    head = git("rev-parse", "HEAD")
    # Execute the real comparison block; mapfile/emitter behavior is covered by
    # the action contracts and is independent of which base commit is selected.
    script = step["run"].split("mapfile -t CHANGED_FILES", 1)[0]
    for expression, value in {
        "github.event_name": event,
        "github.event.pull_request.base.sha": base if event == "pull_request" else "",
        "github.event.before": base if event == "push" else "",
        "github.sha": head,
    }.items():
        script = script.replace("${{ " + expression + " }}", value)
    env = {**os.environ, "RUNNER_TEMP": str(tmp_path)}
    for key, value in step.get("env", {}).items():
        assert value == "${{ github.event.inputs.base_sha }}"
        env[key] = base if dispatch_base == "explicit" else dispatch_base
    # Execute only the checked-in action block with fixed test event metadata.
    result = subprocess.run(  # nosec B603
        [bash_binary, "-c", script],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if expected is None:
        assert result.returncode != 0
    else:
        assert result.returncode == 0, result.stderr
        assert set((tmp_path / "required-gate-changed-files.txt").read_text().splitlines()) == expected
