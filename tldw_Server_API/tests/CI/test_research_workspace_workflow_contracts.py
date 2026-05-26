from pathlib import Path

import yaml


WORKFLOW_PATHS = (
    Path(".github/workflows/ui-research-workspace-parity.yml"),
    Path(".github/workflows/ui-research-workspace-nightly.yml"),
)


def _load(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _frontend_scripts() -> dict:
    return yaml.safe_load(Path("apps/tldw-frontend/package.json").read_text(encoding="utf-8"))[
        "scripts"
    ]


def test_research_workspace_workflows_use_current_names_and_paths() -> None:
    for path in WORKFLOW_PATHS:
        assert path.exists(), f"{path} missing"
        text = path.read_text(encoding="utf-8")
        assert "workspace-playground" not in text
        assert "Workspace Playground" not in text
        assert "WorkspacePlayground" not in text
        assert "research-workspace" in text
        assert "Research Workspace" in text


def test_research_workspace_parity_workflow_calls_existing_package_scripts() -> None:
    workflow = _load(WORKFLOW_PATHS[0])
    scripts = _frontend_scripts()

    webui_steps = workflow["jobs"]["webui-research-workspace-parity"]["steps"]
    webui_run_steps = [
        step for step in webui_steps if step.get("name") == "Run WebUI research workspace parity spec"
    ]
    assert len(webui_run_steps) == 1
    assert webui_run_steps[0]["run"] == "bun run e2e:research-workspace:parity"
    assert "e2e:research-workspace:parity" in scripts


def test_research_workspace_nightly_workflow_calls_existing_package_scripts() -> None:
    workflow = _load(WORKFLOW_PATHS[1])
    scripts = _frontend_scripts()

    webui_steps = workflow["jobs"]["webui-research-workspace-real-backend"]["steps"]
    webui_run_steps = [
        step
        for step in webui_steps
        if step.get("name") == "Run research workspace real-backend WebUI spec"
    ]
    assert len(webui_run_steps) == 1
    assert webui_run_steps[0]["run"] == "bun run e2e:research-workspace:real"
    assert "e2e:research-workspace:real" in scripts
