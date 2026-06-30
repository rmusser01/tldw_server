from pathlib import Path

import yaml


def _load(path: str) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def _workflow_on(workflow: dict) -> dict:
    return workflow[True]


def test_publish_docker_workflow_is_release_driven() -> None:
    workflow = _load(".github/workflows/publish-docker.yml")
    on = _workflow_on(workflow)

    assert "release" in on
    assert on["release"] == {"types": ["published"]}
    assert "workflow_dispatch" in on


def test_publish_docker_matrix_remains_app_worker_audio_worker() -> None:
    workflow = _load(".github/workflows/publish-docker.yml")
    matrix = workflow["jobs"]["push_to_registries"]["strategy"]["matrix"]["include"]

    assert [entry["name"] for entry in matrix] == ["app", "worker", "audio-worker"]


def test_publish_ghcr_main_workflow_remains_push_to_main_driven() -> None:
    workflow = _load(".github/workflows/publish-ghcr-main.yml")
    on = _workflow_on(workflow)

    assert "push" in on
    assert on["push"] == {"branches": ["main"]}
    assert "release" not in on
    assert "workflow_dispatch" not in on


def test_publish_ghcr_main_matrix_remains_app_webui_admin_ui() -> None:
    workflow = _load(".github/workflows/publish-ghcr-main.yml")
    matrix = workflow["jobs"]["publish-ghcr-main"]["strategy"]["matrix"]["include"]

    assert [entry["name"] for entry in matrix] == ["app", "webui", "admin-ui"]
