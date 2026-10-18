import json
from pathlib import Path


MANIFEST_PATH = Path("Docs/Deployment/sidecar_workers_manifest.json")


def test_sidecar_manifest_exists_and_has_expected_shape() -> None:
    assert MANIFEST_PATH.exists(), "Expected default sidecar workers manifest to exist"

    data = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    default_workers = data.get("default_workers")
    workers = data.get("workers")

    assert isinstance(default_workers, list) and default_workers
    assert isinstance(workers, list) and workers

    by_key = {}
    for worker in workers:
        assert isinstance(worker, dict)
        assert isinstance(worker.get("key"), str) and worker["key"]
        assert isinstance(worker.get("slug"), str) and worker["slug"]
        assert isinstance(worker.get("label"), str) and worker["label"]
        assert isinstance(worker.get("module"), str) and worker["module"]
        by_key[worker["key"]] = worker

    for key in default_workers:
        assert key in by_key, f"default worker '{key}' must be present in workers"


def test_sidecar_manifest_runs_the_long_form_tts_worker_by_default() -> None:
    # In sidecar mode the in-process TTS worker is off, so the sidecar must run it (TASK-13849).
    data = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    assert "tts" in data["default_workers"]
    by_key = {worker["key"]: worker for worker in data["workers"]}
    assert by_key["tts"]["module"] == "tldw_Server_API.app.core.TTS.tts_jobs_worker"
