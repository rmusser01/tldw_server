from __future__ import annotations

import json
from pathlib import Path

import pytest

from tldw_Server_API.app.core.Sandbox.image_store import (
    ImageStoreValidationError,
    SandboxImageStore,
)


def _manifest_payload(*, runtime: str, template_name: str, disk_path: str = "/tmp/rootfs.img") -> dict[str, object]:
    return {
        "schema_version": 1,
        "template_id": f"{runtime}:{template_name}",
        "runtime": runtime,
        "template_name": template_name,
        "source_path": None,
        "registered_at": "2026-04-27T00:00:00+00:00",
        "disk_paths": [disk_path],
        "artifacts": [
            {
                "name": Path(disk_path).name,
                "path": disk_path,
                "size_bytes": 4,
                "sha256": "0" * 64,
            }
        ],
        "labels": {},
        "provenance": {},
    }


def test_sandbox_image_store_registers_bundle_reloads_and_plans_gc(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "kernel").write_bytes(b"kernel")
    (bundle / "rootfs.img").write_bytes(b"rootfs")
    (bundle / "manifest.json").write_text(
        json.dumps({"bundle_version": "1", "kernel": "kernel", "rootfs": "rootfs.img"}),
        encoding="utf-8",
    )
    (bundle / "build-info.json").write_text(
        json.dumps({"suite": "bookworm", "architecture": "arm64"}),
        encoding="utf-8",
    )
    run_drop = tmp_path / "store" / "runs" / "run-drop"
    run_drop.mkdir(parents=True)
    (run_drop / "rootfs.img").write_bytes(b"clone")
    store = SandboxImageStore(root_path=tmp_path / "store")

    template_id = store.register_bundle(
        runtime="vz_linux",
        template_name="debian-bookworm-arm64",
        bundle_path=bundle,
        labels={"profile": "minimal"},
    )
    reloaded = SandboxImageStore(root_path=tmp_path / "store")
    record = reloaded.get_template(template_id)
    clone_manifest = reloaded.prepare_run_clone(template_id=template_id, run_id="run-123")
    gc_plan = reloaded.plan_garbage_collection(active_run_ids=set())

    assert record is not None
    assert record.labels == {"profile": "minimal"}
    assert record.provenance == {"suite": "bookworm", "architecture": "arm64"}
    assert [template.template_id for template in reloaded.list_templates(runtime="vz_linux")] == [template_id]
    assert any(item.target_path.endswith("run-123/rootfs.img") for item in clone_manifest.clone_items)
    assert (tmp_path / "store" / "runs" / "run-123" / "manifest.json").exists()
    persisted_manifest = reloaded.get_run_clone_manifest("run-123")
    assert persisted_manifest is not None
    assert persisted_manifest.template_id == template_id
    assert persisted_manifest.run_id == "run-123"
    assert [candidate.run_id for candidate in gc_plan.run_candidates] == [
        "run-123",
        "run-drop",
    ]


def test_sandbox_image_store_rejects_manifest_path_mismatch(tmp_path: Path) -> None:
    artifact = tmp_path / "rootfs.img"
    artifact.write_bytes(b"root")
    manifest_path = tmp_path / "store" / "templates" / "vz_linux" / "actual" / "manifest.json"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(
        json.dumps(_manifest_payload(runtime="vz_linux", template_name="declared", disk_path=str(artifact))),
        encoding="utf-8",
    )

    with pytest.raises(ImageStoreValidationError, match="manifest_path_mismatch"):
        SandboxImageStore(root_path=tmp_path / "store")


def test_sandbox_image_store_rejects_manifest_id_overwrite_vector_on_reload(tmp_path: Path) -> None:
    store_root = tmp_path / "store"
    first_artifact = tmp_path / "first.img"
    second_artifact = tmp_path / "second.img"
    first_artifact.write_bytes(b"one")
    second_artifact.write_bytes(b"two")
    first_manifest = store_root / "templates" / "vz_linux" / "first" / "manifest.json"
    second_manifest = store_root / "templates" / "vz_linux" / "second" / "manifest.json"
    first_manifest.parent.mkdir(parents=True)
    second_manifest.parent.mkdir(parents=True)
    first_manifest.write_text(
        json.dumps(_manifest_payload(runtime="vz_linux", template_name="first", disk_path=str(first_artifact))),
        encoding="utf-8",
    )
    second_payload = _manifest_payload(runtime="vz_linux", template_name="second", disk_path=str(second_artifact))
    second_payload["template_id"] = "vz_linux:first"
    second_manifest.write_text(json.dumps(second_payload), encoding="utf-8")

    with pytest.raises(ImageStoreValidationError, match="manifest_path_mismatch"):
        SandboxImageStore(root_path=store_root)


def test_sandbox_image_store_rejects_non_object_manifest_payload(tmp_path: Path) -> None:
    manifest_path = tmp_path / "store" / "templates" / "vz_linux" / "bad" / "manifest.json"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text("[]", encoding="utf-8")

    with pytest.raises(ImageStoreValidationError, match="manifest_expected_object"):
        SandboxImageStore(root_path=tmp_path / "store")


def test_sandbox_image_store_rejects_missing_manifest_artifact_path(tmp_path: Path) -> None:
    manifest_path = tmp_path / "store" / "templates" / "vz_linux" / "bad" / "manifest.json"
    manifest_path.parent.mkdir(parents=True)
    missing_artifact = tmp_path / "missing-rootfs.img"
    manifest_path.write_text(
        json.dumps(_manifest_payload(runtime="vz_linux", template_name="bad", disk_path=str(missing_artifact))),
        encoding="utf-8",
    )

    with pytest.raises(ImageStoreValidationError, match="manifest_artifact_missing"):
        SandboxImageStore(root_path=tmp_path / "store")


def test_sandbox_image_store_rejects_manifest_disk_paths_mismatch(tmp_path: Path) -> None:
    artifact = tmp_path / "rootfs.img"
    artifact.write_bytes(b"root")
    manifest_path = tmp_path / "store" / "templates" / "vz_linux" / "bad" / "manifest.json"
    manifest_path.parent.mkdir(parents=True)
    payload = _manifest_payload(runtime="vz_linux", template_name="bad", disk_path=str(artifact))
    payload["disk_paths"] = [str(tmp_path / "other-rootfs.img")]
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ImageStoreValidationError, match="manifest_disk_paths_mismatch"):
        SandboxImageStore(root_path=tmp_path / "store")


def test_sandbox_image_store_gc_plan_ignores_files_deleted_during_size_scan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_drop = tmp_path / "store" / "runs" / "run-drop"
    run_drop.mkdir(parents=True)
    vanished = run_drop / "vanished.txt"
    vanished.write_text("gone", encoding="utf-8")
    original_stat = Path.stat
    calls = 0

    def flaky_stat(path: Path, *args, **kwargs):
        nonlocal calls
        if path == vanished:
            calls += 1
            if calls > 1:
                raise FileNotFoundError(path)
        return original_stat(path, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", flaky_stat)

    plan = SandboxImageStore(root_path=tmp_path / "store").plan_garbage_collection(active_run_ids=set())

    assert plan.run_candidates[0].run_id == "run-drop"
    assert plan.run_candidates[0].size_bytes == 0


def test_sandbox_image_store_lists_persisted_run_clone_manifests(tmp_path: Path) -> None:
    disk = tmp_path / "rootfs.img"
    disk.write_bytes(b"rootfs")
    store = SandboxImageStore(root_path=tmp_path / "store")
    template_id = store.register_template(
        runtime="vz_linux",
        template_name="debian-bookworm-arm64",
        disk_paths=[str(disk)],
    )

    store.prepare_run_clone(template_id=template_id, run_id="run-b")
    store.prepare_run_clone(template_id=template_id, run_id="run-a")

    reloaded = SandboxImageStore(root_path=tmp_path / "store")
    assert [manifest.run_id for manifest in reloaded.list_run_clone_manifests()] == [
        "run-a",
        "run-b",
    ]


def test_sandbox_image_store_rejects_invalid_run_id_for_clone_manifest(tmp_path: Path) -> None:
    disk = tmp_path / "rootfs.img"
    disk.write_bytes(b"rootfs")
    store = SandboxImageStore(root_path=tmp_path / "store")
    template_id = store.register_template(
        runtime="vz_linux",
        template_name="debian-bookworm-arm64",
        disk_paths=[str(disk)],
    )

    with pytest.raises(ImageStoreValidationError, match="run_id_invalid"):
        store.prepare_run_clone(template_id=template_id, run_id="../escape")
