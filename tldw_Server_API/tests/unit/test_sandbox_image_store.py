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
    assert any(
        Path(item.target_path).parent.name == "run-123"
        and Path(item.target_path).name == "rootfs.img"
        for item in clone_manifest.clone_items
    )
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


def test_sandbox_image_store_defers_run_manifest_reload_until_needed(tmp_path: Path) -> None:
    manifest_path = tmp_path / "store" / "runs" / "bad-run" / "manifest.json"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text("[]", encoding="utf-8")

    store = SandboxImageStore(root_path=tmp_path / "store")

    with pytest.raises(ImageStoreValidationError, match="run_manifest_expected_object"):
        store.list_run_clone_manifests()


def test_sandbox_image_store_prepare_run_clone_does_not_load_unrelated_manifests(tmp_path: Path) -> None:
    disk = tmp_path / "rootfs.img"
    disk.write_bytes(b"rootfs")
    manifest_path = tmp_path / "store" / "runs" / "bad-run" / "manifest.json"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text("[]", encoding="utf-8")
    store = SandboxImageStore(root_path=tmp_path / "store")
    template_id = store.register_template(
        runtime="vz_linux",
        template_name="debian-bookworm-arm64",
        disk_paths=[str(disk)],
    )

    manifest = store.prepare_run_clone(template_id=template_id, run_id="new-run")

    assert manifest.run_id == "new-run"
    assert (tmp_path / "store" / "runs" / "new-run" / "manifest.json").exists()
    with pytest.raises(ImageStoreValidationError, match="run_manifest_expected_object"):
        store.list_run_clone_manifests()


def test_sandbox_image_store_rejects_duplicate_run_clone_target_names(tmp_path: Path) -> None:
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    first_root.mkdir()
    second_root.mkdir()
    first_disk = first_root / "rootfs.img"
    second_disk = second_root / "rootfs.img"
    first_disk.write_bytes(b"one")
    second_disk.write_bytes(b"two")
    store = SandboxImageStore(root_path=tmp_path / "store")
    template_id = store.register_template(
        runtime="vz_linux",
        template_name="duplicate-targets",
        disk_paths=[str(first_disk), str(second_disk)],
    )

    with pytest.raises(ImageStoreValidationError, match=r"run_clone_target_name_collision: .*rootfs\.img"):
        store.prepare_run_clone(template_id=template_id, run_id="run-duplicate")


def test_sandbox_image_store_rejects_tampered_run_manifest_mode(tmp_path: Path) -> None:
    source = tmp_path / "rootfs.img"
    source.write_bytes(b"rootfs")
    manifest_path = tmp_path / "store" / "runs" / "run-tampered" / "manifest.json"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "template_id": "vz_linux:bundle",
                "run_id": "run-tampered",
                "clone_items": [
                    {
                        "source_path": str(source),
                        "target_path": str(tmp_path / "store" / "runs" / "run-tampered" / "rootfs.img"),
                        "mode": "copy",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ImageStoreValidationError, match="run_manifest_clone_item_mode_invalid"):
        SandboxImageStore(root_path=tmp_path / "store").list_run_clone_manifests()


def test_sandbox_image_store_rejects_tampered_run_manifest_target_layout(tmp_path: Path) -> None:
    source = tmp_path / "rootfs.img"
    source.write_bytes(b"rootfs")
    manifest_path = tmp_path / "store" / "runs" / "run-tampered" / "manifest.json"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "template_id": "vz_linux:bundle",
                "run_id": "run-tampered",
                "clone_items": [
                    {
                        "source_path": str(source),
                        "target_path": str(tmp_path / "outside" / "rootfs.img"),
                        "mode": "clone",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ImageStoreValidationError, match="run_manifest_clone_item_target_invalid"):
        SandboxImageStore(root_path=tmp_path / "store").list_run_clone_manifests()


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


def test_sandbox_image_store_gc_plan_classifies_run_manifest_only_directories(
    tmp_path: Path,
) -> None:
    disk = tmp_path / "rootfs.img"
    disk.write_bytes(b"rootfs")
    store = SandboxImageStore(root_path=tmp_path / "store")
    template_id = store.register_template(
        runtime="vz_linux",
        template_name="debian-bookworm-arm64",
        disk_paths=[str(disk)],
    )

    store.prepare_run_clone(template_id=template_id, run_id="run-manifest-only")
    gc_plan = store.plan_garbage_collection(active_run_ids=set())

    candidate = next(
        item for item in gc_plan.run_candidates if item.run_id == "run-manifest-only"
    )
    assert candidate.reason == "planning_only_run_manifest"
    assert candidate.template_id == template_id


def test_sandbox_image_store_gc_plan_classifies_legacy_run_directories_without_manifest(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "store" / "runs" / "legacy-run"
    run_dir.mkdir(parents=True)
    (run_dir / "rootfs.img").write_bytes(b"clone")

    gc_plan = SandboxImageStore(root_path=tmp_path / "store").plan_garbage_collection(
        active_run_ids=set()
    )

    candidate = next(item for item in gc_plan.run_candidates if item.run_id == "legacy-run")
    assert candidate.reason == "legacy_run_directory"
    assert candidate.template_id is None


def test_sandbox_image_store_cleanup_run_candidate_removes_manifest_only_directory(
    tmp_path: Path,
) -> None:
    disk = tmp_path / "rootfs.img"
    disk.write_bytes(b"rootfs")
    store = SandboxImageStore(root_path=tmp_path / "store")
    template_id = store.register_template(
        runtime="vz_linux",
        template_name="debian-bookworm-arm64",
        disk_paths=[str(disk)],
    )
    store.prepare_run_clone(template_id=template_id, run_id="run-manifest-only")

    deleted = store.cleanup_run_candidate(
        run_id="run-manifest-only",
        reason="planning_only_run_manifest",
    )

    assert deleted is True
    assert not (tmp_path / "store" / "runs" / "run-manifest-only").exists()
    assert store.get_run_clone_manifest("run-manifest-only") is None


def test_sandbox_image_store_cleanup_run_candidate_rejects_manifest_only_reason_when_payload_exists(
    tmp_path: Path,
) -> None:
    disk = tmp_path / "rootfs.img"
    disk.write_bytes(b"rootfs")
    store = SandboxImageStore(root_path=tmp_path / "store")
    template_id = store.register_template(
        runtime="vz_linux",
        template_name="debian-bookworm-arm64",
        disk_paths=[str(disk)],
    )
    store.prepare_run_clone(template_id=template_id, run_id="run-not-empty")
    (tmp_path / "store" / "runs" / "run-not-empty" / "rootfs.img").write_bytes(b"clone")

    with pytest.raises(ImageStoreValidationError, match="gc_reason_mismatch_planning_only_run_manifest"):
        store.cleanup_run_candidate(
            run_id="run-not-empty",
            reason="planning_only_run_manifest",
        )
