from __future__ import annotations

import json
from pathlib import Path

import pytest

from tldw_Server_API.app.core.Sandbox.image_store import (
    ImageStoreValidationError,
    SandboxImageStore,
)


def test_image_store_returns_run_clone_manifest_for_template(tmp_path: Path) -> None:
    disk = tmp_path / "ubuntu-24.04.img"
    disk.write_text("disk", encoding="utf-8")
    store = SandboxImageStore(root_path=tmp_path)
    template_id = store.register_template(
        runtime="vz_linux",
        template_name="ubuntu-24.04",
        disk_paths=[str(disk)],
    )

    manifest = store.prepare_run_clone(template_id=template_id, run_id="run-123")

    assert manifest.template_id == template_id
    assert manifest.run_id == "run-123"
    assert manifest.clone_items[0].source_path.endswith(".img")
    assert manifest.clone_items[0].mode == "clone"


def test_image_store_persists_template_manifest_and_reloads(tmp_path: Path) -> None:
    disk = tmp_path / "rootfs.img"
    disk.write_bytes(b"rootfs-image")
    store = SandboxImageStore(root_path=tmp_path / "store")

    template_id = store.register_template(
        runtime="vz_linux",
        template_name="debian-bookworm-arm64",
        disk_paths=[str(disk)],
        source_path=str(tmp_path),
        labels={"suite": "bookworm"},
    )

    manifest_path = tmp_path / "store" / "templates" / "vz_linux" / "debian-bookworm-arm64" / "manifest.json"
    assert manifest_path.exists()

    reloaded = SandboxImageStore(root_path=tmp_path / "store")
    record = reloaded.get_template(template_id)

    assert record is not None
    assert record.template_id == template_id
    assert record.labels == {"suite": "bookworm"}
    assert record.artifacts[0].sha256 == "d41952dc5f33828e727c4c23bf24a874d0a2e6dd864637c5fa25532f74afd2e2"
    assert record.artifacts[0].size_bytes == len(b"rootfs-image")


def test_image_store_rejects_missing_template_artifacts(tmp_path: Path) -> None:
    store = SandboxImageStore(root_path=tmp_path / "store")

    with pytest.raises(ImageStoreValidationError, match="template_artifact_missing"):
        store.register_template(
            runtime="vz_linux",
            template_name="missing",
            disk_paths=[str(tmp_path / "missing.img")],
        )


def test_image_store_rejects_duplicate_templates_by_default(tmp_path: Path) -> None:
    disk = tmp_path / "rootfs.img"
    disk.write_bytes(b"rootfs")
    store = SandboxImageStore(root_path=tmp_path / "store")
    store.register_template(
        runtime="vz_linux",
        template_name="debian-bookworm-arm64",
        disk_paths=[str(disk)],
    )

    with pytest.raises(ImageStoreValidationError, match="template_duplicate"):
        store.register_template(
            runtime="vz_linux",
            template_name="debian-bookworm-arm64",
            disk_paths=[str(disk)],
        )


def test_image_store_rejects_invalid_persisted_manifest(tmp_path: Path) -> None:
    manifest_path = tmp_path / "store" / "templates" / "vz_linux" / "broken" / "manifest.json"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(json.dumps({"schema_version": 1}), encoding="utf-8")

    with pytest.raises(ImageStoreValidationError, match="manifest_missing_fields"):
        SandboxImageStore(root_path=tmp_path / "store")


def test_image_store_registers_bundle_with_build_provenance(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "kernel").write_bytes(b"kernel")
    (bundle / "rootfs.img").write_bytes(b"rootfs")
    (bundle / "initrd").write_bytes(b"initrd")
    (bundle / "manifest.json").write_text(
        json.dumps({"schema_version": 1, "boot_mode": "linux_direct"}),
        encoding="utf-8",
    )
    (bundle / "build-info.json").write_text(
        json.dumps({"suite": "bookworm", "architecture": "arm64"}),
        encoding="utf-8",
    )
    store = SandboxImageStore(root_path=tmp_path / "store")

    template_id = store.register_bundle(
        runtime="vz_linux",
        template_name="debian-bookworm-arm64",
        bundle_path=bundle,
        labels={"profile": "minimal"},
    )

    record = store.get_template(template_id)
    assert record is not None
    assert record.source_path == str(bundle)
    assert record.provenance == {"suite": "bookworm", "architecture": "arm64"}
    assert {artifact.name for artifact in record.artifacts} == {"kernel", "rootfs.img", "initrd"}


def test_image_store_lists_templates_and_plans_run_gc(tmp_path: Path) -> None:
    disk = tmp_path / "rootfs.img"
    disk.write_bytes(b"rootfs")
    run_keep = tmp_path / "store" / "runs" / "run-keep"
    run_drop = tmp_path / "store" / "runs" / "run-drop"
    run_keep.mkdir(parents=True)
    run_drop.mkdir(parents=True)
    (run_drop / "rootfs.img").write_bytes(b"clone")
    store = SandboxImageStore(root_path=tmp_path / "store")
    template_id = store.register_template(
        runtime="vz_linux",
        template_name="debian-bookworm-arm64",
        disk_paths=[str(disk)],
    )

    templates = store.list_templates(runtime="vz_linux")
    gc_plan = store.plan_garbage_collection(active_run_ids={"run-keep"})

    assert [template.template_id for template in templates] == [template_id]
    assert len(gc_plan.run_candidates) == 1
    assert gc_plan.run_candidates[0].run_id == "run-drop"
    assert gc_plan.run_candidates[0].path == str(run_drop)
    assert run_drop.exists()
