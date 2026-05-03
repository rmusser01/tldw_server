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


def test_image_store_reloads_legacy_manifest_without_artifact_format(tmp_path: Path) -> None:
    disk = tmp_path / "rootfs.img"
    disk.write_bytes(b"rootfs")
    store_root = tmp_path / "store"
    manifest_path = store_root / "templates" / "vz_linux" / "legacy" / "manifest.json"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "template_id": "vz_linux:legacy",
                "runtime": "vz_linux",
                "template_name": "legacy",
                "source_path": None,
                "registered_at": "2026-04-01T00:00:00+00:00",
                "disk_paths": [str(disk)],
                "artifacts": [
                    {
                        "name": disk.name,
                        "path": str(disk),
                        "size_bytes": len(b"rootfs"),
                        "sha256": "d5d75963e365d7b3e74c0e75bc7b5900921c97b6185cd139ab62fc29f6b81b71",
                    }
                ],
                "labels": {},
                "provenance": {},
            }
        ),
        encoding="utf-8",
    )

    record = SandboxImageStore(root_path=store_root).get_template("vz_linux:legacy")

    assert record is not None
    assert record.artifact_format == "unknown"
    assert record.oci_image_ref is None
    assert record.oci_layer_digests == []


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


def test_image_store_registers_bundle_with_tldw_bundle_artifact_format(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "kernel").write_bytes(b"kernel")
    (bundle / "rootfs.img").write_bytes(b"rootfs")
    (bundle / "manifest.json").write_text(
        json.dumps({"schema_version": 1, "boot_mode": "linux_direct"}),
        encoding="utf-8",
    )
    store_root = tmp_path / "store"
    store = SandboxImageStore(root_path=store_root)

    template_id = store.register_bundle(
        runtime="vz_linux",
        template_name="debian-bookworm-arm64",
        bundle_path=bundle,
    )

    manifest_path = store_root / "templates" / "vz_linux" / "debian-bookworm-arm64" / "manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["artifact_format"] == "tldw_bundle"

    reloaded = SandboxImageStore(root_path=store_root).get_template(template_id)
    assert reloaded is not None
    assert reloaded.artifact_format == "tldw_bundle"
    assert reloaded.oci_image_ref is None
    assert reloaded.oci_layer_digests == []


def test_image_store_persists_optional_oci_metadata(tmp_path: Path) -> None:
    disk = tmp_path / "rootfs.img"
    disk.write_bytes(b"rootfs")
    store_root = tmp_path / "store"
    store = SandboxImageStore(root_path=store_root)

    template_id = store.register_template(
        runtime="vz_linux",
        template_name="oci-backed",
        disk_paths=[str(disk)],
        artifact_format="oci_image",
        oci_image_ref="registry.example/tldw/sandbox:bookworm",
        oci_platform="linux/arm64",
        oci_manifest_digest="sha256:" + "a" * 64,
        oci_config_digest="sha256:" + "b" * 64,
        oci_layer_digests=["sha256:" + "c" * 64],
        registry="registry.example",
        imported_at="2026-05-02T00:00:00+00:00",
    )

    manifest_path = store_root / "templates" / "vz_linux" / "oci-backed" / "manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["artifact_format"] == "oci_image"
    assert payload["oci_image_ref"] == "registry.example/tldw/sandbox:bookworm"
    assert payload["oci_platform"] == "linux/arm64"
    assert payload["oci_layer_digests"] == ["sha256:" + "c" * 64]

    reloaded = SandboxImageStore(root_path=store_root).get_template(template_id)
    assert reloaded is not None
    assert reloaded.artifact_format == "oci_image"
    assert reloaded.oci_image_ref == "registry.example/tldw/sandbox:bookworm"
    assert reloaded.oci_platform == "linux/arm64"
    assert reloaded.oci_manifest_digest == "sha256:" + "a" * 64
    assert reloaded.oci_config_digest == "sha256:" + "b" * 64
    assert reloaded.oci_layer_digests == ["sha256:" + "c" * 64]
    assert reloaded.registry == "registry.example"
    assert reloaded.imported_at == "2026-05-02T00:00:00+00:00"


def test_image_store_rejects_unknown_artifact_format(tmp_path: Path) -> None:
    disk = tmp_path / "rootfs.img"
    disk.write_bytes(b"rootfs")
    store = SandboxImageStore(root_path=tmp_path / "store")

    with pytest.raises(ImageStoreValidationError, match="artifact_format_invalid"):
        store.register_template(
            runtime="vz_linux",
            template_name="bad-format",
            disk_paths=[str(disk)],
            artifact_format="tarball",
        )


def test_image_store_rejects_invalid_oci_layer_digests(tmp_path: Path) -> None:
    disk = tmp_path / "rootfs.img"
    disk.write_bytes(b"rootfs")
    store = SandboxImageStore(root_path=tmp_path / "store")

    invalid_values = [[""], [123]]
    for index, oci_layer_digests in enumerate(invalid_values):
        with pytest.raises(ImageStoreValidationError, match="oci_layer_digests_invalid"):
            store.register_template(
                runtime="vz_linux",
                template_name=f"bad-oci-{index}",
                disk_paths=[str(disk)],
                oci_layer_digests=oci_layer_digests,
            )


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
