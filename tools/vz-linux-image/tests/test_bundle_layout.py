from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path


def test_bundle_builder_emits_manifest_and_expected_paths(tmp_path: Path) -> None:
    image_dir = Path(__file__).resolve().parents[1]
    bundle_dir = tmp_path / "bundle"
    rootfs_dir = tmp_path / "rootfs"
    kernel_path = tmp_path / "vmlinuz"
    rootfs_image_path = tmp_path / "rootfs-source.img"
    initrd_path = tmp_path / "initrd.img"

    kernel_path.write_bytes(b"kernel-data")
    rootfs_image_path.write_bytes(b"rootfs-image-data")
    initrd_path.write_bytes(b"initrd-data")

    env = os.environ.copy()
    env["TLDW_VZ_LINUX_IMAGE_ROOTFS"] = str(rootfs_dir)
    env["TLDW_VZ_LINUX_BUNDLE_KERNEL"] = str(kernel_path)
    env["TLDW_VZ_LINUX_BUNDLE_ROOTFS_IMAGE"] = str(rootfs_image_path)
    env["TLDW_VZ_LINUX_BUNDLE_INITRD"] = str(initrd_path)

    subprocess.run(
        [str(image_dir / "scripts" / "build-bundle.sh"), str(bundle_dir)],
        cwd=image_dir,
        env=env,
        check=True,
    )

    manifest_path = bundle_dir / "manifest.json"
    assert manifest_path.is_file()
    assert (bundle_dir / "kernel").is_file()
    assert (bundle_dir / "rootfs.img").is_file()
    assert (bundle_dir / "initrd").is_file()

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["bundle_version"] == "1"
    assert manifest["boot_mode"] == "bundle"
    assert manifest["kernel"] == "kernel"
    assert manifest["rootfs"] == "rootfs.img"
    assert manifest["initrd"] == "initrd"
    assert manifest["guest_agent_path"] == "/usr/local/bin/tldw-agent-guest"
    assert manifest["workspace_mount_tag"] == "workspace"
    assert manifest["vsock_port"] == 1024

    assert (rootfs_dir / "usr/local/bin/tldw-agent-guest").is_file()
    assert (rootfs_dir / "usr/local/bin/tldw-agent-guest-wrapper").is_file()
    assert (rootfs_dir / "etc/systemd/system/tldw-agent-guest.service").is_file()
    assert (rootfs_dir / "etc/systemd/system/workspace.mount").is_file()
    assert (rootfs_dir / "workspace").is_dir()

    guest_service = (rootfs_dir / "etc/systemd/system/tldw-agent-guest.service").read_text(encoding="utf-8")
    assert "Requires=workspace.mount" in guest_service
    assert "After=workspace.mount" in guest_service
    assert "PassEnvironment=TLDW_AGENT_GUEST_VM_ID" in guest_service
    assert "ExecStart=/usr/local/bin/tldw-agent-guest-wrapper" in guest_service

    workspace_mount = (rootfs_dir / "etc/systemd/system/workspace.mount").read_text(encoding="utf-8")
    assert "What=workspace" in workspace_mount
    assert "Where=/workspace" in workspace_mount
    assert "Type=virtiofs" in workspace_mount

    wants_dir = rootfs_dir / "etc/systemd/system/multi-user.target.wants"
    assert wants_dir.is_dir()
    assert (wants_dir / "tldw-agent-guest.service").is_symlink()
    assert (wants_dir / "workspace.mount").is_symlink()
