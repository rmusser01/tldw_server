from __future__ import annotations

import os
import subprocess
from pathlib import Path


IMAGE_DIR = Path(__file__).resolve().parents[1]
BUILD_SCRIPT = IMAGE_DIR / "scripts" / "build-debian-rootfs.sh"
INSTALL_SCRIPT = IMAGE_DIR / "scripts" / "install-agent.sh"


def _run_builder(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(BUILD_SCRIPT), *args],
        cwd=IMAGE_DIR,
        check=False,
        capture_output=True,
        text=True,
        env=os.environ.copy(),
    )


def test_build_rootfs_requires_linux_host() -> None:
    result = _run_builder("--output-rootfs", "/tmp/rootfs")
    assert result.returncode != 0
    assert "Linux host required" in result.stderr


def test_build_rootfs_emits_expected_debootstrap_command_in_dry_run(tmp_path: Path) -> None:
    output_rootfs = tmp_path / "rootfs"
    result = _run_builder("--dry-run", "--profile", "minimal", "--output-rootfs", str(output_rootfs))

    assert result.returncode == 0, result.stderr
    assert "debootstrap" in result.stdout
    assert "--arch=arm64" in result.stdout
    assert "bookworm" in result.stdout
    assert str(output_rootfs) in result.stdout


def test_install_agent_stages_vsock_modules_and_serial_console(tmp_path: Path) -> None:
    rootfs_dir = tmp_path / "rootfs"
    subprocess.run(
        [str(INSTALL_SCRIPT), str(rootfs_dir)],
        cwd=IMAGE_DIR,
        check=True,
    )

    vsock_conf = rootfs_dir / "etc/modules-load.d/vsock.conf"
    assert vsock_conf.is_file()
    contents = vsock_conf.read_text(encoding="utf-8")
    assert "vsock" in contents
    assert "vmw_vsock_virtio_transport" in contents
    assert "virtiofs" in contents
    assert "virtio_console" in contents

    initramfs_modules = rootfs_dir / "etc/initramfs-tools/modules"
    assert initramfs_modules.is_file()
    initramfs_contents = initramfs_modules.read_text(encoding="utf-8")
    assert "virtio_blk" in initramfs_contents
    assert "virtio_console" in initramfs_contents

    wrapper = rootfs_dir / "usr/local/bin/tldw-agent-guest-wrapper"
    assert wrapper.is_file()
    assert os.access(wrapper, os.X_OK)

    serial_getty = rootfs_dir / "etc/systemd/system/getty.target.wants/serial-getty@ttyS0.service"
    assert serial_getty.is_symlink()
