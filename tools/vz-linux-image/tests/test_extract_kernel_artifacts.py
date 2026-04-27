from __future__ import annotations

import os
import subprocess
from pathlib import Path


IMAGE_DIR = Path(__file__).resolve().parents[1]
EXTRACT_SCRIPT = IMAGE_DIR / "scripts" / "extract-kernel-artifacts.sh"


def _run_extractor(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(EXTRACT_SCRIPT), *args],
        cwd=IMAGE_DIR,
        check=False,
        capture_output=True,
        text=True,
        env=os.environ.copy(),
    )


def test_extract_kernel_requires_boot_artifacts_in_rootfs(tmp_path: Path) -> None:
    rootfs_dir = tmp_path / "rootfs"
    output_dir = tmp_path / "out"
    (rootfs_dir / "boot").mkdir(parents=True)

    result = _run_extractor("--rootfs", str(rootfs_dir), "--output-dir", str(output_dir))

    assert result.returncode != 0
    assert "boot artifacts not found" in result.stderr


def test_extract_kernel_supports_dry_run(tmp_path: Path) -> None:
    rootfs_dir = tmp_path / "rootfs"
    boot_dir = rootfs_dir / "boot"
    output_dir = tmp_path / "out"
    boot_dir.mkdir(parents=True)
    (boot_dir / "vmlinuz-6.1.0-arm64").write_bytes(b"kernel")
    (boot_dir / "initrd.img-6.1.0-arm64").write_bytes(b"initrd")

    result = _run_extractor("--dry-run", "--rootfs", str(rootfs_dir), "--output-dir", str(output_dir))

    assert result.returncode == 0, result.stderr
    assert "kernel" in result.stdout
    assert "initrd" in result.stdout
    assert str(output_dir) in result.stdout
