from __future__ import annotations

import os
import subprocess
from pathlib import Path


IMAGE_DIR = Path(__file__).resolve().parents[1]
PACK_SCRIPT = IMAGE_DIR / "scripts" / "pack-rootfs-image.sh"


def _run_packer(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(PACK_SCRIPT), *args],
        cwd=IMAGE_DIR,
        check=False,
        capture_output=True,
        text=True,
        env=os.environ.copy(),
    )


def test_pack_rootfs_requires_existing_rootfs_dir(tmp_path: Path) -> None:
    missing_rootfs = tmp_path / "missing-rootfs"
    output_image = tmp_path / "rootfs.img"

    result = _run_packer("--rootfs", str(missing_rootfs), "--output-image", str(output_image))

    assert result.returncode != 0
    assert "rootfs directory does not exist" in result.stderr


def test_pack_rootfs_supports_dry_run(tmp_path: Path) -> None:
    rootfs_dir = tmp_path / "rootfs"
    output_image = tmp_path / "rootfs.img"
    rootfs_dir.mkdir()

    result = _run_packer("--dry-run", "--rootfs", str(rootfs_dir), "--output-image", str(output_image))

    assert result.returncode == 0, result.stderr
    assert "mke2fs" in result.stdout
    assert str(output_image) in result.stdout
