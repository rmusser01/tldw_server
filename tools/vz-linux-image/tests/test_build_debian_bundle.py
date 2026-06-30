from __future__ import annotations

import os
import subprocess
from pathlib import Path


IMAGE_DIR = Path(__file__).resolve().parents[1]
BUNDLE_BUILDER = IMAGE_DIR / "scripts" / "build-debian-bundle.sh"


def _run_bundle_builder(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(BUNDLE_BUILDER), *args],
        cwd=IMAGE_DIR,
        check=False,
        capture_output=True,
        text=True,
        env=os.environ.copy(),
    )


def test_build_debian_bundle_dry_run_prints_all_artifact_paths(tmp_path: Path) -> None:
    output_dir = tmp_path / "out"

    result = _run_bundle_builder("--dry-run", "--output-dir", str(output_dir))

    assert result.returncode == 0, result.stderr
    assert "rootfs/" in result.stdout
    assert "rootfs.img" in result.stdout
    assert "bundle/" in result.stdout
    assert "build-info.json" in result.stdout
