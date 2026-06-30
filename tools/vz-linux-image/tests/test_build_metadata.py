from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path


IMAGE_DIR = Path(__file__).resolve().parents[1]
BUNDLE_BUILDER = IMAGE_DIR / "scripts" / "build-debian-bundle.sh"


def test_build_metadata_includes_suite_profile_and_kernel_package(tmp_path: Path) -> None:
    output_dir = tmp_path / "out"
    result = subprocess.run(
        [str(BUNDLE_BUILDER), "--dry-run", "--output-dir", str(output_dir)],
        cwd=IMAGE_DIR,
        check=False,
        capture_output=True,
        text=True,
        env=os.environ.copy(),
    )

    assert result.returncode == 0, result.stderr

    metadata = json.loads((output_dir / "build-info.json").read_text(encoding="utf-8"))
    assert metadata["suite"] == "bookworm"
    assert metadata["profile"] == "minimal"
    assert metadata["architecture"] == "arm64"
    assert metadata["kernel_package"] == "linux-image-arm64"
    assert metadata["artifact_kind"] == "canonical_bundle"
