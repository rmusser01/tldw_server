from __future__ import annotations

import os
import subprocess
from pathlib import Path


IMAGE_DIR = Path(__file__).resolve().parents[1]
WRAPPER_SCRIPT = IMAGE_DIR / "scripts" / "run-linux-builder-container.sh"


def test_container_wrapper_invokes_native_builder_script(tmp_path: Path) -> None:
    output_dir = tmp_path / "out"
    result = subprocess.run(
        [str(WRAPPER_SCRIPT), "--dry-run", "--output-dir", str(output_dir)],
        cwd=IMAGE_DIR,
        check=False,
        capture_output=True,
        text=True,
        env=os.environ.copy(),
    )

    assert result.returncode == 0, result.stderr
    assert "build-debian-bundle.sh" in result.stdout
    assert str(output_dir) in result.stdout
