import json
import os
import shutil
import subprocess  # nosec B404
import sys
from pathlib import Path

import pytest


def _bash_executable() -> str:
    if os.name != "nt":
        return shutil.which("bash") or "bash"

    program_files_roots = [
        os.environ.get("ProgramFiles"),
        os.environ.get("ProgramFiles(x86)"),
    ]
    for root in program_files_roots:
        if not root:
            continue
        for relative_path in ("Git/bin/bash.exe", "Git/usr/bin/bash.exe"):
            candidate = Path(root) / Path(relative_path)
            if candidate.exists():
                return str(candidate)

    bash = shutil.which("bash")
    if bash and "WindowsApps" not in bash:
        return bash

    pytest.skip("Git Bash is required to execute start-sidecars.sh on Windows")


def test_start_sidecars_default_health_url_targets_root_health():
    script = Path("start-sidecars.sh").read_text(encoding="utf-8")
    if 'TLDW_SERVER_HEALTH_URL:-http://${UVICORN_HOST}:${UVICORN_PORT}/health}' not in script:
        pytest.fail("Expected start-sidecars.sh to default health probe URL to /health")


def test_start_sidecars_dry_run_profile(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "default_workers": ["media_ingest", "audio_jobs"],
                "workers": [
                    {"key": "media_ingest", "slug": "media", "module": "x.y.z"},
                    {"key": "audio_jobs", "slug": "audio", "module": "x.y.z"},
                ],
            }
        ),
        encoding="utf-8",
    )

    env = os.environ.copy()
    env["TLDW_WORKERS_MANIFEST"] = str(manifest)
    env["TLDW_SIDECAR_PROFILE"] = "tts-only"
    env["TLDW_SIDECAR_DRY_RUN"] = "true"
    env["PYTHON_BIN"] = sys.executable

    result = subprocess.run(  # nosec B603
        [_bash_executable(), "start-sidecars.sh"],
        capture_output=True,
        text=True,
        env=env,
        timeout=10,
    )

    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    assert "audio_jobs" in result.stdout
    assert "media_ingest" not in result.stdout
