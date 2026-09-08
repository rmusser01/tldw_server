"""Opt-in, offline execution of the candidate package-install recipe; no builds."""

import hashlib
import os
import re
import shutil
import subprocess  # nosec B404
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
IMAGE = os.environ.get("TLDW_COMBINED_PACKAGE_TEST_IMAGE", "")
PACKAGE = os.environ.get("TLDW_COMBINED_PACKAGE_TEST_DEB", "")
pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not IMAGE or not PACKAGE, reason="Requires explicit retained candidate image and Expat DEB"),
]


@pytest.mark.parametrize("damage", ["none", "missing-documentation", "changed-library"])
def test_candidate_install_preserves_complete_package_and_detects_damage(damage: str) -> None:
    """Missing install payloads fail before any deliberate damage is introduced."""
    docker = shutil.which("docker")
    assert docker and re.fullmatch(r"sha256:[0-9a-f]{64}", IMAGE)
    package = Path(PACKAGE).resolve(strict=True)
    assert hashlib.sha256(package.read_bytes()).hexdigest() == (
        "0f06b4b09147e9baf790ce6118a6b705b8c44edb780138af179c3f5600b580dd"
    )
    recipe = (ROOT / "Dockerfiles/candidates/combined-expat/Dockerfile").read_text().replace("\\\n", "")
    commands = [line.removeprefix("RUN ") for line in recipe.splitlines() if line.startswith("RUN dpkg ")]
    assert len(commands) == 1
    mutation = {
        "none": ":",
        "missing-documentation": "rm /usr/share/doc/libexpat1/AUTHORS",
        "changed-library": "printf tampered >> /usr/lib/x86_64-linux-gnu/libexpat.so.1",
    }[damage]
    script = "\n".join(
        [
            "mkdir -p /opt/combined-evidence",
            "cp /input.deb /tmp/qualified-libexpat1.deb",
            commands[0],
            "dpkg --verify libexpat1 > /tmp/verify-before",
            "cat /tmp/verify-before",
            "test ! -s /tmp/verify-before",
            "printf 'complete-package-verified\\n'",
            mutation,
            "printf 'package-damage-applied\\n'",
            "dpkg --verify libexpat1 > /tmp/verify-after",
            "cat /tmp/verify-after",
            "test ! -s /tmp/verify-after",
        ]
    )
    # Execute the checked-in RUN in a disposable container, never in the host shell.
    result = subprocess.run(  # nosec B603
        [
            docker,
            "run",
            "--rm",
            "--pull",
            "never",
            "--platform",
            "linux/amd64",
            "--network",
            "none",
            "--user",
            "0:0",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--pids-limit",
            "64",
            "--memory",
            "256m",
            "--cpus",
            "1",
            "--mount",
            f"type=bind,src={package},dst=/input.deb,readonly",
            "--entrypoint",
            "/bin/bash",
            IMAGE,
            "-euo",
            "pipefail",
            "-c",
            script,
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    output = result.stdout + result.stderr
    assert "complete-package-verified\n" in output, output
    assert "package-damage-applied\n" in output, output
    assert (result.returncode == 0) == (damage == "none"), output
