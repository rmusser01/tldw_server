"""Behavioral safety checks for test-only offline rootfs preparation."""

import os
import subprocess  # nosec B404
from pathlib import Path

import pytest


@pytest.mark.unit
@pytest.mark.parametrize("name", ["install-agent.sh", "install-missing-agent.sh"])
@pytest.mark.parametrize("recovery_status", [0, 1, 4])
def test_installer_recovers_journal_before_raw_writes_and_rejects_unsafe_status(
    tmp_path: Path, name: str, recovery_status: int
) -> None:
    """Never write through debugfs while a pending journal can overwrite new data."""
    fixture = Path(__file__).resolve().parent / "failure_drill" / name
    script = tmp_path / "installer.sh"
    log = tmp_path / "operations"
    # Stub guest tools, not the shell control flow; no real filesystem is touched.
    script.write_text(
        """set -eu
recovered=0
e2fsck() {
    printf 'fsck:%s\\n' "$*" >> "$TEST_LOG"
    if [ "$1" = -p ]; then
        recovered=1
        return "$RECOVERY_STATUS"
    fi
}
debugfs() {
    printf 'debugfs\\n' >> "$TEST_LOG"
    [ "$recovered" = 1 ] || return 77
}
cmp() { :; }
sha256sum() { :; }
"""
        + fixture.read_text()
    )
    result = subprocess.run(  # nosec B603
        ["/bin/sh", str(script)],
        env={**os.environ, "TEST_LOG": str(log), "RECOVERY_STATUS": str(recovery_status)},
        capture_output=True,
        timeout=5,
    )
    operations = log.read_text().splitlines()
    if recovery_status in (0, 1):
        assert result.returncode == 0, result.stderr.decode()
        assert operations[0] == "fsck:-p /workspace/rootfs.img"
        assert "debugfs" in operations
        assert operations[-1] == "fsck:-fn /workspace/rootfs.img"
    else:
        assert result.returncode == recovery_status
        assert operations == ["fsck:-p /workspace/rootfs.img"]
