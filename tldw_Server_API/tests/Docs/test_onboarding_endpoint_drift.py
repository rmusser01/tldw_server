import subprocess
import sys


def test_onboarding_has_no_legacy_media_process_endpoint() -> None:
    proc = subprocess.run(
        [sys.executable, "Helper_Scripts/docs/check_onboarding_endpoint_drift.py"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
