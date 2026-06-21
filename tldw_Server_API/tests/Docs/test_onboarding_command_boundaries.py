import subprocess
import sys


def test_onboarding_command_boundary_script_passes() -> None:
    proc = subprocess.run(
        [sys.executable, "Helper_Scripts/docs/check_onboarding_command_boundaries.py"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
