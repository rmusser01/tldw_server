"""Executable validation must fail before optimized Python removes its checks."""
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize('name', [
    'email_archive_throughput_sqlite_2026_09_25.py',
    'email_archive_throughput_postgres_2026_09_25.py',
    'email_local_release_checks_13376.py',
])
def test_probe_rejects_optimized_python_before_environment_setup(name):
    script = Path(__file__).resolve().parents[3] / 'Docs/Operations/probes' / name
    result = subprocess.run(
        [sys.executable, '-O', str(script)], capture_output=True, text=True, timeout=10, check=False,
    )
    assert result.returncode != 0
    assert 'Synthetic validation requires Python without optimization' in result.stderr
    assert 'EMAIL_PROBE_PG_MANIFEST' not in result.stderr
