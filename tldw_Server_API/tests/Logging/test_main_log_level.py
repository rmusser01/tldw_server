from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit
REPO_ROOT = Path(__file__).resolve().parents[3]


def test_main_uses_log_level_environment(tmp_path: Path) -> None:
    env = os.environ.copy()
    env.update(
        {
            "AUTH_MODE": "single_user",
            "SINGLE_USER_API_KEY": "main-log-level-test-key-0123456789",
            "MCP_JWT_SECRET": "m" * 32,
            "MCP_API_KEY_SALT": "s" * 32,
            "DATABASE_URL": f"sqlite:///{tmp_path / 'users.db'}",
            "TEST_MODE": "1",
            "ULTRA_MINIMAL_APP": "1",
            "AUTO_DOWNLOAD_MODELS": "false",
            "DISABLE_NLTK_DOWNLOADS": "1",
            "TLDW_CAPTURE_STDERR": "0",
            "LOG_LEVEL": "WARNING",
            "PYTHONPATH": str(REPO_ROOT),
        }
    )

    result = subprocess.run(  # nosec B603
        [
            sys.executable,
            "-c",
            "from tldw_Server_API.app import main; print(f'resolved_log_level={main._log_level}')",
        ],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=90,
        check=False,
    )

    assert result.returncode == 0, result.stderr[-4000:]
    assert "resolved_log_level=WARNING" in result.stdout
