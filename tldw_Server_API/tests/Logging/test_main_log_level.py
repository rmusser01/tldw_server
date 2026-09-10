"""Regression tests for the main application's startup log level."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit
REPO_ROOT = Path(__file__).resolve().parents[3]


@pytest.mark.parametrize(
    ("process_log_level", "dotenv_log_level", "expected_log_level"),
    [
        pytest.param("WARNING", None, "WARNING", id="process-environment"),
        pytest.param(None, "ERROR", "ERROR", id="selected-dotenv-file"),
        pytest.param("CRITICAL", "DEBUG", "CRITICAL", id="process-environment-wins"),
    ],
)
def test_main_uses_log_level_environment(
    tmp_path: Path,
    process_log_level: str | None,
    dotenv_log_level: str | None,
    expected_log_level: str,
) -> None:
    """Resolve LOG_LEVEL from the selected dotenv without overriding the process."""
    env_file = tmp_path / "runtime.env"
    dotenv_lines = [
        "AUTH_MODE=single_user",
        "SINGLE_USER_API_KEY=main-log-level-test-key-0123456789",
        f"MCP_JWT_SECRET={'m' * 32}",
        f"MCP_API_KEY_SALT={'s' * 32}",
    ]
    if dotenv_log_level is not None:
        dotenv_lines.append(f"LOG_LEVEL={dotenv_log_level}")
    env_file.write_text("\n".join(dotenv_lines) + "\n", encoding="utf-8")

    env = os.environ.copy()
    env.pop("LOG_LEVEL", None)
    env.update(
        {
            "DATABASE_URL": f"sqlite:///{tmp_path / 'users.db'}",
            "TEST_MODE": "1",
            "ULTRA_MINIMAL_APP": "1",
            "AUTO_DOWNLOAD_MODELS": "false",
            "DISABLE_NLTK_DOWNLOADS": "1",
            "TLDW_CAPTURE_STDERR": "0",
            "TLDW_ENV_FILE": str(env_file),
            "TLDW_ENV_FILE_EXCLUSIVE": "1",
            "PYTHONPATH": str(REPO_ROOT),
        }
    )
    if process_log_level is not None:
        env["LOG_LEVEL"] = process_log_level

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
    assert f"resolved_log_level={expected_log_level}" in result.stdout
