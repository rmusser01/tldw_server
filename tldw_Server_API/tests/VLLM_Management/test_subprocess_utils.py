from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from tldw_Server_API.app.core.Workflows import subprocess_utils


@pytest.mark.unit
def test_start_process_redacts_api_key_from_logged_command(monkeypatch, tmp_path: Path):
    logged: list[str] = []

    def fake_open(path: Path, mode: str, buffering: int = 0):  # noqa: ANN001
        return SimpleNamespace(close=lambda: None)

    def fake_popen(cmd, **kwargs):  # noqa: ANN001, ANN003
        return SimpleNamespace(pid=12345)

    def fake_info(message: str) -> None:
        logged.append(message)

    monkeypatch.setattr("builtins.open", fake_open)
    monkeypatch.setattr(subprocess_utils.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(subprocess_utils.logger, "info", fake_info)
    monkeypatch.setattr(subprocess_utils.os, "getpgid", lambda pid: 54321)

    task = subprocess_utils.start_process(
        ["python", "-m", "vllm.entrypoints.openai.api_server", "--api-key", "super-secret"],
        workdir=tmp_path,
        log_dir=tmp_path / "logs",
    )

    assert task.pid == 12345
    assert logged
    assert "super-secret" not in logged[0]
    assert "[REDACTED]" in logged[0]
