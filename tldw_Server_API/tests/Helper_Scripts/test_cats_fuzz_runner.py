from __future__ import annotations

import json
import subprocess
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from Helper_Scripts.cats_fuzz import DEFAULT_TEST_API_KEY
from Helper_Scripts.cats_fuzz.cats_cli import CatsProcessResult
from Helper_Scripts.cats_fuzz.manifest import get_builtin_block
from Helper_Scripts.cats_fuzz.runner import (
    get_default_runtime_block,
    run_contract_block,
    run_runtime_block,
)
from Helper_Scripts.cats_fuzz.server import (
    UvicornServer,
    start_server,
    stop_server,
    wait_for_health,
    wait_for_readiness,
)


@pytest.mark.unit
def test_contract_block_writes_summary_for_validate_and_stats(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    openapi = tmp_path / "openapi.json"
    openapi.write_text("{}", encoding="utf-8")

    calls: list[list[str]] = []

    def fake_run(
        command: list[str],
        *,
        timeout_seconds: int,
        env: dict[str, str] | None = None,
    ) -> CatsProcessResult:
        calls.append(command)
        return CatsProcessResult(command=command, exit_code=0, stdout="{}", stderr="")

    monkeypatch.setattr("Helper_Scripts.cats_fuzz.runner.run_command", fake_run)

    result = run_contract_block(
        contract_path=openapi,
        output_dir=tmp_path / "out",
        cats_version="13.8.0",
        openapi_sha256="abc",
    )

    assert result.exit_code == 0
    assert any(command[:2] == ["cats", "validate"] for command in calls)
    assert any(command[:2] == ["cats", "stats"] for command in calls)
    assert (tmp_path / "out" / "contract" / "summary.json").exists()


@pytest.mark.unit
def test_runtime_block_writes_artifacts_passes_env_and_uses_top_level_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    openapi = tmp_path / "openapi.json"
    openapi.write_text("{}", encoding="utf-8")
    block = replace(get_builtin_block("public-read"), requires_readiness=False)
    process_env = {"CATS_HOME": str(tmp_path / "cats-home")}
    calls: list[tuple[list[str], dict[str, str] | None]] = []

    def fake_run(
        command: list[str],
        *,
        timeout_seconds: int,
        env: dict[str, str] | None = None,
    ) -> CatsProcessResult:
        calls.append((command, env))
        return CatsProcessResult(
            command=command,
            exit_code=0,
            stdout="runtime stdout",
            stderr="runtime stderr",
        )

    monkeypatch.setattr("Helper_Scripts.cats_fuzz.runner.run_command", fake_run)

    summary = run_runtime_block(
        block=block,
        contract_path=openapi,
        server_url="http://127.0.0.1:8000",
        output_dir=tmp_path / "out",
        cats_version="13.8.0",
        env=process_env,
    )

    block_dir = tmp_path / "out" / "public-read"
    command = calls[0][0]
    assert command[:5] == [
        "cats",
        "-c",
        str(openapi),
        "-s",
        "http://127.0.0.1:8000",
    ]
    assert command[1] != "run"
    assert command[command.index("--output") + 1] == str(block_dir / "cats-report")
    assert calls == [(command, process_env)]
    assert (block_dir / "stdout.log").read_text(encoding="utf-8") == "runtime stdout"
    assert (block_dir / "stderr.log").read_text(encoding="utf-8") == "runtime stderr"
    assert (block_dir / "summary.json").exists()
    assert summary.report_dir == str(block_dir / "cats-report")


@pytest.mark.unit
def test_runtime_summary_masks_api_key_in_memory_and_on_disk(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    openapi = tmp_path / "openapi.json"
    openapi.write_text("{}", encoding="utf-8")
    block = replace(get_builtin_block("public-read"), requires_readiness=False)

    def fake_run(
        command: list[str],
        *,
        timeout_seconds: int,
        env: dict[str, str] | None = None,
    ) -> CatsProcessResult:
        return CatsProcessResult(command=command, exit_code=0, stdout="", stderr="")

    monkeypatch.setattr("Helper_Scripts.cats_fuzz.runner.run_command", fake_run)

    summary = run_runtime_block(
        block=block,
        contract_path=openapi,
        server_url="http://127.0.0.1:8000",
        output_dir=tmp_path / "out",
        cats_version="13.8.0",
        api_key=DEFAULT_TEST_API_KEY,
    )

    summary_json = (tmp_path / "out" / "public-read" / "summary.json").read_text(encoding="utf-8")
    assert DEFAULT_TEST_API_KEY not in " ".join(summary.command)
    assert DEFAULT_TEST_API_KEY not in " ".join(summary.masked_command)
    assert DEFAULT_TEST_API_KEY not in summary_json
    assert "X-API-KEY=$X-API-KEY" in summary_json


@pytest.mark.unit
def test_runtime_block_waits_for_readiness_when_required(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    openapi = tmp_path / "openapi.json"
    openapi.write_text("{}", encoding="utf-8")
    ready_urls: list[str] = []

    def fake_wait_for_readiness(server_url: str) -> None:
        ready_urls.append(server_url)

    def fake_run(
        command: list[str],
        *,
        timeout_seconds: int,
        env: dict[str, str] | None = None,
    ) -> CatsProcessResult:
        return CatsProcessResult(command=command, exit_code=0, stdout="", stderr="")

    monkeypatch.setattr("Helper_Scripts.cats_fuzz.runner.wait_for_readiness", fake_wait_for_readiness)
    monkeypatch.setattr("Helper_Scripts.cats_fuzz.runner.run_command", fake_run)

    run_runtime_block(
        block=get_builtin_block("public-read"),
        contract_path=openapi,
        server_url="http://127.0.0.1:8000",
        output_dir=tmp_path / "out",
        cats_version="13.8.0",
    )

    assert ready_urls == ["http://127.0.0.1:8000"]


@pytest.mark.unit
def test_runtime_block_skips_readiness_when_not_required(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    openapi = tmp_path / "openapi.json"
    openapi.write_text("{}", encoding="utf-8")
    block = replace(get_builtin_block("public-read"), requires_readiness=False, name="no-ready")

    def fail_wait_for_readiness(server_url: str) -> None:
        raise AssertionError("readiness wait should not be called")

    def fake_run(
        command: list[str],
        *,
        timeout_seconds: int,
        env: dict[str, str] | None = None,
    ) -> CatsProcessResult:
        return CatsProcessResult(command=command, exit_code=0, stdout="", stderr="")

    monkeypatch.setattr("Helper_Scripts.cats_fuzz.runner.wait_for_readiness", fail_wait_for_readiness)
    monkeypatch.setattr("Helper_Scripts.cats_fuzz.runner.run_command", fake_run)

    summary = run_runtime_block(
        block=block,
        contract_path=openapi,
        server_url="http://127.0.0.1:8000",
        output_dir=tmp_path / "out",
        cats_version="13.8.0",
    )

    assert summary.block == "no-ready"


@pytest.mark.unit
def test_get_default_runtime_block_is_public_read() -> None:
    assert get_default_runtime_block() == get_builtin_block("public-read")


class _Response:
    def __init__(self, status: int) -> None:
        self.status = status

    def __enter__(self) -> "_Response":
        return self

    def __exit__(self, *exc_info: Any) -> None:
        return None


@pytest.mark.unit
@pytest.mark.parametrize("status", [200, 204])
def test_wait_for_health_accepts_2xx_status(status: int, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "Helper_Scripts.cats_fuzz.server.urlopen",
        lambda request, timeout: _Response(status),
    )

    wait_for_health("http://127.0.0.1:8000", timeout_seconds=0.01)


@pytest.mark.unit
@pytest.mark.parametrize("status", [404, 503])
def test_wait_for_health_rejects_non_2xx_status(status: int, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "Helper_Scripts.cats_fuzz.server.urlopen",
        lambda request, timeout: _Response(status),
    )
    monkeypatch.setattr("Helper_Scripts.cats_fuzz.server.time.sleep", lambda seconds: None)

    with pytest.raises(TimeoutError):
        wait_for_health("http://127.0.0.1:8000", timeout_seconds=0.01)


@pytest.mark.unit
def test_wait_for_readiness_uses_ready_before_health_ready(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requested_urls: list[str] = []

    def fake_urlopen(request: Any, timeout: float) -> _Response:
        requested_urls.append(request.full_url)
        if request.full_url == "http://127.0.0.1:8000/ready":
            return _Response(503)
        return _Response(204)

    monkeypatch.setattr("Helper_Scripts.cats_fuzz.server.urlopen", fake_urlopen)

    wait_for_readiness("http://127.0.0.1:8000", timeout_seconds=0.01)

    assert requested_urls == [
        "http://127.0.0.1:8000/ready",
        "http://127.0.0.1:8000/health/ready",
    ]


@pytest.mark.unit
@pytest.mark.parametrize("ready_status", [404, 503])
def test_wait_for_readiness_falls_back_from_ready_to_health_ready(
    ready_status: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    requested_urls: list[str] = []

    def fake_urlopen(request: Any, timeout: float) -> _Response:
        requested_urls.append(request.full_url)
        if request.full_url == "http://127.0.0.1:8000/ready":
            return _Response(ready_status)
        return _Response(204)

    monkeypatch.setattr("Helper_Scripts.cats_fuzz.server.urlopen", fake_urlopen)

    wait_for_readiness("http://127.0.0.1:8000", timeout_seconds=0.01)

    assert requested_urls == [
        "http://127.0.0.1:8000/ready",
        "http://127.0.0.1:8000/health/ready",
    ]


@pytest.mark.unit
@pytest.mark.parametrize("status", [404, 500])
def test_wait_for_readiness_rejects_all_non_2xx_statuses(status: int, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "Helper_Scripts.cats_fuzz.server.urlopen",
        lambda request, timeout: _Response(status),
    )
    monkeypatch.setattr("Helper_Scripts.cats_fuzz.server.time.sleep", lambda seconds: None)

    with pytest.raises(TimeoutError):
        wait_for_readiness("http://127.0.0.1:8000", timeout_seconds=0.01)


class _FakeProcess:
    def __init__(self, exited: bool = False) -> None:
        self.exited = exited
        self.terminated = False
        self.killed = False

    def poll(self) -> int | None:
        return 0 if self.exited else None

    def terminate(self) -> None:
        self.terminated = True
        self.exited = True

    def kill(self) -> None:
        self.killed = True
        self.exited = True

    def wait(self, timeout: int) -> int:
        self.exited = True
        return 0


@pytest.mark.unit
def test_start_server_discards_output_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    popen_calls: list[dict[str, Any]] = []
    fake_process = _FakeProcess()

    def fake_popen(command: list[str], **kwargs: Any) -> _FakeProcess:
        popen_calls.append(kwargs)
        return fake_process

    monkeypatch.setattr("Helper_Scripts.cats_fuzz.server.subprocess.Popen", fake_popen)
    monkeypatch.setattr("Helper_Scripts.cats_fuzz.server.wait_for_health", lambda url: None)

    server = start_server(env={"AUTH_MODE": "single_user"}, port=1234)

    assert server.process is fake_process
    assert popen_calls[0]["stdout"] == subprocess.DEVNULL
    assert popen_calls[0]["stderr"] == subprocess.DEVNULL


@pytest.mark.unit
def test_start_server_with_log_dir_redirects_output_and_stop_closes_streams(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    popen_calls: list[dict[str, Any]] = []
    fake_process = _FakeProcess()

    def fake_popen(command: list[str], **kwargs: Any) -> _FakeProcess:
        popen_calls.append(kwargs)
        return fake_process

    monkeypatch.setattr("Helper_Scripts.cats_fuzz.server.subprocess.Popen", fake_popen)
    monkeypatch.setattr("Helper_Scripts.cats_fuzz.server.wait_for_health", lambda url: None)

    server = start_server(env={"AUTH_MODE": "single_user"}, port=1234, log_dir=tmp_path)

    assert (tmp_path / "uvicorn.stdout.log").exists()
    assert (tmp_path / "uvicorn.stderr.log").exists()
    assert popen_calls[0]["stdout"] is server.stdout_stream
    assert popen_calls[0]["stderr"] is server.stderr_stream
    assert server.stdout_stream is not None
    assert server.stderr_stream is not None

    stop_server(server)

    assert fake_process.terminated is True
    assert server.stdout_stream.closed is True
    assert server.stderr_stream.closed is True


@pytest.mark.unit
def test_stop_server_closes_streams_for_already_exited_process(tmp_path: Path) -> None:
    stdout_stream = (tmp_path / "stdout.log").open("w", encoding="utf-8")
    stderr_stream = (tmp_path / "stderr.log").open("w", encoding="utf-8")
    server = UvicornServer(
        process=_FakeProcess(exited=True),
        url="http://127.0.0.1:1234",
        stdout_stream=stdout_stream,
        stderr_stream=stderr_stream,
    )

    stop_server(server)

    assert stdout_stream.closed is True
    assert stderr_stream.closed is True


@pytest.mark.unit
def test_contract_block_summary_combines_validate_and_stats_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    openapi = tmp_path / "openapi.json"
    openapi.write_text("{}", encoding="utf-8")

    def fake_run(
        command: list[str],
        *,
        timeout_seconds: int,
        env: dict[str, str] | None = None,
    ) -> CatsProcessResult:
        stdout = '{"command": "%s"}' % command[1]
        return CatsProcessResult(command=command, exit_code=0, stdout=stdout, stderr="")

    monkeypatch.setattr("Helper_Scripts.cats_fuzz.runner.run_command", fake_run)

    summary = run_contract_block(
        contract_path=openapi,
        output_dir=tmp_path / "out",
        cats_version="13.8.0",
    )
    data = json.loads((tmp_path / "out" / "contract" / "summary.json").read_text())

    assert summary.openapi_sha256
    assert "validate" in (tmp_path / "out" / "contract" / "stdout.log").read_text()
    assert "stats" in (tmp_path / "out" / "contract" / "stdout.log").read_text()
    assert data["block"] == "contract"
