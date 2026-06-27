from __future__ import annotations

import json
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
from Helper_Scripts.cats_fuzz.server import wait_for_health, wait_for_readiness


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
@pytest.mark.parametrize("status", [200, 302, 404])
def test_wait_for_health_accepts_non_5xx_status(status: int, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "Helper_Scripts.cats_fuzz.server.urlopen",
        lambda request, timeout: _Response(status),
    )

    wait_for_health("http://127.0.0.1:8000", timeout_seconds=0.01)


@pytest.mark.unit
def test_wait_for_health_rejects_5xx_status(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "Helper_Scripts.cats_fuzz.server.urlopen",
        lambda request, timeout: _Response(503),
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
def test_wait_for_readiness_rejects_all_5xx(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "Helper_Scripts.cats_fuzz.server.urlopen",
        lambda request, timeout: _Response(500),
    )
    monkeypatch.setattr("Helper_Scripts.cats_fuzz.server.time.sleep", lambda seconds: None)

    with pytest.raises(TimeoutError):
        wait_for_readiness("http://127.0.0.1:8000", timeout_seconds=0.01)


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
