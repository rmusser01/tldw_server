from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from Helper_Scripts.cats_fuzz import DEFAULT_TEST_API_KEY
from Helper_Scripts.cats_fuzz.cats_cli import (
    build_cats_run_command,
    build_cats_stats_command,
    build_cats_validate_command,
    classify_cats_exit,
    run_command,
)
from Helper_Scripts.cats_fuzz.manifest import get_builtin_block


@pytest.mark.unit
def test_public_read_command_uses_blackbox_and_junit_reports(tmp_path: Path) -> None:
    block = get_builtin_block("public-read")
    command = build_cats_run_command(
        block,
        contract_path=tmp_path / "openapi.json",
        server_url="http://127.0.0.1:8000",
        output_dir=tmp_path / "reports",
        api_key=DEFAULT_TEST_API_KEY,
    )

    assert "--blackbox" in command
    assert "--skipReportingForIgnored" in command
    assert "--reportFormat" in command
    assert command[command.index("--reportFormat") + 1] == "HTML_ONLY,JUNIT"
    assert "-H" not in command
    assert f"X-API-KEY={DEFAULT_TEST_API_KEY}" not in command
    assert "--path" in command
    assert "/" in command[command.index("--path") + 1].split(",")


@pytest.mark.unit
def test_auth_read_command_includes_api_key_header(tmp_path: Path) -> None:
    block = get_builtin_block("auth-read")
    command = build_cats_run_command(
        block,
        contract_path=tmp_path / "openapi.json",
        server_url="http://127.0.0.1:8000",
        output_dir=tmp_path / "reports",
        api_key=DEFAULT_TEST_API_KEY,
    )

    assert "-H" in command
    assert f"X-API-KEY={DEFAULT_TEST_API_KEY}" in command
    assert command.index("-H") < command.index("--maskHeaders")


@pytest.mark.unit
def test_run_command_uses_top_level_cats_fuzz_command(tmp_path: Path) -> None:
    block = get_builtin_block("public-read")
    contract_path = tmp_path / "openapi.json"
    command = build_cats_run_command(
        block,
        contract_path=contract_path,
        server_url="http://127.0.0.1:8000",
        output_dir=tmp_path / "reports",
        api_key=DEFAULT_TEST_API_KEY,
    )

    assert command[:5] == [
        "cats",
        "-c",
        str(contract_path),
        "-s",
        "http://127.0.0.1:8000",
    ]
    assert command[1] != "run"


@pytest.mark.unit
def test_dry_run_stays_on_top_level_cats_fuzz_command(tmp_path: Path) -> None:
    block = get_builtin_block("public-read")
    contract_path = tmp_path / "openapi.json"
    command = build_cats_run_command(
        block,
        contract_path=contract_path,
        server_url="http://127.0.0.1:8000",
        output_dir=tmp_path / "reports",
        api_key=DEFAULT_TEST_API_KEY,
        dry_run=True,
    )

    assert command[:5] == [
        "cats",
        "-c",
        str(contract_path),
        "-s",
        "http://127.0.0.1:8000",
    ]
    assert "--dryRun" in command


@pytest.mark.unit
def test_cats_exit_classification_separates_usage_tool_and_api_failures() -> None:
    assert classify_cats_exit(0, "") == "ok"
    assert classify_cats_exit(2, "Invalid value for option") == "usage"
    assert classify_cats_exit(1, "Internal execution error") == "tool"
    assert classify_cats_exit(124, "Command timed out after 10 seconds") == "tool"
    assert classify_cats_exit(1, "Some tests failed with 500") == "api"


@pytest.mark.unit
def test_validate_and_stats_commands_default_to_json_output(tmp_path: Path) -> None:
    contract_path = tmp_path / "openapi.json"

    assert build_cats_validate_command(contract_path) == [
        "cats",
        "validate",
        "-c",
        str(contract_path),
        "-j",
    ]
    assert build_cats_stats_command(contract_path) == [
        "cats",
        "stats",
        "-c",
        str(contract_path),
        "-j",
    ]


@pytest.mark.unit
def test_run_command_captures_stdout_stderr_and_exit_code() -> None:
    command = [
        sys.executable,
        "-c",
        "import sys; print('out'); print('err', file=sys.stderr); raise SystemExit(3)",
    ]

    result = run_command(command, timeout_seconds=10)

    assert result.command == command
    assert result.exit_code == 3
    assert result.stdout == "out\n"
    assert result.stderr == "err\n"


@pytest.mark.unit
def test_run_command_returns_structured_result_on_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    command = ["cats", "-c", "openapi.json"]

    def raise_timeout(*_args: object, **kwargs: object) -> None:
        raise subprocess.TimeoutExpired(
            cmd=command,
            timeout=kwargs["timeout"],
            output=b"partial stdout",
            stderr=b"partial stderr",
        )

    monkeypatch.setattr("Helper_Scripts.cats_fuzz.cats_cli.subprocess.run", raise_timeout)

    result = run_command(command, timeout_seconds=10)

    assert result.command == command
    assert result.exit_code == 124
    assert result.stdout == "partial stdout"
    assert "Command timed out after 10 seconds" in result.stderr
    assert "partial stderr" in result.stderr
    assert classify_cats_exit(result.exit_code, result.stderr) == "tool"


@pytest.mark.unit
def test_run_command_returns_structured_result_when_binary_is_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    command = ["missing-cats", "-c", "openapi.json"]

    def raise_missing_binary(*_args: object, **_kwargs: object) -> None:
        raise FileNotFoundError("missing-cats")

    monkeypatch.setattr("Helper_Scripts.cats_fuzz.cats_cli.subprocess.run", raise_missing_binary)

    result = run_command(command, timeout_seconds=10)

    assert result.command == command
    assert result.exit_code == 127
    assert result.stdout == ""
    assert "Failed to execute command" in result.stderr
    assert "missing-cats" in result.stderr
    assert classify_cats_exit(result.exit_code, result.stderr) == "tool"
