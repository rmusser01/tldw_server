from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from Helper_Scripts.cats_fuzz.summary import CatsRunSummary


@pytest.mark.unit
def test_cli_defaults_to_contract_and_public_read() -> None:
    from Helper_Scripts.cats_fuzz.cli import parse_args

    args = parse_args([])

    assert args.block == ["contract", "public-read"]
    assert args.output == "artifacts/cats-fuzz"
    assert args.start_server is True


@pytest.mark.unit
def test_cli_accepts_existing_server_url() -> None:
    from Helper_Scripts.cats_fuzz.cli import parse_args

    args = parse_args(["--server-url", "http://127.0.0.1:8000", "--no-start-server"])

    assert args.server_url == "http://127.0.0.1:8000"
    assert args.start_server is False


@pytest.mark.unit
def test_cli_server_url_without_start_flag_uses_existing_server() -> None:
    from Helper_Scripts.cats_fuzz.cli import parse_args

    args = parse_args(["--server-url", "http://127.0.0.1:8000"])

    assert args.server_url == "http://127.0.0.1:8000"
    assert args.start_server is False


@pytest.mark.unit
@pytest.mark.parametrize(
    "server_url",
    [
        "http://127.0.0.1:8000/prefix",
        "http://127.0.0.1:8000?debug=1",
        "http://127.0.0.1:8000#fragment",
    ],
)
def test_cli_rejects_existing_server_url_with_path_query_or_fragment(
    server_url: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from Helper_Scripts.cats_fuzz.cli import parse_args

    with pytest.raises(SystemExit) as exc_info:
        parse_args(["--block", "public-read", "--server-url", server_url])

    assert exc_info.value.code == 2
    assert "must be an origin URL without a path, query, or fragment" in capsys.readouterr().err


def _summary(block: str, exit_code: int) -> CatsRunSummary:
    return CatsRunSummary(
        block=block,
        cats_version="cats 13.8.0",
        openapi_sha256="abc123",
        command=["cats"],
        masked_command=["cats"],
        exit_code=exit_code,
        failure_class="ok" if exit_code == 0 else "api",
        stdout_path=f"{block}/stdout.log",
        stderr_path=f"{block}/stderr.log",
        report_dir=f"{block}/cats-report",
    )


@pytest.mark.unit
def test_contract_only_run_exports_openapi_and_returns_contract_exit_code(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from Helper_Scripts.cats_fuzz import cli

    calls: list[tuple[str, object]] = []

    def fake_run(
        command: list[str],
        *,
        check: bool = False,
        capture_output: bool = False,
        text: bool = False,
        timeout: int | None = None,
        env: dict[str, str] | None = None,
    ) -> object:
        calls.append(("subprocess", (command, check, capture_output, text, timeout, env)))
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    def fail_start_server(*args: object, **kwargs: object) -> None:
        raise AssertionError("contract-only run should not start a server")

    def fake_run_contract_block(
        contract_path: Path,
        output_dir: Path,
        cats_version: str,
        *,
        cats_bin: str,
    ) -> CatsRunSummary:
        calls.append(("contract", (contract_path, output_dir, cats_version, cats_bin)))
        return _summary("contract", 7)

    monkeypatch.setattr(cli, "build_child_env", lambda output_dir, allow_external=False: {"SAFE": "1"})
    monkeypatch.setattr(cli, "build_openapi_export_command", lambda path: ["python", "export", str(path)])
    monkeypatch.setattr(cli.subprocess, "run", fake_run)
    monkeypatch.setattr(cli, "_cats_version", lambda cats_bin: "cats 13.8.0")
    monkeypatch.setattr(cli, "start_server", fail_start_server)
    monkeypatch.setattr(cli, "run_contract_block", fake_run_contract_block)

    result = cli.main(["--block", "contract", "--output", str(tmp_path)])

    assert result == 7
    assert calls[0] == (
        "subprocess",
        (["python", "export", str(tmp_path / "openapi.json")], False, True, True, 120, {"SAFE": "1"}),
    )
    assert calls[1] == ("contract", (tmp_path / "openapi.json", tmp_path, "cats 13.8.0", "cats"))


@pytest.mark.unit
def test_runtime_default_blocks_start_server_use_started_url_and_stop_finally(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from Helper_Scripts.cats_fuzz import cli

    events: list[tuple[str, object]] = []
    server = SimpleNamespace(url="http://127.0.0.1:9123")
    child_env = {
        "SAFE": "1",
        "TEST_MODE": "true",
        "TLDW_ENV_FILE": str(tmp_path / "runtime" / ".env"),
    }
    server_env = {
        "SAFE": "1",
        "TLDW_ENV_FILE": str(tmp_path / "runtime" / "cats-server.env"),
    }

    def fake_run(
        command: list[str],
        *,
        check: bool = False,
        capture_output: bool = False,
        text: bool = False,
        timeout: int | None = None,
        env: dict[str, str] | None = None,
    ) -> object:
        events.append(("export", (command, check, timeout, env)))
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    def fake_start_server(env: dict[str, str], *, log_dir: Path) -> SimpleNamespace:
        events.append(("start", (env, log_dir)))
        return server

    def fake_stop_server(started_server: SimpleNamespace) -> None:
        events.append(("stop", started_server))

    def fake_run_contract_block(
        contract_path: Path,
        output_dir: Path,
        cats_version: str,
        *,
        cats_bin: str,
    ) -> CatsRunSummary:
        events.append(("contract", (contract_path, output_dir, cats_version, cats_bin)))
        return _summary("contract", 0)

    def fake_run_runtime_block(
        block: object,
        contract_path: Path,
        server_url: str,
        output_dir: Path,
        cats_version: str,
        *,
        cats_bin: str,
        dry_run: bool,
        env: dict[str, str],
    ) -> CatsRunSummary:
        events.append(
            (
                "runtime",
                (getattr(block, "name"), contract_path, server_url, output_dir, cats_version, cats_bin, dry_run, env),
            )
        )
        return _summary(getattr(block, "name"), 5)

    def fake_build_server_env(output_dir: Path, env: dict[str, str]) -> dict[str, str]:
        assert output_dir == tmp_path
        assert env is child_env
        return server_env

    monkeypatch.setattr(cli, "build_child_env", lambda output_dir, allow_external=False: child_env)
    monkeypatch.setattr(cli, "build_server_env", fake_build_server_env)
    monkeypatch.setattr(cli, "build_openapi_export_command", lambda path: ["python", "export", str(path)])
    monkeypatch.setattr(cli.subprocess, "run", fake_run)
    monkeypatch.setattr(cli, "_cats_version", lambda cats_bin: "cats 13.8.0")
    monkeypatch.setattr(cli, "start_server", fake_start_server)
    monkeypatch.setattr(cli, "stop_server", fake_stop_server)
    monkeypatch.setattr(cli, "run_contract_block", fake_run_contract_block)
    monkeypatch.setattr(cli, "run_runtime_block", fake_run_runtime_block)

    result = cli.main(["--output", str(tmp_path)])

    assert result == 5
    assert events[0] == (
        "export",
        (["python", "export", str(tmp_path / "openapi.json")], False, 120, child_env),
    )
    assert events[1] == ("start", (server_env, tmp_path / "server"))
    started_env = events[1][1][0]
    assert started_env is server_env
    assert started_env.get("TEST_MODE", "") == ""
    assert started_env["TLDW_ENV_FILE"] != child_env["TLDW_ENV_FILE"]
    assert events[2][0] == "contract"
    assert events[3] == (
        "runtime",
        (
            "public-read",
            tmp_path / "openapi.json",
            "http://127.0.0.1:9123",
            tmp_path,
            "cats 13.8.0",
            "cats",
            False,
            child_env,
        ),
    )
    assert events[-1] == ("stop", server)


@pytest.mark.unit
def test_runtime_run_with_existing_server_url_does_not_start_server(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from Helper_Scripts.cats_fuzz import cli

    runtime_calls: list[tuple[str, str]] = []

    def fail_start_server(*args: object, **kwargs: object) -> None:
        raise AssertionError("existing server URL should skip server startup")

    def fake_run_runtime_block(
        block: object,
        contract_path: Path,
        server_url: str,
        output_dir: Path,
        cats_version: str,
        *,
        cats_bin: str,
        dry_run: bool,
        env: dict[str, str],
    ) -> CatsRunSummary:
        runtime_calls.append((getattr(block, "name"), server_url))
        return _summary(getattr(block, "name"), 0)

    monkeypatch.setattr(cli, "build_child_env", lambda output_dir, allow_external=False: {"SAFE": "1"})
    monkeypatch.setattr(cli, "build_openapi_export_command", lambda path: ["python", "export", str(path)])
    monkeypatch.setattr(
        cli.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout="", stderr="")
    )
    monkeypatch.setattr(cli, "_cats_version", lambda cats_bin: "cats 13.8.0")
    monkeypatch.setattr(cli, "start_server", fail_start_server)
    monkeypatch.setattr(cli, "run_runtime_block", fake_run_runtime_block)

    result = cli.main(
        [
            "--block",
            "public-read",
            "--server-url",
            "http://127.0.0.1:8000",
            "--no-start-server",
            "--output",
            str(tmp_path),
        ]
    )

    assert result == 0
    assert runtime_calls == [("public-read", "http://127.0.0.1:8000")]


@pytest.mark.unit
def test_runtime_run_with_server_url_only_does_not_start_server(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from Helper_Scripts.cats_fuzz import cli

    runtime_calls: list[tuple[str, str]] = []

    def fail_start_server(*args: object, **kwargs: object) -> None:
        raise AssertionError("server URL should skip server startup unless --start-server is explicit")

    def fake_run_runtime_block(
        block: object,
        contract_path: Path,
        server_url: str,
        output_dir: Path,
        cats_version: str,
        *,
        cats_bin: str,
        dry_run: bool,
        env: dict[str, str],
    ) -> CatsRunSummary:
        runtime_calls.append((getattr(block, "name"), server_url))
        return _summary(getattr(block, "name"), 0)

    monkeypatch.setattr(cli, "build_child_env", lambda output_dir, allow_external=False: {"SAFE": "1"})
    monkeypatch.setattr(cli, "build_openapi_export_command", lambda path: ["python", "export", str(path)])
    monkeypatch.setattr(
        cli.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout="", stderr="")
    )
    monkeypatch.setattr(cli, "_cats_version", lambda cats_bin: "cats 13.8.0")
    monkeypatch.setattr(cli, "start_server", fail_start_server)
    monkeypatch.setattr(cli, "run_runtime_block", fake_run_runtime_block)

    result = cli.main(
        [
            "--block",
            "public-read",
            "--server-url",
            "http://127.0.0.1:8000",
            "--output",
            str(tmp_path),
        ]
    )

    assert result == 0
    assert runtime_calls == [("public-read", "http://127.0.0.1:8000")]


@pytest.mark.unit
def test_runtime_without_server_url_and_no_start_server_raises_before_env_or_export(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from Helper_Scripts.cats_fuzz import cli

    def fail_build_child_env(*args: object, **kwargs: object) -> None:
        raise AssertionError("child env should not be built for invalid runtime invocation")

    def fail_run(*args: object, **kwargs: object) -> None:
        raise AssertionError("OpenAPI export should not run for invalid runtime invocation")

    monkeypatch.setattr(cli, "build_child_env", fail_build_child_env)
    monkeypatch.setattr(cli.subprocess, "run", fail_run)

    with pytest.raises(SystemExit) as exc_info:
        cli.main(["--block", "public-read", "--no-start-server", "--output", str(tmp_path)])
    assert exc_info.value.code == 2
    assert "public-read requires --server-url or --start-server" in capsys.readouterr().err


@pytest.mark.unit
def test_runtime_rejects_non_loopback_existing_server_url_before_env_or_export(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from Helper_Scripts.cats_fuzz import cli

    def fail_build_child_env(*args: object, **kwargs: object) -> None:
        raise AssertionError("child env should not be built for remote runtime invocation")

    def fail_run(*args: object, **kwargs: object) -> None:
        raise AssertionError("OpenAPI export should not run for remote runtime invocation")

    monkeypatch.setattr(cli, "build_child_env", fail_build_child_env)
    monkeypatch.setattr(cli.subprocess, "run", fail_run)

    with pytest.raises(SystemExit) as exc_info:
        cli.main(
            [
                "--block",
                "public-read",
                "--server-url",
                "https://prod.example",
                "--no-start-server",
                "--output",
                str(tmp_path),
            ]
        )
    assert exc_info.value.code == 2
    assert "public-read only allows loopback --server-url values" in capsys.readouterr().err


@pytest.mark.unit
def test_openapi_export_failure_writes_logs_and_stops_before_blocks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from Helper_Scripts.cats_fuzz import cli

    errors: list[str] = []

    def fake_run(
        command: list[str],
        *,
        check: bool = False,
        capture_output: bool = False,
        text: bool = False,
        timeout: int | None = None,
        env: dict[str, str] | None = None,
    ) -> subprocess.CompletedProcess[str]:
        assert command == ["python", "export", str(tmp_path / "openapi.json")]
        assert check is False
        assert capture_output is True
        assert text is True
        assert timeout == 120
        assert env == {"SAFE": "1"}
        return subprocess.CompletedProcess(command, 9, stdout="export out", stderr="export err")

    def fail_start_or_run(*args: object, **kwargs: object) -> None:
        raise AssertionError("server and CATS blocks should not run after export failure")

    monkeypatch.setattr(cli, "build_child_env", lambda output_dir, allow_external=False: {"SAFE": "1"})
    monkeypatch.setattr(cli, "build_openapi_export_command", lambda path: ["python", "export", str(path)])
    monkeypatch.setattr(cli.subprocess, "run", fake_run)
    monkeypatch.setattr(
        cli, "logger", SimpleNamespace(error=lambda message, *args: errors.append(message.format(*args)))
    )
    monkeypatch.setattr(cli, "start_server", fail_start_or_run)
    monkeypatch.setattr(cli, "run_contract_block", fail_start_or_run)
    monkeypatch.setattr(cli, "run_runtime_block", fail_start_or_run)

    result = cli.main(["--output", str(tmp_path)])

    assert result == 9
    assert (tmp_path / "openapi-export.stdout.log").read_text(encoding="utf-8") == "export out"
    assert (tmp_path / "openapi-export.stderr.log").read_text(encoding="utf-8") == "export err"
    assert errors == [f"OpenAPI export failed; see {tmp_path / 'openapi-export.stderr.log'}"]


@pytest.mark.unit
def test_openapi_export_timeout_writes_partial_logs_and_returns_timeout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from Helper_Scripts.cats_fuzz import cli

    errors: list[str] = []

    def fake_run(
        command: list[str],
        *,
        check: bool = False,
        capture_output: bool = False,
        text: bool = False,
        timeout: int | None = None,
        env: dict[str, str] | None = None,
    ) -> subprocess.CompletedProcess[str]:
        raise subprocess.TimeoutExpired(command, timeout or 0, output="partial stdout", stderr=b"partial stderr")

    monkeypatch.setattr(cli, "build_openapi_export_command", lambda path: ["python", "export", str(path)])
    monkeypatch.setattr(cli.subprocess, "run", fake_run)
    monkeypatch.setattr(
        cli, "logger", SimpleNamespace(error=lambda message, *args: errors.append(message.format(*args)))
    )

    result = cli._export_openapi_contract(tmp_path / "openapi.json", {"SAFE": "1"}, tmp_path)

    assert result == 124
    assert (tmp_path / "openapi-export.stdout.log").read_text(encoding="utf-8") == "partial stdout"
    assert (tmp_path / "openapi-export.stderr.log").read_text(encoding="utf-8") == "partial stderr"
    assert errors == [f"OpenAPI export timed out after 120 seconds; see {tmp_path / 'openapi-export.stderr.log'}"]


@pytest.mark.unit
def test_cats_version_uses_first_output_line(monkeypatch: pytest.MonkeyPatch) -> None:
    from Helper_Scripts.cats_fuzz import cli

    cats_banner_output = """# # # # # # # # # # # # # # # # # # # # # # # # # #
#             _____   ___ _____ _____             #
# # # # # # # # # # # # # # # # # # # # # # # # # #

CATS version 13.8.0
Built on: 2026-04-01 17:42:27 UTC
"""

    def fake_run(
        command: list[str],
        *,
        check: bool = False,
        capture_output: bool = False,
        text: bool = False,
    ) -> subprocess.CompletedProcess[str]:
        assert command == ["cats", "--version"]
        assert check is False
        assert capture_output is True
        assert text is True
        return subprocess.CompletedProcess(command, 0, stdout=cats_banner_output, stderr="")

    monkeypatch.setattr(cli.subprocess, "run", fake_run)

    assert cli._cats_version("cats") == "CATS version 13.8.0"


@pytest.mark.unit
def test_cats_version_falls_back_to_unknown(monkeypatch: pytest.MonkeyPatch) -> None:
    from Helper_Scripts.cats_fuzz import cli

    def fake_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(["cats", "--version"], 127, stdout="", stderr="cats error")

    monkeypatch.setattr(cli.subprocess, "run", fake_run)

    assert cli._cats_version("cats") == "unknown"
