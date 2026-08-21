"""CATS command builders, subprocess execution, and exit classification."""

from __future__ import annotations

import subprocess  # nosec B404
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from Helper_Scripts.cats_fuzz.manifest import CatsBlock


@dataclass(frozen=True)
class CatsProcessResult:
    """Captured result for one CATS or harness subprocess invocation."""

    command: list[str]
    exit_code: int
    stdout: str
    stderr: str


_TIMEOUT_EXIT_CODE = 124


def _normalize_timeout_output(value: str | bytes | None) -> str:
    """Convert partial timeout output to text for artifact persistence."""
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def _append_csv_option(command: list[str], option: str, values: Sequence[str]) -> None:
    """Append a comma-separated CATS option when values are configured."""
    if values:
        command.extend([option, ",".join(values)])


def build_cats_run_command(
    block: CatsBlock,
    contract_path: Path,
    server_url: str,
    output_dir: Path,
    api_key: str,
    cats_bin: str = "cats",
    dry_run: bool = False,
) -> list[str]:
    """Build a runtime CATS fuzzing command for one manifest block."""
    command = [
        cats_bin,
        "-c",
        str(contract_path),
        "-s",
        server_url,
        "--maskHeaders",
        "X-API-KEY,Authorization",
        "--skipReportingForIgnored",
        "--maxRequestsPerMinute",
        str(block.max_requests_per_minute),
        "--connectionTimeout",
        str(block.connection_timeout),
        "--readTimeout",
        str(block.read_timeout),
        "--writeTimeout",
        str(block.write_timeout),
        "--reportFormat",
        ",".join(block.report_formats),
        "--output",
        str(output_dir),
    ]
    if block.include_api_key:
        command[5:5] = ["-H", f"X-API-KEY={api_key}"]
    if block.blackbox:
        command.append("--blackbox")
    _append_csv_option(command, "--path", block.paths)
    _append_csv_option(command, "--tag", block.tags)
    _append_csv_option(command, "--skipPath", block.skip_paths)
    _append_csv_option(command, "--skipTag", block.skip_tags)
    _append_csv_option(command, "--skipHttpMethod", block.skip_methods)
    if dry_run:
        command.append("--dryRun")
    return command


def build_cats_validate_command(
    contract_path: Path,
    cats_bin: str = "cats",
    json_output: bool = True,
) -> list[str]:
    """Build a CATS OpenAPI validation command."""
    command = [cats_bin, "validate", "-c", str(contract_path)]
    if json_output:
        command.append("-j")
    return command


def build_cats_stats_command(
    contract_path: Path,
    cats_bin: str = "cats",
    json_output: bool = True,
) -> list[str]:
    """Build a CATS OpenAPI statistics command."""
    command = [cats_bin, "stats", "-c", str(contract_path)]
    if json_output:
        command.append("-j")
    return command


def classify_cats_exit(exit_code: int, stderr: str) -> str:
    """Classify a CATS exit as ok, usage, tool, or API behavior."""
    if exit_code == 0:
        return "ok"

    normalized_stderr = stderr.lower()
    usage_markers = (
        "invalid value",
        "unknown option",
        "missing required",
        "unmatched argument",
        "usage:",
    )
    if exit_code == 2 or any(marker in normalized_stderr for marker in usage_markers):
        return "usage"

    tool_markers = (
        "failed to execute command",
        "internal execution error",
        "exception",
        "stacktrace",
        "stack trace",
        "timed out",
        "timeout",
    )
    if exit_code == _TIMEOUT_EXIT_CODE or any(marker in normalized_stderr for marker in tool_markers):
        return "tool"

    return "api"


def run_command(
    command: Sequence[str],
    timeout_seconds: int,
    env: Mapping[str, str] | None = None,
) -> CatsProcessResult:
    """Run a CATS subprocess with timeout and structured failure capture."""
    # Command arguments are built by the local harness from controlled values.
    try:
        completed = subprocess.run(  # nosec B603
            list(command),
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            env=dict(env) if env is not None else None,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = _normalize_timeout_output(exc.output)
        partial_stderr = _normalize_timeout_output(exc.stderr)
        timeout_message = f"Command timed out after {timeout_seconds} seconds"
        stderr = timeout_message if not partial_stderr else f"{timeout_message}\n{partial_stderr}"
        return CatsProcessResult(
            command=list(command),
            exit_code=_TIMEOUT_EXIT_CODE,
            stdout=stdout,
            stderr=stderr,
        )
    except OSError as exc:
        return CatsProcessResult(
            command=list(command),
            exit_code=127,
            stdout="",
            stderr=f"Failed to execute command: {exc}",
        )
    return CatsProcessResult(
        command=list(command),
        exit_code=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
    )


__all__ = [
    "CatsProcessResult",
    "build_cats_run_command",
    "build_cats_stats_command",
    "build_cats_validate_command",
    "classify_cats_exit",
    "run_command",
]
