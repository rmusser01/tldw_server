from __future__ import annotations

import hashlib
from collections.abc import Mapping
from pathlib import Path

from Helper_Scripts.cats_fuzz import DEFAULT_TEST_API_KEY
from Helper_Scripts.cats_fuzz.cats_cli import (
    CatsProcessResult,
    build_cats_run_command,
    build_cats_stats_command,
    build_cats_validate_command,
    classify_cats_exit,
    run_command,
)
from Helper_Scripts.cats_fuzz.manifest import CatsBlock, get_builtin_block
from Helper_Scripts.cats_fuzz.server import wait_for_readiness
from Helper_Scripts.cats_fuzz.summary import CatsRunSummary, mask_command, write_summary


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_process_artifacts(result: CatsProcessResult, output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = output_dir / "stdout.log"
    stderr_path = output_dir / "stderr.log"
    stdout_path.write_text(result.stdout, encoding="utf-8")
    stderr_path.write_text(result.stderr, encoding="utf-8")
    return stdout_path, stderr_path


def _summarize(
    *,
    block_name: str,
    cats_version: str,
    openapi_sha256: str,
    result: CatsProcessResult,
    output_dir: Path,
    report_dir: Path,
) -> CatsRunSummary:
    stdout_path, stderr_path = _write_process_artifacts(result, output_dir)
    masked_command = mask_command(result.command)
    summary = CatsRunSummary(
        block=block_name,
        cats_version=cats_version,
        openapi_sha256=openapi_sha256,
        command=masked_command,
        masked_command=masked_command,
        exit_code=result.exit_code,
        failure_class=classify_cats_exit(result.exit_code, result.stderr),
        stdout_path=str(stdout_path),
        stderr_path=str(stderr_path),
        report_dir=str(report_dir),
    )
    write_summary(summary, output_dir / "summary.json")
    return summary


def _merge_contract_results(
    validate_result: CatsProcessResult,
    stats_result: CatsProcessResult,
) -> CatsProcessResult:
    exit_code = validate_result.exit_code or stats_result.exit_code
    return CatsProcessResult(
        command=validate_result.command + [";"] + stats_result.command,
        exit_code=exit_code,
        stdout="\n".join(part for part in (validate_result.stdout, stats_result.stdout) if part),
        stderr="\n".join(part for part in (validate_result.stderr, stats_result.stderr) if part),
    )


def run_contract_block(
    contract_path: Path,
    output_dir: Path,
    cats_version: str,
    openapi_sha256: str | None = None,
    cats_bin: str = "cats",
) -> CatsRunSummary:
    validate_result = run_command(
        build_cats_validate_command(contract_path, cats_bin=cats_bin),
        timeout_seconds=60,
    )
    stats_result = run_command(
        build_cats_stats_command(contract_path, cats_bin=cats_bin),
        timeout_seconds=60,
    )
    result = _merge_contract_results(validate_result, stats_result)
    block_dir = output_dir / "contract"
    return _summarize(
        block_name="contract",
        cats_version=cats_version,
        openapi_sha256=openapi_sha256 or _sha256(contract_path),
        result=result,
        output_dir=block_dir,
        report_dir=block_dir,
    )


def run_runtime_block(
    block: CatsBlock,
    contract_path: Path,
    server_url: str,
    output_dir: Path,
    cats_version: str,
    api_key: str = DEFAULT_TEST_API_KEY,
    cats_bin: str = "cats",
    dry_run: bool = False,
    env: Mapping[str, str] | None = None,
) -> CatsRunSummary:
    if block.requires_readiness:
        wait_for_readiness(server_url)

    block_dir = output_dir / block.name
    report_dir = block_dir / "cats-report"
    command = build_cats_run_command(
        block=block,
        contract_path=contract_path,
        server_url=server_url,
        output_dir=report_dir,
        api_key=api_key,
        cats_bin=cats_bin,
        dry_run=dry_run,
    )
    result = run_command(command, timeout_seconds=block.timeout_seconds, env=env)
    return _summarize(
        block_name=block.name,
        cats_version=cats_version,
        openapi_sha256=_sha256(contract_path),
        result=result,
        output_dir=block_dir,
        report_dir=report_dir,
    )


def get_default_runtime_block() -> CatsBlock:
    return get_builtin_block("public-read")


__all__ = [
    "get_default_runtime_block",
    "run_contract_block",
    "run_runtime_block",
]
