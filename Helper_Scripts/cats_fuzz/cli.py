from __future__ import annotations

import argparse
import sys

# This wrapper invokes fixed local CLI argv with shell=False.
import subprocess  # nosec B404
from pathlib import Path
from collections.abc import Mapping

from Helper_Scripts.cats_fuzz.env import build_child_env, build_server_env
from Helper_Scripts.cats_fuzz.manifest import get_builtin_block
from Helper_Scripts.cats_fuzz.openapi_export import build_openapi_export_command
from Helper_Scripts.cats_fuzz.runner import run_contract_block, run_runtime_block
from Helper_Scripts.cats_fuzz.server import start_server, stop_server


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run local CATS OpenAPI fuzzing blocks.")
    parser.add_argument(
        "--block",
        action="append",
        choices=("contract", "public-read", "auth-read"),
        help="CATS harness block to run; repeat to select multiple blocks.",
    )
    parser.add_argument("--output", default="artifacts/cats-fuzz", help="Artifact output directory.")
    parser.add_argument("--cats-bin", default="cats", help="CATS executable to invoke.")
    parser.add_argument("--server-url", help="Existing server URL for runtime blocks.")
    parser.add_argument(
        "--start-server",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Start a local loopback uvicorn server for runtime blocks.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Pass dry-run mode to runtime CATS blocks.")
    parser.add_argument(
        "--allow-external",
        action="store_true",
        help=(
            "Permit parent env to contain real credentials; known sensitive values are "
            "still scrubbed from the child env."
        ),
    )
    args = parser.parse_args(argv)
    if args.block is None:
        args.block = ["contract", "public-read"]
    return args


def _first_output_line(value: str) -> str | None:
    for line in value.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped
    return None


def _cats_version_line(value: str) -> str | None:
    for line in value.splitlines():
        stripped = line.strip()
        if stripped.startswith("CATS version "):
            return stripped
    return None


def _cats_version(cats_bin: str) -> str:
    try:
        # Fixed argv, shell=False; cats_bin selects the local CATS executable.
        result = subprocess.run(  # nosec B603
            [cats_bin, "--version"],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return "unknown"

    if result.returncode != 0:
        return "unknown"
    return (
        _cats_version_line(result.stdout)
        or _cats_version_line(result.stderr)
        or _first_output_line(result.stdout)
        or _first_output_line(result.stderr)
        or "unknown"
    )


def _first_runtime_block(selected_blocks: list[str]) -> str | None:
    for block_name in selected_blocks:
        if block_name != "contract":
            return block_name
    return None


def _export_openapi_contract(contract_path: Path, child_env: Mapping[str, str], output_dir: Path) -> int:
    # Local helper argv, shell=False, executed with an isolated child env.
    result = subprocess.run(  # nosec B603
        build_openapi_export_command(contract_path),
        check=False,
        capture_output=True,
        text=True,
        env=dict(child_env),
    )
    stdout_path = output_dir / "openapi-export.stdout.log"
    stderr_path = output_dir / "openapi-export.stderr.log"
    stdout_path.write_text(result.stdout, encoding="utf-8")
    stderr_path.write_text(result.stderr, encoding="utf-8")
    if result.returncode != 0:
        print(f"OpenAPI export failed; see {stderr_path}", file=sys.stderr)
    return result.returncode


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    selected_blocks = list(args.block)
    first_runtime_block = _first_runtime_block(selected_blocks)
    needs_runtime = first_runtime_block is not None
    if needs_runtime and not args.start_server and not args.server_url:
        raise ValueError(f"{first_runtime_block} requires --server-url or --start-server")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    child_env = build_child_env(output_dir, allow_external=args.allow_external)
    contract_path = output_dir / "openapi.json"

    export_exit_code = _export_openapi_contract(contract_path, child_env, output_dir)
    if export_exit_code != 0:
        return export_exit_code

    cats_version = _cats_version(args.cats_bin)
    started_server = None
    exit_code = 0

    try:
        if needs_runtime and args.start_server:
            server_env = build_server_env(output_dir, child_env)
            started_server = start_server(server_env, log_dir=output_dir / "server")

        for block_name in selected_blocks:
            if block_name == "contract":
                summary = run_contract_block(
                    contract_path,
                    output_dir,
                    cats_version,
                    cats_bin=args.cats_bin,
                )
            else:
                server_url = started_server.url if started_server is not None else args.server_url
                summary = run_runtime_block(
                    get_builtin_block(block_name),
                    contract_path,
                    server_url,
                    output_dir,
                    cats_version,
                    cats_bin=args.cats_bin,
                    dry_run=args.dry_run,
                    env=child_env,
                )

            if exit_code == 0 and summary.exit_code != 0:
                exit_code = summary.exit_code
    finally:
        if started_server is not None:
            stop_server(started_server)

    return exit_code


__all__ = ["_cats_version", "_export_openapi_contract", "main", "parse_args"]
