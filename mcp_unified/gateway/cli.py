"""Command-line helpers for standalone MCP gateway configuration workflows."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, NoReturn, TextIO

from mcp_unified.profiles.presets import get_builtin_preset, list_builtin_presets

from .config import (
    GatewayConfigFormat,
    GatewayProfileBootstrapConfig,
    load_gateway_profile_bootstrap_config,
)


class _CliArgumentError(ValueError):
    """Raised when CLI arguments cannot be parsed into a command."""


class _CliHelpRequested(Exception):
    """Raised after argparse has printed human-readable help output."""

    def __init__(self, status: int) -> None:
        """Store the intended process exit status."""

        self.status = status


class _JsonArgumentParser(argparse.ArgumentParser):
    """Argument parser that lets `main()` format parse failures as JSON."""

    def error(self, message: str) -> NoReturn:
        """Raise a normal exception instead of printing usage and exiting."""

        raise _CliArgumentError(message)

    def exit(self, status: int = 0, message: str | None = None) -> NoReturn:
        """Raise exceptions instead of terminating the Python process."""

        if status == 0:
            if message:
                self._print_message(message, sys.stdout)
            raise _CliHelpRequested(status)

        raise _CliArgumentError(
            message.strip() if message else f"argument parsing failed with status {status}"
        )


def main(argv: Sequence[str] | None = None) -> int:
    """Run the standalone MCP gateway CLI and return a process exit code."""

    parser = _build_parser()
    try:
        args = parser.parse_args(argv)
    except _CliHelpRequested as exc:
        return exc.status
    except _CliArgumentError as exc:
        _emit_json(
            {
                "error": str(exc),
                "ok": False,
            },
            sys.stderr,
        )
        return 2
    return args.handler(args)


def _build_parser() -> _JsonArgumentParser:
    """Build the CLI argument parser."""

    parser = _JsonArgumentParser(
        prog="mcp-unified-gateway",
        description="Standalone MCP Unified gateway configuration utilities.",
    )
    subparsers = parser.add_subparsers(
        dest="command",
        parser_class=_JsonArgumentParser,
        required=True,
    )

    validate_config = subparsers.add_parser(
        "validate-config",
        help="Validate a gateway profile bootstrap config file.",
    )
    validate_config.add_argument(
        "path",
        type=Path,
        help="Path to a JSON or TOML gateway config file.",
    )
    validate_config.add_argument(
        "--format",
        choices=("json", "toml"),
        dest="config_format",
        help="Config format override when the file suffix is unavailable.",
    )
    validate_config.set_defaults(handler=_handle_validate_config)

    list_presets = subparsers.add_parser(
        "list-presets",
        help="List bundled MCP profile presets.",
    )
    list_presets.set_defaults(handler=_handle_list_presets)

    show_preset = subparsers.add_parser(
        "show-preset",
        help="Show one bundled MCP profile preset.",
    )
    show_preset.add_argument(
        "preset_id",
        help="Bundled preset id to inspect.",
    )
    show_preset.set_defaults(handler=_handle_show_preset)

    return parser


def _handle_validate_config(args: argparse.Namespace) -> int:
    """Validate one config file and emit a deterministic JSON result."""

    config_path: Path = args.path
    config_format: GatewayConfigFormat | str | None = args.config_format
    try:
        config = load_gateway_profile_bootstrap_config(
            config_path,
            format=config_format,
        )
    except Exception as exc:  # noqa: BLE001
        # CLI boundary: config loader failures should be machine-readable.
        _emit_json(
            {
                "error": str(exc),
                "ok": False,
                "path": str(config_path),
            },
            sys.stderr,
        )
        return 1

    _emit_json(_validated_config_payload(config, config_path), sys.stdout)
    return 0


def _handle_list_presets(_args: argparse.Namespace) -> int:
    """Emit bundled profile preset summaries as deterministic JSON."""

    presets = sorted(list_builtin_presets(), key=lambda preset: preset.id)
    _emit_json(
        {
            "presets": [
                {
                    "description": preset.profile.description,
                    "id": preset.id,
                    "name": preset.profile.name,
                    "version": preset.version,
                }
                for preset in presets
            ]
        },
        sys.stdout,
    )
    return 0


def _handle_show_preset(args: argparse.Namespace) -> int:
    """Emit one bundled profile preset with its full profile policy."""

    preset_id: str = args.preset_id
    preset = get_builtin_preset(preset_id)
    if preset is None:
        _emit_json(
            {
                "error": f"Unknown MCP profile preset: {preset_id}",
                "ok": False,
                "preset_id": preset_id,
            },
            sys.stderr,
        )
        return 1

    _emit_json(
        {
            "ok": True,
            "preset": preset.model_dump(mode="json"),
        },
        sys.stdout,
    )
    return 0


def _validated_config_payload(
    config: GatewayProfileBootstrapConfig,
    path: Path,
) -> dict[str, Any]:
    """Build the JSON payload for a successfully validated config."""

    sqlite_path = config.store.sqlite_path
    return {
        "default_preset_id": config.default_preset_id,
        "default_profile_id": config.default_profile_id,
        "ok": True,
        "path": str(path),
        "profiles": len(config.profiles),
        "store": {
            "kind": config.store.kind,
            "sqlite_path": str(sqlite_path) if sqlite_path is not None else None,
        },
    }


def _emit_json(payload: Mapping[str, Any], stream: TextIO) -> None:
    """Write one JSON object to the selected stream."""

    stream.write(json.dumps(payload, sort_keys=True))
    stream.write("\n")


if __name__ == "__main__":  # pragma: no cover - exercised through console script.
    raise SystemExit(main())
