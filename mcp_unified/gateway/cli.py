"""Command-line helpers for standalone MCP gateway configuration workflows."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from collections.abc import Callable, Coroutine, Mapping, Sequence
from pathlib import Path
from typing import Any, NoReturn, TextIO

from mcp_unified.profiles.presets import (
    duplicate_builtin_preset,
    get_builtin_preset,
    list_builtin_presets,
)

from .config import (
    GatewayConfigFormat,
    GatewayProfileBootstrapConfig,
    GatewayProfileStorageBundle,
    build_gateway_profile_storage,
    load_gateway_profile_bootstrap_config,
)
from .profiles import GatewayProfileManagementError, GatewayProfileManager

_ProfileOperation = Callable[
    [GatewayProfileManager],
    Coroutine[Any, Any, dict[str, Any]],
]


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
        return args.handler(args)
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

    list_profiles = subparsers.add_parser(
        "list-profiles",
        help="List profiles from a configured gateway profile store.",
    )
    _add_profile_config_argument(list_profiles)
    list_profiles.set_defaults(handler=_handle_list_profiles)

    show_profile = subparsers.add_parser(
        "show-profile",
        help="Show one profile from a configured gateway profile store.",
    )
    show_profile.add_argument(
        "profile_id",
        help="Profile id to inspect.",
    )
    _add_profile_config_argument(show_profile)
    show_profile.set_defaults(handler=_handle_show_profile)

    duplicate_preset = subparsers.add_parser(
        "duplicate-preset",
        help="Duplicate a bundled preset into a persistent gateway profile store.",
    )
    duplicate_preset.add_argument(
        "preset_id",
        help="Bundled preset id to duplicate.",
    )
    duplicate_preset.add_argument(
        "--profile-id",
        help="Optional stored profile id for the duplicate.",
    )
    duplicate_preset.add_argument(
        "--name",
        help="Optional stored profile display name for the duplicate.",
    )
    _add_profile_config_argument(duplicate_preset)
    duplicate_preset.set_defaults(handler=_handle_duplicate_preset)

    get_default_profile = subparsers.add_parser(
        "get-default-profile",
        help="Show the active gateway default profile.",
    )
    _add_profile_config_argument(get_default_profile)
    get_default_profile.set_defaults(handler=_handle_get_default_profile)

    set_default_profile = subparsers.add_parser(
        "set-default-profile",
        help="Persist the gateway default profile assignment.",
    )
    set_default_profile.add_argument(
        "profile_id",
        help="Profile id to make the gateway default.",
    )
    _add_profile_config_argument(set_default_profile)
    set_default_profile.set_defaults(handler=_handle_set_default_profile)

    return parser


def _add_profile_config_argument(parser: argparse.ArgumentParser) -> None:
    """Add the common config option for profile-management commands."""

    parser.add_argument(
        "--config",
        type=Path,
        help=(
            "Path to a JSON or TOML gateway config file. "
            "Falls back to MCP_UNIFIED_GATEWAY_CONFIG."
        ),
    )


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


def _handle_list_profiles(args: argparse.Namespace) -> int:
    """List profiles from a configured gateway profile store."""

    return _handle_profile_management_command(
        args,
        lambda manager: manager.list_profiles(),
        seed_memory_readonly=True,
    )


def _handle_show_profile(args: argparse.Namespace) -> int:
    """Show one profile from a configured gateway profile store."""

    profile_id = _require_cli_text(args.profile_id, field="profile_id")
    return _handle_profile_management_command(
        args,
        lambda manager: manager.show_profile(profile_id),
        seed_memory_readonly=True,
    )


def _handle_duplicate_preset(args: argparse.Namespace) -> int:
    """Duplicate a bundled preset into a persistent gateway profile store."""

    preset_id = _require_cli_text(args.preset_id, field="preset_id")
    profile_id = _optional_cli_text(args.profile_id, field="profile_id")
    name = _optional_cli_text(args.name, field="name")
    return _handle_profile_management_command(
        args,
        lambda manager: _duplicate_preset_for_cli(
            manager,
            preset_id,
            profile_id=profile_id,
            name=name,
        ),
        require_persistent=True,
    )


def _handle_get_default_profile(args: argparse.Namespace) -> int:
    """Show the active gateway default profile."""

    return _handle_profile_management_command(
        args,
        lambda manager: manager.get_default_profile(),
        seed_memory_readonly=True,
    )


def _handle_set_default_profile(args: argparse.Namespace) -> int:
    """Persist the gateway default profile assignment."""

    profile_id = _require_cli_text(args.profile_id, field="profile_id")
    return _handle_profile_management_command(
        args,
        lambda manager: _set_default_profile_for_cli(manager, profile_id),
        require_persistent=True,
    )


def _handle_profile_management_command(
    args: argparse.Namespace,
    operation: _ProfileOperation,
    *,
    seed_memory_readonly: bool = False,
    require_persistent: bool = False,
) -> int:
    """Run a profile-management command against the configured profile store."""

    config_path = _config_path_from_args(args)
    bundle: GatewayProfileStorageBundle | None = None
    try:
        config = load_gateway_profile_bootstrap_config(config_path)
        bundle = build_gateway_profile_storage(config.store)
        if seed_memory_readonly and not bundle.metadata.persistent:
            _run_async(_seed_readonly_memory_store(bundle, config))
        manager = _manager_from_bundle(
            bundle,
            fallback_default_profile_id=_memory_fallback_default_profile_id(
                config,
                bundle,
                seed_memory_readonly=seed_memory_readonly,
            ),
        )
    except Exception as exc:  # noqa: BLE001
        # CLI boundary: config, storage construction, and read-only memory
        # seeding failures should be machine-readable.
        _emit_json(
            {
                "error": str(exc),
                "ok": False,
                "path": str(config_path),
            },
            sys.stderr,
        )
        if bundle is not None:
            _run_async(_close_storage_bundle(bundle))
        return 1

    try:
        if require_persistent and not bundle.metadata.persistent:
            raise GatewayProfileManagementError(
                "Profile management mutation requires a persistent gateway store",
                reason_code="profile_store_unavailable",
            )
        payload = _run_async(operation(manager))
    except GatewayProfileManagementError as exc:
        _emit_json(exc.to_payload(), sys.stderr)
        return 1
    finally:
        _run_async(_close_storage_bundle(bundle))

    _emit_json(_cli_payload(payload), sys.stdout)
    return 0


def _config_path_from_args(args: argparse.Namespace) -> Path:
    """Return the explicit or environment-provided gateway config path."""

    if args.config is not None:
        return args.config
    env_value = os.environ.get("MCP_UNIFIED_GATEWAY_CONFIG")
    if env_value and env_value.strip():
        return Path(env_value)
    raise _CliArgumentError(
        "--config is required unless MCP_UNIFIED_GATEWAY_CONFIG is set"
    )


def _manager_from_bundle(
    bundle: GatewayProfileStorageBundle,
    *,
    fallback_default_profile_id: str | None,
) -> GatewayProfileManager:
    """Build a profile manager around already-resolved gateway stores."""

    return GatewayProfileManager(
        profile_store=bundle.profile_store,
        assignment_store=bundle.assignment_store,
        audit_store=bundle.audit_store,
        store_metadata=bundle.metadata,
        fallback_default_profile_id=fallback_default_profile_id,
    )


async def _duplicate_preset_for_cli(
    manager: GatewayProfileManager,
    preset_id: str,
    *,
    profile_id: str | None = None,
    name: str | None = None,
) -> dict[str, Any]:
    """Duplicate a preset and expose CLI-level preset metadata fields."""

    payload = await manager.duplicate_preset(
        preset_id,
        profile_id=profile_id,
        name=name,
    )
    profile = payload.get("profile")
    if isinstance(profile, Mapping):
        payload = dict(payload)
        if profile.get("preset_id") is not None:
            payload["preset_id"] = profile["preset_id"]
        if profile.get("preset_version") is not None:
            payload["preset_version"] = profile["preset_version"]
    return payload


async def _set_default_profile_for_cli(
    manager: GatewayProfileManager,
    profile_id: str,
) -> dict[str, Any]:
    """Set the default profile and return the compact CLI write envelope."""

    payload = await manager.set_default_profile(profile_id)
    assignment = payload.get("assignment")
    assigned_profile_id = profile_id
    if isinstance(assignment, Mapping) and isinstance(assignment.get("profile_id"), str):
        assigned_profile_id = assignment["profile_id"]
    return {
        "ok": payload["ok"],
        "profile_id": assigned_profile_id,
        "assignment": assignment,
        "store": payload["store"],
    }


async def _seed_readonly_memory_store(
    bundle: GatewayProfileStorageBundle,
    config: GatewayProfileBootstrapConfig,
) -> dict[str, Any]:
    """Seed config-defined profiles into the transient memory store for reads."""

    for profile in config.profiles:
        await bundle.profile_store.upsert_profile(profile)

    if config.default_preset_id is not None:
        if await bundle.profile_store.get_profile(config.default_preset_id) is not None:
            raise ValueError(
                f"Cannot seed MCP profile preset '{config.default_preset_id}': "
                f"profile id '{config.default_preset_id}' already exists"
            )
        await bundle.profile_store.upsert_profile(
            duplicate_builtin_preset(
                config.default_preset_id,
                profile_id=config.default_preset_id,
            )
        )
    return {}


def _memory_fallback_default_profile_id(
    config: GatewayProfileBootstrapConfig,
    bundle: GatewayProfileStorageBundle,
    *,
    seed_memory_readonly: bool,
) -> str | None:
    """Return the configured fallback default id for read-only memory stores."""

    if seed_memory_readonly and not bundle.metadata.persistent:
        return config.default_profile_id or config.default_preset_id
    return None


async def _close_storage_bundle(
    bundle: GatewayProfileStorageBundle,
) -> dict[str, Any]:
    """Close unique stores that expose an async close method."""

    seen: set[int] = set()
    for store in (bundle.profile_store, bundle.assignment_store, bundle.audit_store):
        if store is None or id(store) in seen:
            continue
        seen.add(id(store))
        aclose = getattr(store, "aclose", None)
        if callable(aclose):
            await aclose()
    return {}


def _run_async(coro: Coroutine[Any, Any, dict[str, Any]]) -> dict[str, Any]:
    """Run an async profile-management operation from a sync CLI handler."""

    return asyncio.run(coro)


def _cli_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize manager payloads to the CLI response envelope."""

    return {key: value for key, value in payload.items() if key != "default"}


def _require_cli_text(value: str, *, field: str) -> str:
    """Require a non-blank CLI text value."""

    normalized = _optional_cli_text(value, field=field)
    if normalized is None:
        raise _CliArgumentError(f"{field} is required")
    return normalized


def _optional_cli_text(value: str | None, *, field: str) -> str | None:
    """Normalize optional CLI text values and reject blanks."""

    if value is None:
        return None
    normalized = value.strip()
    if not normalized:
        raise _CliArgumentError(f"{field} cannot be empty")
    return normalized


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
