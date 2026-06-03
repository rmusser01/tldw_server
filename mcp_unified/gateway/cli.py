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

from mcp_unified.package_metadata import package_metadata_summary
from mcp_unified.profiles.presets import (
    duplicate_builtin_preset,
    get_builtin_preset,
    list_builtin_presets,
)

from .config import (
    ExternalRegistryStorageConfigurationError,
    GatewayConfigFormat,
    GatewayExternalRegistryStorageBundle,
    GatewayProfileBootstrapConfig,
    GatewayProfileStorageBundle,
    build_gateway_external_registry_storage,
    build_gateway_profile_storage,
    credential_grant_manager_from_storage,
    external_registry_manager_from_storage,
    gateway_config_snapshot_manager_from_storage,
    load_gateway_profile_bootstrap_config,
)
from .credential_grants import (
    GatewayCredentialGrantManagementError,
    GatewayCredentialGrantManager,
)
from .external_registry import (
    GatewayExternalRegistryManagementError,
    GatewayExternalRegistryManager,
)
from .profiles import GatewayProfileManagementError, GatewayProfileManager
from .remote_admin import (
    RemoteGatewayAdminClient,
    RemoteGatewayAdminConfig,
    RemoteGatewayAdminError,
)
from .snapshots import (
    GatewayConfigSnapshotManagementError,
    GatewayConfigSnapshotManager,
)

_ProfileOperation = Callable[
    [GatewayProfileManager],
    Coroutine[Any, Any, dict[str, Any]],
]
_ExternalRegistryOperation = Callable[
    [GatewayExternalRegistryManager],
    Coroutine[Any, Any, dict[str, Any]],
]
_CredentialGrantOperation = Callable[
    [GatewayCredentialGrantManager],
    Coroutine[Any, Any, dict[str, Any]],
]
_ConfigSnapshotOperation = Callable[
    [GatewayConfigSnapshotManager],
    Coroutine[Any, Any, Mapping[str, Any]],
]
_RemoteRuntimeOperation = Callable[[RemoteGatewayAdminClient], dict[str, Any]]


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

    package_info = subparsers.add_parser(
        "package-info",
        help="Show MCP Unified package release-readiness metadata.",
    )
    package_info.set_defaults(handler=_handle_package_info)

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

    create_profile = subparsers.add_parser(
        "create-profile",
        help="Create a user-editable profile in a persistent gateway store.",
    )
    create_profile.add_argument(
        "--profile-file",
        type=Path,
        required=True,
        help="Path to a JSON profile object, or '-' to read from stdin.",
    )
    _add_profile_config_argument(create_profile)
    create_profile.set_defaults(handler=_handle_create_profile)

    patch_profile = subparsers.add_parser(
        "patch-profile",
        help="Patch a stored profile in a persistent gateway store.",
    )
    patch_profile.add_argument(
        "profile_id",
        help="Profile id to patch.",
    )
    patch_profile.add_argument(
        "--patch-file",
        type=Path,
        required=True,
        help="Path to a JSON patch object, or '-' to read from stdin.",
    )
    _add_profile_config_argument(patch_profile)
    patch_profile.set_defaults(handler=_handle_patch_profile)

    delete_profile = subparsers.add_parser(
        "delete-profile",
        help="Delete an unassigned non-default profile in a persistent gateway store.",
    )
    delete_profile.add_argument(
        "profile_id",
        help="Profile id to delete.",
    )
    _add_profile_config_argument(delete_profile)
    delete_profile.set_defaults(handler=_handle_delete_profile)

    list_external_servers = subparsers.add_parser(
        "list-external-servers",
        help="List external MCP servers from a configured gateway registry store.",
    )
    list_external_servers.add_argument(
        "--enabled",
        choices=("true", "false"),
        help="Optional enabled-state filter.",
    )
    _add_profile_config_argument(list_external_servers)
    list_external_servers.set_defaults(handler=_handle_list_external_servers)

    show_external_server = subparsers.add_parser(
        "show-external-server",
        help="Show one external MCP server from a configured registry store.",
    )
    show_external_server.add_argument(
        "server_id",
        help="External server id to inspect.",
    )
    _add_profile_config_argument(show_external_server)
    show_external_server.set_defaults(handler=_handle_show_external_server)

    create_external_server = subparsers.add_parser(
        "create-external-server",
        help="Create an external MCP server in a persistent gateway registry store.",
    )
    create_external_server.add_argument(
        "--server-file",
        type=Path,
        required=True,
        help="Path to a JSON external server object, or '-' to read from stdin.",
    )
    _add_profile_config_argument(create_external_server)
    create_external_server.set_defaults(handler=_handle_create_external_server)

    patch_external_server = subparsers.add_parser(
        "patch-external-server",
        help="Patch an external MCP server in a persistent gateway registry store.",
    )
    patch_external_server.add_argument(
        "server_id",
        help="External server id to patch.",
    )
    patch_external_server.add_argument(
        "--patch-file",
        type=Path,
        required=True,
        help="Path to a JSON patch object, or '-' to read from stdin.",
    )
    _add_profile_config_argument(patch_external_server)
    patch_external_server.set_defaults(handler=_handle_patch_external_server)

    delete_external_server = subparsers.add_parser(
        "delete-external-server",
        help="Delete an ungranted external MCP server from a persistent registry store.",
    )
    delete_external_server.add_argument(
        "server_id",
        help="External server id to delete.",
    )
    _add_profile_config_argument(delete_external_server)
    delete_external_server.set_defaults(handler=_handle_delete_external_server)

    list_credential_grants = subparsers.add_parser(
        "list-credential-grants",
        help="List credential grants from a configured gateway store.",
    )
    list_credential_grants.add_argument(
        "--profile-id",
        help="Optional profile id filter.",
    )
    list_credential_grants.add_argument(
        "--external-server-id",
        help="Optional external server id filter.",
    )
    _add_profile_config_argument(list_credential_grants)
    list_credential_grants.set_defaults(handler=_handle_list_credential_grants)

    show_credential_grant = subparsers.add_parser(
        "show-credential-grant",
        help="Show one credential grant from a configured gateway store.",
    )
    show_credential_grant.add_argument(
        "grant_id",
        help="Credential grant id to inspect.",
    )
    _add_profile_config_argument(show_credential_grant)
    show_credential_grant.set_defaults(handler=_handle_show_credential_grant)

    create_credential_grant = subparsers.add_parser(
        "create-credential-grant",
        help="Create a credential grant in a persistent gateway store.",
    )
    create_credential_grant.add_argument(
        "--grant-file",
        type=Path,
        required=True,
        help="Path to a JSON credential grant object, or '-' to read from stdin.",
    )
    _add_profile_config_argument(create_credential_grant)
    create_credential_grant.set_defaults(handler=_handle_create_credential_grant)

    patch_credential_grant = subparsers.add_parser(
        "patch-credential-grant",
        help="Patch a credential grant in a persistent gateway store.",
    )
    patch_credential_grant.add_argument(
        "grant_id",
        help="Credential grant id to patch.",
    )
    patch_credential_grant.add_argument(
        "--patch-file",
        type=Path,
        required=True,
        help="Path to a JSON patch object, or '-' to read from stdin.",
    )
    _add_profile_config_argument(patch_credential_grant)
    patch_credential_grant.set_defaults(handler=_handle_patch_credential_grant)

    delete_credential_grant = subparsers.add_parser(
        "delete-credential-grant",
        help="Delete a credential grant from a persistent gateway store.",
    )
    delete_credential_grant.add_argument(
        "grant_id",
        help="Credential grant id to delete.",
    )
    _add_profile_config_argument(delete_credential_grant)
    delete_credential_grant.set_defaults(handler=_handle_delete_credential_grant)

    export_config = subparsers.add_parser(
        "export-config",
        help="Export a persistent gateway config snapshot.",
    )
    export_config.add_argument(
        "--output",
        type=Path,
        help="Optional file path to write the snapshot JSON.",
    )
    _add_profile_config_argument(export_config)
    export_config.set_defaults(handler=_handle_export_config)

    import_config = subparsers.add_parser(
        "import-config",
        help="Import a persistent gateway config snapshot.",
    )
    import_config.add_argument(
        "--snapshot-file",
        type=Path,
        required=True,
        help="Path to a JSON config snapshot object, or '-' to read from stdin.",
    )
    import_config.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and report planned mutations without writing.",
    )
    _add_profile_config_argument(import_config)
    import_config.set_defaults(handler=_handle_import_config)

    runtime_list = subparsers.add_parser(
        "runtime-list",
        help="List external runtime state from a running gateway.",
    )
    _add_remote_runtime_arguments(runtime_list)
    runtime_list.set_defaults(handler=_handle_runtime_list)

    runtime_start = subparsers.add_parser(
        "runtime-start",
        help="Start one external server through a running gateway.",
    )
    runtime_start.add_argument(
        "server_id",
        help="External server id to start.",
    )
    _add_remote_runtime_arguments(runtime_start)
    runtime_start.set_defaults(handler=_handle_runtime_start)

    runtime_stop = subparsers.add_parser(
        "runtime-stop",
        help="Stop one external server through a running gateway.",
    )
    runtime_stop.add_argument(
        "server_id",
        help="External server id to stop.",
    )
    _add_remote_runtime_arguments(runtime_stop)
    runtime_stop.set_defaults(handler=_handle_runtime_stop)

    runtime_restart = subparsers.add_parser(
        "runtime-restart",
        help="Restart one external server through a running gateway.",
    )
    runtime_restart.add_argument(
        "server_id",
        help="External server id to restart.",
    )
    _add_remote_runtime_arguments(runtime_restart)
    runtime_restart.set_defaults(handler=_handle_runtime_restart)

    runtime_refresh = subparsers.add_parser(
        "runtime-refresh",
        help="Refresh one external runtime or all runtimes from a running gateway.",
    )
    runtime_refresh.add_argument(
        "server_id",
        nargs="?",
        help="Optional external server id to refresh.",
    )
    _add_remote_runtime_arguments(runtime_refresh)
    runtime_refresh.set_defaults(handler=_handle_runtime_refresh)

    runtime_reconcile = subparsers.add_parser(
        "runtime-reconcile",
        help="Reconcile one external runtime or all runtimes from a running gateway.",
    )
    runtime_reconcile.add_argument(
        "server_id",
        nargs="?",
        help="Optional external server id to reconcile.",
    )
    _add_remote_runtime_arguments(runtime_reconcile)
    runtime_reconcile.set_defaults(handler=_handle_runtime_reconcile)

    runtime_install = subparsers.add_parser(
        "runtime-install",
        help="Run the configured install flow through a running gateway.",
    )
    runtime_install.add_argument(
        "server_id",
        help="External server id to install.",
    )
    _add_remote_runtime_arguments(runtime_install)
    runtime_install.set_defaults(handler=_handle_runtime_install)

    runtime_update = subparsers.add_parser(
        "runtime-update",
        help="Run the configured update flow through a running gateway.",
    )
    runtime_update.add_argument(
        "server_id",
        help="External server id to update.",
    )
    _add_remote_runtime_arguments(runtime_update)
    runtime_update.set_defaults(handler=_handle_runtime_update)

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
            "Falls back to MCP_UNIFIED_GATEWAY_CONFIG or MCP_GATEWAY_CONFIG."
        ),
    )


def _add_remote_runtime_arguments(parser: argparse.ArgumentParser) -> None:
    """Add common remote runtime options without command-line secrets."""

    parser.add_argument(
        "--gateway-url",
        help=(
            "Mounted gateway base URL, for example http://127.0.0.1:8000/mcp. "
            "Falls back to MCP_UNIFIED_GATEWAY_URL."
        ),
    )
    parser.add_argument(
        "--admin-header-name",
        default="X-MCP-Gateway-Admin-Key",
        help="Admin auth header name used with MCP_UNIFIED_GATEWAY_ADMIN_KEY.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=30.0,
        help="Remote gateway request timeout in seconds.",
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


def _handle_package_info(_args: argparse.Namespace) -> int:
    """Emit MCP Unified package release-readiness metadata."""

    _emit_json(package_metadata_summary(), sys.stdout)
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


def _handle_create_profile(args: argparse.Namespace) -> int:
    """Create one user-editable profile from a JSON file or stdin payload."""

    return _handle_profile_management_command(
        args,
        lambda manager: manager.create_profile(
            _load_json_argument_file(args.profile_file, label="profile"),
        ),
        require_persistent=True,
    )


def _handle_patch_profile(args: argparse.Namespace) -> int:
    """Patch one stored profile using a JSON file or stdin payload."""

    profile_id = _require_cli_text(args.profile_id, field="profile_id")
    return _handle_profile_management_command(
        args,
        lambda manager: manager.patch_profile(
            profile_id,
            _load_json_argument_file(args.patch_file, label="patch"),
        ),
        require_persistent=True,
    )


def _handle_delete_profile(args: argparse.Namespace) -> int:
    """Delete one stored profile from a persistent gateway store."""

    profile_id = _require_cli_text(args.profile_id, field="profile_id")
    return _handle_profile_management_command(
        args,
        lambda manager: manager.delete_profile(profile_id),
        require_persistent=True,
    )


def _handle_list_external_servers(args: argparse.Namespace) -> int:
    """List external servers from a configured gateway registry store."""

    enabled = _optional_bool_choice(args.enabled)
    return _handle_external_registry_command(
        args,
        lambda manager: manager.list_servers(enabled=enabled),
    )


def _handle_show_external_server(args: argparse.Namespace) -> int:
    """Show one external server from a configured gateway registry store."""

    server_id = _require_cli_text(args.server_id, field="server_id")
    return _handle_external_registry_command(
        args,
        lambda manager: manager.show_server(server_id),
    )


def _handle_create_external_server(args: argparse.Namespace) -> int:
    """Create one external server from a JSON file or stdin payload."""

    return _handle_external_registry_command(
        args,
        lambda manager: manager.create_server(
            _load_json_argument_file(args.server_file, label="server"),
        ),
    )


def _handle_patch_external_server(args: argparse.Namespace) -> int:
    """Patch one external server using a JSON file or stdin payload."""

    server_id = _require_cli_text(args.server_id, field="server_id")
    return _handle_external_registry_command(
        args,
        lambda manager: manager.patch_server(
            server_id,
            _load_json_argument_file(args.patch_file, label="patch"),
        ),
    )


def _handle_delete_external_server(args: argparse.Namespace) -> int:
    """Delete one external server from a persistent gateway registry store."""

    server_id = _require_cli_text(args.server_id, field="server_id")
    return _handle_external_registry_command(
        args,
        lambda manager: manager.delete_server(server_id),
    )


def _handle_list_credential_grants(args: argparse.Namespace) -> int:
    """List credential grants from a configured gateway store."""

    profile_id = _optional_cli_text(args.profile_id, field="profile_id")
    external_server_id = _optional_cli_text(
        args.external_server_id,
        field="external_server_id",
    )
    return _handle_credential_grant_command(
        args,
        lambda manager: manager.list_grants(
            profile_id=profile_id,
            external_server_id=external_server_id,
        ),
    )


def _handle_show_credential_grant(args: argparse.Namespace) -> int:
    """Show one credential grant from a configured gateway store."""

    grant_id = _require_cli_text(args.grant_id, field="grant_id")
    return _handle_credential_grant_command(
        args,
        lambda manager: manager.show_grant(grant_id),
    )


def _handle_create_credential_grant(args: argparse.Namespace) -> int:
    """Create one credential grant from a JSON file or stdin payload."""

    return _handle_credential_grant_command(
        args,
        lambda manager: manager.create_grant(
            _load_json_argument_file(args.grant_file, label="grant"),
        ),
    )


def _handle_patch_credential_grant(args: argparse.Namespace) -> int:
    """Patch one credential grant using a JSON file or stdin payload."""

    grant_id = _require_cli_text(args.grant_id, field="grant_id")
    return _handle_credential_grant_command(
        args,
        lambda manager: manager.patch_grant(
            grant_id,
            _load_json_argument_file(args.patch_file, label="patch"),
        ),
    )


def _handle_delete_credential_grant(args: argparse.Namespace) -> int:
    """Delete one credential grant from a persistent gateway store."""

    grant_id = _require_cli_text(args.grant_id, field="grant_id")
    return _handle_credential_grant_command(
        args,
        lambda manager: manager.delete_grant(grant_id),
    )


def _handle_export_config(args: argparse.Namespace) -> int:
    """Export a gateway config snapshot to stdout or a JSON file."""

    return _handle_config_snapshot_command(
        args,
        lambda manager: _export_config_for_cli(
            manager,
            output_path=args.output,
        ),
    )


def _handle_import_config(args: argparse.Namespace) -> int:
    """Import a gateway config snapshot from a JSON file or stdin."""

    return _handle_config_snapshot_command(
        args,
        lambda manager: _import_config_for_cli(
            manager,
            snapshot_file=args.snapshot_file,
            dry_run=bool(args.dry_run),
        ),
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


def _handle_runtime_list(args: argparse.Namespace) -> int:
    """List external runtime state from a running gateway."""

    return _handle_remote_runtime_command(
        args,
        lambda client: client.list_runtime_servers(),
    )


def _handle_runtime_start(args: argparse.Namespace) -> int:
    """Start one external server through a running gateway."""

    server_id = _require_cli_text(args.server_id, field="server_id")
    return _handle_remote_runtime_command(
        args,
        lambda client: client.start_server(server_id),
    )


def _handle_runtime_stop(args: argparse.Namespace) -> int:
    """Stop one external server through a running gateway."""

    server_id = _require_cli_text(args.server_id, field="server_id")
    return _handle_remote_runtime_command(
        args,
        lambda client: client.stop_server(server_id),
    )


def _handle_runtime_restart(args: argparse.Namespace) -> int:
    """Restart one external server through a running gateway."""

    server_id = _require_cli_text(args.server_id, field="server_id")
    return _handle_remote_runtime_command(
        args,
        lambda client: client.restart_server(server_id),
    )


def _handle_runtime_refresh(args: argparse.Namespace) -> int:
    """Refresh one external runtime or all runtimes through a running gateway."""

    server_id = _optional_cli_text(args.server_id, field="server_id")
    return _handle_remote_runtime_command(
        args,
        lambda client: client.refresh_server(server_id),
    )


def _handle_runtime_reconcile(args: argparse.Namespace) -> int:
    """Reconcile one external runtime or all runtimes through a running gateway."""

    server_id = _optional_cli_text(args.server_id, field="server_id")
    return _handle_remote_runtime_command(
        args,
        lambda client: client.reconcile(server_id),
    )


def _handle_runtime_install(args: argparse.Namespace) -> int:
    """Run one external server install flow through a running gateway."""

    server_id = _require_cli_text(args.server_id, field="server_id")
    return _handle_remote_runtime_command(
        args,
        lambda client: client.install_server(server_id),
    )


def _handle_runtime_update(args: argparse.Namespace) -> int:
    """Run one external server update flow through a running gateway."""

    server_id = _require_cli_text(args.server_id, field="server_id")
    return _handle_remote_runtime_command(
        args,
        lambda client: client.update_server(server_id),
    )


def _handle_remote_runtime_command(
    args: argparse.Namespace,
    operation: _RemoteRuntimeOperation,
) -> int:
    """Run one remote runtime command against an already-running gateway."""

    try:
        config = _remote_runtime_config_from_args(args)
        payload = operation(RemoteGatewayAdminClient(config))
    except _CliArgumentError:
        raise
    except RemoteGatewayAdminError as exc:
        _emit_json(exc.to_payload(), sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001
        _emit_json(
            {
                "error": "Remote gateway command failed",
                "error_type": exc.__class__.__name__,
                "ok": False,
                "reason_code": "remote_gateway_command_failed",
            },
            sys.stderr,
        )
        return 1

    _emit_json(payload, sys.stdout)
    return 0


def _remote_runtime_config_from_args(
    args: argparse.Namespace,
) -> RemoteGatewayAdminConfig:
    """Build a remote admin config from CLI args and environment."""

    gateway_url = _optional_cli_text(args.gateway_url, field="gateway_url")
    if gateway_url is None:
        gateway_url = _optional_cli_text(
            os.environ.get("MCP_UNIFIED_GATEWAY_URL"),
            field="MCP_UNIFIED_GATEWAY_URL",
        )
    if gateway_url is None:
        raise _CliArgumentError(
            "--gateway-url is required unless MCP_UNIFIED_GATEWAY_URL is set"
        )

    admin_key = _optional_cli_text(
        os.environ.get("MCP_UNIFIED_GATEWAY_ADMIN_KEY"),
        field="MCP_UNIFIED_GATEWAY_ADMIN_KEY",
    )
    try:
        return RemoteGatewayAdminConfig(
            gateway_url=gateway_url,
            admin_header_name=args.admin_header_name,
            admin_key=admin_key,
            timeout_seconds=args.timeout_seconds,
        )
    except ValueError as exc:
        raise _CliArgumentError(str(exc)) from exc


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


def _handle_external_registry_command(
    args: argparse.Namespace,
    operation: _ExternalRegistryOperation,
    *,
    require_persistent: bool = True,
) -> int:
    """Run an external-registry command against the configured gateway store."""

    config_path = _config_path_from_args(args)
    bundle: GatewayExternalRegistryStorageBundle | None = None
    try:
        try:
            config = load_gateway_profile_bootstrap_config(config_path)
            bundle = build_gateway_external_registry_storage(config.store)
            if require_persistent and not bundle.metadata.persistent:
                raise GatewayExternalRegistryManagementError(
                    "External registry management requires a persistent gateway store",
                    reason_code="external_registry_store_unavailable",
                )
            manager = external_registry_manager_from_storage(bundle)
        except _CliArgumentError:
            raise
        except GatewayExternalRegistryManagementError as exc:
            _emit_json(exc.to_payload(), sys.stderr)
            return 1
        except Exception as exc:  # noqa: BLE001
            unavailable_error = _external_registry_storage_unavailable_error(exc)
            if unavailable_error is not None:
                _emit_json(unavailable_error.to_payload(), sys.stderr)
                return 1
            # CLI boundary: config loading and storage construction failures
            # should be machine-readable.
            _emit_json(
                {
                    "error": str(exc),
                    "ok": False,
                    "path": str(config_path),
                },
                sys.stderr,
            )
            return 1

        try:
            payload = _run_async(operation(manager))
        except _CliArgumentError:
            raise
        except GatewayExternalRegistryManagementError as exc:
            _emit_json(exc.to_payload(), sys.stderr)
            return 1
        except Exception:  # noqa: BLE001
            unavailable_error = GatewayExternalRegistryManagementError(
                "External registry store unavailable",
                reason_code="external_registry_store_unavailable",
            )
            _emit_json(unavailable_error.to_payload(), sys.stderr)
            return 1

        _emit_json(_cli_payload(payload), sys.stdout)
        return 0
    finally:
        if bundle is not None:
            _run_async(_close_external_registry_bundle(bundle))


def _handle_credential_grant_command(
    args: argparse.Namespace,
    operation: _CredentialGrantOperation,
    *,
    require_persistent: bool = True,
) -> int:
    """Run a credential-grant command against the configured gateway store."""

    config_path = _config_path_from_args(args)
    profile_bundle: GatewayProfileStorageBundle | None = None
    external_bundle: GatewayExternalRegistryStorageBundle | None = None
    try:
        try:
            config = load_gateway_profile_bootstrap_config(config_path)
            profile_bundle = build_gateway_profile_storage(config.store)
            external_bundle = build_gateway_external_registry_storage(config.store)
            if require_persistent and not external_bundle.metadata.persistent:
                raise GatewayCredentialGrantManagementError(
                    "Credential grant management requires a persistent gateway store",
                    reason_code="credential_grant_store_unavailable",
                )
            manager = credential_grant_manager_from_storage(
                external_bundle,
                profile_storage=profile_bundle,
            )
        except _CliArgumentError:
            raise
        except GatewayCredentialGrantManagementError as exc:
            _emit_json(exc.to_payload(), sys.stderr)
            return 1
        except Exception as exc:  # noqa: BLE001
            unavailable_error = _credential_grant_storage_unavailable_error(exc)
            if unavailable_error is not None:
                _emit_json(unavailable_error.to_payload(), sys.stderr)
                return 1
            _emit_json(
                {
                    "error": str(exc),
                    "ok": False,
                    "path": str(config_path),
                },
                sys.stderr,
            )
            return 1

        try:
            payload = _run_async(operation(manager))
        except _CliArgumentError:
            raise
        except GatewayCredentialGrantManagementError as exc:
            _emit_json(exc.to_payload(), sys.stderr)
            return 1
        except Exception:  # noqa: BLE001
            unavailable_error = GatewayCredentialGrantManagementError(
                "Credential grant store unavailable",
                reason_code="credential_grant_store_unavailable",
            )
            _emit_json(unavailable_error.to_payload(), sys.stderr)
            return 1

        _emit_json(_cli_payload(payload), sys.stdout)
        return 0
    finally:
        if profile_bundle is not None:
            _run_async(_close_storage_bundle(profile_bundle))
        if external_bundle is not None:
            _run_async(_close_external_registry_bundle(external_bundle))


def _handle_config_snapshot_command(
    args: argparse.Namespace,
    operation: _ConfigSnapshotOperation,
) -> int:
    """Run a config snapshot command against persistent gateway stores."""

    config_path = _config_path_from_args(args)
    profile_bundle: GatewayProfileStorageBundle | None = None
    external_bundle: GatewayExternalRegistryStorageBundle | None = None
    try:
        try:
            config = load_gateway_profile_bootstrap_config(config_path)
            profile_bundle = build_gateway_profile_storage(config.store)
            external_bundle = build_gateway_external_registry_storage(config.store)
            if (
                not profile_bundle.metadata.persistent
                or not external_bundle.metadata.persistent
            ):
                raise GatewayConfigSnapshotManagementError(
                    "Config snapshots require a persistent gateway store",
                    reason_code="config_snapshot_store_unavailable",
                )
            manager = gateway_config_snapshot_manager_from_storage(
                profile_bundle,
                external_bundle,
            )
        except _CliArgumentError:
            raise
        except GatewayConfigSnapshotManagementError as exc:
            _emit_json(exc.to_payload(), sys.stderr)
            return 1
        except Exception as exc:  # noqa: BLE001
            unavailable_error = _config_snapshot_storage_unavailable_error(exc)
            if unavailable_error is not None:
                _emit_json(unavailable_error.to_payload(), sys.stderr)
                return 1
            _emit_json(
                {
                    "error": str(exc),
                    "ok": False,
                    "path": str(config_path),
                },
                sys.stderr,
            )
            return 1

        try:
            payload = _run_async(operation(manager))
        except _CliArgumentError:
            raise
        except GatewayConfigSnapshotManagementError as exc:
            _emit_json(exc.to_payload(), sys.stderr)
            return 1
        except Exception:  # noqa: BLE001
            unavailable_error = GatewayConfigSnapshotManagementError(
                "Config snapshot store unavailable",
                reason_code="config_snapshot_store_unavailable",
            )
            _emit_json(unavailable_error.to_payload(), sys.stderr)
            return 1

        _emit_json(dict(payload), sys.stdout)
        return 0
    finally:
        if profile_bundle is not None:
            _run_async(_close_storage_bundle(profile_bundle))
        if external_bundle is not None:
            _run_async(_close_external_registry_bundle(external_bundle))


def _credential_grant_storage_unavailable_error(
    exc: Exception,
) -> GatewayCredentialGrantManagementError | None:
    """Map expected unavailable credential-grant storage build failures."""

    if not isinstance(exc, ExternalRegistryStorageConfigurationError):
        return None
    return GatewayCredentialGrantManagementError(
        str(exc),
        reason_code="credential_grant_store_unavailable",
    )


def _external_registry_storage_unavailable_error(
    exc: Exception,
) -> GatewayExternalRegistryManagementError | None:
    """Map expected unavailable external-registry storage build failures."""

    if not isinstance(exc, ExternalRegistryStorageConfigurationError):
        return None
    return GatewayExternalRegistryManagementError(
        str(exc),
        reason_code="external_registry_store_unavailable",
    )


def _config_snapshot_storage_unavailable_error(
    exc: Exception,
) -> GatewayConfigSnapshotManagementError | None:
    """Map expected unavailable config snapshot storage build failures."""

    if not isinstance(exc, ExternalRegistryStorageConfigurationError):
        return None
    return GatewayConfigSnapshotManagementError(
        str(exc),
        reason_code="config_snapshot_store_unavailable",
    )


def _config_path_from_args(args: argparse.Namespace) -> Path:
    """Return the explicit or environment-provided gateway config path."""

    if args.config is not None:
        return args.config
    env_value = os.environ.get("MCP_UNIFIED_GATEWAY_CONFIG") or os.environ.get(
        "MCP_GATEWAY_CONFIG"
    )
    if env_value and env_value.strip():
        return Path(env_value)
    raise _CliArgumentError(
        "--config is required unless MCP_UNIFIED_GATEWAY_CONFIG or "
        "MCP_GATEWAY_CONFIG is set"
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


async def _export_config_for_cli(
    manager: GatewayConfigSnapshotManager,
    *,
    output_path: Path | None,
) -> dict[str, Any]:
    """Export a snapshot, optionally writing it to a file."""

    snapshot = await manager.export_snapshot()
    payload = snapshot.model_dump(mode="json")
    if output_path is None:
        return payload
    try:
        output_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    except OSError as exc:
        raise _CliArgumentError(f"Unable to write snapshot JSON: {exc}") from exc
    return {
        "ok": True,
        "output": str(output_path),
        "schema": payload["schema"],
        "version": payload["version"],
    }


async def _import_config_for_cli(
    manager: GatewayConfigSnapshotManager,
    *,
    snapshot_file: Path,
    dry_run: bool,
) -> dict[str, Any]:
    """Import a config snapshot from a JSON file or stdin."""

    snapshot = _load_json_argument_file(snapshot_file, label="snapshot")
    return await manager.import_snapshot(snapshot, dry_run=dry_run)


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


async def _close_external_registry_bundle(
    bundle: GatewayExternalRegistryStorageBundle,
) -> dict[str, Any]:
    """Close unique external-registry stores that expose an async close method."""

    seen: set[int] = set()
    for store in (
        bundle.external_registry_store,
        bundle.credential_grant_store,
        bundle.audit_store,
    ):
        if store is None or id(store) in seen:
            continue
        seen.add(id(store))
        aclose = getattr(store, "aclose", None)
        if callable(aclose):
            await aclose()
    return {}


def _run_async(coro: Coroutine[Any, Any, Any]) -> Any:
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


def _optional_bool_choice(value: str | None) -> bool | None:
    """Convert optional argparse true/false choices into booleans."""

    if value is None:
        return None
    if value == "true":
        return True
    if value == "false":
        return False
    raise _CliArgumentError("enabled must be true or false")


def _load_json_argument_file(path: Path, *, label: str) -> dict[str, Any]:
    """Load one JSON object payload from a file path or stdin marker '-'."""

    try:
        if str(path) == "-":
            payload_text = sys.stdin.read()
        else:
            payload_text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise _CliArgumentError(f"Unable to read {label} JSON: {exc}") from exc

    try:
        payload = json.loads(payload_text)
    except json.JSONDecodeError as exc:
        raise _CliArgumentError(f"Invalid {label} JSON: {exc.msg}") from exc

    if not isinstance(payload, dict):
        raise _CliArgumentError(f"{label} JSON must be an object")
    return payload


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
        "external_runtime": {
            "enabled": config.external_runtime.enabled,
            "process_policy": _process_policy_summary(config.external_runtime),
            "reconcile_on_startup": config.external_runtime.reconcile_on_startup,
            "stop_on_shutdown": config.external_runtime.stop_on_shutdown,
            "transport_factory": config.external_runtime.transport_factory,
        },
        "ok": True,
        "path": str(path),
        "profiles": len(config.profiles),
        "store": {
            "kind": config.store.kind,
            "sqlite_path": str(sqlite_path) if sqlite_path is not None else None,
        },
    }


def _process_policy_summary(
    external_runtime: Any,
) -> dict[str, Any]:
    """Return a compact process-policy summary without path or command values."""

    policy = getattr(external_runtime, "process_policy", None)
    if policy is None:
        return {
            "allow_path_lookup": False,
            "allowed_cwd_roots": 0,
            "allowed_env_names": None,
            "allowed_executables": 0,
            "configured": getattr(
                external_runtime,
                "process_policy_configured",
                False,
            ),
            "default_cwd": False,
            "reject_shell_executables": False,
        }

    allowed_env_names = getattr(policy, "allowed_env_names", None)
    return {
        "allow_path_lookup": getattr(policy, "allow_path_lookup", False),
        "allowed_cwd_roots": len(getattr(policy, "allowed_cwd_roots", ())),
        "allowed_env_names": (
            None if allowed_env_names is None else len(allowed_env_names)
        ),
        "allowed_executables": len(getattr(policy, "allowed_executables", ())),
        "configured": getattr(external_runtime, "process_policy_configured", False),
        "default_cwd": getattr(policy, "default_cwd", None) is not None,
        "reject_shell_executables": getattr(
            policy,
            "reject_shell_executables",
            False,
        ),
    }


def _emit_json(payload: Mapping[str, Any], stream: TextIO) -> None:
    """Write one JSON object to the selected stream."""

    stream.write(json.dumps(payload, sort_keys=True))
    stream.write("\n")


if __name__ == "__main__":  # pragma: no cover - exercised through console script.
    raise SystemExit(main())
