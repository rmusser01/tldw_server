"""Offline operator commands for the canonical admin-webhook control plane."""

from __future__ import annotations

import asyncio
import json
import os
import secrets
from collections.abc import Awaitable, Callable
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, TypeVar

import click

from tldw_Server_API.app.core.Admin_Webhooks.audit import (
    emit_mandatory_webhook_operation_audit,
)
from tldw_Server_API.app.core.Admin_Webhooks.config import AdminWebhookSettings
from tldw_Server_API.app.core.Admin_Webhooks.crypto import load_webhook_key_ring
from tldw_Server_API.app.core.Admin_Webhooks.domain import WebhookError
from tldw_Server_API.app.core.Admin_Webhooks.key_rotation import (
    WebhookKeyRotationService,
)
from tldw_Server_API.app.core.Admin_Webhooks.legacy_import import (
    LegacyImportError,
    LegacyImportRequest,
    LegacyImportService,
    LegacyRejectionReason,
)
from tldw_Server_API.app.core.Admin_Webhooks.repository import AdminWebhookRepository
from tldw_Server_API.app.core.AuthNZ.database import DatabasePool
from tldw_Server_API.app.core.AuthNZ.settings import Settings
from tldw_Server_API.app.services import admin_system_ops_service as system_ops

T = TypeVar("T")


def _json_default(value: object) -> object:
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc).isoformat()
    raise TypeError("value is not JSON serializable")


def _print_json(value: object) -> None:
    click.echo(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            default=_json_default,
        )
    )


async def _with_runtime(
    operation: Callable[
        [LegacyImportService, WebhookKeyRotationService, AdminWebhookRepository],
        Awaitable[T],
    ],
) -> T:
    pool = DatabasePool(Settings())
    await pool.initialize()
    try:
        repository = AdminWebhookRepository(pool)
        ring = load_webhook_key_ring().require_ring()
        settings = AdminWebhookSettings.from_environment(os.environ)
        roots = [system_ops._STORE_PATH.parent.resolve(strict=False)]
        if repository.database_path is not None:
            roots.append(repository.database_path.parent.resolve(strict=False))
        importer = LegacyImportService(
            repository=repository,
            key_ring=ring,
            settings=settings,
            application_data_paths=tuple(dict.fromkeys(roots)),
        )
        rotation = WebhookKeyRotationService(
            repository=repository,
            key_ring=ring,
        )
        return await operation(importer, rotation, repository)
    finally:
        await pool.close()


def _run(
    operation: Callable[
        [LegacyImportService, WebhookKeyRotationService, AdminWebhookRepository],
        Awaitable[T],
    ],
) -> T:
    try:
        return asyncio.run(_with_runtime(operation))
    except LegacyImportError as exc:
        raise click.ClickException(exc.code.value) from None
    except WebhookError as exc:
        raise click.ClickException(exc.code.value) from None
    except click.ClickException:
        raise
    except Exception:  # noqa: BLE001 - never expose runtime exception details
        raise click.ClickException("admin_webhook_operation_failed") from None


def _request_id(prefix: str) -> str:
    return f"{prefix}_{secrets.token_hex(12)}"


@click.group(name="admin-webhooks")
def admin_webhooks_group() -> None:
    """Manage canonical outgoing-webhook migration and key operations."""


@admin_webhooks_group.command("import-legacy")
@click.option("--dry-run", is_flag=True, help="Publish a redacted migration report.")
@click.option("--apply", "apply_import", is_flag=True, help="Apply an approved report.")
@click.option("--all-writers-quiesced", is_flag=True)
@click.option("--allow-legacy-credential-decryption", is_flag=True)
@click.option("--approved-report-digest")
@click.option("--report", "report_path", required=True, type=click.Path(path_type=Path))
@click.option("--backup", "backup_path", type=click.Path(path_type=Path))
@click.option(
    "--rollback-key-file",
    "rollback_key_path",
    type=click.Path(path_type=Path),
)
@click.option("--operator-id", required=True, type=click.IntRange(min=1))
def import_legacy(
    *,
    dry_run: bool,
    apply_import: bool,
    all_writers_quiesced: bool,
    allow_legacy_credential_decryption: bool,
    approved_report_digest: str | None,
    report_path: Path,
    backup_path: Path | None,
    rollback_key_path: Path | None,
    operator_id: int,
) -> None:
    """Dry-run or apply the deterministic two-source legacy import."""
    if dry_run == apply_import:
        raise click.UsageError("Choose exactly one of --dry-run or --apply.")
    if apply_import and not all_writers_quiesced:
        raise click.UsageError("--apply requires --all-writers-quiesced.")
    if apply_import and approved_report_digest is None:
        raise click.UsageError("--apply requires --approved-report-digest.")
    if dry_run and approved_report_digest is not None:
        raise click.UsageError("--approved-report-digest is valid only with --apply.")
    request = LegacyImportRequest(
        report_path=report_path,
        backup_path=backup_path,
        rollback_key_path=rollback_key_path,
        operator_id=operator_id,
        now=datetime.now(timezone.utc),
        allow_legacy_credential_decryption=allow_legacy_credential_decryption,
    )

    async def operation(
        importer: LegacyImportService,
        _rotation: WebhookKeyRotationService,
        _repository: AdminWebhookRepository,
    ) -> object:
        if dry_run:
            plan = await importer.build_plan(request)
            return {
                "accepted_count": len(plan.accepted),
                "explicitly_rejected_count": len(plan.explicitly_rejected),
                "operation_id": plan.operation_id,
                "projected_non_deleted_count": plan.projected_non_deleted_count,
                "report_digest": plan.report_digest,
                "requires_system_ops_backup": plan.requires_system_ops_backup,
                "unresolved_count": len(plan.unresolved),
            }
        if approved_report_digest is None:
            raise click.UsageError("--apply requires --approved-report-digest.")
        state = await importer.apply_plan(
            request,
            approved_report_digest=approved_report_digest,
            request_id=_request_id("whimport"),
        )
        return {
            "operation_id": state.import_operation_id,
            "phase": state.phase,
            "rollback_expires_at": state.rollback_expires_at,
            "rollback_retirement_phase": state.rollback_retirement_phase,
        }

    _print_json(_run(operation))


@admin_webhooks_group.command("reject-source")
@click.option(
    "--source-kind",
    required=True,
    type=click.Choice(["system_ops", "database"]),
)
@click.option("--source-identity", required=True)
@click.option("--source-record-fingerprint", required=True)
@click.option(
    "--reason-code",
    required=True,
    type=click.Choice([value.value for value in LegacyRejectionReason]),
)
@click.option("--operator-id", required=True, type=click.IntRange(min=1))
def reject_source(
    *,
    source_kind: str,
    source_identity: str,
    source_record_fingerprint: str,
    reason_code: str,
    operator_id: int,
) -> None:
    """Reject one exact current legacy source record."""

    async def operation(
        importer: LegacyImportService,
        _rotation: WebhookKeyRotationService,
        _repository: AdminWebhookRepository,
    ) -> object:
        state = await importer.reject_source(
            source_kind=source_kind,
            source_identity=source_identity,
            source_record_fingerprint=source_record_fingerprint,
            reason_code=LegacyRejectionReason(reason_code),
            operator_id=operator_id,
            now=datetime.now(timezone.utc),
            request_id=_request_id("whreject"),
        )
        return {"phase": state.phase, "state_revision": state.state_revision}

    _print_json(_run(operation))


@admin_webhooks_group.command("extract-rollback-backup")
@click.option("--backup", "backup_path", required=True, type=click.Path(path_type=Path))
@click.option(
    "--rollback-key-file",
    "rollback_key_path",
    required=True,
    type=click.Path(path_type=Path),
)
@click.option("--output", "output_path", required=True, type=click.Path(path_type=Path))
@click.option("--operator-id", required=True, type=click.IntRange(min=1))
@click.option("--confirm", is_flag=True, required=True)
def extract_rollback_backup(
    *,
    backup_path: Path,
    rollback_key_path: Path,
    output_path: Path,
    operator_id: int,
    confirm: bool,
) -> None:
    """Extract a retained encrypted backup to one new private file."""

    async def operation(
        importer: LegacyImportService,
        _rotation: WebhookKeyRotationService,
        _repository: AdminWebhookRepository,
    ) -> object:
        result = await importer.extract_rollback_backup(
            backup_path=backup_path,
            rollback_key_path=rollback_key_path,
            output_path=output_path,
            operator_id=operator_id,
            now=datetime.now(timezone.utc),
            confirmed=confirm,
            request_id=_request_id("whextract"),
        )
        return {"result": result}

    _print_json(_run(operation))


@admin_webhooks_group.command("destroy-rollback-key")
@click.option("--backup", "backup_path", required=True, type=click.Path(path_type=Path))
@click.option(
    "--rollback-key-file",
    "rollback_key_path",
    required=True,
    type=click.Path(path_type=Path),
)
@click.option("--operator-id", required=True, type=click.IntRange(min=1))
@click.option("--confirm", is_flag=True, required=True)
def destroy_rollback_key(
    *,
    backup_path: Path,
    rollback_key_path: Path,
    operator_id: int,
    confirm: bool,
) -> None:
    """Retire the active rollback key and encrypted backup after expiry."""

    async def operation(
        importer: LegacyImportService,
        _rotation: WebhookKeyRotationService,
        _repository: AdminWebhookRepository,
    ) -> object:
        result = await importer.destroy_rollback_key(
            backup_path=backup_path,
            rollback_key_path=rollback_key_path,
            operator_id=operator_id,
            now=datetime.now(timezone.utc),
            confirmed=confirm,
            request_id=_request_id("whdestroy"),
        )
        return {"result": result}

    _print_json(_run(operation))


@admin_webhooks_group.group("rotate-key")
def rotate_key_group() -> None:
    """Run the forward-only canonical protected-value key rotation."""


def _rotation_command(
    method: Literal["start", "resume", "verify", "finalize"],
    operation_id: str,
    operator_id: int,
    source_key_id: str | None = None,
    target_key_id: str | None = None,
) -> None:
    if method == "start" and (source_key_id is None or target_key_id is None):
        raise click.UsageError("start requires source and target key IDs.")
    request_id = _request_id("whrotate")

    async def operation(
        _importer: LegacyImportService,
        rotation: WebhookKeyRotationService,
        _repository: AdminWebhookRepository,
    ) -> object:
        if method == "start":
            if source_key_id is None or target_key_id is None:
                raise click.UsageError("start requires source and target key IDs.")
            result = await rotation.start(
                operation_id,
                source_key_id,
                target_key_id,
                operator_id=operator_id,
                request_id=request_id,
                audit_sink=emit_mandatory_webhook_operation_audit,
            )
        elif method == "resume":
            result = await rotation.resume(
                operation_id,
                operator_id=operator_id,
                request_id=request_id,
                audit_sink=emit_mandatory_webhook_operation_audit,
            )
        elif method == "verify":
            result = await rotation.verify(
                operation_id,
                operator_id=operator_id,
                request_id=request_id,
                audit_sink=emit_mandatory_webhook_operation_audit,
            )
        else:
            result = await rotation.finalize(
                operation_id,
                operator_id=operator_id,
                request_id=request_id,
                audit_sink=emit_mandatory_webhook_operation_audit,
            )
        return asdict(result)

    _print_json(_run(operation))


@rotate_key_group.command("start")
@click.option("--operation-id", required=True)
@click.option("--source-key-id", required=True)
@click.option("--target-key-id", required=True)
@click.option("--operator-id", required=True, type=click.IntRange(min=1))
def rotate_key_start(
    *,
    operation_id: str,
    source_key_id: str,
    target_key_id: str,
    operator_id: int,
) -> None:
    """Start a protected-value key rotation."""
    _rotation_command(
        "start",
        operation_id,
        operator_id,
        source_key_id,
        target_key_id,
    )


def _register_rotation_command(
    name: Literal["resume", "verify", "finalize"],
) -> None:
    @rotate_key_group.command(name)
    @click.option("--operation-id", required=True)
    @click.option("--operator-id", required=True, type=click.IntRange(min=1))
    def command(*, operation_id: str, operator_id: int) -> None:
        _rotation_command(name, operation_id, operator_id)


_register_rotation_command("resume")
_register_rotation_command("verify")
_register_rotation_command("finalize")


@admin_webhooks_group.command("rotation-status")
def rotation_status() -> None:
    """Show sanitized durable key-rotation progress."""

    async def operation(
        _importer: LegacyImportService,
        _rotation: WebhookKeyRotationService,
        repository: AdminWebhookRepository,
    ) -> object:
        state = await repository.get_migration_state()
        return {
            "active_primary_key_id": state.active_primary_key_id,
            "operation_id": state.rotation_operation_id,
            "phase": state.rotation_phase,
            "processed_count": state.rotation_processed_count,
            "source_key_id": state.rotation_source_key_id,
            "target_key_id": state.rotation_target_key_id,
            "verified_count": state.rotation_verified_count,
        }

    _print_json(_run(operation))
