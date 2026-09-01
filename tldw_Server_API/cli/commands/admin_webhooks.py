"""Offline operator commands for the canonical admin-webhook control plane."""

from __future__ import annotations

import asyncio
import json
import os
import secrets
import sqlite3
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, TypeVar
from urllib.parse import quote

import aiosqlite
import click

from tldw_Server_API.app.core.Admin_Webhooks.audit import (
    emit_mandatory_webhook_operation_audit,
)
from tldw_Server_API.app.core.Admin_Webhooks.config import AdminWebhookSettings
from tldw_Server_API.app.core.Admin_Webhooks.control_plane import (
    AdminWebhookControlPlane,
    evaluate_activation_readiness,
)
from tldw_Server_API.app.core.Admin_Webhooks.crypto import load_webhook_key_ring
from tldw_Server_API.app.core.Admin_Webhooks.domain import (
    AdminWebhookActivationCheck,
    AdminWebhookActivationPhase,
    AdminWebhookActivationReasonCode,
    WebhookError,
)
from tldw_Server_API.app.core.Admin_Webhooks.key_rotation import (
    WebhookKeyRotationService,
)
from tldw_Server_API.app.core.Admin_Webhooks.legacy_import import (
    LegacyImportError,
    LegacyImportRequest,
    LegacyImportService,
    LegacyRejectionReason,
)
from tldw_Server_API.app.core.Admin_Webhooks.observability import (
    AdminWebhookDeliveryCapability,
    JobsCapabilityStatus,
)
from tldw_Server_API.app.core.AuthNZ.database import (
    DatabasePool,
    _apply_single_user_fallback,
)
from tldw_Server_API.app.core.AuthNZ.settings import Settings
from tldw_Server_API.app.core.DB_Management.admin_webhooks_repository import (
    AdminWebhookRepository,
)
from tldw_Server_API.app.core.Jobs.operations.contracts import (
    ADMIN_WEBHOOK_DELIVERY_DOMAIN,
    ADMIN_WEBHOOK_DELIVERY_JOB_TYPE,
    ADMIN_WEBHOOK_DELIVERY_QUEUE,
)
from tldw_Server_API.app.core.Utils.Utils import get_project_root
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


class _ReadOnlyJobsCapabilityProbe:
    """Inspect configured Jobs readiness without creating or migrating tables."""

    @staticmethod
    def _configured_job_type_ready() -> bool:
        allowed = {
            item.strip()
            for variable in (
                "JOBS_ALLOWED_JOB_TYPES",
                "JOBS_ALLOWED_JOB_TYPES_ADMIN_WEBHOOKS",
            )
            for item in os.getenv(variable, "").split(",")
            if item.strip()
        }
        return not allowed or ADMIN_WEBHOOK_DELIVERY_JOB_TYPE in allowed

    @staticmethod
    def _sqlite_path() -> Path:
        configured = str(os.getenv("JOBS_DB_PATH") or "").strip()
        if configured:
            path = Path(configured).expanduser()
            if not path.is_absolute():
                path = Path(get_project_root()) / path
            return path.resolve(strict=False)
        return (Path(get_project_root()) / "Databases" / "jobs.db").resolve(strict=False)

    @staticmethod
    def _sqlite_database_ready(path: Path) -> bool:
        if not path.is_file():
            return False
        uri = f"file:{quote(str(path), safe='/')}?mode=ro"
        try:
            with sqlite3.connect(uri, uri=True) as connection:
                connection.execute("SELECT 1 FROM jobs LIMIT 0").fetchone()
        except (OSError, sqlite3.Error):
            return False
        return True

    @staticmethod
    def _postgres_database_ready(dsn: str) -> bool:
        try:
            import psycopg

            with psycopg.connect(
                dsn,
                autocommit=True,
                connect_timeout=5,
            ) as connection:
                connection.execute("SELECT 1 FROM jobs LIMIT 0").fetchone()
        except Exception:  # noqa: BLE001 - readiness is deliberately fail-closed
            return False
        return True

    async def status(self) -> JobsCapabilityStatus:
        from tldw_Server_API.app.core.Jobs.manager import JobManager

        queues = getattr(JobManager, "DOMAIN_ALLOWED_QUEUES", {})
        queue_ready = isinstance(queues, dict) and ADMIN_WEBHOOK_DELIVERY_QUEUE in queues.get(
            ADMIN_WEBHOOK_DELIVERY_DOMAIN, ()
        )
        job_type_ready = bool(callable(getattr(JobManager, "admit_job", None)) and self._configured_job_type_ready())
        dsn = str(os.getenv("JOBS_DB_URL") or "").strip()
        if dsn.startswith(("postgres://", "postgresql://")):
            backend = "postgres"
            database_ready = await asyncio.to_thread(
                self._postgres_database_ready,
                dsn,
            )
        elif dsn:
            backend = "unavailable"
            database_ready = False
        else:
            backend = "sqlite"
            database_ready = await asyncio.to_thread(
                self._sqlite_database_ready,
                self._sqlite_path(),
            )
        return JobsCapabilityStatus(
            database_ready=database_ready,
            queue_ready=queue_ready,
            job_type_ready=job_type_ready,
            backend=backend,
        )


class _ReadOnlyAdminWebhookPool:
    """Minimal AuthNZ pool adapter that cannot initialize or mutate schema."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.pool: object | None = None
        self.db_path: str | None = None
        self._sqlite_uri = False
        self._sqlite_fs_path: str | None = None
        self._initialized = False
        self._resolver = DatabasePool(settings)

    async def initialize(self) -> None:
        if self._initialized:
            return
        if self._resolver._should_use_postgres():
            import asyncpg

            self.pool = await asyncpg.create_pool(
                self.settings.DATABASE_URL,
                min_size=getattr(self.settings, "DATABASE_POOL_MIN_SIZE", 1),
                max_size=getattr(self.settings, "DATABASE_POOL_MAX_SIZE", 10),
                max_queries=getattr(self.settings, "DATABASE_MAX_QUERIES", 50_000),
                max_inactive_connection_lifetime=getattr(
                    self.settings,
                    "DATABASE_MAX_INACTIVE_CONNECTION_LIFETIME",
                    300,
                ),
                command_timeout=60,
                server_settings={"default_transaction_read_only": "on"},
            )
            self._initialized = True
            return

        raw_url = _apply_single_user_fallback(
            self.settings.DATABASE_URL,
            auth_mode=getattr(self.settings, "AUTH_MODE", "single_user"),
        )
        _db_path, _uri, filesystem_path = DatabasePool._resolve_sqlite_paths(raw_url)
        if not filesystem_path or filesystem_path == ":memory:":
            raise OSError("admin webhook database unavailable")
        path = Path(filesystem_path).expanduser().resolve(strict=False)
        if not path.is_file():
            raise OSError("admin webhook database unavailable")
        self._sqlite_fs_path = str(path)
        self.db_path = f"file:{quote(str(path), safe='/')}?mode=ro"
        self._sqlite_uri = True
        self._initialized = True

    @asynccontextmanager
    async def acquire(self, *, timeout: float | None = None) -> AsyncIterator[object]:
        if not self._initialized:
            await self.initialize()
        if self.pool is not None:
            connection = await self.pool.acquire(timeout=timeout)  # type: ignore[attr-defined]
            try:
                yield connection
            finally:
                await self.pool.release(connection)  # type: ignore[attr-defined]
            return

        connection = await aiosqlite.connect(self.db_path, uri=self._sqlite_uri)
        try:
            await connection.execute("PRAGMA query_only=ON")
            await connection.execute("PRAGMA foreign_keys=ON")
            await connection.execute("PRAGMA busy_timeout=5000")
            connection.row_factory = aiosqlite.Row
            yield connection
        finally:
            await connection.close()

    async def close(self) -> None:
        if self.pool is not None:
            await self.pool.close()  # type: ignore[attr-defined]
        self.pool = None
        self._initialized = False


async def _with_activation_check(
    *,
    phase: AdminWebhookActivationPhase,
    settings: AdminWebhookSettings,
) -> AdminWebhookActivationCheck:
    pool = _ReadOnlyAdminWebhookPool(Settings())
    await pool.initialize()
    try:
        repository = AdminWebhookRepository(pool)
        key_ring_result = load_webhook_key_ring()
        delivery_capability = AdminWebhookDeliveryCapability(
            repository=repository,
            key_ring_result=key_ring_result,
            jobs_probe=_ReadOnlyJobsCapabilityProbe(),
            heartbeat_freshness_seconds=(settings.delivery_heartbeat_freshness_seconds),
        )
        control_plane = AdminWebhookControlPlane(
            repository=repository,
            settings=settings,
            key_ring_result=key_ring_result,
            delivery_capability=delivery_capability,
        )
        status = await control_plane.status(now=datetime.now(timezone.utc))
        return evaluate_activation_readiness(
            status,
            phase=phase,
            max_backlog_age_seconds=(settings.activation_max_backlog_age_seconds),
        )
    finally:
        await pool.close()


def _closed_activation_check(
    *,
    phase: AdminWebhookActivationPhase,
    settings: AdminWebhookSettings,
) -> AdminWebhookActivationCheck:
    expected_mode = "migrate" if phase is AdminWebhookActivationPhase.PREDEPLOY else "on"
    reasons = []
    if settings.mode.value != expected_mode:
        reasons.append(AdminWebhookActivationReasonCode.PHASE_MISMATCH)
    reasons.append(AdminWebhookActivationReasonCode.DATABASE_UNAVAILABLE)
    return AdminWebhookActivationCheck(
        phase=phase,
        ready=False,
        mode=settings.mode.value,
        schema_ready=False,
        migration_complete=False,
        key_ready=False,
        jobs_ready=False,
        limits_ready=False,
        worker_ready=False,
        reconciler_ready=False,
        retention_ready=False,
        runtime_ready=False,
        backlog_age_ready=False,
        oldest_nonterminal_age_seconds=None,
        max_backlog_age_seconds=settings.activation_max_backlog_age_seconds,
        reason_codes=tuple(reasons),
    )


def _run_activation_check(
    phase: AdminWebhookActivationPhase,
) -> AdminWebhookActivationCheck:
    try:
        settings = AdminWebhookSettings.from_environment(os.environ)
    except Exception:  # noqa: BLE001 - never expose configuration details
        raise click.ClickException("admin_webhook_configuration_invalid") from None
    try:
        return asyncio.run(_with_activation_check(phase=phase, settings=settings))
    except Exception:  # noqa: BLE001 - unavailable state is closed and sanitized
        return _closed_activation_check(phase=phase, settings=settings)


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


@admin_webhooks_group.command("activation-check")
@click.option(
    "--phase",
    required=True,
    type=click.Choice([value.value for value in AdminWebhookActivationPhase]),
)
def activation_check(*, phase: str) -> None:
    """Evaluate one read-only canonical activation phase."""
    result = _run_activation_check(AdminWebhookActivationPhase(phase))
    _print_json(asdict(result))
    if not result.ready:
        raise click.exceptions.Exit(1)


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
