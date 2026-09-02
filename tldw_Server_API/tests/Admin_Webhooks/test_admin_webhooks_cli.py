from __future__ import annotations

import json
import sqlite3
from types import SimpleNamespace

import click
import pytest
from click.testing import CliRunner

from tldw_Server_API.app.core.Admin_Webhooks.domain import (
    AdminWebhookActivationCheck,
    AdminWebhookActivationPhase,
    AdminWebhookActivationReasonCode,
    WebhookError,
    WebhookErrorCode,
)
from tldw_Server_API.cli.commands import admin_webhooks


@pytest.mark.unit
def test_command_tree_exposes_import_rollback_and_rotation_operations() -> None:
    runner = CliRunner()

    root = runner.invoke(admin_webhooks.admin_webhooks_group, ["--help"])
    rotation = runner.invoke(
        admin_webhooks.admin_webhooks_group,
        ["rotate-key", "--help"],
    )

    assert root.exit_code == 0
    assert {
        "destroy-rollback-key",
        "extract-rollback-backup",
        "import-legacy",
        "activation-check",
        "reject-source",
        "rotate-key",
        "rotation-status",
    }.issubset(root.output.split())
    assert rotation.exit_code == 0
    assert {"finalize", "resume", "start", "verify"}.issubset(rotation.output.split())


@pytest.mark.unit
def test_import_apply_requires_quiescence_before_runtime_initialization(
    monkeypatch,
) -> None:
    runtime_called = False

    def fail_if_called(_operation):
        nonlocal runtime_called
        runtime_called = True
        raise AssertionError("runtime must not initialize")

    monkeypatch.setattr(admin_webhooks, "_run", fail_if_called)

    result = CliRunner().invoke(
        admin_webhooks.admin_webhooks_group,
        [
            "import-legacy",
            "--apply",
            "--approved-report-digest",
            "sha256:" + ("a" * 64),
            "--report",
            "report.json",
            "--operator-id",
            "9",
        ],
    )

    assert result.exit_code == 2
    assert "--apply requires --all-writers-quiesced" in result.output
    assert runtime_called is False


@pytest.mark.unit
def test_import_apply_requires_literal_digest_before_runtime_initialization(
    monkeypatch,
) -> None:
    runtime_called = False

    def fail_if_called(_operation):
        nonlocal runtime_called
        runtime_called = True
        raise AssertionError("runtime must not initialize")

    monkeypatch.setattr(admin_webhooks, "_run", fail_if_called)

    result = CliRunner().invoke(
        admin_webhooks.admin_webhooks_group,
        [
            "import-legacy",
            "--apply",
            "--all-writers-quiesced",
            "--report",
            "report.json",
            "--operator-id",
            "9",
        ],
    )

    assert result.exit_code == 2
    assert "--apply requires --approved-report-digest" in result.output
    assert runtime_called is False


@pytest.mark.unit
def test_runtime_preserves_closed_key_rotation_error_code(monkeypatch) -> None:
    async def failing_runtime(_operation):
        raise WebhookError(WebhookErrorCode.KEY_UNAVAILABLE)

    monkeypatch.setattr(admin_webhooks, "_with_runtime", failing_runtime)

    with pytest.raises(click.ClickException) as caught:
        admin_webhooks._run(lambda _importer, _rotation, _repository: None)

    assert str(caught.value) == "admin_webhook_key_unavailable"


def _activation_result(
    *,
    ready: bool,
) -> AdminWebhookActivationCheck:
    reasons = () if ready else (AdminWebhookActivationReasonCode.WORKER_UNAVAILABLE,)
    return AdminWebhookActivationCheck(
        phase=AdminWebhookActivationPhase.LIVE,
        ready=ready,
        mode="on",
        schema_ready=True,
        migration_complete=True,
        key_ready=True,
        jobs_ready=True,
        limits_ready=True,
        worker_ready=ready,
        reconciler_ready=True,
        retention_ready=True,
        runtime_ready=ready,
        backlog_age_ready=True,
        oldest_nonterminal_age_seconds=0,
        max_backlog_age_seconds=300,
        reason_codes=reasons,
    )


@pytest.mark.unit
@pytest.mark.parametrize(("ready", "expected_exit"), [(True, 0), (False, 1)])
def test_activation_check_outputs_closed_readiness_and_exit_status(
    monkeypatch,
    ready: bool,
    expected_exit: int,
) -> None:
    monkeypatch.setattr(
        admin_webhooks,
        "_run_activation_check",
        lambda phase: _activation_result(ready=ready),
        raising=False,
    )

    result = CliRunner().invoke(
        admin_webhooks.admin_webhooks_group,
        ["activation-check", "--phase", "live"],
    )

    assert result.exit_code == expected_exit
    payload = json.loads(result.output)
    assert payload["phase"] == "live"
    assert payload["ready"] is ready
    assert payload["reason_codes"] == ([] if ready else ["worker_unavailable"])
    assert set(payload) == {
        "backlog_age_ready",
        "jobs_ready",
        "key_ready",
        "limits_ready",
        "max_backlog_age_seconds",
        "migration_complete",
        "mode",
        "oldest_nonterminal_age_seconds",
        "phase",
        "ready",
        "reason_codes",
        "reconciler_ready",
        "retention_ready",
        "runtime_ready",
        "schema_ready",
        "worker_ready",
    }


@pytest.mark.unit
def test_activation_check_requires_explicit_phase() -> None:
    result = CliRunner().invoke(
        admin_webhooks.admin_webhooks_group,
        ["activation-check"],
    )

    assert result.exit_code == 2
    assert "Missing option '--phase'" in result.output


@pytest.mark.unit
def test_activation_check_logs_unexpected_failure_with_exception_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    failure = RuntimeError("private activation failure")
    observed: dict[str, object] = {}

    class BoundLogger:
        def error(self, message: str, *values: object) -> None:
            observed["message"] = message
            observed["values"] = values

    class Logger:
        def opt(self, *, exception: BaseException) -> BoundLogger:
            observed["exception"] = exception
            return BoundLogger()

    async def fail_activation_check(**_kwargs: object) -> AdminWebhookActivationCheck:
        raise failure

    settings = admin_webhooks.AdminWebhookSettings.from_environment({})
    monkeypatch.setattr(
        admin_webhooks.AdminWebhookSettings,
        "from_environment",
        lambda _env: settings,
    )
    monkeypatch.setattr(admin_webhooks, "_with_activation_check", fail_activation_check)
    monkeypatch.setattr(admin_webhooks, "logger", Logger(), raising=False)

    result = admin_webhooks._run_activation_check(AdminWebhookActivationPhase.LIVE)

    assert result.ready is False
    assert observed == {
        "exception": failure,
        "message": "Admin webhook activation check failed phase={}",
        "values": ("live",),
    }


@pytest.mark.unit
@pytest.mark.asyncio
async def test_activation_jobs_probe_is_read_only_and_does_not_create_missing_database(
    monkeypatch,
    tmp_path,
) -> None:
    jobs_path = tmp_path / "jobs.db"
    with sqlite3.connect(jobs_path) as connection:
        connection.execute("CREATE TABLE jobs (id INTEGER PRIMARY KEY)")
    original = jobs_path.read_bytes()
    monkeypatch.setenv("JOBS_DB_PATH", str(jobs_path))
    monkeypatch.delenv("JOBS_DB_URL", raising=False)
    monkeypatch.delenv("JOBS_ALLOWED_JOB_TYPES", raising=False)
    monkeypatch.delenv("JOBS_ALLOWED_JOB_TYPES_ADMIN_WEBHOOKS", raising=False)

    available = await admin_webhooks._ReadOnlyJobsCapabilityProbe().status()

    assert available.database_ready is True
    assert available.queue_ready is True
    assert available.job_type_ready is True
    assert available.backend == "sqlite"
    assert jobs_path.read_bytes() == original

    missing_path = tmp_path / "missing.db"
    monkeypatch.setenv("JOBS_DB_PATH", str(missing_path))

    unavailable = await admin_webhooks._ReadOnlyJobsCapabilityProbe().status()

    assert unavailable.database_ready is False
    assert unavailable.backend == "sqlite"
    assert missing_path.exists() is False


@pytest.mark.unit
@pytest.mark.asyncio
async def test_activation_authnz_pool_is_read_only_and_does_not_create_missing_database(
    tmp_path,
) -> None:
    database_path = tmp_path / "users.db"
    with sqlite3.connect(database_path) as connection:
        connection.execute("CREATE TABLE readiness_marker (value INTEGER)")
        connection.execute("INSERT INTO readiness_marker VALUES (7)")
    original = database_path.read_bytes()
    settings = SimpleNamespace(
        AUTH_MODE="single_user",
        DATABASE_URL=str(database_path),
    )
    pool = admin_webhooks._ReadOnlyAdminWebhookPool(settings)

    await pool.initialize()
    try:
        async with pool.acquire() as connection:
            row = await (await connection.execute("SELECT value FROM readiness_marker")).fetchone()
    finally:
        await pool.close()

    assert row[0] == 7
    assert database_path.read_bytes() == original

    missing_path = tmp_path / "missing-users.db"
    missing_pool = admin_webhooks._ReadOnlyAdminWebhookPool(
        SimpleNamespace(
            AUTH_MODE="single_user",
            DATABASE_URL=str(missing_path),
        )
    )
    with pytest.raises(OSError, match="unavailable"):
        await missing_pool.initialize()
    assert missing_path.exists() is False
