import asyncio
import contextlib

import pytest

from tldw_Server_API.app.services import claims_alerts_scheduler
from tldw_Server_API.app.services import claims_review_metrics_scheduler
from tldw_Server_API.app.services import admin_backup_jobs_worker
from tldw_Server_API.app.services import admin_byok_validation_jobs_worker
from tldw_Server_API.app.services import connectors_worker
from tldw_Server_API.app.services import file_artifacts_export_gc_service
from tldw_Server_API.app.services import ingestion_sources_cleanup_service
from tldw_Server_API.app.services import kanban_activity_cleanup_service
from tldw_Server_API.app.services import kanban_purge_service
from tldw_Server_API.app.services import notifications_prune_service
from tldw_Server_API.app.services import quality_eval_scheduler
from tldw_Server_API.app.services import reminder_jobs_worker
from tldw_Server_API.app.services import reminders_scheduler


pytestmark = pytest.mark.unit


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("env_var", "start_fn"),
    [
        ("INGESTION_SOURCES_CLEANUP_ENABLED", ingestion_sources_cleanup_service.start_ingestion_sources_cleanup_scheduler),
        ("KANBAN_ACTIVITY_CLEANUP_ENABLED", kanban_activity_cleanup_service.start_kanban_activity_cleanup_scheduler),
        ("KANBAN_PURGE_ENABLED", kanban_purge_service.start_kanban_purge_scheduler),
        ("FILES_EXPORT_GC_ENABLED", file_artifacts_export_gc_service.start_file_artifacts_export_gc_scheduler),
        ("NOTIFICATIONS_PRUNE_ENABLED", notifications_prune_service.start_notifications_prune_scheduler),
        ("REMINDERS_SCHEDULER_ENABLED", reminders_scheduler.start_reminders_scheduler),
        ("RAG_QUALITY_EVAL_ENABLED", quality_eval_scheduler.start_quality_eval_scheduler),
        ("REMINDER_JOBS_WORKER_ENABLED", reminder_jobs_worker.start_reminder_jobs_worker),
        ("ADMIN_BACKUP_JOBS_WORKER_ENABLED", admin_backup_jobs_worker.start_admin_backup_jobs_worker),
        (
            "ADMIN_BYOK_VALIDATION_JOBS_WORKER_ENABLED",
            admin_byok_validation_jobs_worker.start_admin_byok_validation_jobs_worker,
        ),
        ("CONNECTORS_WORKER_ENABLED", connectors_worker.start_connectors_worker),
        ("CLAIMS_ALERTS_SCHEDULER_ENABLED", claims_alerts_scheduler.start_claims_alerts_scheduler),
        ("CLAIMS_REVIEW_METRICS_SCHEDULER_ENABLED", claims_review_metrics_scheduler.start_claims_review_metrics_scheduler),
    ],
)
async def test_service_startup_flags_accept_single_letter_y(monkeypatch: pytest.MonkeyPatch, env_var: str, start_fn):
    monkeypatch.setenv(env_var, "y")

    if start_fn is connectors_worker.start_connectors_worker:
        async def _fake_connectors_worker(_stop_event=None):
            await asyncio.sleep(3600)

        monkeypatch.setattr(connectors_worker, "run_connectors_worker", _fake_connectors_worker)
    elif start_fn is reminder_jobs_worker.start_reminder_jobs_worker:
        async def _fake_reminder_jobs_worker(_stop_event=None):
            await asyncio.sleep(3600)

        monkeypatch.setattr(
            reminder_jobs_worker,
            "run_reminder_jobs_worker",
            _fake_reminder_jobs_worker,
        )
    elif start_fn is admin_backup_jobs_worker.start_admin_backup_jobs_worker:
        async def _fake_admin_backup_jobs_worker(_stop_event=None):
            await asyncio.sleep(3600)

        monkeypatch.setattr(
            admin_backup_jobs_worker,
            "run_admin_backup_jobs_worker",
            _fake_admin_backup_jobs_worker,
        )
    elif start_fn is admin_byok_validation_jobs_worker.start_admin_byok_validation_jobs_worker:
        async def _fake_admin_byok_jobs_worker(_stop_event=None):
            await asyncio.sleep(3600)

        monkeypatch.setattr(
            admin_byok_validation_jobs_worker,
            "run_admin_byok_validation_jobs_worker",
            _fake_admin_byok_jobs_worker,
        )

    task = await start_fn()
    assert task is not None
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("env_var", "start_fn"),
    [
        ("INGESTION_SOURCES_CLEANUP_ENABLED", ingestion_sources_cleanup_service.start_ingestion_sources_cleanup_scheduler),
        ("KANBAN_ACTIVITY_CLEANUP_ENABLED", kanban_activity_cleanup_service.start_kanban_activity_cleanup_scheduler),
        ("KANBAN_PURGE_ENABLED", kanban_purge_service.start_kanban_purge_scheduler),
        ("FILES_EXPORT_GC_ENABLED", file_artifacts_export_gc_service.start_file_artifacts_export_gc_scheduler),
        ("NOTIFICATIONS_PRUNE_ENABLED", notifications_prune_service.start_notifications_prune_scheduler),
        ("REMINDERS_SCHEDULER_ENABLED", reminders_scheduler.start_reminders_scheduler),
        ("RAG_QUALITY_EVAL_ENABLED", quality_eval_scheduler.start_quality_eval_scheduler),
        ("REMINDER_JOBS_WORKER_ENABLED", reminder_jobs_worker.start_reminder_jobs_worker),
        ("ADMIN_BACKUP_JOBS_WORKER_ENABLED", admin_backup_jobs_worker.start_admin_backup_jobs_worker),
        (
            "ADMIN_BYOK_VALIDATION_JOBS_WORKER_ENABLED",
            admin_byok_validation_jobs_worker.start_admin_byok_validation_jobs_worker,
        ),
        ("CONNECTORS_WORKER_ENABLED", connectors_worker.start_connectors_worker),
    ],
)
async def test_service_startup_flags_remain_disabled_when_unset(monkeypatch: pytest.MonkeyPatch, env_var: str, start_fn):
    monkeypatch.delenv(env_var, raising=False)

    if start_fn is connectors_worker.start_connectors_worker:
        async def _should_not_run(*_args, **_kwargs):
            raise AssertionError("connectors worker should not start when disabled")

        monkeypatch.setattr(connectors_worker, "run_connectors_worker", _should_not_run)
    elif start_fn is reminder_jobs_worker.start_reminder_jobs_worker:
        async def _should_not_run(*_args, **_kwargs):
            raise AssertionError("reminder jobs worker should not start when disabled")

        monkeypatch.setattr(reminder_jobs_worker, "run_reminder_jobs_worker", _should_not_run)
    elif start_fn is admin_backup_jobs_worker.start_admin_backup_jobs_worker:
        async def _should_not_run(*_args, **_kwargs):
            raise AssertionError("admin backup jobs worker should not start when disabled")

        monkeypatch.setattr(admin_backup_jobs_worker, "run_admin_backup_jobs_worker", _should_not_run)
    elif start_fn is admin_byok_validation_jobs_worker.start_admin_byok_validation_jobs_worker:
        async def _should_not_run(*_args, **_kwargs):
            raise AssertionError("admin byok validation jobs worker should not start when disabled")

        monkeypatch.setattr(
            admin_byok_validation_jobs_worker,
            "run_admin_byok_validation_jobs_worker",
            _should_not_run,
        )

    task = await start_fn()
    assert task is None
