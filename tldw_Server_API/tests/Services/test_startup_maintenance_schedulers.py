from __future__ import annotations

import importlib
import sys

import pytest


pytestmark = pytest.mark.unit


def _import_startup_maintenance_schedulers():
    sys.modules.pop("tldw_Server_API.app.services.startup_maintenance_schedulers", None)
    return importlib.import_module("tldw_Server_API.app.services.startup_maintenance_schedulers")


@pytest.mark.asyncio
async def test_start_maintenance_schedulers_combines_handles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    startup_maintenance = _import_startup_maintenance_schedulers()
    calls: list[str] = []

    async def _fake_quality():
        calls.append("quality")
        return "quality-task"

    async def _fake_outputs():
        calls.append("outputs")
        return "outputs-task"

    async def _fake_kanban_activity():
        calls.append("kanban-activity")
        return "kanban-activity-task"

    async def _fake_ingestion_sources():
        calls.append("ingestion-sources")
        return "ingestion-sources-task"

    async def _fake_kanban_purge():
        calls.append("kanban-purge")
        return "kanban-purge-task"

    async def _fake_files_gc():
        calls.append("files-gc")
        return "files-gc-task"

    async def _fake_notifications():
        calls.append("notifications")
        return "notifications-task"

    async def _fake_jobs_prune():
        calls.append("jobs-prune")
        return "jobs-prune-task"

    monkeypatch.setattr(startup_maintenance, "_start_quality_eval_scheduler", _fake_quality)
    monkeypatch.setattr(startup_maintenance, "_start_outputs_purge_scheduler", _fake_outputs)
    monkeypatch.setattr(startup_maintenance, "_start_kanban_activity_cleanup_scheduler", _fake_kanban_activity)
    monkeypatch.setattr(startup_maintenance, "_start_ingestion_sources_cleanup_scheduler", _fake_ingestion_sources)
    monkeypatch.setattr(startup_maintenance, "_start_kanban_purge_scheduler", _fake_kanban_purge)
    monkeypatch.setattr(startup_maintenance, "_start_file_artifacts_export_gc_scheduler", _fake_files_gc)
    monkeypatch.setattr(startup_maintenance, "_start_notifications_prune_scheduler", _fake_notifications)
    monkeypatch.setattr(startup_maintenance, "_start_jobs_prune_scheduler", _fake_jobs_prune)

    handles = await startup_maintenance.start_maintenance_schedulers()

    assert calls == [
        "quality",
        "outputs",
        "kanban-activity",
        "ingestion-sources",
        "kanban-purge",
        "files-gc",
        "notifications",
        "jobs-prune",
    ]
    assert handles.quality_eval_task == "quality-task"
    assert handles.outputs_purge_task == "outputs-task"
    assert handles.kanban_activity_cleanup_task == "kanban-activity-task"
    assert handles.ingestion_sources_cleanup_task == "ingestion-sources-task"
    assert handles.kanban_purge_task == "kanban-purge-task"
    assert handles.files_export_gc_task == "files-gc-task"
    assert handles.notifications_prune_task == "notifications-task"
    assert handles.jobs_prune_task == "jobs-prune-task"


@pytest.mark.asyncio
async def test_start_quality_eval_scheduler_skips_when_flag_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    startup_maintenance = _import_startup_maintenance_schedulers()

    monkeypatch.setattr(startup_maintenance, "_env_enabled", lambda key: False)

    task = await startup_maintenance._start_quality_eval_scheduler()

    assert task is None


@pytest.mark.asyncio
async def test_start_jobs_prune_scheduler_starts_when_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    startup_maintenance = _import_startup_maintenance_schedulers()

    async def _fake_start():
        return "jobs-prune-task"

    monkeypatch.setattr(startup_maintenance, "_env_enabled", lambda key: True)
    monkeypatch.setattr(startup_maintenance, "_start_jobs_prune_scheduler_service", _fake_start)

    task = await startup_maintenance._start_jobs_prune_scheduler()

    assert task == "jobs-prune-task"
