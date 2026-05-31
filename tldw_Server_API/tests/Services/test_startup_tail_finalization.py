from __future__ import annotations

import importlib
import sys
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.unit


def _import_startup_tail_finalization():
    sys.modules.pop("tldw_Server_API.app.services.startup_tail_finalization", None)
    return importlib.import_module("tldw_Server_API.app.services.startup_tail_finalization")


@pytest.mark.asyncio
async def test_finalize_startup_tail_refreshes_inventory_and_starts_recurring_schedulers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    startup_tail = _import_startup_tail_finalization()
    app = object()
    owned_job_pollers: list[object] = []
    observed: dict[str, object] = {}

    worker_group_handles = SimpleNamespace(
        core_jobs_task="core-task",
        core_jobs_stop_event="core-stop",
        files_jobs_task="files-task",
        files_jobs_stop_event="files-stop",
        data_tables_jobs_task="data-tables-task",
        data_tables_jobs_stop_event="data-tables-stop",
        prompt_studio_jobs_task="prompt-studio-task",
        prompt_studio_jobs_stop_event="prompt-studio-stop",
        study_pack_jobs_task="study-pack-task",
        study_pack_jobs_stop_event="study-pack-stop",
        study_suggestions_jobs_task="study-suggestions-task",
        study_suggestions_jobs_stop_event="study-suggestions-stop",
        privilege_snapshot_task="privilege-task",
        privilege_snapshot_stop_event="privilege-stop",
        audio_jobs_task="audio-task",
        audio_jobs_stop_event="audio-stop",
        audiobook_jobs_task="audiobook-task",
        audiobook_jobs_stop_event="audiobook-stop",
        presentation_render_jobs_task="presentation-task",
        presentation_render_jobs_stop_event="presentation-stop",
        media_ingest_jobs_task="media-ingest-task",
        media_ingest_jobs_stop_event="media-ingest-stop",
        media_ingest_heavy_jobs_task="media-heavy-task",
        media_ingest_heavy_jobs_stop_event="media-heavy-stop",
        reading_digest_jobs_task="reading-digest-task",
        reading_digest_jobs_stop_event="reading-digest-stop",
        vn_asset_jobs_task="vn-asset-task",
        vn_asset_jobs_stop_event="vn-asset-stop",
        vn_asset_generation_jobs_task="vn-generation-task",
        vn_asset_generation_jobs_stop_event="vn-generation-stop",
        companion_reflection_jobs_task="companion-task",
        companion_reflection_jobs_stop_event="companion-stop",
        reminder_jobs_task="reminder-task",
        reminder_jobs_stop_event="reminder-stop",
        admin_backup_jobs_task="admin-backup-task",
        admin_backup_jobs_stop_event="admin-backup-stop",
        admin_byok_validation_jobs_task="admin-byok-task",
        admin_byok_validation_jobs_stop_event="admin-byok-stop",
        admin_maintenance_rotation_jobs_task="admin-rotation-task",
        admin_maintenance_rotation_jobs_stop_event="admin-rotation-stop",
        recipe_run_jobs_task="recipe-task",
        recipe_run_jobs_stop_event="recipe-stop",
        evals_abtest_jobs_task="evals-task",
        evals_abtest_jobs_stop_event="evals-stop",
    )
    service_group_handles = SimpleNamespace(
        connectors_jobs_task="connectors-task",
        connectors_jobs_stop_event="connectors-stop",
    )

    def _fake_replace_owned_job_poller_inventory(
        seen_app,
        seen_owned_job_pollers,
        *,
        registrations,
    ):
        observed["app"] = seen_app
        observed["owned_job_pollers"] = seen_owned_job_pollers
        observed["registrations"] = registrations

    async def _fake_start_recurring_schedulers(*, test_mode: bool):
        observed["test_mode"] = test_mode
        return SimpleNamespace(
            authnz_scheduler_started=True,
            workflows_sched_task="workflows-task",
            reading_digest_sched_task="reading-digest-scheduler-task",
            admin_backup_sched_task="admin-backup-scheduler-task",
            companion_reflection_sched_task="companion-scheduler-task",
            reminders_sched_task="reminders-scheduler-task",
            connectors_sync_sched_task="connectors-sync-scheduler-task",
        )

    monkeypatch.setattr(
        startup_tail,
        "_start_recurring_schedulers",
        _fake_start_recurring_schedulers,
    )

    handles = await startup_tail.finalize_startup_tail(
        app=app,
        owned_job_pollers=owned_job_pollers,
        startup_worker_group_handles=worker_group_handles,
        startup_service_group_handles=service_group_handles,
        replace_owned_job_poller_inventory=_fake_replace_owned_job_poller_inventory,
        test_mode=True,
    )

    assert observed["app"] is app
    assert observed["owned_job_pollers"] is owned_job_pollers
    assert observed["test_mode"] is True
    assert observed["registrations"] == [
        ("core_jobs_task", "core-task", "core-stop", 5.0),
        ("files_jobs_task", "files-task", "files-stop", 5.0),
        ("data_tables_jobs_task", "data-tables-task", "data-tables-stop", 5.0),
        ("prompt_studio_jobs_task", "prompt-studio-task", "prompt-studio-stop", 5.0),
        ("study_pack_jobs_task", "study-pack-task", "study-pack-stop", 5.0),
        ("study_suggestions_jobs_task", "study-suggestions-task", "study-suggestions-stop", 5.0),
        ("privilege_snapshot_task", "privilege-task", "privilege-stop", 5.0),
        ("audio_jobs_task", "audio-task", "audio-stop", 5.0),
        ("audiobook_jobs_task", "audiobook-task", "audiobook-stop", 5.0),
        ("presentation_render_jobs_task", "presentation-task", "presentation-stop", 5.0),
        ("media_ingest_jobs_task", "media-ingest-task", "media-ingest-stop", 5.0),
        ("media_ingest_heavy_jobs_task", "media-heavy-task", "media-heavy-stop", 5.0),
        ("reading_digest_jobs_task", "reading-digest-task", "reading-digest-stop", 5.0),
        ("vn_asset_jobs_task", "vn-asset-task", "vn-asset-stop", 5.0),
        ("vn_asset_generation_jobs_task", "vn-generation-task", "vn-generation-stop", 5.0),
        ("companion_reflection_jobs_task", "companion-task", "companion-stop", 5.0),
        ("reminder_jobs_task", "reminder-task", "reminder-stop", 5.0),
        ("admin_backup_jobs_task", "admin-backup-task", "admin-backup-stop", 5.0),
        ("admin_byok_validation_jobs_task", "admin-byok-task", "admin-byok-stop", 5.0),
        ("admin_maintenance_rotation_jobs_task", "admin-rotation-task", "admin-rotation-stop", 5.0),
        ("recipe_run_jobs_task", "recipe-task", "recipe-stop", 5.0),
        ("evals_abtest_jobs_task", "evals-task", "evals-stop", 5.0),
        ("connectors_jobs_task", "connectors-task", "connectors-stop", 5.0),
    ]
    assert handles.authnz_scheduler_started is True
    assert handles.workflows_sched_task == "workflows-task"
    assert handles.connectors_sync_sched_task == "connectors-sync-scheduler-task"


@pytest.mark.asyncio
async def test_finalize_startup_tail_passes_worker_inventory_to_recurring_schedulers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    startup_tail = _import_startup_tail_finalization()
    worker_inventory = object()
    observed: dict[str, object] = {}

    def _fake_replace_owned_job_poller_inventory(*args, **kwargs):
        return None

    async def _fake_start_recurring_schedulers(**kwargs):
        observed.update(kwargs)
        return SimpleNamespace(
            authnz_scheduler_started=False,
            workflows_sched_task=None,
            reading_digest_sched_task=None,
            admin_backup_sched_task=None,
            companion_reflection_sched_task=None,
            reminders_sched_task=None,
            connectors_sync_sched_task=None,
        )

    empty_worker_group_handles = SimpleNamespace(
        core_jobs_task=None,
        core_jobs_stop_event=None,
        files_jobs_task=None,
        files_jobs_stop_event=None,
        data_tables_jobs_task=None,
        data_tables_jobs_stop_event=None,
        prompt_studio_jobs_task=None,
        prompt_studio_jobs_stop_event=None,
        study_pack_jobs_task=None,
        study_pack_jobs_stop_event=None,
        study_suggestions_jobs_task=None,
        study_suggestions_jobs_stop_event=None,
        privilege_snapshot_task=None,
        privilege_snapshot_stop_event=None,
        audio_jobs_task=None,
        audio_jobs_stop_event=None,
        audiobook_jobs_task=None,
        audiobook_jobs_stop_event=None,
        presentation_render_jobs_task=None,
        presentation_render_jobs_stop_event=None,
        media_ingest_jobs_task=None,
        media_ingest_jobs_stop_event=None,
        media_ingest_heavy_jobs_task=None,
        media_ingest_heavy_jobs_stop_event=None,
        reading_digest_jobs_task=None,
        reading_digest_jobs_stop_event=None,
        vn_asset_jobs_task=None,
        vn_asset_jobs_stop_event=None,
        vn_asset_generation_jobs_task=None,
        vn_asset_generation_jobs_stop_event=None,
        companion_reflection_jobs_task=None,
        companion_reflection_jobs_stop_event=None,
        reminder_jobs_task=None,
        reminder_jobs_stop_event=None,
        admin_backup_jobs_task=None,
        admin_backup_jobs_stop_event=None,
        admin_byok_validation_jobs_task=None,
        admin_byok_validation_jobs_stop_event=None,
        admin_maintenance_rotation_jobs_task=None,
        admin_maintenance_rotation_jobs_stop_event=None,
        recipe_run_jobs_task=None,
        recipe_run_jobs_stop_event=None,
        evals_abtest_jobs_task=None,
        evals_abtest_jobs_stop_event=None,
    )

    monkeypatch.setattr(
        startup_tail,
        "_start_recurring_schedulers",
        _fake_start_recurring_schedulers,
    )

    await startup_tail.finalize_startup_tail(
        app=object(),
        owned_job_pollers=[],
        startup_worker_group_handles=empty_worker_group_handles,
        startup_service_group_handles=SimpleNamespace(
            connectors_jobs_task=None,
            connectors_jobs_stop_event=None,
        ),
        replace_owned_job_poller_inventory=_fake_replace_owned_job_poller_inventory,
        test_mode=True,
        worker_inventory=worker_inventory,
    )

    assert observed == {"test_mode": True, "worker_inventory": worker_inventory}
