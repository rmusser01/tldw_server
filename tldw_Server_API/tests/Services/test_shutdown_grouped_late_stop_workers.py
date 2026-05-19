from __future__ import annotations

from types import SimpleNamespace

import pytest


pytestmark = pytest.mark.unit


@pytest.mark.asyncio
async def test_shutdown_grouped_late_stop_workers_runs_helpers_in_order_and_returns_handles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app.services import shutdown_grouped_late_stop_workers as shutdown_workers

    calls: list[tuple[str, dict[str, object]]] = []
    should_run_late_stop = lambda *args, **kwargs: True
    guard_exceptions = (RuntimeError,)

    async def _record_media_ingest(**kwargs):
        calls.append(("media_ingest", kwargs))
        return SimpleNamespace(
            media_ingest_jobs_task=kwargs["media_ingest_jobs_task"],
            media_ingest_jobs_stop_event=kwargs["media_ingest_jobs_stop_event"],
            media_ingest_heavy_jobs_task=kwargs["media_ingest_heavy_jobs_task"],
            media_ingest_heavy_jobs_stop_event=kwargs["media_ingest_heavy_jobs_stop_event"],
        )

    async def _record_reading_study_companion(**kwargs):
        calls.append(("reading_study_companion", kwargs))
        return SimpleNamespace(
            reading_digest_jobs_task=kwargs["reading_digest_jobs_task"],
            reading_digest_jobs_stop_event=kwargs["reading_digest_jobs_stop_event"],
            study_pack_jobs_task=kwargs["study_pack_jobs_task"],
            study_pack_jobs_stop_event=kwargs["study_pack_jobs_stop_event"],
            study_suggestions_jobs_task=kwargs["study_suggestions_jobs_task"],
            study_suggestions_jobs_stop_event=kwargs["study_suggestions_jobs_stop_event"],
            companion_reflection_jobs_task=kwargs["companion_reflection_jobs_task"],
            companion_reflection_jobs_stop_event=kwargs["companion_reflection_jobs_stop_event"],
        )

    async def _record_reminder_admin(**kwargs):
        calls.append(("reminder_admin", kwargs))
        return SimpleNamespace(
            reminder_jobs_task=kwargs["reminder_jobs_task"],
            admin_backup_jobs_task=kwargs["admin_backup_jobs_task"],
            admin_maintenance_rotation_jobs_task=kwargs["admin_maintenance_rotation_jobs_task"],
            admin_maintenance_rotation_jobs_stop_event=kwargs["admin_maintenance_rotation_jobs_stop_event"],
        )

    async def _record_recipe_abtest(**kwargs):
        calls.append(("recipe_abtest", kwargs))
        return SimpleNamespace(
            recipe_run_jobs_task=kwargs["recipe_run_jobs_task"],
            recipe_run_jobs_stop_event=kwargs["recipe_run_jobs_stop_event"],
            evals_abtest_jobs_task=kwargs["evals_abtest_jobs_task"],
            evals_abtest_jobs_stop_event=kwargs["evals_abtest_jobs_stop_event"],
        )

    monkeypatch.setattr(
        shutdown_workers,
        "_shutdown_media_ingest_jobs_workers",
        _record_media_ingest,
    )
    monkeypatch.setattr(
        shutdown_workers,
        "_shutdown_reading_study_companion_jobs_workers",
        _record_reading_study_companion,
    )
    monkeypatch.setattr(
        shutdown_workers,
        "_shutdown_reminder_admin_jobs_workers",
        _record_reminder_admin,
    )
    monkeypatch.setattr(
        shutdown_workers,
        "_shutdown_recipe_abtest_jobs_workers",
        _record_recipe_abtest,
    )

    handles = await shutdown_workers.shutdown_grouped_late_stop_workers(
        media_ingest_jobs_task="media-task",
        media_ingest_jobs_stop_event="media-stop",
        media_ingest_heavy_jobs_task="media-heavy-task",
        media_ingest_heavy_jobs_stop_event="media-heavy-stop",
        reading_digest_jobs_task="reading-task",
        reading_digest_jobs_stop_event="reading-stop",
        study_pack_jobs_task="study-pack-task",
        study_pack_jobs_stop_event="study-pack-stop",
        study_suggestions_jobs_task="study-suggestions-task",
        study_suggestions_jobs_stop_event="study-suggestions-stop",
        companion_reflection_jobs_task="companion-task",
        companion_reflection_jobs_stop_event="companion-stop",
        reminder_jobs_task="reminder-task",
        admin_backup_jobs_task="admin-backup-task",
        admin_maintenance_rotation_jobs_task="admin-maintenance-task",
        admin_maintenance_rotation_jobs_stop_event="admin-maintenance-stop",
        recipe_run_jobs_task="recipe-task",
        recipe_run_jobs_stop_event="recipe-stop",
        evals_abtest_jobs_task="abtest-task",
        evals_abtest_jobs_stop_event="abtest-stop",
        should_run_late_stop=should_run_late_stop,
        guard_exceptions=guard_exceptions,
    )

    assert [name for name, _ in calls] == [
        "media_ingest",
        "reading_study_companion",
        "reminder_admin",
        "recipe_abtest",
    ]
    assert all(call_kwargs["should_run_late_stop"] is should_run_late_stop for _, call_kwargs in calls)
    assert all(call_kwargs["guard_exceptions"] == guard_exceptions for _, call_kwargs in calls)
    assert handles.media_ingest_jobs_task == "media-task"
    assert handles.media_ingest_jobs_stop_event == "media-stop"
    assert handles.media_ingest_heavy_jobs_task == "media-heavy-task"
    assert handles.media_ingest_heavy_jobs_stop_event == "media-heavy-stop"
    assert handles.reading_digest_jobs_task == "reading-task"
    assert handles.reading_digest_jobs_stop_event == "reading-stop"
    assert handles.study_pack_jobs_task == "study-pack-task"
    assert handles.study_pack_jobs_stop_event == "study-pack-stop"
    assert handles.study_suggestions_jobs_task == "study-suggestions-task"
    assert handles.study_suggestions_jobs_stop_event == "study-suggestions-stop"
    assert handles.companion_reflection_jobs_task == "companion-task"
    assert handles.companion_reflection_jobs_stop_event == "companion-stop"
    assert handles.reminder_jobs_task == "reminder-task"
    assert handles.admin_backup_jobs_task == "admin-backup-task"
    assert handles.admin_maintenance_rotation_jobs_task == "admin-maintenance-task"
    assert handles.admin_maintenance_rotation_jobs_stop_event == "admin-maintenance-stop"
    assert handles.recipe_run_jobs_task == "recipe-task"
    assert handles.recipe_run_jobs_stop_event == "recipe-stop"
    assert handles.evals_abtest_jobs_task == "abtest-task"
    assert handles.evals_abtest_jobs_stop_event == "abtest-stop"


@pytest.mark.asyncio
async def test_run_shutdown_grouped_late_stop_workers_delegates_and_returns_handles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app.services import shutdown_grouped_late_stop_workers as shutdown_workers

    should_run_late_stop = lambda *args, **kwargs: True
    recorded_calls: list[dict[str, object]] = []
    expected_handles = shutdown_workers.GroupedLateStopWorkerHandles(
        media_ingest_jobs_task="media-task",
        media_ingest_jobs_stop_event="media-stop",
        media_ingest_heavy_jobs_task="media-heavy-task",
        media_ingest_heavy_jobs_stop_event="media-heavy-stop",
        reading_digest_jobs_task="reading-task",
        reading_digest_jobs_stop_event="reading-stop",
        study_pack_jobs_task="study-pack-task",
        study_pack_jobs_stop_event="study-pack-stop",
        study_suggestions_jobs_task="study-suggestions-task",
        study_suggestions_jobs_stop_event="study-suggestions-stop",
        companion_reflection_jobs_task="companion-task",
        companion_reflection_jobs_stop_event="companion-stop",
        reminder_jobs_task="reminder-task",
        admin_backup_jobs_task="admin-backup-task",
        admin_maintenance_rotation_jobs_task="admin-maintenance-task",
        admin_maintenance_rotation_jobs_stop_event="admin-maintenance-stop",
        recipe_run_jobs_task="recipe-task",
        recipe_run_jobs_stop_event="recipe-stop",
        evals_abtest_jobs_task="abtest-task",
        evals_abtest_jobs_stop_event="abtest-stop",
    )

    async def _fake_shutdown_grouped_late_stop_workers(**kwargs):
        recorded_calls.append(kwargs)
        return expected_handles

    monkeypatch.setattr(
        shutdown_workers,
        "shutdown_grouped_late_stop_workers",
        _fake_shutdown_grouped_late_stop_workers,
    )

    handles = await shutdown_workers.run_shutdown_grouped_late_stop_workers(
        media_ingest_jobs_task="media-task",
        media_ingest_jobs_stop_event="media-stop",
        media_ingest_heavy_jobs_task="media-heavy-task",
        media_ingest_heavy_jobs_stop_event="media-heavy-stop",
        reading_digest_jobs_task="reading-task",
        reading_digest_jobs_stop_event="reading-stop",
        study_pack_jobs_task="study-pack-task",
        study_pack_jobs_stop_event="study-pack-stop",
        study_suggestions_jobs_task="study-suggestions-task",
        study_suggestions_jobs_stop_event="study-suggestions-stop",
        companion_reflection_jobs_task="companion-task",
        companion_reflection_jobs_stop_event="companion-stop",
        reminder_jobs_task="reminder-task",
        admin_backup_jobs_task="admin-backup-task",
        admin_maintenance_rotation_jobs_task="admin-maintenance-task",
        admin_maintenance_rotation_jobs_stop_event="admin-maintenance-stop",
        recipe_run_jobs_task="recipe-task",
        recipe_run_jobs_stop_event="recipe-stop",
        evals_abtest_jobs_task="abtest-task",
        evals_abtest_jobs_stop_event="abtest-stop",
        should_run_late_stop=should_run_late_stop,
        guard_exceptions=(RuntimeError,),
    )

    assert handles is expected_handles
    assert len(recorded_calls) == 1
    assert recorded_calls[0]["should_run_late_stop"] is should_run_late_stop
    assert recorded_calls[0]["media_ingest_jobs_task"] == "media-task"
    assert recorded_calls[0]["evals_abtest_jobs_stop_event"] == "abtest-stop"


@pytest.mark.asyncio
async def test_run_shutdown_grouped_late_stop_workers_logs_and_returns_original_handles_on_guard_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app.services import shutdown_grouped_late_stop_workers as shutdown_workers

    debug_messages: list[str] = []

    async def _raise_guard_failure(**_kwargs):
        raise RuntimeError("grouped late-stop unavailable")

    monkeypatch.setattr(
        shutdown_workers,
        "shutdown_grouped_late_stop_workers",
        _raise_guard_failure,
    )
    monkeypatch.setattr(
        shutdown_workers.logger,
        "debug",
        lambda message, *args, **kwargs: debug_messages.append(str(message)),
    )

    handles = await shutdown_workers.run_shutdown_grouped_late_stop_workers(
        media_ingest_jobs_task="media-task",
        media_ingest_jobs_stop_event="media-stop",
        media_ingest_heavy_jobs_task="media-heavy-task",
        media_ingest_heavy_jobs_stop_event="media-heavy-stop",
        reading_digest_jobs_task="reading-task",
        reading_digest_jobs_stop_event="reading-stop",
        study_pack_jobs_task="study-pack-task",
        study_pack_jobs_stop_event="study-pack-stop",
        study_suggestions_jobs_task="study-suggestions-task",
        study_suggestions_jobs_stop_event="study-suggestions-stop",
        companion_reflection_jobs_task="companion-task",
        companion_reflection_jobs_stop_event="companion-stop",
        reminder_jobs_task="reminder-task",
        admin_backup_jobs_task="admin-backup-task",
        admin_maintenance_rotation_jobs_task="admin-maintenance-task",
        admin_maintenance_rotation_jobs_stop_event="admin-maintenance-stop",
        recipe_run_jobs_task="recipe-task",
        recipe_run_jobs_stop_event="recipe-stop",
        evals_abtest_jobs_task="abtest-task",
        evals_abtest_jobs_stop_event="abtest-stop",
        should_run_late_stop=lambda *args, **kwargs: True,
        guard_exceptions=(RuntimeError,),
    )

    assert handles.media_ingest_jobs_task == "media-task"
    assert handles.reading_digest_jobs_task == "reading-task"
    assert handles.evals_abtest_jobs_stop_event == "abtest-stop"
    assert any("Grouped late-stop workers skipped" in message for message in debug_messages)
