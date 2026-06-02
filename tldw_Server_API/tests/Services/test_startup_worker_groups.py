from __future__ import annotations

import importlib
import sys
from collections.abc import Callable
from types import SimpleNamespace

import pytest

from tldw_Server_API.app.services.lifecycle_worker_specs import WorkerLifecycleContext

pytestmark = pytest.mark.unit


def _import_startup_worker_groups():
    sys.modules.pop("tldw_Server_API.app.services.startup_worker_groups", None)
    return importlib.import_module("tldw_Server_API.app.services.startup_worker_groups")


def _context() -> WorkerLifecycleContext:
    return WorkerLifecycleContext(
        app=object(),
        settings={},
        test_mode=True,
        route_enabled=lambda *_args, **_kwargs: True,
        logger=None,
        startup_guard_exceptions=(),
        import_exceptions=(),
    )


def test_collect_startup_worker_specs_uses_declarative_provider_catalog(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    startup_groups = _import_startup_worker_groups()
    provider_calls: list[str] = []

    def _provider(name: str):
        def _collect(_context: WorkerLifecycleContext):
            provider_calls.append(name)
            return ()

        return _collect

    providers = (
        _provider("primary"),
        _provider("study"),
        _provider("content"),
        _provider("sidecar"),
        _provider("notifications"),
        _provider("cleanup"),
        _provider("compactor"),
        _provider("claims"),
        _provider("usage"),
        _provider("llm-usage"),
        _provider("runtime"),
        _provider("optional"),
        _provider("auxiliary"),
        _provider("infra"),
        _provider("maintenance"),
        _provider("recurring"),
    )
    monkeypatch.setattr(
        startup_groups,
        "startup_worker_spec_providers",
        lambda: providers,
    )

    specs = startup_groups.collect_startup_worker_specs(_context())

    assert specs == ()
    assert provider_calls == [
        "primary",
        "study",
        "content",
        "sidecar",
        "notifications",
        "cleanup",
        "compactor",
        "claims",
        "usage",
        "llm-usage",
        "runtime",
        "optional",
        "auxiliary",
        "infra",
        "maintenance",
        "recurring",
    ]


def test_collect_startup_worker_specs_accepts_real_provider_graph() -> None:
    startup_groups = _import_startup_worker_groups()

    specs = startup_groups.collect_startup_worker_specs(_context())
    spec_names = {spec.name for spec in specs}

    assert len(specs) == len(spec_names)
    assert {
        "core_jobs_task",
        "claims_rebuild",
        "jobs_metrics_task",
        "connectors_sync_sched_task",
    }.issubset(spec_names)


@pytest.mark.asyncio
async def test_start_worker_groups_requires_worker_inventory_before_starting_legacy_workers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    startup_groups = _import_startup_worker_groups()
    calls: list[str] = []

    async def _record_cleanup_workers(**_kwargs: object) -> SimpleNamespace:
        calls.append("cleanup")
        return SimpleNamespace()

    async def _record_primary_jobs_pollers(**_kwargs: object) -> SimpleNamespace:
        calls.append("primary")
        return SimpleNamespace(
            core_jobs_stop_event=None,
            core_jobs_task=None,
            files_jobs_stop_event=None,
            files_jobs_task=None,
            data_tables_jobs_stop_event=None,
            data_tables_jobs_task=None,
            prompt_studio_jobs_stop_event=None,
            prompt_studio_jobs_task=None,
        )

    async def _record_study_privilege_jobs_pollers(**_kwargs: object) -> SimpleNamespace:
        calls.append("study")
        return SimpleNamespace(
            study_pack_jobs_stop_event=None,
            study_pack_jobs_task=None,
            study_suggestions_jobs_stop_event=None,
            study_suggestions_jobs_task=None,
            privilege_snapshot_stop_event=None,
            privilege_snapshot_task=None,
        )

    async def _record_compactor_websub_workers(**_kwargs: object) -> SimpleNamespace:
        calls.append("compactor")
        return SimpleNamespace()

    async def _record_content_jobs_pollers(**_kwargs: object) -> SimpleNamespace:
        calls.append("content")
        return SimpleNamespace(
            audio_jobs_stop_event=None,
            audio_jobs_task=None,
            audiobook_jobs_stop_event=None,
            audiobook_jobs_task=None,
            presentation_render_jobs_stop_event=None,
            presentation_render_jobs_task=None,
            media_ingest_jobs_stop_event=None,
            media_ingest_jobs_task=None,
            media_ingest_heavy_jobs_stop_event=None,
            media_ingest_heavy_jobs_task=None,
            reading_digest_jobs_stop_event=None,
            reading_digest_jobs_task=None,
            vn_asset_jobs_stop_event=None,
            vn_asset_jobs_task=None,
            vn_asset_generation_jobs_stop_event=None,
            vn_asset_generation_jobs_task=None,
            companion_reflection_jobs_stop_event=None,
            companion_reflection_jobs_task=None,
        )

    async def _record_sidecar_owned_jobs_pollers(**_kwargs: object) -> SimpleNamespace:
        calls.append("sidecar")
        return SimpleNamespace(
            reminder_jobs_stop_event=None,
            reminder_jobs_task=None,
            admin_backup_jobs_stop_event=None,
            admin_backup_jobs_task=None,
            admin_byok_validation_jobs_stop_event=None,
            admin_byok_validation_jobs_task=None,
            admin_maintenance_rotation_jobs_stop_event=None,
            admin_maintenance_rotation_jobs_task=None,
            recipe_run_jobs_stop_event=None,
            recipe_run_jobs_task=None,
        )

    async def _record_notifications_abtest_workers(**_kwargs: object) -> SimpleNamespace:
        calls.append("notifications")
        return SimpleNamespace(
            jobs_notifications_bridge_task=None,
            evals_abtest_jobs_stop_event=None,
            evals_abtest_jobs_task=None,
        )

    monkeypatch.setattr(startup_groups, "_start_cleanup_workers", _record_cleanup_workers)
    monkeypatch.setattr(startup_groups, "_start_primary_jobs_pollers", _record_primary_jobs_pollers)
    monkeypatch.setattr(
        startup_groups,
        "_start_study_privilege_jobs_pollers",
        _record_study_privilege_jobs_pollers,
    )
    monkeypatch.setattr(
        startup_groups,
        "_start_compactor_websub_workers",
        _record_compactor_websub_workers,
    )
    monkeypatch.setattr(startup_groups, "_start_content_jobs_pollers", _record_content_jobs_pollers)
    monkeypatch.setattr(
        startup_groups,
        "_start_sidecar_owned_jobs_pollers",
        _record_sidecar_owned_jobs_pollers,
    )
    monkeypatch.setattr(
        startup_groups,
        "_start_notifications_abtest_workers",
        _record_notifications_abtest_workers,
    )

    with pytest.raises(RuntimeError, match="worker_inventory is required"):
        await startup_groups.start_worker_groups(
            app=object(),
            app_settings={"SINGLE_USER_FIXED_ID": "7"},
            test_mode=True,
            route_enabled=lambda *_args, **_kwargs: False,
            startup_guard_exceptions=(RuntimeError,),
            owned_job_pollers=[],
            register_owned_job_poller=object(),
            worker_inventory=None,
        )

    assert calls == []


@pytest.mark.asyncio
async def test_start_worker_groups_runs_helpers_in_order_and_returns_handles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    startup_groups = _import_startup_worker_groups()
    calls: list[str] = []
    route_enabled_calls: list[tuple[str, bool]] = []
    app = object()
    owned_job_pollers: list[object] = []
    register_owned_job_poller = object()
    worker_inventory = object()
    worker_inventory_ref = worker_inventory

    async def _record_cleanup_workers(
        *,
        app_settings: dict[str, str],
        test_mode: bool,
        worker_inventory: object | None = None,
    ) -> SimpleNamespace:
        assert app_settings == {"SINGLE_USER_FIXED_ID": "7"}
        assert test_mode is True
        assert worker_inventory is worker_inventory_ref
        calls.append("cleanup")
        return SimpleNamespace()

    async def _record_primary_jobs_pollers(
        *,
        app,
        owned_job_pollers,
        register_owned_job_poller,
        should_start_worker,
        sidecar_mode,
        worker_inventory,
    ):
        assert sidecar_mode is False
        assert worker_inventory is worker_inventory_ref
        assert should_start_worker("FILES_JOBS_WORKER_ENABLED", "files") is False
        calls.append("primary")
        return SimpleNamespace(
            core_jobs_stop_event="core-stop",
            core_jobs_task="core-task",
            files_jobs_stop_event="files-stop",
            files_jobs_task="files-task",
            data_tables_jobs_stop_event="data-stop",
            data_tables_jobs_task="data-task",
            prompt_studio_jobs_stop_event="prompt-stop",
            prompt_studio_jobs_task="prompt-task",
        )

    async def _record_study_privilege_jobs_pollers(
        *,
        app: object,
        owned_job_pollers: list[object],
        register_owned_job_poller: object,
        should_start_worker: Callable[..., bool],
        worker_inventory: object | None,
    ) -> SimpleNamespace:
        """Record the study/privilege startup group call."""

        del app, owned_job_pollers, register_owned_job_poller
        assert worker_inventory is worker_inventory_ref
        assert should_start_worker("STUDY_PACK_JOBS_WORKER_ENABLED", "flashcards") is False
        calls.append("study")
        return SimpleNamespace(
            study_pack_jobs_stop_event="study-pack-stop",
            study_pack_jobs_task="study-pack-task",
            study_suggestions_jobs_stop_event="study-suggestions-stop",
            study_suggestions_jobs_task="study-suggestions-task",
            privilege_snapshot_stop_event="privilege-stop",
            privilege_snapshot_task="privilege-task",
        )

    async def _record_compactor_websub_workers(*, should_start_worker, worker_inventory):
        assert should_start_worker("AUDIO_JOBS_WORKER_ENABLED", "audio-jobs") is True
        assert worker_inventory is worker_inventory_ref
        calls.append("compactor")
        return SimpleNamespace()

    async def _record_content_jobs_pollers(
        *,
        app: object,
        owned_job_pollers: list[object],
        register_owned_job_poller: object,
        should_start_worker: Callable[..., bool],
        worker_inventory: object | None,
    ) -> SimpleNamespace:
        """Record the content jobs startup group call."""

        assert worker_inventory is worker_inventory_ref
        del app, owned_job_pollers, register_owned_job_poller
        assert should_start_worker("READING_DIGEST_JOBS_WORKER_ENABLED", "collections-websub") is False
        calls.append("content")
        return SimpleNamespace(
            audio_jobs_stop_event="audio-stop",
            audio_jobs_task="audio-task",
            audiobook_jobs_stop_event="audiobook-stop",
            audiobook_jobs_task="audiobook-task",
            presentation_render_jobs_stop_event="slides-stop",
            presentation_render_jobs_task="slides-task",
            media_ingest_jobs_stop_event="media-stop",
            media_ingest_jobs_task="media-task",
            media_ingest_heavy_jobs_stop_event="media-heavy-stop",
            media_ingest_heavy_jobs_task="media-heavy-task",
            reading_digest_jobs_stop_event="reading-stop",
            reading_digest_jobs_task="reading-task",
            vn_asset_jobs_stop_event="vn-asset-stop",
            vn_asset_jobs_task="vn-asset-task",
            vn_asset_generation_jobs_stop_event="vn-generation-stop",
            vn_asset_generation_jobs_task="vn-generation-task",
            companion_reflection_jobs_stop_event="companion-stop",
            companion_reflection_jobs_task="companion-task",
        )

    async def _record_sidecar_owned_jobs_pollers(
        *,
        app: object,
        owned_job_pollers: list[object],
        register_owned_job_poller: object,
        sidecar_mode: bool,
        worker_inventory: object | None,
    ) -> SimpleNamespace:
        """Record the sidecar-owned jobs startup group call."""

        del app, owned_job_pollers, register_owned_job_poller
        assert sidecar_mode is False
        assert worker_inventory is worker_inventory_ref
        calls.append("sidecar")
        return SimpleNamespace(
            reminder_jobs_stop_event="reminder-stop",
            reminder_jobs_task="reminder-task",
            admin_backup_jobs_stop_event="backup-stop",
            admin_backup_jobs_task="backup-task",
            admin_byok_validation_jobs_stop_event="byok-stop",
            admin_byok_validation_jobs_task="byok-task",
            admin_maintenance_rotation_jobs_stop_event="maintenance-stop",
            admin_maintenance_rotation_jobs_task="maintenance-task",
            recipe_run_jobs_stop_event="recipe-stop",
            recipe_run_jobs_task="recipe-task",
        )

    async def _record_notifications_abtest_workers(
        *,
        app,
        owned_job_pollers,
        register_owned_job_poller,
        sidecar_mode,
        worker_inventory,
    ):
        del app, owned_job_pollers, register_owned_job_poller
        assert sidecar_mode is False
        assert worker_inventory is worker_inventory_ref
        calls.append("notifications")
        return SimpleNamespace(
            jobs_notifications_bridge_task="bridge-task",
            evals_abtest_jobs_stop_event="abtest-stop",
            evals_abtest_jobs_task="abtest-task",
        )

    monkeypatch.setenv("AUDIO_JOBS_WORKER_ENABLED", "1")
    monkeypatch.delenv("TLDW_WORKERS_SIDECAR_MODE", raising=False)
    worker_inventory_ref = worker_inventory
    monkeypatch.setattr(startup_groups, "_start_cleanup_workers", _record_cleanup_workers)
    monkeypatch.setattr(startup_groups, "_start_primary_jobs_pollers", _record_primary_jobs_pollers)
    monkeypatch.setattr(
        startup_groups,
        "_start_study_privilege_jobs_pollers",
        _record_study_privilege_jobs_pollers,
    )
    monkeypatch.setattr(
        startup_groups,
        "_start_compactor_websub_workers",
        _record_compactor_websub_workers,
    )
    monkeypatch.setattr(startup_groups, "_start_content_jobs_pollers", _record_content_jobs_pollers)
    monkeypatch.setattr(
        startup_groups,
        "_start_sidecar_owned_jobs_pollers",
        _record_sidecar_owned_jobs_pollers,
    )
    monkeypatch.setattr(
        startup_groups,
        "_start_notifications_abtest_workers",
        _record_notifications_abtest_workers,
    )

    handles = await startup_groups.start_worker_groups(
        app=app,
        app_settings={"SINGLE_USER_FIXED_ID": "7"},
        test_mode=True,
        route_enabled=lambda route_key, *, default_stable=True: route_enabled_calls.append((route_key, default_stable)),
        startup_guard_exceptions=(RuntimeError,),
        owned_job_pollers=owned_job_pollers,
        register_owned_job_poller=register_owned_job_poller,
        worker_inventory=worker_inventory,
    )

    assert calls == [
        "cleanup",
        "primary",
        "study",
        "compactor",
        "content",
        "sidecar",
        "notifications",
    ]
    assert route_enabled_calls == []
    assert not hasattr(handles, "cleanup_task")
    assert not hasattr(handles, "chatbooks_cleanup_task")
    assert handles.core_jobs_task == "core-task"
    assert handles.study_pack_jobs_task == "study-pack-task"
    assert not hasattr(handles, "embeddings_compactor_task")
    assert handles.audio_jobs_task == "audio-task"
    assert handles.vn_asset_jobs_task == "vn-asset-task"
    assert handles.vn_asset_generation_jobs_task == "vn-generation-task"
    assert handles.recipe_run_jobs_task == "recipe-task"
    assert handles.evals_abtest_jobs_task == "abtest-task"
