from __future__ import annotations

import contextlib

import pytest
from fastapi import FastAPI

from tldw_Server_API.app.services.lifecycle_worker_specs import (
    ShutdownPhase,
    WorkerLifecycleContext,
    WorkerStrategy,
)


def _context(settings: dict[str, object] | None = None) -> WorkerLifecycleContext:
    return WorkerLifecycleContext(
        app=FastAPI(),
        settings=settings or {},
        test_mode=True,
        route_enabled=lambda *_args, **_kwargs: True,
        logger=None,
        startup_guard_exceptions=(),
        import_exceptions=(),
    )


def test_claims_rebuild_worker_spec_matches_legacy_worker_contract() -> None:
    from tldw_Server_API.app.services import startup_claims_rebuild

    [spec] = startup_claims_rebuild.provide_claims_rebuild_worker_specs()

    assert spec.name == "claims_rebuild"
    assert spec.task_name == "claims_task"
    assert spec.category == "claims"
    assert spec.phase is ShutdownPhase.BACKGROUND_WORKER_SHUTDOWN
    assert spec.timeout_sec == 5.0
    assert spec.strategy is WorkerStrategy.STOP_EVENT_TASK
    assert spec.factory is not None


@pytest.mark.asyncio
async def test_claims_rebuild_worker_spec_factory_delegates_to_existing_loop(monkeypatch) -> None:
    from tldw_Server_API.app.services import startup_claims_rebuild

    calls: list[dict[str, object]] = []

    async def _fake_loop(app_settings, *, stop_event, interval_sec, policy) -> str:
        calls.append(
            {
                "settings": app_settings,
                "stop_event": stop_event,
                "interval_sec": interval_sec,
                "policy": policy,
            }
        )
        return "claims"

    monkeypatch.setattr(startup_claims_rebuild, "_run_claims_rebuild_loop", _fake_loop)

    [spec] = startup_claims_rebuild.provide_claims_rebuild_worker_specs()

    assert spec.factory is not None
    result = await spec.factory(
        _context(
            {
                "CLAIMS_REBUILD_INTERVAL_SEC": 123,
                "CLAIMS_REBUILD_POLICY": "stale",
            }
        ),
        "claims-stop",
    )

    assert result == "claims"
    assert calls == [
        {
            "settings": {
                "CLAIMS_REBUILD_INTERVAL_SEC": 123,
                "CLAIMS_REBUILD_POLICY": "stale",
            },
            "stop_event": "claims-stop",
            "interval_sec": 123,
            "policy": "stale",
        }
    ]


def test_claims_rebuild_worker_spec_enabled_uses_settings() -> None:
    from tldw_Server_API.app.services import startup_claims_rebuild

    [spec] = startup_claims_rebuild.provide_claims_rebuild_worker_specs()

    assert spec.enabled(_context({"CLAIMS_REBUILD_ENABLED": True})) is True
    assert spec.enabled(_context({"CLAIMS_REBUILD_ENABLED": False})) is False


def test_claims_rebuild_db_session_uses_managed_media_database(monkeypatch, tmp_path) -> None:
    from tldw_Server_API.app.services import startup_claims_rebuild

    captured: dict[str, object] = {}
    expected_db_path = str(tmp_path / "media-17.db")

    @contextlib.contextmanager
    def _fake_managed_media_database(
        client_id: str,
        *,
        db_path: str,
        initialize: bool = True,
    ):
        captured["client_id"] = client_id
        captured["db_path"] = db_path
        captured["initialize"] = initialize
        yield "db-sentinel"

    monkeypatch.setattr(
        startup_claims_rebuild,
        "_get_user_media_db_path",
        lambda user_id: str(tmp_path / f"media-{user_id}.db"),
    )
    monkeypatch.setattr(
        startup_claims_rebuild,
        "_managed_media_database",
        _fake_managed_media_database,
    )

    settings = {
        "SINGLE_USER_FIXED_ID": "17",
        "SERVER_CLIENT_ID": "startup-client",
    }

    with startup_claims_rebuild._claims_rebuild_db_session(settings) as (user_id, session_db_path, db):
        assert user_id == 17
        assert session_db_path == expected_db_path
        assert db == "db-sentinel"

    assert captured == {
        "client_id": "startup-client",
        "db_path": expected_db_path,
        "initialize": False,
    }


def test_startup_claims_rebuild_media_id_helper_delegates_to_claims_service(monkeypatch) -> None:
    from tldw_Server_API.app.core.Claims_Extraction import claims_service
    from tldw_Server_API.app.services import startup_claims_rebuild

    captured: dict[str, object] = {}

    def _fake_list_claims_rebuild_media_ids(
        db,
        *,
        policy,
        stale_days,
        compare_media_last_modified,
        limit,
    ):
        captured["db"] = db
        captured["policy"] = policy
        captured["stale_days"] = stale_days
        captured["compare_media_last_modified"] = compare_media_last_modified
        captured["limit"] = limit
        return [101, 202]

    monkeypatch.setattr(
        claims_service,
        "list_claims_rebuild_media_ids",
        _fake_list_claims_rebuild_media_ids,
    )

    db = object()
    result = startup_claims_rebuild._list_claims_rebuild_media_ids(
        db,
        policy="stale",
        stale_days=7,
        compare_media_last_modified=False,
        limit=25,
    )

    assert result == [101, 202]
    assert captured == {
        "db": db,
        "policy": "stale",
        "stale_days": 7,
        "compare_media_last_modified": False,
        "limit": 25,
    }
