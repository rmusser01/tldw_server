"""Operator recovery endpoints for Watchlists runs."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tldw_Server_API.app.api.v1.endpoints.watchlists import router as watchlists_router
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User, get_request_user
from tldw_Server_API.app.core.config import API_V1_PREFIX
from tldw_Server_API.app.core.DB_Management.Watchlists_DB import WatchlistsDatabase
from tldw_Server_API.app.core.Watchlists import pipeline
from tldw_Server_API.app.core.Watchlists.pipeline import _safe_source_error_text, run_watchlist_job

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _isolated_watchlists_db(monkeypatch, tmp_path):
    base_dir = tmp_path / "watchlists_operator_recovery_dbs"
    base_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("USER_DB_BASE_DIR", str(base_dir))
    monkeypatch.delenv("TEST_MODE", raising=False)
    monkeypatch.delenv("TLDW_TEST_MODE", raising=False)


@pytest.fixture()
def client_with_user():
    user_id = 9442

    async def override_user():
        return User(id=user_id, username="wluser", email=None, is_active=True)

    app = FastAPI()
    app.include_router(watchlists_router, prefix=f"{API_V1_PREFIX}")
    app.dependency_overrides[get_request_user] = override_user
    with TestClient(app) as client:
        yield client, user_id
    app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_retry_run_audio_reuses_job_audio_config_without_rerunning_ingestion(monkeypatch):
    from tldw_Server_API.app.api.v1.endpoints import watchlists
    from tldw_Server_API.app.core.Watchlists.audio_briefing_workflow import AudioBriefingTriggerResult

    retry_run_audio = watchlists.retry_run_audio

    run = SimpleNamespace(
        id=10,
        job_id=7,
        stats_json=json.dumps(
            {
                "items_ingested": 2,
                "audio_briefing_task_id": "stale-task",
                "audio_briefing_retry_task_id": "stale-retry-task",
                "audio_briefing_reason": "old_reason",
                "audio_request_id": "wla_old_request",
                "audio": {
                    "status": "completed",
                    "audio_request_id": "wla_old_request",
                    "final_artifact": {"artifact_id": "old-final"},
                },
            }
        ),
        error_msg=None,
    )
    job = SimpleNamespace(
        id=7,
        output_prefs_json=json.dumps(
            {
                "generate_audio": True,
                "target_audio_minutes": 8,
                "audio_cast": {
                    "speaker_count": 2,
                    "speakers": [
                        {"id": "host", "label": "Host", "voice": "af_bella"},
                        {"id": "analyst", "label": "Analyst", "voice": "am_adam"},
                    ],
                },
            }
        ),
    )
    db = MagicMock()
    db.get_run.return_value = run
    db.get_job.return_value = job
    db.update_run.return_value = run
    trigger = AsyncMock(
        return_value=AudioBriefingTriggerResult(
            status="submitted",
            task_id="task-retry-10",
            audio_request_id="wla_retry_request",
        )
    )
    rerun_job = AsyncMock()
    monkeypatch.setattr(watchlists, "run_watchlist_job", rerun_job)
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Watchlists.audio_briefing_workflow.trigger_audio_briefing",
        trigger,
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.api.v1.endpoints.watchlists.resolve_user_id_for_request",
        lambda *args, **kwargs: 945,
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.api.v1.endpoints.watchlists._resolve_target_watchlists_context",
        AsyncMock(return_value=(945, db)),
    )

    user = SimpleNamespace(id=945, role="admin")
    result = await retry_run_audio(run_id=10, target_user_id=None, current_user=user, db=db)

    assert result.run_id == 10
    assert result.stage == "audio"
    assert result.retried is True
    assert result.task_id == "task-retry-10"
    trigger.assert_awaited_once()
    rerun_job.assert_not_called()
    db.update_run.assert_called_once()
    persisted_stats = json.loads(db.update_run.call_args.kwargs["stats_json"])
    assert persisted_stats["audio_briefing_status"] == "queued"
    assert persisted_stats["audio_briefing_task_id"] == "task-retry-10"
    assert persisted_stats["audio_briefing_retry_task_id"] == "task-retry-10"
    assert persisted_stats["audio_request_id"] == "wla_retry_request"
    assert persisted_stats["audio"]["status"] == "queued"
    assert persisted_stats["audio"]["audio_request_id"] == "wla_retry_request"
    assert persisted_stats["audio"]["final_artifact"] is None
    assert persisted_stats["audio"].get("artifact_id") is None
    assert persisted_stats["previous_audio"]["stale"] is True
    assert persisted_stats["previous_audio"]["superseded_by"] == "wla_retry_request"
    assert persisted_stats["previous_audio"]["final_artifact"]["artifact_id"] == "old-final"


@pytest.mark.asyncio
async def test_retry_run_audio_marks_output_audio_stale(monkeypatch):
    from tldw_Server_API.app.api.v1.endpoints import watchlists
    from tldw_Server_API.app.core.Watchlists.audio_briefing_workflow import AudioBriefingTriggerResult

    run = SimpleNamespace(
        id=10,
        job_id=7,
        stats_json=json.dumps(
            {
                "items_ingested": 2,
                "audio_briefing_task_id": "stale-task",
                "audio_request_id": "wla_old_request",
                "audio": {
                    "status": "completed",
                    "audio_request_id": "wla_old_request",
                    "artifact_id": "old-final",
                    "final_artifact": {"artifact_id": "old-final"},
                },
            }
        ),
        error_msg=None,
    )
    job = SimpleNamespace(id=7, output_prefs_json=json.dumps({"generate_audio": True}))
    db = MagicMock()
    db.get_run.return_value = run
    db.get_job.return_value = job
    db.update_run.return_value = run

    output_row = SimpleNamespace(
        id=55,
        run_id=10,
        job_id=7,
        type="briefing_markdown",
        format="md",
        metadata_json=json.dumps(
            {
                "origin": "watchlists",
                "delivery_status": "sent",
                "audio_request_id": "wla_old_request",
                "audio": {
                    "status": "completed",
                    "audio_request_id": "wla_old_request",
                    "artifact_id": "old-final",
                    "final_artifact": {"artifact_id": "old-final"},
                },
            }
        ),
    )
    collections_db = MagicMock()
    collections_db.list_output_artifacts.return_value = ([output_row], 1)
    collections_db.update_output_artifact_metadata.return_value = output_row

    trigger = AsyncMock(
        return_value=AudioBriefingTriggerResult(
            status="submitted",
            task_id="task-retry-10",
            audio_request_id="wla_retry_request",
        )
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Watchlists.audio_briefing_workflow.trigger_audio_briefing",
        trigger,
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.api.v1.endpoints.watchlists._resolve_target_watchlists_context",
        AsyncMock(return_value=(945, db)),
    )

    user = SimpleNamespace(id=945, role="admin")
    await watchlists.retry_run_audio(
        run_id=10,
        target_user_id=None,
        current_user=user,
        db=db,
        collections_db=collections_db,
    )

    collections_db.update_output_artifact_metadata.assert_called_once()
    persisted_output = json.loads(collections_db.update_output_artifact_metadata.call_args.kwargs["metadata_json"])
    assert persisted_output["delivery_status"] == "sent"
    assert persisted_output["audio_request_id"] == "wla_retry_request"
    assert persisted_output["audio"]["status"] == "queued"
    assert persisted_output["audio"]["audio_request_id"] == "wla_retry_request"
    assert persisted_output["audio"]["final_artifact"] is None
    assert persisted_output["audio"].get("artifact_id") is None
    assert persisted_output["previous_audio"]["stale"] is True
    assert persisted_output["previous_audio"]["superseded_by"] == "wla_retry_request"
    assert persisted_output["previous_audio"]["final_artifact"]["artifact_id"] == "old-final"


@pytest.mark.asyncio
async def test_retry_run_audio_does_not_mutate_state_when_queue_submit_fails(monkeypatch):
    from fastapi import HTTPException

    from tldw_Server_API.app.api.v1.endpoints import watchlists
    from tldw_Server_API.app.core.Watchlists.audio_briefing_workflow import AudioBriefingTriggerResult

    run = SimpleNamespace(id=10, job_id=7, stats_json=json.dumps({"items_ingested": 2}), error_msg=None)
    job = SimpleNamespace(id=7, output_prefs_json=json.dumps({"generate_audio": True}))
    db = MagicMock()
    db.get_run.return_value = run
    db.get_job.return_value = job
    db.update_run.return_value = run
    trigger = AsyncMock(
        return_value=AudioBriefingTriggerResult(
            status="queue_unavailable",
            reason="workflows_queue_has_no_workers",
        )
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Watchlists.audio_briefing_workflow.trigger_audio_briefing",
        trigger,
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.api.v1.endpoints.watchlists._resolve_target_watchlists_context",
        AsyncMock(return_value=(945, db)),
    )

    user = SimpleNamespace(id=945, role="admin")
    with pytest.raises(HTTPException) as exc_info:
        await watchlists.retry_run_audio(run_id=10, target_user_id=None, current_user=user, db=db)

    assert exc_info.value.status_code == 409
    assert exc_info.value.detail == "audio_retry_not_queued"
    trigger.assert_awaited_once()
    db.update_run.assert_not_called()


@pytest.mark.asyncio
async def test_retry_run_delivery_redelivers_latest_output_and_updates_metadata(monkeypatch):
    from tldw_Server_API.app.api.v1.endpoints.watchlists import retry_run_delivery

    run = SimpleNamespace(id=10, job_id=7, stats_json="{}", error_msg=None)
    db = MagicMock()
    db.get_run.return_value = run
    output_row_old = SimpleNamespace(
        id=54,
        run_id=10,
        job_id=7,
        title="Older Daily Digest",
        format="md",
        metadata_json=json.dumps(
            {
                "origin": "watchlists",
                "delivery_plan": {
                    "email": {"enabled": True, "recipients": ["old@example.com"], "attach_file": False}
                },
            }
        ),
        chatbook_path=None,
    )
    output_row = SimpleNamespace(
        id=55,
        run_id=10,
        job_id=7,
        title="Daily Digest",
        format="md",
        metadata_json=json.dumps(
            {
                "origin": "watchlists",
                "delivery_plan": {
                    "email": {"enabled": True, "recipients": ["digest@example.com"], "attach_file": False}
                },
            }
        ),
        chatbook_path=None,
    )
    output_row_audio = SimpleNamespace(
        id=56,
        run_id=10,
        job_id=7,
        title="Daily Digest (Audio)",
        format="mp3",
        type="tts_audio",
        metadata_json=json.dumps(
            {
                "origin": "watchlists",
                "variant_of": 55,
                "variant_kind": "tts",
                "delivery_plan": {
                    "email": {"enabled": True, "recipients": ["digest@example.com"], "attach_file": False}
                },
            }
        ),
        chatbook_path=None,
    )
    collections_db = MagicMock()
    collections_db.list_output_artifacts.return_value = ([output_row_audio, output_row, output_row_old], 3)
    collections_db.update_output_artifact_metadata.return_value = output_row

    monkeypatch.setattr(
        "tldw_Server_API.app.api.v1.endpoints.watchlists._resolve_target_watchlists_context",
        AsyncMock(return_value=(945, db)),
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.api.v1.endpoints.watchlists.resolve_user_id_for_request",
        lambda *args, **kwargs: 945,
    )
    row_to_output = AsyncMock(
        return_value=SimpleNamespace(
            id=55,
            title="Daily Digest",
            format="md",
            content="# Digest",
            metadata=json.loads(output_row.metadata_json),
            chatbook_path=None,
        )
    )
    monkeypatch.setattr("tldw_Server_API.app.api.v1.endpoints.watchlists._row_to_output", row_to_output)

    class FakeNotificationsService:
        def __init__(self, *, user_id: int, user_email: str | None = None) -> None:
            self.user_id = user_id
            self.user_email = user_email

        async def deliver_email(self, **kwargs):
            return SimpleNamespace(
                channel="email",
                status="sent",
                details={
                    "provider": "fake",
                    "subject": "Daily Digest",
                    "deliveries": [{"recipient": "digest@example.com", "status": "sent"}],
                },
            )

        def deliver_chatbook(self, **kwargs):
            return SimpleNamespace(channel="chatbook", status="skipped", details={"reason": "disabled"})

    monkeypatch.setattr(
        "tldw_Server_API.app.api.v1.endpoints.watchlists.NotificationsService",
        FakeNotificationsService,
    )

    user = SimpleNamespace(id=945, email="wl@example.com", role="admin")
    result = await retry_run_delivery(
        run_id=10,
        target_user_id=None,
        current_user=user,
        db=db,
        collections_db=collections_db,
    )

    assert result.run_id == 10
    assert result.stage == "delivery"
    assert result.retried is True
    assert result.output_id == 55
    row_to_output.assert_awaited_once()
    assert row_to_output.await_args.args[0].id == 55
    assert result.delivery_results == [
        {
            "channel": "email",
            "status": "sent",
            "delivery_count": 1,
            "delivery_status_counts": {"sent": 1},
        }
    ]
    metadata_update = json.loads(collections_db.update_output_artifact_metadata.call_args.kwargs["metadata_json"])
    assert metadata_update["delivery_retry_results"][0]["channel"] == "email"
    assert "provider" not in metadata_update["delivery_retry_results"][0]
    assert "subject" not in metadata_update["delivery_retry_results"][0]


@pytest.mark.asyncio
async def test_retry_run_delivery_uses_target_collections_db_and_does_not_fallback_to_actor_email(monkeypatch):
    from tldw_Server_API.app.api.v1.endpoints import watchlists

    run = SimpleNamespace(id=10, job_id=None, stats_json="{}", error_msg=None)
    db = MagicMock()
    db.get_run.return_value = run
    output_row = SimpleNamespace(
        id=55,
        run_id=10,
        job_id=None,
        title="Delegated Digest",
        format="md",
        metadata_json=json.dumps(
            {
                "origin": "watchlists",
                "delivery_plan": {
                    "email": {"enabled": True, "recipients": [], "attach_file": False},
                    "chatbook": {"enabled": True, "metadata": {"source": "retry"}},
                },
            }
        ),
        chatbook_path=None,
    )
    current_collections_db = MagicMock()
    current_collections_db.list_output_artifacts.return_value = ([], 0)
    target_collections_db = MagicMock()
    target_collections_db.list_output_artifacts.return_value = ([output_row], 1)
    target_collections_db.update_output_artifact_metadata.return_value = output_row
    monkeypatch.setattr(
        watchlists.CollectionsDatabase,
        "for_user",
        classmethod(lambda cls, user_id: target_collections_db),
    )
    monkeypatch.setattr(
        watchlists,
        "_resolve_target_watchlists_context",
        AsyncMock(return_value=(947, db)),
    )
    monkeypatch.setattr(
        watchlists,
        "_row_to_output",
        AsyncMock(
            return_value=SimpleNamespace(
                id=55,
                title="Delegated Digest",
                format="md",
                content="# Digest",
                metadata=json.loads(output_row.metadata_json),
                chatbook_path=None,
            )
        ),
    )
    service_instances: list[Any] = []
    email_kwargs: list[dict[str, Any]] = []
    chatbook_kwargs: list[dict[str, Any]] = []

    class FakeNotificationsService:
        def __init__(self, *, user_id: int, user_email: str | None = None) -> None:
            self.user_id = user_id
            self.user_email = user_email
            service_instances.append(self)

        async def deliver_email(self, **kwargs):
            email_kwargs.append(kwargs)
            return SimpleNamespace(channel="email", status="skipped", details={"reason": "no_recipients"})

        def deliver_chatbook(self, **kwargs):
            chatbook_kwargs.append(kwargs)
            return SimpleNamespace(
                channel="chatbook",
                status="stored",
                details={"document_id": 222, "provider": "watchlists", "model": "watchlists"},
            )

    monkeypatch.setattr(watchlists, "NotificationsService", FakeNotificationsService)

    user = SimpleNamespace(id=945, email="admin@example.com", roles=["admin"], permissions=[])
    result = await watchlists.retry_run_delivery(
        run_id=10,
        target_user_id=947,
        current_user=user,
        db=db,
        collections_db=current_collections_db,
    )

    assert result.output_id == 55
    current_collections_db.list_output_artifacts.assert_not_called()
    target_collections_db.list_output_artifacts.assert_called_once()
    target_collections_db.update_output_artifact_metadata.assert_called_once()
    assert service_instances[0].user_id == 947
    assert service_instances[0].user_email is None
    assert email_kwargs[0]["fallback_to_user_email"] is False
    assert chatbook_kwargs[0]["metadata"]["job_id"] == 0
    metadata_update = json.loads(target_collections_db.update_output_artifact_metadata.call_args.kwargs["metadata_json"])
    assert metadata_update["chatbook_document_id"] == 222
    assert metadata_update["delivery_retry_results"][1] == {
        "channel": "chatbook",
        "status": "stored",
        "document_id": 222,
    }


@pytest.mark.asyncio
async def test_run_diagnostics_bundle_includes_run_and_latest_output_metadata(monkeypatch):
    from tldw_Server_API.app.api.v1.endpoints.watchlists import get_run_diagnostics

    run = SimpleNamespace(
        id=10,
        job_id=7,
        status="failed",
        started_at="2026-05-19T04:00:00Z",
        finished_at="2026-05-19T04:03:00Z",
        stats_json=json.dumps(
            {
                "items_found": 5,
                "audio_briefing_status": "queue_unavailable",
                "audio_briefing_reason": "workflows_queue_has_no_workers",
            }
        ),
        error_msg="delivery_failed",
        log_path="/private/tmp/watchlists.log",
    )
    db = MagicMock()
    db.get_run.return_value = run
    db.get_job.return_value = SimpleNamespace(id=7, name="Daily Digest", schedule_expr="0 */5 * * *")
    output_row = SimpleNamespace(
        id=55,
        run_id=10,
        job_id=7,
        title="Daily Digest",
        format="md",
        metadata_json=json.dumps(
            {
                "origin": "watchlists",
                "deliveries": [
                    {
                        "channel": "email",
                        "status": "failed",
                        "provider": "smtp",
                        "subject": "Daily Digest",
                        "deliveries": [{"recipient": "digest@example.com", "status": "failed"}],
                    }
                ],
                "audio_briefing_status": "enqueue_failed",
                "audio_briefing_reason": "scheduler_submit_failed",
            }
        ),
    )
    collections_db = MagicMock()
    collections_db.list_output_artifacts.return_value = ([output_row], 1)
    monkeypatch.setattr(
        "tldw_Server_API.app.api.v1.endpoints.watchlists._resolve_target_watchlists_context",
        AsyncMock(return_value=(945, db)),
    )

    user = SimpleNamespace(id=945, role="admin")
    result = await get_run_diagnostics(
        run_id=10,
        target_user_id=None,
        current_user=user,
        db=db,
        collections_db=collections_db,
    )

    assert result.run_id == 10
    assert result.run["status"] == "failed"
    assert result.job["id"] == 7
    assert result.outputs[0]["id"] == 55
    assert result.outputs[0]["deliveries"][0]["status"] == "failed"
    assert result.outputs[0]["deliveries"][0]["delivery_count"] == 1
    assert result.outputs[0]["audio_briefing_status"] == "enqueue_failed"
    assert result.outputs[0]["audio_briefing_reason"] == "scheduler_submit_failed"
    assert "provider" not in result.outputs[0]["deliveries"][0]
    assert "subject" not in result.outputs[0]["deliveries"][0]
    assert result.audio == {
        "task_id": None,
        "status": "queue_unavailable",
        "reason": "workflows_queue_has_no_workers",
    }
    assert "log_path" not in json.dumps(result.model_dump())


@pytest.mark.asyncio
async def test_run_diagnostics_uses_target_collections_db_for_delegated_user(monkeypatch):
    from tldw_Server_API.app.api.v1.endpoints import watchlists

    run = SimpleNamespace(
        id=10,
        job_id=7,
        status="failed",
        started_at="2026-05-19T04:00:00Z",
        finished_at="2026-05-19T04:03:00Z",
        stats_json=json.dumps({"items_found": 5}),
        error_msg="delivery_failed",
    )
    db = MagicMock()
    db.get_run.return_value = run
    db.get_job.return_value = SimpleNamespace(id=7, name="Daily Digest", schedule_expr="0 */5 * * *")
    output_row = SimpleNamespace(
        id=55,
        run_id=10,
        job_id=7,
        title="Daily Digest",
        format="md",
        metadata_json=json.dumps({"origin": "watchlists"}),
    )
    current_collections_db = MagicMock()
    current_collections_db.list_output_artifacts.return_value = ([], 0)
    target_collections_db = MagicMock()
    target_collections_db.list_output_artifacts.return_value = ([output_row], 1)
    monkeypatch.setattr(
        watchlists.CollectionsDatabase,
        "for_user",
        classmethod(lambda cls, user_id: target_collections_db),
    )
    monkeypatch.setattr(
        watchlists,
        "_resolve_target_watchlists_context",
        AsyncMock(return_value=(947, db)),
    )

    user = SimpleNamespace(id=945, roles=["admin"], permissions=[])
    result = await watchlists.get_run_diagnostics(
        run_id=10,
        target_user_id=947,
        current_user=user,
        db=db,
        collections_db=current_collections_db,
    )

    assert result.outputs[0]["id"] == 55
    current_collections_db.list_output_artifacts.assert_not_called()
    target_collections_db.list_output_artifacts.assert_called_once()


@pytest.mark.asyncio
async def test_run_stats_record_safe_source_error_when_active_source_fetch_fails(monkeypatch):
    user_id = 9442
    db = WatchlistsDatabase.for_user(user_id)
    source = db.create_source(
        name="Private Feed",
        url="https://news.example/feed.xml?api_key=value_to_redact&token=another_value_to_redact",
        source_type="rss",
        active=True,
        settings_json=json.dumps({"limit": 5}),
        tags=["news"],
        group_ids=[],
    )
    job = db.create_job(
        name="Recovery Digest",
        description=None,
        scope_json=json.dumps({"sources": [source.id]}),
        schedule_expr=None,
        schedule_timezone="UTC",
        active=True,
        max_concurrency=None,
        per_host_delay_ms=None,
        retry_policy_json=None,
        output_prefs_json=None,
    )

    async def _forbidden_feed(*args, **kwargs):
        return {
            "status": 403,
            "items": [],
            "error": "403 forbidden for api_key=value_to_redact token=another_value_to_redact",
        }

    monkeypatch.setattr(pipeline, "fetch_rss_feed_history", _forbidden_feed)

    result = await run_watchlist_job(user_id, job.id)

    assert result["items_ingested"] == 0
    run = db.get_run(result["run_id"])
    assert run.status in {"completed", "succeeded", "partial", "warning"}
    stats = json.loads(run.stats_json or "{}")
    assert stats["source_errors"] >= 1
    assert stats["source_statuses"][0]["source_id"] == source.id
    assert stats["source_statuses"][0]["name"] == "Private Feed"
    assert stats["source_statuses"][0]["status"].startswith("error:")
    assert stats["source_statuses"][0]["items_found"] == 0
    assert stats["source_statuses"][0]["items_ingested"] == 0
    error_text = str(stats["source_statuses"][0].get("error") or "")
    assert error_text
    assert "value_to_redact" not in error_text
    assert "another_value_to_redact" not in error_text
    assert "api_key" not in error_text.lower()
    assert "token=" not in error_text.lower()


def test_safe_source_error_text_redacts_common_secret_formats():
    text = _safe_source_error_text(
        "Authorization: Bearer value_to_redact password: password_value_to_redact "
        "token=token_value_to_redact https://example.test/feed?api_key=query_value_to_redact"
    )

    assert "value_to_redact" not in text
    assert "password_value_to_redact" not in text
    assert "token_value_to_redact" not in text
    assert "query_value_to_redact" not in text
    assert "api_key" not in text.lower()
    assert "password:" not in text.lower()
    assert "token=" not in text.lower()
    assert "Bearer [redacted]" in text


def test_safe_source_error_text_redacts_json_style_secret_formats():
    text = _safe_source_error_text(
        "fetch failed {\"token\":\"json token value;still\", \"api_key\": \"json,api&value\", "
        "'password': 'json password value'} token=\"quoted token value;still\" "
        "Authorization: Bearer \"auth bearer value;still\" "
        "{\"Authorization\": \"Bearer json bearer value\", \"bearer\": \"Bearer nested bearer value\"}"
    )

    for leaked in (
        "json token",
        "token value",
        "json,api",
        "api&value",
        "json password",
        "quoted token",
        "auth bearer",
        "json bearer",
        "nested bearer",
    ):
        assert leaked not in text
    assert text.count("[redacted]") >= 6
    assert "api_key" not in text.lower()
    assert "password" not in text.lower()
    assert "token=" not in text.lower()


def test_safe_source_error_text_removes_url_basic_auth():
    text = _safe_source_error_text(
        "https://demo_user:value_to_redact@example.test/feed?api_key=query_value_to_redact"
    )

    assert text == "https://example.test/feed"
    assert "demo_user" not in text
    assert "value_to_redact" not in text
    assert "query_value_to_redact" not in text


@pytest.mark.asyncio
async def test_site_source_records_partial_extraction_status(monkeypatch):
    user_id = 9442
    db = WatchlistsDatabase.for_user(user_id)
    source = db.create_source(
        name="Partial Site",
        url="https://news.example/",
        source_type="site",
        active=True,
        settings_json=json.dumps({"top_n": 2, "discover_method": "frontpage"}),
        tags=["partial-site"],
        group_ids=[],
    )
    job = db.create_job(
        name="Partial Site Digest",
        description=None,
        scope_json=json.dumps({"sources": [source.id]}),
        schedule_expr=None,
        schedule_timezone="UTC",
        active=True,
        max_concurrency=None,
        per_host_delay_ms=None,
        retry_policy_json=None,
        output_prefs_json=None,
    )

    async def _top_links(*args, **kwargs):
        return ["https://news.example/ok", "https://news.example/missing"]

    def _article(url: str):
        if url.endswith("/missing"):
            return None
        return {
            "title": "Extracted Story",
            "url": url,
            "content": "Extracted story body",
            "author": None,
        }

    monkeypatch.setattr(
        "tldw_Server_API.app.core.Watchlists.fetchers.fetch_site_top_links",
        _top_links,
    )
    monkeypatch.setattr(pipeline, "fetch_site_article", _article)

    result = await run_watchlist_job(user_id, job.id)

    assert result["items_found"] == 2
    assert result["items_ingested"] == 1
    run = db.get_run(result["run_id"])
    stats = json.loads(run.stats_json or "{}")
    assert stats["source_errors"] == 1
    assert stats["source_statuses"][0]["status"] == "partial:extraction"
    assert stats["source_statuses"][0]["error"] == "1 of 2 URLs failed extraction"


def test_run_details_preserve_source_failure_stats(client_with_user):
    client, user_id = client_with_user
    db = WatchlistsDatabase.for_user(user_id)
    job = db.create_job(
        name="Detail Recovery Digest",
        description=None,
        scope_json=json.dumps({"sources": []}),
        schedule_expr=None,
        schedule_timezone="UTC",
        active=True,
        max_concurrency=None,
        per_host_delay_ms=None,
        retry_policy_json=None,
        output_prefs_json=None,
    )
    run = db.create_run(job_id=job.id, status="succeeded")
    db.update_run(
        run.id,
        stats_json=json.dumps(
            {
                "items_found": 0,
                "items_ingested": 0,
                "source_errors": 1,
                "source_statuses": [
                    {
                        "source_id": 123,
                        "name": "Blocked Feed",
                        "status": "error:403",
                        "error": "HTTP 403",
                        "items_found": 0,
                        "items_ingested": 0,
                    }
                ],
            }
        ),
    )

    response = client.get(f"/api/v1/watchlists/runs/{run.id}/details")

    assert response.status_code == 200, response.text
    stats = response.json()["stats"]
    assert stats["source_errors"] == 1
    assert stats["source_statuses"][0]["status"] == "error:403"
