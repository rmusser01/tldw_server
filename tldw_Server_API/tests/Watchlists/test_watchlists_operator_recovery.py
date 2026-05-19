"""Operator recovery endpoints for Watchlists runs."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

pytestmark = pytest.mark.unit


@pytest.mark.asyncio
async def test_retry_run_audio_reuses_job_audio_config_without_rerunning_ingestion(monkeypatch):
    from tldw_Server_API.app.api.v1.endpoints import watchlists

    retry_run_audio = watchlists.retry_run_audio

    run = SimpleNamespace(id=10, job_id=7, stats_json=json.dumps({"items_ingested": 2}), error_msg=None)
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
    trigger = AsyncMock(return_value="task-retry-10")
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
        stats_json=json.dumps({"items_found": 5, "audio_briefing_task_id": "task-10"}),
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
    assert "provider" not in result.outputs[0]["deliveries"][0]
    assert "subject" not in result.outputs[0]["deliveries"][0]
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
