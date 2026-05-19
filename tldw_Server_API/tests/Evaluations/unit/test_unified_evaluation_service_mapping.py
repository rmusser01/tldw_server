import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from tldw_Server_API.app.core.Evaluations.unified_evaluation_service import UnifiedEvaluationService


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("eval_type", "expected_sub_type"),
    [
        ("geval", "summarization"),
        ("rag", "rag"),
        ("response_quality", "response_quality"),
    ],
)
async def test_create_evaluation_maps_to_model_graded(tmp_path, eval_type, expected_sub_type):
    svc = UnifiedEvaluationService(db_path=str(tmp_path / "evals.db"), enable_webhooks=False)

    evaluation = await svc.create_evaluation(
        name=f"test_{eval_type}",
        eval_type=eval_type,
        eval_spec={"metrics": ["relevance"]},
        created_by="tester",
    )

    assert evaluation["eval_type"] == "model_graded"
    assert evaluation["eval_spec"].get("sub_type") == expected_sub_type


def test_unified_service_uses_backend_adapter_for_postgres_webhooks(monkeypatch, tmp_path):
    from tldw_Server_API.app.core.DB_Management.backends.base import BackendType
    from tldw_Server_API.app.core.Evaluations import db_adapter as db_adapter_mod
    from tldw_Server_API.app.core.Evaluations import eval_runner as eval_runner_mod
    from tldw_Server_API.app.core.Evaluations import unified_evaluation_service as svc_mod
    from tldw_Server_API.app.core.Evaluations import webhook_manager as webhook_manager_mod

    class _FakeBackend:
        backend_type = BackendType.POSTGRESQL

    class _FakeDB:
        backend_type = BackendType.POSTGRESQL
        backend = _FakeBackend()

    class _DummyRunner:
        def __init__(self, _db_path):
            self.running_tasks = {}

    captured = {}
    sentinel_adapter = object()

    class _DummyWebhookManager:
        def __init__(self, **kwargs):
            captured["kwargs"] = kwargs

    monkeypatch.setattr(eval_runner_mod, "EvaluationRunner", _DummyRunner)
    monkeypatch.setattr(svc_mod, "_create_evals_db", lambda db_path: _FakeDB())
    monkeypatch.setattr(
        db_adapter_mod,
        "create_adapter_from_backend",
        lambda backend: sentinel_adapter,
    )
    monkeypatch.setattr(webhook_manager_mod, "WebhookManager", _DummyWebhookManager)

    UnifiedEvaluationService(db_path=str(tmp_path / "evals.db"), enable_webhooks=True)

    kwargs = captured["kwargs"]
    assert kwargs.get("adapter") is sentinel_adapter
    assert "db_path" not in kwargs


@pytest.mark.asyncio
async def test_run_evaluation_async_skips_cancelled_webhook_when_status_is_already_terminal(
    monkeypatch,
    tmp_path,
):
    service = UnifiedEvaluationService(
        db_path=str(tmp_path / "evals.db"),
        enable_webhooks=True,
    )
    service.webhook_manager = SimpleNamespace(send_webhook=AsyncMock())

    async def _cancelled_call(*_args, **_kwargs):
        raise asyncio.CancelledError

    monkeypatch.setattr(service.circuit_breaker, "call", _cancelled_call)
    monkeypatch.setattr(
        service.db,
        "get_run",
        lambda run_id, created_by=None: {"id": run_id, "status": "completed"},
    )

    status_updates: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def _record_status(*args, **kwargs):
        status_updates.append((args, kwargs))

    monkeypatch.setattr(service.db, "update_run_status", _record_status)

    with pytest.raises(asyncio.CancelledError):
        await service._run_evaluation_async(
            run_id="run_terminal",
            eval_id="eval_1",
            eval_config={"webhook_url": "https://example.com/webhook"},
            created_by="tenant-user",
            webhook_user_id="user_tenant-user",
        )

    assert status_updates == []
    service.webhook_manager.send_webhook.assert_not_awaited()


@pytest.mark.asyncio
async def test_cancel_run_cleans_up_terminal_task_without_rewriting_status_or_logging_cancel(
    monkeypatch,
    tmp_path,
):
    service = UnifiedEvaluationService(
        db_path=str(tmp_path / "evals.db"),
        enable_webhooks=False,
    )

    class _Task:
        def __init__(self) -> None:
            self.cancel_called = False

        def cancel(self) -> None:
            self.cancel_called = True

    task = _Task()
    service.runner.running_tasks["run_terminal"] = task

    terminal_run = {"id": "run_terminal", "status": "completed"}
    monkeypatch.setattr(service.db, "get_run", lambda run_id, created_by=None: terminal_run)
    monkeypatch.setattr(service.runner.db, "get_run", lambda run_id: terminal_run)

    status_updates: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def _record_status(*args, **kwargs):
        status_updates.append((args, kwargs))
        return True

    monkeypatch.setattr(service.db, "update_run_status", _record_status)
    monkeypatch.setattr(service.runner.db, "update_run_status", _record_status)

    audit_log = AsyncMock()
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Evaluations.unified_evaluation_service.log_run_cancelled_async",
        audit_log,
    )

    assert await service.cancel_run(
        "run_terminal",
        cancelled_by="tester",
        created_by="tester",
    ) is True
    assert task.cancel_called is True
    assert "run_terminal" not in service.runner.running_tasks
    assert status_updates == []
    audit_log.assert_not_awaited()
