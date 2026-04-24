from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from tldw_Server_API.app.core.Evaluations import unified_evaluation_service as eval_service
from tldw_Server_API.app.core.Evaluations.audit_adapter import MandatoryAuditWriteError
from tldw_Server_API.app.core.Evaluations.unified_evaluation_service import UnifiedEvaluationService


@pytest.mark.asyncio
async def test_create_run_leaves_no_row_webhook_or_async_task_when_mandatory_audit_fails(tmp_path, monkeypatch):
    service = UnifiedEvaluationService(
        db_path=str(tmp_path / "evaluations.db"),
        enable_webhooks=True,
    )
    await service.initialize()
    try:
        service.db.create_evaluation(
            name="Strict Eval",
            eval_type="exact_match",
            eval_spec={"metrics": ["exact_match"]},
            created_by="17",
            eval_id="eval_1",
        )

        service.webhook_manager = SimpleNamespace(send_webhook=AsyncMock())

        def _unexpected_create_run(*_args, **_kwargs):
            raise AssertionError("db.create_run should not be called when mandatory audit fails")

        monkeypatch.setattr(service.db, "create_run", _unexpected_create_run)

        def _unexpected_create_task(*_args, **_kwargs):
            raise AssertionError("asyncio.create_task should not be called when mandatory audit fails")

        monkeypatch.setattr(eval_service.asyncio, "create_task", _unexpected_create_task)

        async def _boom(*_args, **_kwargs):
            raise MandatoryAuditWriteError("Mandatory audit persistence unavailable")

        monkeypatch.setattr(eval_service, "log_run_started_async", _boom)

        with pytest.raises(MandatoryAuditWriteError) as exc_info:
            await service.create_run(
                eval_id="eval_1",
                target_model="gpt-4o-mini",
                config={"temperature": 0.0},
                webhook_url="https://example.com/hook",
                created_by="17",
                webhook_user_id="29",
            )

        assert "Mandatory audit persistence unavailable" in str(exc_info.value)
        assert service.db.list_runs(eval_id="eval_1", created_by="17") == []
        service.webhook_manager.send_webhook.assert_not_awaited()
    finally:
        await service.shutdown()
