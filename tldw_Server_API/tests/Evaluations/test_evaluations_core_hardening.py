"""Regression coverage for Evaluations core hardening findings."""

from __future__ import annotations

import asyncio
import ipaddress

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tldw_Server_API.app.api.v1.schemas.synthetic_eval_schemas import (
    SyntheticEvalProvenance,
    SyntheticEvalReviewActionType,
    SyntheticEvalReviewState,
)
from tldw_Server_API.app.core.DB_Management.Evaluations_DB import EvaluationsDatabase
from tldw_Server_API.app.core.Evaluations.db_adapter import (
    DatabaseAdapterFactory,
    DatabaseConfig,
    DatabaseType,
)
from tldw_Server_API.app.core.Evaluations.eval_runner import EvaluationRunner
from tldw_Server_API.app.core.Evaluations.synthetic_eval_service import SyntheticEvalWorkflowService
from tldw_Server_API.app.core.Evaluations.unified_evaluation_service import UnifiedEvaluationService
from tldw_Server_API.app.core.Evaluations.user_rate_limiter import UserRateLimiter, UserTier
from tldw_Server_API.app.core.Evaluations.webhook_manager import WebhookPayload, WebhookManager
from tldw_Server_API.app.core.Evaluations.webhook_security import WebhookSecurityValidator


@pytest.mark.asyncio
async def test_create_evaluation_rejects_dataset_owned_by_another_user(temp_db_path) -> None:
    service = UnifiedEvaluationService(db_path=str(temp_db_path), enable_webhooks=False)

    dataset_id = service.db.create_dataset(
        name="private-dataset",
        samples=[{"input": {"text": "tenant-a secret"}, "expected": {"text": "ok"}}],
        created_by="user-a",
    )

    with pytest.raises(ValueError, match="Dataset .* not found"):
        await service.create_evaluation(
            name="cross-owner",
            eval_type="exact_match",
            eval_spec={"metrics": ["exact_match"]},
            dataset_id=dataset_id,
            created_by="user-b",
        )


@pytest.mark.asyncio
async def test_store_evaluation_result_raises_when_standard_persistence_fails(temp_db_path, monkeypatch) -> None:
    service = UnifiedEvaluationService(db_path=str(temp_db_path), enable_webhooks=False)

    def _fail_create_evaluation(*_args, **_kwargs):
        raise RuntimeError("db create failed")

    monkeypatch.setattr(service.db, "create_evaluation", _fail_create_evaluation)

    with pytest.raises(RuntimeError, match="db create failed"):
        await service._store_evaluation_result(
            "geval",
            input_data={"transcript": "input", "summary": "summary"},
            results={"score": 0.5},
            metadata={"user_id": "user-a"},
        )


@pytest.mark.asyncio
async def test_runner_does_not_load_dataset_owned_by_another_user(temp_db_path) -> None:
    runner = EvaluationRunner(str(temp_db_path), max_concurrent_evals=1, eval_timeout=5)
    dataset_id = runner.db.create_dataset(
        name="private-dataset",
        samples=[{"input": {"text": "tenant-a secret"}, "expected": {"text": "ok"}}],
        created_by="user-a",
    )
    eval_id = runner.db.create_evaluation(
        name="bad-reference",
        eval_type="exact_match",
        eval_spec={"metrics": ["exact_match"]},
        dataset_id=dataset_id,
        created_by="user-b",
    )
    evaluation = runner.db.get_evaluation(eval_id, created_by="user-b")

    with pytest.raises(ValueError, match="No samples found"):
        await runner._get_samples(evaluation, {"created_by": "user-b"})


def _synthetic_service(db: EvaluationsDatabase, user_id: str) -> SyntheticEvalWorkflowService:
    return SyntheticEvalWorkflowService(db=db, user_id=user_id)


def _create_synthetic_sample(
    service: SyntheticEvalWorkflowService,
    *,
    sample_id: str,
    created_by: str,
    review_state: str = SyntheticEvalReviewState.DRAFT.value,
) -> None:
    service.repository.create_draft_sample(
        sample_id=sample_id,
        recipe_kind="rag_retrieval_tuning",
        sample_payload={"query": sample_id},
        provenance=SyntheticEvalProvenance.SYNTHETIC_FROM_CORPUS.value,
        review_state=review_state,
        created_by=created_by,
    )


def test_synthetic_queue_lists_only_current_users_draft_samples(tmp_path) -> None:
    db = EvaluationsDatabase(str(tmp_path / "evaluations.db"))
    service_a = _synthetic_service(db, "user-a")
    service_b = _synthetic_service(db, "user-b")
    _create_synthetic_sample(service_a, sample_id="sample-a", created_by="user-a")
    _create_synthetic_sample(service_b, sample_id="sample-b", created_by="user-b")

    queue = service_b.list_queue()

    assert [row["sample_id"] for row in queue["data"]] == ["sample-b"]  # nosec B101
    assert queue["total"] == 1  # nosec B101


def test_synthetic_review_rejects_sample_owned_by_another_user(tmp_path) -> None:
    db = EvaluationsDatabase(str(tmp_path / "evaluations.db"))
    service_a = _synthetic_service(db, "user-a")
    service_b = _synthetic_service(db, "user-b")
    _create_synthetic_sample(service_a, sample_id="sample-a", created_by="user-a")

    with pytest.raises(ValueError, match="sample does not exist"):
        service_b.review_sample(
            "sample-a",
            action=SyntheticEvalReviewActionType.APPROVE.value,
        )


def test_synthetic_promotion_rejects_sample_owned_by_another_user(tmp_path) -> None:
    db = EvaluationsDatabase(str(tmp_path / "evaluations.db"))
    service_a = _synthetic_service(db, "user-a")
    service_b = _synthetic_service(db, "user-b")
    _create_synthetic_sample(
        service_a,
        sample_id="sample-a",
        created_by="user-a",
        review_state=SyntheticEvalReviewState.APPROVED.value,
    )

    with pytest.raises(ValueError, match="sample does not exist"):
        service_b.promote_samples(
            sample_ids=["sample-a"],
            dataset_name="should-not-promote",
        )


@pytest.mark.asyncio
async def test_runner_direct_webhook_rejects_private_targets_without_posting(temp_db_path) -> None:
    runner = EvaluationRunner(str(temp_db_path), max_concurrent_evals=1, eval_timeout=5)

    with patch(
        "tldw_Server_API.app.core.Evaluations.eval_runner.afetch",
        new_callable=AsyncMock,
    ) as mock_fetch:
        mock_fetch.return_value = SimpleNamespace(status_code=200, text="ok")
        await runner._send_webhook(
            "http://127.0.0.1/webhook",
            "run-private",
            "eval-private",
            "completed",
            {"ok": True},
        )

    mock_fetch.assert_not_awaited()


@pytest.mark.asyncio
async def test_registered_webhook_delivery_uses_resolved_safe_target(temp_db_path, monkeypatch) -> None:
    manager = WebhookManager(str(temp_db_path))
    original_url = "http://attacker.test/webhook"
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    webhook = {
        "id": 1,
        "url": original_url,
        "secret": "test-secret",
        "retry_count": 1,
        "timeout_seconds": 5,
    }
    payload = WebhookPayload(
        event="evaluation.completed",
        evaluation_id="eval-safe",
        timestamp="2026-06-23T00:00:00+00:00",
        data={"score": 1.0},
    )

    async def _ok_fetch(**kwargs):
        return SimpleNamespace(status_code=200, text="ok")

    with (
        patch.object(manager, "_create_delivery_record", return_value=1),
        patch.object(manager, "_update_delivery_record", MagicMock()),
        patch.object(manager, "_update_webhook_stats", MagicMock()),
        patch("tldw_Server_API.app.core.testing.is_test_mode", return_value=False),
        patch(
            "socket.getaddrinfo",
            return_value=[(None, None, None, None, ("93.184.216.34", 80))],
        ),
        patch(
            "tldw_Server_API.app.core.Evaluations.webhook_manager.afetch",
            new=AsyncMock(side_effect=_ok_fetch),
        ) as mock_fetch,
    ):
        await manager._deliver_webhook(webhook, payload)

    mock_fetch.assert_awaited_once()
    _, kwargs = mock_fetch.await_args
    assert kwargs["url"] == "http://93.184.216.34/webhook"  # nosec B101
    assert kwargs["headers"]["Host"] == "attacker.test"  # nosec B101


@pytest.mark.asyncio
async def test_registered_https_webhook_delivery_preserves_hostname_for_tls(temp_db_path, monkeypatch) -> None:
    manager = WebhookManager(str(temp_db_path))
    original_url = "https://example.test/webhook"
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    webhook = {
        "id": 1,
        "url": original_url,
        "secret": "test-secret",
        "retry_count": 1,
        "timeout_seconds": 5,
    }
    payload = WebhookPayload(
        event="evaluation.completed",
        evaluation_id="eval-safe",
        timestamp="2026-06-23T00:00:00+00:00",
        data={"score": 1.0},
    )

    async def _ok_fetch(**kwargs):
        return SimpleNamespace(status_code=200, text="ok")

    with (
        patch.object(manager, "_create_delivery_record", return_value=1),
        patch.object(manager, "_update_delivery_record", MagicMock()),
        patch.object(manager, "_update_webhook_stats", MagicMock()),
        patch("tldw_Server_API.app.core.testing.is_test_mode", return_value=False),
        patch(
            "socket.getaddrinfo",
            return_value=[(None, None, None, None, ("93.184.216.34", 443))],
        ),
        patch(
            "tldw_Server_API.app.core.Evaluations.webhook_manager.afetch",
            new=AsyncMock(side_effect=_ok_fetch),
        ) as mock_fetch,
    ):
        await manager._deliver_webhook(webhook, payload)

    mock_fetch.assert_awaited_once()
    _, kwargs = mock_fetch.await_args
    assert kwargs["url"] == original_url  # nosec B101
    assert "Host" not in kwargs["headers"]  # nosec B101


@pytest.mark.asyncio
async def test_webhook_domain_filters_match_label_boundaries(monkeypatch) -> None:
    validator = WebhookSecurityValidator()
    monkeypatch.setattr(
        "socket.getaddrinfo",
        lambda *_args, **_kwargs: [(None, None, None, None, ("93.184.216.34", 80))],
    )

    validator.blocked_domains = {"example.com"}
    validator.allowed_domains = set()
    result = await validator.validate_webhook_url(
        "http://badexample.com/webhook",
        user_id="user-a",
        check_connectivity=False,
    )
    assert result.valid  # nosec B101
    assert not any(error.code == "BLOCKED_DOMAIN" for error in result.errors)  # nosec B101

    validator.blocked_domains = set()
    validator.allowed_domains = {".example.com"}
    result = await validator.validate_webhook_url(
        "http://badexample.com/webhook",
        user_id="user-a",
        check_connectivity=False,
    )
    assert not result.valid  # nosec B101
    assert any(error.code == "DOMAIN_NOT_ALLOWED" for error in result.errors)  # nosec B101

    result = await validator.validate_webhook_url(
        "http://api.example.com/webhook",
        user_id="user-a",
        check_connectivity=False,
    )
    assert result.valid  # nosec B101
    assert result.metadata["domain_status"] == "allowed"  # nosec B101


def test_webhook_private_ip_check_handles_mixed_ip_versions() -> None:
    validator = WebhookSecurityValidator()

    validator._raise_if_private_ip(ipaddress.ip_address("2001:4860:4860::8888"))

    with pytest.raises(ValueError, match="private IP"):
        validator._raise_if_private_ip(ipaddress.ip_address("::1"))


def test_postgresql_adapter_is_not_constructible_until_implemented() -> None:
    with pytest.raises(ValueError, match="Unsupported database type"):
        DatabaseAdapterFactory.create(
            DatabaseConfig(
                db_type=DatabaseType.POSTGRESQL,
                connection_string="postgresql://user:pass@example.com/evaluations",
            )
        )


@pytest.mark.asyncio
async def test_rate_limiter_cost_reservation_is_atomic(tmp_path, monkeypatch) -> None:
    limiter = UserRateLimiter(db_path=str(tmp_path / "rate_limits.db"))
    user_id = "cost-user"
    endpoint = "/api/v1/evaluations/geval"
    await limiter.upgrade_user_tier(
        user_id=user_id,
        new_tier=UserTier.CUSTOM,
        custom_limits={
            "evaluations_per_minute": 100,
            "batch_evaluations_per_minute": 100,
            "evaluations_per_day": 100,
            "total_tokens_per_day": 100_000,
            "burst_size": 0,
            "max_cost_per_day": 0.05,
            "max_cost_per_month": 10.0,
        },
    )

    barrier = asyncio.Event()
    waiters = 0
    original_record_request = limiter._record_request

    async def _delayed_record_request(*args, **kwargs):
        nonlocal waiters
        waiters += 1
        if waiters == 2:
            barrier.set()
        await asyncio.wait_for(barrier.wait(), timeout=1)
        return await original_record_request(*args, **kwargs)

    monkeypatch.setattr(limiter, "_record_request", _delayed_record_request)

    results = await asyncio.gather(
        limiter.check_rate_limit(user_id, endpoint, estimated_cost=0.04),
        limiter.check_rate_limit(user_id, endpoint, estimated_cost=0.04),
    )

    allowed_count = sum(1 for allowed, _metadata in results if allowed)
    denied_metadata = [metadata for allowed, metadata in results if not allowed]
    assert allowed_count == 1  # nosec B101
    assert len(denied_metadata) == 1  # nosec B101
    assert denied_metadata[0]["error"] == "Daily cost limit exceeded"  # nosec B101


def test_synthetic_workflow_requires_user_for_owner_scoped_operations(tmp_path) -> None:
    db = EvaluationsDatabase(str(tmp_path / "evaluations.db"))
    service = SyntheticEvalWorkflowService(db=db, user_id=None)

    with pytest.raises(ValueError, match="user_id is required"):
        service.list_queue()
