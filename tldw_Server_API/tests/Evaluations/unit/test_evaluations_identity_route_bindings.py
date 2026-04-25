from __future__ import annotations

import pytest
from fastapi import Response

from tldw_Server_API.app.api.v1.endpoints.evaluations import (
    evaluations_benchmarks as benchmarks_ep,
)
from tldw_Server_API.app.api.v1.endpoints.evaluations import (
    evaluations_crud as crud_ep,
)
from tldw_Server_API.app.api.v1.endpoints.evaluations import (
    evaluations_datasets as datasets_ep,
)
from tldw_Server_API.app.api.v1.endpoints.evaluations import (
    evaluations_embeddings_abtest as abtest_ep,
)
from tldw_Server_API.app.api.v1.endpoints.evaluations import (
    evaluations_webhooks as webhooks_ep,
)
from tldw_Server_API.app.api.v1.schemas.embeddings_abtest_schemas import (
    EmbeddingsABTestCreateRequest,
)
from tldw_Server_API.app.api.v1.schemas.evaluation_schemas_unified import (
    CreateDatasetRequest,
    WebhookRegistrationRequest,
)
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User
from tldw_Server_API.app.core.Evaluations.identity import EvaluationIdentity


@pytest.mark.unit
def test_benchmark_helper_uses_canonical_string_scope(monkeypatch: pytest.MonkeyPatch):
    captured: dict[str, object] = {}

    class _Manager:
        def __init__(self, *, user_id=None, **_kwargs):
            captured["user_id"] = user_id

    monkeypatch.setattr(benchmarks_ep, "EvaluationManager", _Manager)

    identity = EvaluationIdentity(
        user_scope="tenant-user",
        created_by="tenant-user",
        rate_limit_subject="tenant-user",
        webhook_user_id="user_tenant-user",
    )

    benchmarks_ep._get_evaluation_manager_for_user(
        identity
    )

    assert captured["user_id"] == "tenant-user"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_create_dataset_uses_canonical_identity_for_service_and_idempotency(
    monkeypatch: pytest.MonkeyPatch,
):
    captured: dict[str, object] = {}

    class _DB:
        def lookup_idempotency(self, _scope, _idempotency_key, user_id):
            captured["lookup_user"] = user_id
            return None

        def record_idempotency(self, _scope, _idempotency_key, _resource_id, user_id):
            captured["record_user"] = user_id

    class _Service:
        def __init__(self):
            self.db = _DB()

        async def create_dataset(self, *, name, samples, description, metadata, created_by):
            captured["created_by"] = created_by
            captured["dataset_name"] = name
            captured["sample_count"] = len(samples)
            captured["metadata"] = metadata
            captured["description"] = description
            return "ds_identity_1"

        async def get_dataset(self, dataset_id, created_by):
            captured["get_created_by"] = created_by
            return {
                "id": dataset_id,
                "object": "dataset",
                "name": "new_ds",
                "description": "for tests",
                "sample_count": 1,
                "samples": [{"input": {"text": "foo"}, "expected": "bar", "metadata": {}}],
                "created": 1700000000,
                "created_at": 1700000000,
                "created_by": created_by,
                "metadata": {"k": "v"},
            }

    def _get_service(user_id):
        captured["service_user"] = user_id
        return _Service()

    monkeypatch.setattr(datasets_ep, "get_unified_evaluation_service_for_user", _get_service)

    current_user = User(id="tenant-user", username="tenant", email=None, is_active=True)
    payload = CreateDatasetRequest(
        name="new_ds",
        description="for tests",
        samples=[{"input": {"text": "foo"}, "expected": "bar", "metadata": {}}],
        metadata={"k": "v"},
    )

    response = Response()
    result = await datasets_ep.create_dataset(
        payload,
        user_id="super-secret-api-key",
        current_user=current_user,
        idempotency_key="idem-dataset",
        response=response,
    )

    assert result.id == "ds_identity_1"
    assert captured["service_user"] == "tenant-user"
    assert captured["lookup_user"] == "tenant-user"
    assert captured["created_by"] == "tenant-user"
    assert captured["get_created_by"] == "tenant-user"
    assert captured["record_user"] == "tenant-user"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_create_embeddings_abtest_uses_canonical_identity_for_storage_and_idempotency(
    monkeypatch: pytest.MonkeyPatch,
):
    captured: dict[str, object] = {}

    class _DB:
        def lookup_idempotency(self, _scope, _idempotency_key, user_id):
            captured["lookup_user"] = user_id
            return None

        def create_abtest(self, *, name, config, created_by):
            captured["created_by"] = created_by
            captured["name"] = name
            captured["config"] = config
            return "abtest_identity_1"

        def upsert_abtest_arm(self, **kwargs):
            captured.setdefault("arms", []).append(kwargs)

        def insert_abtest_queries(self, test_id, queries):
            captured["queries_test_id"] = test_id
            captured["queries"] = queries

        def record_idempotency(self, _scope, _idempotency_key, _resource_id, user_id):
            captured["record_user"] = user_id

    class _Service:
        def __init__(self):
            self.db = _DB()

    def _get_service(user_id):
        captured["service_user"] = user_id
        return _Service()

    monkeypatch.setattr(abtest_ep, "get_unified_evaluation_service_for_user", _get_service)
    monkeypatch.setattr(abtest_ep, "validate_abtest_policy", lambda cfg, user: None)

    def _capture_audit(*, user_id, eval_id, name, eval_type):
        captured["audit_user_id"] = user_id
        captured["audit_eval_id"] = eval_id
        captured["audit_name"] = name
        captured["audit_eval_type"] = eval_type

    monkeypatch.setattr(abtest_ep, "log_evaluation_created", _capture_audit)

    current_user = User(id="tenant-user", username="tenant", email=None, is_active=True)
    payload = EmbeddingsABTestCreateRequest.model_validate(
        {
            "name": "identity-check",
            "config": {
                "arms": [{"provider": "openai", "model": "text-embedding-3-small"}],
                "media_ids": [],
                "retrieval": {"k": 3, "search_mode": "vector"},
                "queries": [{"text": "hello"}],
                "metric_level": "media",
            },
        }
    )

    response = Response()
    result = await abtest_ep.create_embeddings_abtest(
        payload,
        user_ctx="super-secret-api-key",
        _=None,
        current_user=current_user,
        idempotency_key="idem-abtest",
        response=response,
    )

    assert result.test_id == "abtest_identity_1"
    assert captured["service_user"] == "tenant-user"
    assert captured["lookup_user"] == "tenant-user"
    assert captured["created_by"] == "tenant-user"
    assert captured["record_user"] == "tenant-user"
    assert captured["audit_user_id"] == "tenant-user"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_create_run_uses_canonical_identity_for_service_and_webhook_owner(
    monkeypatch: pytest.MonkeyPatch,
):
    captured: dict[str, object] = {}
    identity = EvaluationIdentity(
        user_scope="tenant-scope",
        created_by="tenant-created",
        rate_limit_subject="tenant-rate-limit",
        webhook_user_id="user_tenant-scope",
    )

    class _DB:
        def lookup_idempotency(self, _scope, _idempotency_key, user_id):
            captured["lookup_user"] = user_id
            return None

        def record_idempotency(self, _scope, _idempotency_key, _resource_id, user_id):
            captured["record_user"] = user_id

    class _Service:
        def __init__(self):
            self.db = _DB()

        async def create_run(
            self,
            *,
            eval_id,
            target_model,
            config,
            dataset_override,
            webhook_url,
            created_by,
            webhook_user_id,
        ):
            captured["eval_id"] = eval_id
            captured["target_model"] = target_model
            captured["config"] = config
            captured["dataset_override"] = dataset_override
            captured["webhook_url"] = webhook_url
            captured["created_by"] = created_by
            captured["webhook_user_id"] = webhook_user_id
            return {
                "id": "run_identity_1",
                "object": "run",
                "eval_id": eval_id,
                "status": "pending",
                "target_model": target_model or "gpt-4o-mini",
                "created": 1700000000,
            }

    monkeypatch.setattr(crud_ep, "get_evaluation_identity", lambda _user: identity, raising=False)

    def _get_service(user_id):
        captured["service_user"] = user_id
        return _Service()

    monkeypatch.setattr(crud_ep, "get_unified_evaluation_service_for_user", _get_service)

    result = await crud_ep.create_run(
        eval_id="eval_identity_1",
        request=crud_ep.CreateRunSimpleRequest(
            target_model="gpt-4o-mini",
            config={"max_workers": 2},
            webhook_url="https://example.com/hook",
        ),
        user_id="super-secret-api-key",
        current_user=User(id=7, username="tenant", email=None, is_active=True),
        idempotency_key="idem-run",
        response=Response(),
    )

    assert result.id == "run_identity_1"
    assert captured["service_user"] == "tenant-scope"
    assert captured["lookup_user"] == "tenant-created"
    assert captured["created_by"] == "tenant-created"
    assert captured["webhook_user_id"] == "user_tenant-scope"
    assert captured["record_user"] == "tenant-created"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_register_webhook_uses_canonical_identity_for_manager_scope(
    monkeypatch: pytest.MonkeyPatch,
):
    captured: dict[str, object] = {}
    identity = EvaluationIdentity(
        user_scope="tenant-scope",
        created_by="tenant-created",
        rate_limit_subject="tenant-rate-limit",
        webhook_user_id="user_tenant-scope",
    )

    class _Manager:
        async def register_webhook(self, *, user_id, url, secret, events, retry_count, timeout_seconds):
            captured["register_user_id"] = user_id
            captured["register_url"] = url
            captured["register_events"] = events
            captured["register_retry_count"] = retry_count
            captured["register_timeout_seconds"] = timeout_seconds
            return {
                "webhook_id": 1,
                "url": url,
                "events": [str(event.value if hasattr(event, "value") else event) for event in events],
                "secret": secret,
                "created_at": "2024-01-01T00:00:00",
                "status": "active",
            }

    monkeypatch.setattr(webhooks_ep, "get_evaluation_identity", lambda _user: identity, raising=False)

    def _get_manager(user_id):
        captured["manager_scope"] = user_id
        return _Manager()

    monkeypatch.setattr(webhooks_ep, "_get_webhook_manager_for_user", _get_manager)

    result = await webhooks_ep.register_webhook(
        request=WebhookRegistrationRequest(
            url="https://example.com/webhook",
            events=["evaluation.completed"],
            secret="s" * 32,
        ),
        _user_ctx="super-secret-api-key",
        current_user=User(id=7, username="tenant", email=None, is_active=True),
    )

    assert result.webhook_id == 1
    assert captured["manager_scope"] == "tenant-scope"
    assert captured["register_user_id"] == "user_tenant-scope"
