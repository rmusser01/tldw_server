# Evals Identity Remediation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Canonicalize evaluations user identity so auth context tokens are never used as storage or limiter subjects, and both numeric and tenant-style string user ids resolve to the correct per-user evaluations state across every Evals surface.

**Architecture:** Introduce one canonical evaluations identity helper that derives a stable user scope from `current_user`, then make every per-user factory and route consume that scope for DB binding, idempotency, `created_by`, webhook ownership, and Jobs ownership. Keep `verify_api_key()` as an authentication dependency only; routes may still require it, but all persistence and rate-limit accounting must come from the canonical scope instead of the raw auth token.

**Tech Stack:** Python 3, FastAPI, Pydantic, pytest, SQLite/PostgreSQL-aware DB path helpers, Loguru, Bandit, Markdown

---

## Implementation File Map

**Create:**
- `tldw_Server_API/app/core/Evaluations/identity.py`: Canonical evaluations identity helpers and `EvaluationIdentity` dataclass.
- `tldw_Server_API/tests/Evaluations/unit/test_evals_identity.py`: Unit coverage for canonical scope derivation, per-user service cache keys, per-user rate-limiter cache keys, and `EvaluationManager` string-id DB binding.
- `tldw_Server_API/tests/Evaluations/integration/test_embeddings_abtest_multi_user_api.py`: Integration coverage proving string user ids create and read A/B state from the correct per-user evaluations DB with the right `created_by`.
- `tldw_Server_API/tests/Evaluations/unit/test_evaluations_identity_route_bindings.py`: Focused route-binding regressions for CRUD and RAG pipeline endpoints that must use stable string ids.

**Modify:**
- `tldw_Server_API/app/api/v1/endpoints/evaluations/evaluations_auth.py`: Route-facing wrapper for canonical evaluations identity and rate-limit header/status helpers.
- `tldw_Server_API/app/api/v1/endpoints/evaluations/evaluations_unified.py`: Replace raw `user_ctx` usage in rate-limit, G-Eval, RAG, response-quality, OCR, batch, A/B SSE/export/delete, metrics, and service-binding paths.
- `tldw_Server_API/app/api/v1/endpoints/evaluations/evaluations_crud.py`: Use canonical scope for service binding, idempotency lookup/recording, `created_by`, and webhook ownership.
- `tldw_Server_API/app/api/v1/endpoints/evaluations/evaluations_datasets.py`: Use canonical scope for service binding, idempotency, dataset ownership, and listing filters.
- `tldw_Server_API/app/api/v1/endpoints/evaluations/evaluations_rag_pipeline.py`: Use canonical scope for service binding and preset ownership.
- `tldw_Server_API/app/api/v1/endpoints/evaluations/evaluations_benchmarks.py`: Bind `EvaluationManager` with the canonical scope instead of numeric coercion.
- `tldw_Server_API/app/api/v1/endpoints/evaluations/evaluations_webhooks.py`: Bind per-user webhook managers with the canonical scope and keep webhook ownership derived from that same scope.
- `tldw_Server_API/app/api/v1/endpoints/evaluations/evaluations_embeddings_abtest.py`: Use canonical scope for service binding, idempotency, `created_by`, sync execution, and Jobs payload ownership.
- `tldw_Server_API/app/core/Evaluations/unified_evaluation_service.py`: Change per-user service cache keys from `int` to canonical `str` scope and preserve tenant-style ids.
- `tldw_Server_API/app/core/Evaluations/user_rate_limiter.py`: Change per-user limiter cache keys from `int` to canonical `str` scope and preserve tenant-style ids outside test shims.
- `tldw_Server_API/app/core/Evaluations/evaluation_manager.py`: Accept `user_id: int | str`, preserve canonical string scopes, and resolve DB paths without numeric fallback.
- `tldw_Server_API/app/core/Evaluations/embeddings_abtest_jobs_worker.py`: Preserve string owner ids when rehydrating Jobs and opening per-user services/media DBs.
- `tldw_Server_API/app/core/Evaluations/webhook_identity.py`: Route webhook owner id generation through the canonical evaluations scope helper.
- `tldw_Server_API/tests/AuthNZ/unit/test_evaluations_auth_runtime_guards.py`: Add unit coverage for the route-facing identity helper wrapper.
- `tldw_Server_API/tests/Evaluations/test_evaluations_stage1_route_and_error_regressions.py`: Add rate-limit status regression for canonical user scope instead of auth token.
- `tldw_Server_API/tests/Evaluations/test_evaluations_stage4_auth_policy_and_dataset_permissions.py`: Add dataset route regression proving service binding and `created_by` use the stable string user id.
- `tldw_Server_API/tests/Evaluations/test_evaluations_benchmarks_api.py`: Add benchmark helper regression for string user ids.
- `tldw_Server_API/tests/Evaluations/test_embeddings_abtest_jobs_worker.py`: Add worker regression proving `owner_user_id="tenant-user"` stays `tenant-user`.
- `tldw_Server_API/tests/Evaluations/integration/test_webhook_multi_user_api.py`: Extend multi-user webhook coverage to string ids, not only integers.
- `tldw_Server_API/tests/Evaluations/conftest.py`: Seed per-user service cache under canonical string keys so fixtures stay aligned with the new cache contract.
- `tldw_Server_API/tests/Evaluations/test_evaluations_unified.py`: Align cache seeding with canonical string keys where tests pre-populate `_service_instances_by_user`.
- `tldw_Server_API/app/core/Evaluations/README.md`: Document the canonical evaluations identity contract.
- `tldw_Server_API/app/core/Evaluations/SECURITY.md`: Document that auth tokens must never be reused as ownership or rate-limit subjects.

## Task 1: Introduce the Canonical Evaluations Identity Helper and Core Factory Support

**Files:**
- Create: `tldw_Server_API/app/core/Evaluations/identity.py`
- Create: `tldw_Server_API/tests/Evaluations/unit/test_evals_identity.py`
- Modify: `tldw_Server_API/app/core/Evaluations/unified_evaluation_service.py`
- Modify: `tldw_Server_API/app/core/Evaluations/user_rate_limiter.py`
- Modify: `tldw_Server_API/app/core/Evaluations/evaluation_manager.py`
- Modify: `tldw_Server_API/tests/Evaluations/conftest.py`
- Modify: `tldw_Server_API/tests/Evaluations/test_evaluations_unified.py`
- Test: `tldw_Server_API/tests/Evaluations/unit/test_evals_identity.py`

- [ ] **Step 1: Write the failing core identity tests**

Create `tldw_Server_API/tests/Evaluations/unit/test_evals_identity.py` with:

```python
import pytest

from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User
from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths
from tldw_Server_API.app.core.Evaluations import unified_evaluation_service as service_module
from tldw_Server_API.app.core.Evaluations import user_rate_limiter as limiter_module
from tldw_Server_API.app.core.Evaluations.evaluation_manager import EvaluationManager


def test_evaluations_identity_preserves_string_user_ids():
    from tldw_Server_API.app.core.Evaluations.identity import evaluations_identity_from_user

    identity = evaluations_identity_from_user(
        User(id="tenant-user", username="tenant-user", is_active=True)
    )

    assert identity.user_scope == "tenant-user"
    assert identity.created_by == "tenant-user"
    assert identity.rate_limit_subject == "tenant-user"
    assert identity.webhook_user_id == "user_tenant-user"


def test_unified_service_cache_uses_canonical_string_scope(monkeypatch, tmp_path):
    created = {}

    class _DummyService:
        def __init__(self, db_path: str, **_kwargs):
            created["db_path"] = db_path

    monkeypatch.setattr(service_module, "UnifiedEvaluationService", _DummyService)
    service_module._service_instances_by_user.clear()
    service_module._service_instance = None

    svc = service_module.get_unified_evaluation_service_for_user("tenant-user")

    assert isinstance(svc, _DummyService)
    assert created["db_path"] == str(DatabasePaths.get_evaluations_db_path("tenant-user"))
    assert "tenant-user" in service_module._service_instances_by_user


def test_rate_limiter_cache_uses_canonical_string_scope(monkeypatch, tmp_path):
    created = {}

    class _DummyLimiter:
        def __init__(self, db_path: str):
            created["db_path"] = db_path

    monkeypatch.setattr(limiter_module, "UserRateLimiter", _DummyLimiter)
    monkeypatch.setattr(limiter_module, "is_test_mode", lambda: False)
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    limiter_module._user_rate_limiter_instances.clear()

    limiter = limiter_module.get_user_rate_limiter_for_user("tenant-user")

    assert isinstance(limiter, _DummyLimiter)
    assert created["db_path"] == str(DatabasePaths.get_evaluations_db_path("tenant-user"))
    assert "tenant-user" in limiter_module._user_rate_limiter_instances


def test_evaluation_manager_uses_string_user_scope_for_db_path(tmp_path, monkeypatch):
    monkeypatch.setattr(EvaluationManager, "_init_database", lambda self: None, raising=False)

    manager = EvaluationManager(user_id="tenant-user")

    assert manager.db_path == DatabasePaths.get_evaluations_db_path("tenant-user")
```

- [ ] **Step 2: Run the new tests to verify the current code fails**

Run:
```bash
source .venv/bin/activate && python -m pytest -v tldw_Server_API/tests/Evaluations/unit/test_evals_identity.py
```

Expected: failures showing one or more of these current problems: missing `identity.py`, per-user service cache coerces `"tenant-user"` to the single-user DB, per-user rate-limiter cache coerces `"tenant-user"` to the single-user DB, or `EvaluationManager(user_id="tenant-user")` falls back to numeric single-user behavior.

- [ ] **Step 3: Implement the canonical helper and switch the core factories to canonical string scopes**

Create `tldw_Server_API/app/core/Evaluations/identity.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths


@dataclass(frozen=True)
class EvaluationIdentity:
    user_scope: str
    created_by: str
    rate_limit_subject: str
    webhook_user_id: str


def canonical_evaluations_user_scope(
    user_or_id: Any,
    *,
    fallback: str | int | None = None,
) -> str:
    raw = ""
    if hasattr(user_or_id, "id_str"):
        raw = str(getattr(user_or_id, "id_str") or "").strip()
    elif hasattr(user_or_id, "id"):
        raw = str(getattr(user_or_id, "id") or "").strip()
    elif user_or_id is not None:
        raw = str(user_or_id).strip()

    if not raw and fallback is not None:
        raw = str(fallback).strip()
    if not raw:
        raw = str(DatabasePaths.get_single_user_id())
    return raw


def evaluations_identity_from_user(
    user: Any,
    *,
    fallback: str | int | None = None,
) -> EvaluationIdentity:
    user_scope = canonical_evaluations_user_scope(user, fallback=fallback)
    webhook_user_id = user_scope if user_scope.startswith("user_") else f"user_{user_scope}"
    return EvaluationIdentity(
        user_scope=user_scope,
        created_by=user_scope,
        rate_limit_subject=user_scope,
        webhook_user_id=webhook_user_id,
    )
```

Update `tldw_Server_API/app/core/Evaluations/unified_evaluation_service.py`:

```python
from tldw_Server_API.app.core.Evaluations.identity import canonical_evaluations_user_scope

_service_instances_by_user: "OrderedDict[str, UnifiedEvaluationService]" = OrderedDict()


def get_unified_evaluation_service_for_user(user_id: str | int) -> UnifiedEvaluationService:
    global _service_instances_lock
    if _service_instances_lock is None:
        import threading as _threading
        _service_instances_lock = _threading.Lock()

    with _service_instances_lock:
        uid_key = canonical_evaluations_user_scope(user_id)
        if uid_key in _service_instances_by_user:
            svc = _service_instances_by_user.pop(uid_key)
            _service_instances_by_user[uid_key] = svc
            return svc

        db_path = str(DatabasePaths.get_evaluations_db_path(uid_key))
        svc = UnifiedEvaluationService(db_path=db_path)
        _service_instances_by_user[uid_key] = svc
        return svc
```

Update `tldw_Server_API/app/core/Evaluations/user_rate_limiter.py`:

```python
from tldw_Server_API.app.core.Evaluations.identity import canonical_evaluations_user_scope


def get_user_rate_limiter_for_user(user_id: str | int) -> UserRateLimiter:
    try:
        import os as _os
        if is_test_mode() or "PYTEST_CURRENT_TEST" in _os.environ:
            return user_rate_limiter
    except _USER_RATE_LIMIT_NONCRITICAL_EXCEPTIONS:
        pass

    global _user_rate_limiter_lock
    if _user_rate_limiter_lock is None:
        _user_rate_limiter_lock = threading.Lock()

    with _user_rate_limiter_lock:
        uid_key = canonical_evaluations_user_scope(user_id)
        inst = _user_rate_limiter_instances.get(uid_key)
        if inst is not None:
            return inst
        db_path = str(DatabasePaths.get_evaluations_db_path(uid_key))
        inst = UserRateLimiter(db_path=db_path)
        _user_rate_limiter_instances[uid_key] = inst
        return inst
```

Update `tldw_Server_API/app/core/Evaluations/evaluation_manager.py`:

```python
from tldw_Server_API.app.core.Evaluations.identity import canonical_evaluations_user_scope


class EvaluationManager:
    def __init__(self, db_path: Optional[Union[str, Path]] = None, *, user_id: Optional[int | str] = None):
        self.config = load_comprehensive_config()
        self._user_id = canonical_evaluations_user_scope(
            user_id,
            fallback=DatabasePaths.get_single_user_id(),
        )
        self.db_path = self._get_db_path(explicit_path=db_path)
        self._init_database()
```

Align the test fixtures in `tldw_Server_API/tests/Evaluations/conftest.py` and `tldw_Server_API/tests/Evaluations/test_evaluations_unified.py` so any seeded `_service_instances_by_user` entry uses `canonical_evaluations_user_scope(_test_user_id)` instead of the raw integer key.

- [ ] **Step 4: Run the unit tests again and one existing service-mapping check**

Run:
```bash
source .venv/bin/activate && python -m pytest -v \
  tldw_Server_API/tests/Evaluations/unit/test_evals_identity.py \
  tldw_Server_API/tests/Evaluations/unit/test_unified_evaluation_service_mapping.py
```

Expected: PASS. The new identity tests should prove canonical string scopes survive all three core factories, and the existing service mapping test should still pass with string cache keys.

- [ ] **Step 5: Commit the core identity foundation**

Run:
```bash
git add \
  tldw_Server_API/app/core/Evaluations/identity.py \
  tldw_Server_API/app/core/Evaluations/unified_evaluation_service.py \
  tldw_Server_API/app/core/Evaluations/user_rate_limiter.py \
  tldw_Server_API/app/core/Evaluations/evaluation_manager.py \
  tldw_Server_API/tests/Evaluations/unit/test_evals_identity.py \
  tldw_Server_API/tests/Evaluations/conftest.py \
  tldw_Server_API/tests/Evaluations/test_evaluations_unified.py
git commit -m "fix: canonicalize evals identity core factories"
```

Expected: one commit capturing the new helper plus the core per-user factory changes.

## Task 2: Thread Canonical Identity Through Shared Auth, Rate-Limit, Dataset, and Unified Route Paths

**Files:**
- Modify: `tldw_Server_API/app/api/v1/endpoints/evaluations/evaluations_auth.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/evaluations/evaluations_unified.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/evaluations/evaluations_datasets.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/evaluations/evaluations_crud.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/evaluations/evaluations_rag_pipeline.py`
- Modify: `tldw_Server_API/tests/AuthNZ/unit/test_evaluations_auth_runtime_guards.py`
- Modify: `tldw_Server_API/tests/Evaluations/test_evaluations_stage1_route_and_error_regressions.py`
- Modify: `tldw_Server_API/tests/Evaluations/test_evaluations_stage4_auth_policy_and_dataset_permissions.py`
- Test: `tldw_Server_API/tests/Evaluations/test_evaluations_stage1_route_and_error_regressions.py`
- Test: `tldw_Server_API/tests/Evaluations/test_evaluations_stage4_auth_policy_and_dataset_permissions.py`

- [ ] **Step 1: Add failing route regressions for auth-token leakage**

Append this test to `tldw_Server_API/tests/Evaluations/test_evaluations_stage1_route_and_error_regressions.py`:

```python
def test_rate_limits_endpoint_uses_stable_user_id_instead_of_auth_context_token(monkeypatch):
    app = _build_eval_only_app(monkeypatch)

    async def _verify_api_key_override():
        return "super-secret-api-key"

    async def _get_user_override():
        return User(id="tenant-user", username="tenant-user", email=None, is_active=True)

    app.dependency_overrides[eval_unified.verify_api_key] = _verify_api_key_override
    app.dependency_overrides[eval_unified.get_eval_request_user] = _get_user_override

    captured = {}

    class _Limiter:
        async def get_usage_summary(self, user_id):
            captured["user_id"] = user_id
            return {
                "tier": "free",
                "limits": {"per_minute": {"evaluations": 10}, "daily": {"evaluations": 100, "tokens": 1000, "cost": 1}, "monthly": {"cost": 10}},
                "usage": {"today": {"evaluations": 1, "tokens": 10, "cost": 0}, "month": {"cost": 0}},
                "remaining": {"daily_evaluations": 99, "daily_tokens": 990, "daily_cost": 1, "monthly_cost": 10},
            }

    monkeypatch.setattr(eval_unified, "get_user_rate_limiter_for_user", lambda _uid: _Limiter())

    with TestClient(app) as client:
        response = client.get("/api/v1/evaluations/rate-limits")

    assert response.status_code == 200, response.text
    assert captured["user_id"] == "tenant-user"
```

Append this test to `tldw_Server_API/tests/Evaluations/test_evaluations_stage4_auth_policy_and_dataset_permissions.py`:

```python
def test_dataset_create_uses_stable_string_user_id_for_service_and_idempotency(monkeypatch):
    app = FastAPI()
    app.include_router(eval_unified.router, prefix="/api/v1")

    captured = {}

    class _FakeDB:
        def lookup_idempotency(self, _scope, _idempotency_key, user_id):
            captured["lookup_user_id"] = user_id
            return None

        def record_idempotency(self, _scope, _idempotency_key, _resource_id, user_id):
            captured["record_user_id"] = user_id
            return None

    class _Service:
        def __init__(self):
            self.db = _FakeDB()

        async def create_dataset(self, *, name, samples, description, metadata, created_by):
            captured["created_by"] = created_by
            return "ds_tenant"

        async def get_dataset(self, dataset_id, created_by):
            return {
                "id": dataset_id,
                "object": "dataset",
                "name": "tenant-dataset",
                "description": "",
                "sample_count": 1,
                "samples": [{"input": {"text": "hello"}, "expected": "hello", "metadata": {}}],
                "created": 1700000000,
                "created_at": 1700000000,
                "created_by": created_by,
                "metadata": {},
            }

    def _service_factory(user_id):
        captured["service_user_id"] = user_id
        return _Service()

    monkeypatch.setattr(eval_datasets, "get_unified_evaluation_service_for_user", _service_factory)

    async def _verify_api_key_override():
        return "super-secret-api-key"

    async def _get_user_override():
        return User(
            id="tenant-user",
            username="tenant-user",
            email=None,
            is_active=True,
            permissions=[EVALS_READ, EVALS_MANAGE],
            is_admin=False,
        )

    app.dependency_overrides[eval_datasets.verify_api_key] = _verify_api_key_override
    app.dependency_overrides[eval_datasets.get_eval_request_user] = _get_user_override
    app.dependency_overrides[eval_unified.verify_api_key] = _verify_api_key_override
    app.dependency_overrides[eval_unified.get_eval_request_user] = _get_user_override

    payload = {
        "name": "tenant-dataset",
        "description": "tenant scoped",
        "samples": [{"input": {"text": "foo"}, "expected": "bar", "metadata": {}}],
        "metadata": {"scope": "tenant"},
    }

    with TestClient(app) as client:
        response = client.post("/api/v1/evaluations/datasets", json=payload)

    assert response.status_code == 201, response.text
    assert captured["service_user_id"] == "tenant-user"
    assert captured["lookup_user_id"] == "tenant-user"
    assert captured["record_user_id"] == "tenant-user"
    assert captured["created_by"] == "tenant-user"
```

Append this test to `tldw_Server_API/tests/AuthNZ/unit/test_evaluations_auth_runtime_guards.py`:

```python
def test_get_evaluations_identity_uses_request_user_not_auth_token() -> None:
    user = SimpleNamespace(id="tenant-user", id_str="tenant-user")

    identity = eval_auth.get_evaluations_identity(user)

    assert identity.user_scope == "tenant-user"
    assert identity.created_by == "tenant-user"
    assert identity.rate_limit_subject == "tenant-user"
```

- [ ] **Step 2: Run the shared-route regressions to confirm they fail first**

Run:
```bash
source .venv/bin/activate && python -m pytest -v \
  tldw_Server_API/tests/AuthNZ/unit/test_evaluations_auth_runtime_guards.py \
  tldw_Server_API/tests/Evaluations/test_evaluations_stage1_route_and_error_regressions.py \
  tldw_Server_API/tests/Evaluations/test_evaluations_stage4_auth_policy_and_dataset_permissions.py
```

Expected: failures showing that `/rate-limits` still uses `verify_api_key()` output as the usage subject, dataset idempotency still uses the auth token instead of the stable user scope, or the route-facing identity helper does not exist yet.

- [ ] **Step 3: Add a route-facing identity wrapper and replace raw `user_ctx` ownership across the shared routes**

Update `tldw_Server_API/app/api/v1/endpoints/evaluations/evaluations_auth.py`:

```python
from tldw_Server_API.app.core.Evaluations.identity import (
    EvaluationIdentity,
    evaluations_identity_from_user,
)


def get_evaluations_identity(current_user: User) -> EvaluationIdentity:
    return evaluations_identity_from_user(current_user)
```

Update the rate-limit status route in `tldw_Server_API/app/api/v1/endpoints/evaluations/evaluations_unified.py`:

```python
async def get_rate_limit_status(
    user_id: str = Depends(verify_api_key),
    current_user: User = Depends(get_eval_request_user),
):
    try:
        _ = user_id
        identity = get_evaluations_identity(current_user)
        limiter = get_user_rate_limiter_for_user(identity.user_scope)
        summary = await limiter.get_usage_summary(identity.rate_limit_subject)
```

Update the shared evaluation routes in `tldw_Server_API/app/api/v1/endpoints/evaluations/evaluations_unified.py` so every limiter and service call uses the identity wrapper:

```python
identity = get_evaluations_identity(current_user)
limiter = get_user_rate_limiter_for_user(identity.user_scope)
allowed, meta = await limiter.check_rate_limit(
    identity.rate_limit_subject,
    endpoint="evals:geval",
    is_batch=False,
    tokens_requested=tokens_est,
    estimated_cost=0.0,
)
svc = get_unified_evaluation_service_for_user(identity.user_scope)
result = await svc.evaluate_geval(
    source_text=request.source_text,
    summary=request.summary,
    metrics=request.metrics,
    api_name=provider_name,
    api_key=provider_api_key,
    user_id=identity.created_by,
    webhook_user_id=identity.webhook_user_id,
)
```

For each remaining `evaluations_unified.py` route that still mixes `user_ctx`, `current_user.id`, or raw inline `stable_user_id` values, make the same four replacements:

```python
identity = get_evaluations_identity(current_user)
limiter = get_user_rate_limiter_for_user(identity.user_scope)
svc = get_unified_evaluation_service_for_user(identity.user_scope)
created_by = identity.created_by
```

Apply that exact pattern to `/rag`, `/response-quality`, `/batch`, `/ocr`, `/ocr-pdf`, `/history`, `/metrics`, and the A/B SSE/export/delete routes.

Update `tldw_Server_API/app/api/v1/endpoints/evaluations/evaluations_datasets.py`:

```python
identity = get_evaluations_identity(current_user)
svc = get_unified_evaluation_service_for_user(identity.user_scope)
if idempotency_key:
    existing_id = svc.db.lookup_idempotency("dataset", idempotency_key, identity.created_by)
dataset_id = await svc.create_dataset(
    name=dataset_request.name,
    samples=[model_dump_compat(s) for s in dataset_request.samples],
    description=dataset_request.description or "",
    metadata=model_dump_compat(dataset_request.metadata) if dataset_request.metadata else None,
    created_by=identity.created_by,
)
svc.db.record_idempotency("dataset", idempotency_key, dataset_id, identity.created_by)
```

Update `tldw_Server_API/app/api/v1/endpoints/evaluations/evaluations_crud.py` everywhere it currently binds `get_unified_evaluation_service_for_user(current_user.id)` or records idempotency with `stable_user_id` derived inline:

```python
identity = get_evaluations_identity(current_user)
svc = get_unified_evaluation_service_for_user(identity.user_scope)
existing_id = svc.db.lookup_idempotency("run", idempotency_key, identity.created_by)
run = await svc.create_run(
    eval_id,
    target_model=request.target_model,
    config=request.config,
    dataset_override=request.dataset_override,
    webhook_url=request.webhook_url,
    created_by=identity.created_by,
    webhook_user_id=identity.webhook_user_id,
)
svc.db.record_idempotency("run", idempotency_key, run["id"], identity.created_by)
```

Update `tldw_Server_API/app/api/v1/endpoints/evaluations/evaluations_rag_pipeline.py`:

```python
identity = get_evaluations_identity(current_user)
svc = get_unified_evaluation_service_for_user(identity.user_scope)
db.upsert_pipeline_preset(preset.name, preset.config, user_id=identity.created_by)
row = db.get_pipeline_preset(preset.name, user_id=identity.created_by)
```

- [ ] **Step 4: Re-run the shared-route tests plus an existing CRUD smoke test**

Run:
```bash
source .venv/bin/activate && python -m pytest -v \
  tldw_Server_API/tests/AuthNZ/unit/test_evaluations_auth_runtime_guards.py \
  tldw_Server_API/tests/Evaluations/test_evaluations_stage1_route_and_error_regressions.py \
  tldw_Server_API/tests/Evaluations/test_evaluations_stage4_auth_policy_and_dataset_permissions.py \
  tldw_Server_API/tests/Evaluations/test_evaluations_crud_create_run_api.py
```

Expected: PASS. Shared routes should now ignore the raw auth token for storage and limiter identity, while the existing CRUD create-run smoke test should continue to pass.

- [ ] **Step 5: Commit the shared-route identity fix**

Run:
```bash
git add \
  tldw_Server_API/app/api/v1/endpoints/evaluations/evaluations_auth.py \
  tldw_Server_API/app/api/v1/endpoints/evaluations/evaluations_unified.py \
  tldw_Server_API/app/api/v1/endpoints/evaluations/evaluations_datasets.py \
  tldw_Server_API/app/api/v1/endpoints/evaluations/evaluations_crud.py \
  tldw_Server_API/app/api/v1/endpoints/evaluations/evaluations_rag_pipeline.py \
  tldw_Server_API/tests/AuthNZ/unit/test_evaluations_auth_runtime_guards.py \
  tldw_Server_API/tests/Evaluations/test_evaluations_stage1_route_and_error_regressions.py \
  tldw_Server_API/tests/Evaluations/test_evaluations_stage4_auth_policy_and_dataset_permissions.py
git commit -m "fix: use stable eval identity in shared routes"
```

Expected: one commit that removes auth-token identity leakage from the shared routes.

## Task 3: Fix the Remaining Feature-Specific Identity Leaks in Benchmarks, Webhooks, Embeddings A/B, and the Jobs Worker

**Files:**
- Modify: `tldw_Server_API/app/api/v1/endpoints/evaluations/evaluations_benchmarks.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/evaluations/evaluations_webhooks.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/evaluations/evaluations_embeddings_abtest.py`
- Modify: `tldw_Server_API/app/core/Evaluations/embeddings_abtest_jobs_worker.py`
- Modify: `tldw_Server_API/app/core/Evaluations/webhook_identity.py`
- Modify: `tldw_Server_API/tests/Evaluations/test_evaluations_benchmarks_api.py`
- Modify: `tldw_Server_API/tests/Evaluations/integration/test_webhook_multi_user_api.py`
- Modify: `tldw_Server_API/tests/Evaluations/test_embeddings_abtest_jobs_worker.py`
- Create: `tldw_Server_API/tests/Evaluations/integration/test_embeddings_abtest_multi_user_api.py`
- Test: `tldw_Server_API/tests/Evaluations/test_evaluations_benchmarks_api.py`
- Test: `tldw_Server_API/tests/Evaluations/integration/test_webhook_multi_user_api.py`
- Test: `tldw_Server_API/tests/Evaluations/test_embeddings_abtest_jobs_worker.py`
- Test: `tldw_Server_API/tests/Evaluations/integration/test_embeddings_abtest_multi_user_api.py`

- [ ] **Step 1: Add failing feature-surface regressions for string user ids**

Append this test to `tldw_Server_API/tests/Evaluations/test_evaluations_benchmarks_api.py`:

```python
def test_benchmark_manager_preserves_string_user_id(monkeypatch):
    captured = {}

    class _Manager:
        pass

    def _manager_factory(*, user_id=None):
        captured["user_id"] = user_id
        return _Manager()

    monkeypatch.setattr(benchmarks_ep, "EvaluationManager", _manager_factory)

    user = User(
        id="tenant-user",
        username="tenant-user",
        email=None,
        is_active=True,
        roles=["admin"],
        permissions=["system.configure", "evals.read", "evals.manage"],
    )

    manager = benchmarks_ep._get_evaluation_manager_for_user(user)

    assert isinstance(manager, _Manager)
    assert captured["user_id"] == "tenant-user"
```

Append this test to `tldw_Server_API/tests/Evaluations/test_embeddings_abtest_jobs_worker.py`:

```python
@pytest.mark.unit
@pytest.mark.asyncio
async def test_handle_abtest_job_preserves_string_owner_user_id(monkeypatch):
    captured = {}

    async def _fake_run_abtest_full(db, config, test_id, user_id, media_db):
        captured["test_id"] = test_id
        captured["user_id"] = user_id

    class _Svc:
        def __init__(self):
            self.db = object()

    def _service_factory(uid):
        captured["service_uid"] = uid
        return _Svc()

    def _media_db_factory(user_id):
        captured["media_db_uid"] = user_id
        return object()

    monkeypatch.setattr(worker, "get_unified_evaluation_service_for_user", _service_factory)
    monkeypatch.setattr(worker, "_build_media_db", _media_db_factory)
    monkeypatch.setattr(worker, "run_abtest_full", _fake_run_abtest_full)

    job = {
        "job_type": ABTEST_JOBS_JOB_TYPE,
        "payload": {
            "test_id": "abtest_tenant",
            "config": {
                "arms": [{"provider": "openai", "model": "text-embedding-3-small"}],
                "media_ids": [],
                "retrieval": {"k": 3, "search_mode": "vector"},
                "queries": [{"text": "hello"}],
                "metric_level": "media",
            },
        },
        "owner_user_id": "tenant-user",
    }

    result = await worker.handle_abtest_job(job)

    assert result["test_id"] == "abtest_tenant"
    assert captured["service_uid"] == "tenant-user"
    assert captured["media_db_uid"] == "tenant-user"
    assert captured["user_id"] == "tenant-user"
```

In the `multi_user_webhook_client` fixture inside `tldw_Server_API/tests/Evaluations/integration/test_webhook_multi_user_api.py`, replace the `_get_eval_request_user` override with:

```python
    async def _get_eval_request_user(
        request: Request,
        _user_ctx: str = Depends(_verify_api_key),
    ) -> User:
        user_id = request.headers.get("X-User-Id", "1")
        parsed_user_id = int(user_id) if str(user_id).isdigit() else str(user_id)
        return User(
            id=parsed_user_id,
            username=f"user_{user_id}",
            roles=["admin"],
            permissions=["evals.read", "evals.manage"],
            is_admin=True,
        )
```

Then append this test to the same file:

```python
def test_webhook_list_preserves_string_user_scopes(multi_user_webhook_client, monkeypatch):
    client = multi_user_webhook_client

    db_path = DatabasePaths.get_evaluations_db_path("tenant-user")
    _seed_webhook(db_path, user_id="tenant-user", url="https://example.com/tenant")

    resp = client.get(
        "/api/v1/evaluations/webhooks",
        headers={"X-User-Id": "tenant-user", "X-API-KEY": "test"},
    )

    assert resp.status_code == 200, resp.text
    payload = resp.json()
    assert len(payload) == 1
    assert payload[0]["url"] == "https://example.com/tenant"
```

Create `tldw_Server_API/tests/Evaluations/integration/test_embeddings_abtest_multi_user_api.py`:

```python
import pytest
from fastapi import Depends, FastAPI, Header, Request
from fastapi.testclient import TestClient

from tldw_Server_API.app.api.v1.endpoints.evaluations import evaluations_auth as eval_auth
from tldw_Server_API.app.api.v1.endpoints.evaluations.evaluations_unified import router as evals_router
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User
from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths
from tldw_Server_API.app.core.Evaluations.unified_evaluation_service import get_unified_evaluation_service_for_user


pytestmark = [pytest.mark.integration]


def test_embeddings_abtest_create_uses_string_user_scope(tmp_path, monkeypatch):
    monkeypatch.setenv("AUTH_MODE", "multi_user")
    monkeypatch.setenv("TESTING", "true")
    monkeypatch.setenv("TEST_MODE", "true")
    monkeypatch.setenv("EVALS_HEAVY_ADMIN_ONLY", "false")
    monkeypatch.setenv("USER_DB_BASE_DIR", str(tmp_path / "user_dbs"))

    app = FastAPI()
    app.include_router(evals_router, prefix="/api/v1")

    async def _verify_api_key(request: Request, x_api_key: str = Header(None, alias="X-API-KEY")) -> str:
        _ = x_api_key
        return "super-secret-api-key"

    async def _get_eval_request_user(
        request: Request,
        _user_ctx: str = Depends(_verify_api_key),
    ) -> User:
        return User(
            id="tenant-user",
            username="tenant-user",
            roles=["admin"],
            permissions=["evals.read", "evals.manage", "system.configure"],
            is_admin=True,
            is_active=True,
        )

    app.dependency_overrides[eval_auth.verify_api_key] = _verify_api_key
    app.dependency_overrides[eval_auth.get_eval_request_user] = _get_eval_request_user

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/evaluations/embeddings/abtest",
            json={
                "name": "tenant-abtest",
                "config": {
                    "arms": [{"provider": "openai", "model": "text-embedding-3-small"}],
                    "media_ids": [],
                    "retrieval": {"k": 3, "search_mode": "vector"},
                    "queries": [{"text": "hello"}],
                    "metric_level": "media",
                    "reuse_existing": True,
                },
            },
            headers={"X-API-KEY": "test"},
        )

    assert response.status_code == 200, response.text
    test_id = response.json()["test_id"]
    svc = get_unified_evaluation_service_for_user("tenant-user")
    row = svc.db.get_abtest(test_id, created_by="tenant-user")
    assert row is not None
    assert str(DatabasePaths.get_evaluations_db_path("tenant-user")).endswith("evaluations.db")
```

- [ ] **Step 2: Run the feature-surface regressions to verify they fail**

Run:
```bash
source .venv/bin/activate && python -m pytest -v \
  tldw_Server_API/tests/Evaluations/test_evaluations_benchmarks_api.py \
  tldw_Server_API/tests/Evaluations/integration/test_webhook_multi_user_api.py \
  tldw_Server_API/tests/Evaluations/test_embeddings_abtest_jobs_worker.py \
  tldw_Server_API/tests/Evaluations/integration/test_embeddings_abtest_multi_user_api.py
```

Expected: failures showing numeric coercion in benchmarks, webhook manager DB binding defaulting to the single-user DB for string ids, A/B creation storing ownership under the auth token or wrong DB, or the Jobs worker collapsing `"tenant-user"` to user `1`.

- [ ] **Step 3: Switch the remaining feature endpoints and worker to the canonical evaluations scope**

Update `tldw_Server_API/app/api/v1/endpoints/evaluations/evaluations_benchmarks.py`:

```python
from tldw_Server_API.app.core.Evaluations.identity import canonical_evaluations_user_scope


def _get_evaluation_manager_for_user(current_user: User) -> EvaluationManager:
    user_scope = canonical_evaluations_user_scope(current_user)
    return EvaluationManager(user_id=user_scope)
```

Update `tldw_Server_API/app/api/v1/endpoints/evaluations/evaluations_webhooks.py`:

```python
from tldw_Server_API.app.api.v1.endpoints.evaluations.evaluations_auth import get_evaluations_identity
from tldw_Server_API.app.core.Evaluations.identity import canonical_evaluations_user_scope


def _get_webhook_manager_for_user(user_id: str | int) -> WebhookManager:
    user_scope = canonical_evaluations_user_scope(user_id)
    service = get_unified_evaluation_service_for_user(user_scope)
    manager = getattr(service, "webhook_manager", None)
    if manager is None:
        service.webhook_manager = webhook_manager
        return webhook_manager
    return manager


identity = get_evaluations_identity(current_user)
wm = _get_webhook_manager_for_user(identity.user_scope)
records = wm.get_webhook_status(user_id=identity.webhook_user_id)
```

Update `tldw_Server_API/app/core/Evaluations/webhook_identity.py`:

```python
from tldw_Server_API.app.core.Evaluations.identity import canonical_evaluations_user_scope


def webhook_user_id_from_user(user: User, *, fallback: str = "1") -> str:
    raw = canonical_evaluations_user_scope(user, fallback=fallback)
    if raw.startswith("user_"):
        return raw
    return f"user_{raw}"
```

Update `tldw_Server_API/app/api/v1/endpoints/evaluations/evaluations_embeddings_abtest.py`:

```python
from tldw_Server_API.app.api.v1.endpoints.evaluations.evaluations_auth import get_evaluations_identity


identity = get_evaluations_identity(current_user)
svc = get_unified_evaluation_service_for_user(identity.user_scope)
db = svc.db
if idempotency_key:
    existing_id = db.lookup_idempotency("emb_abtest", idempotency_key, identity.created_by)
test_id = db.create_abtest(
    name=payload.name,
    config=cfg.model_dump(),
    created_by=identity.created_by,
)
log_evaluation_created(
    user_id=identity.created_by,
    eval_id=test_id,
    name=payload.name,
    eval_type="embeddings_abtest",
)
job_payload = {
    "test_id": test_id,
    "config": cfg.model_dump(),
    "user_id": identity.user_scope,
}
job_row = jm.create_job(
    domain=ABTEST_JOBS_DOMAIN,
    queue=abtest_jobs_queue(),
    job_type=ABTEST_JOBS_JOB_TYPE,
    payload=job_payload,
    owner_user_id=identity.user_scope,
    priority=5,
    max_retries=3,
    idempotency_key=abtest_jobs_idempotency_key(test_id, idempotency_key),
)
```

Update `tldw_Server_API/app/core/Evaluations/embeddings_abtest_jobs_worker.py`:

```python
from tldw_Server_API.app.core.Evaluations.identity import canonical_evaluations_user_scope


def _normalize_user_id(value: Any) -> str:
    return canonical_evaluations_user_scope(
        value,
        fallback=DatabasePaths.get_single_user_id(),
    )


owner = job.get("owner_user_id") or payload.get("user_id")
user_scope = _normalize_user_id(owner)
job_logger = logger.bind(
    test_id=str(test_id),
    job_id=str(job_id) if job_id is not None else None,
    user_id=user_scope,
)
svc = get_unified_evaluation_service_for_user(user_scope)
media_db = _build_media_db(user_scope)
await run_abtest_full(svc.db, config, str(test_id), user_scope, media_db)
```

- [ ] **Step 4: Re-run the feature-surface tests**

Run:
```bash
source .venv/bin/activate && python -m pytest -v \
  tldw_Server_API/tests/Evaluations/test_evaluations_benchmarks_api.py \
  tldw_Server_API/tests/Evaluations/integration/test_webhook_multi_user_api.py \
  tldw_Server_API/tests/Evaluations/test_embeddings_abtest_jobs_worker.py \
  tldw_Server_API/tests/Evaluations/integration/test_embeddings_abtest_multi_user_api.py
```

Expected: PASS. Benchmarks, webhooks, A/B creation, and the A/B Jobs worker should all preserve `"tenant-user"` instead of falling back to user `1`.

- [ ] **Step 5: Commit the feature-surface identity fixes**

Run:
```bash
git add \
  tldw_Server_API/app/api/v1/endpoints/evaluations/evaluations_benchmarks.py \
  tldw_Server_API/app/api/v1/endpoints/evaluations/evaluations_webhooks.py \
  tldw_Server_API/app/api/v1/endpoints/evaluations/evaluations_embeddings_abtest.py \
  tldw_Server_API/app/core/Evaluations/embeddings_abtest_jobs_worker.py \
  tldw_Server_API/app/core/Evaluations/webhook_identity.py \
  tldw_Server_API/tests/Evaluations/test_evaluations_benchmarks_api.py \
  tldw_Server_API/tests/Evaluations/integration/test_webhook_multi_user_api.py \
  tldw_Server_API/tests/Evaluations/test_embeddings_abtest_jobs_worker.py \
  tldw_Server_API/tests/Evaluations/integration/test_embeddings_abtest_multi_user_api.py
git commit -m "fix: preserve eval identity in feature surfaces"
```

Expected: one commit capturing the remaining benchmark, webhook, A/B, and worker identity fixes.

## Task 4: Add a Route-Binding Identity Matrix, Document the Contract, and Run Final Verification

**Files:**
- Create: `tldw_Server_API/tests/Evaluations/unit/test_evaluations_identity_route_bindings.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/evaluations/evaluations_crud.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/evaluations/evaluations_rag_pipeline.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/evaluations/evaluations_unified.py`
- Modify: `tldw_Server_API/app/core/Evaluations/README.md`
- Modify: `tldw_Server_API/app/core/Evaluations/SECURITY.md`
- Test: `tldw_Server_API/tests/Evaluations/unit/test_evaluations_identity_route_bindings.py`

- [ ] **Step 1: Add a focused route-binding matrix for the remaining CRUD and RAG pipeline call sites**

Create `tldw_Server_API/tests/Evaluations/unit/test_evaluations_identity_route_bindings.py`:

```python
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tldw_Server_API.app.api.v1.endpoints.evaluations import evaluations_crud as eval_crud
from tldw_Server_API.app.api.v1.endpoints.evaluations import evaluations_rag_pipeline as eval_pipeline
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User


def test_create_run_uses_stable_string_user_id_for_service_and_webhook(monkeypatch):
    app = FastAPI()
    app.include_router(eval_crud.crud_router, prefix="/api/v1/evaluations")

    captured = {}

    class _Svc:
        class db:
            @staticmethod
            def lookup_idempotency(*_args, **_kwargs):
                return None

            @staticmethod
            def record_idempotency(*_args, **_kwargs):
                return None

        async def create_run(self, eval_id, target_model, config=None, dataset_override=None, webhook_url=None, created_by=None, webhook_user_id=None):
            captured["created_by"] = created_by
            captured["webhook_user_id"] = webhook_user_id
            return {
                "id": "run_tenant",
                "object": "run",
                "eval_id": eval_id,
                "status": "pending",
                "target_model": target_model,
                "created": 1700000000,
            }

    async def _verify_api_key_override():
        return "super-secret-api-key"

    async def _get_user_override():
        return User(
            id="tenant-user",
            username="tenant-user",
            email=None,
            is_active=True,
            roles=["admin"],
            permissions=["evals.read", "evals.manage"],
        )

    def _service_factory(user_id):
        captured["service_user_id"] = user_id
        return _Svc()

    monkeypatch.setattr(eval_crud, "get_unified_evaluation_service_for_user", _service_factory)
    app.dependency_overrides[eval_crud.verify_api_key] = _verify_api_key_override
    app.dependency_overrides[eval_crud.get_eval_request_user] = _get_user_override

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/evaluations/eval_123/runs",
            json={"target_model": "gpt-4o-mini", "config": {"max_workers": 1}},
        )

    assert response.status_code == 202, response.text
    assert captured["service_user_id"] == "tenant-user"
    assert captured["created_by"] == "tenant-user"
    assert captured["webhook_user_id"] == "user_tenant-user"


def test_rag_pipeline_preset_routes_use_stable_string_user_id(monkeypatch):
    app = FastAPI()
    app.include_router(eval_pipeline.pipeline_router, prefix="/api/v1/evaluations")

    captured = {"list_user_ids": []}

    class _DB:
        def upsert_pipeline_preset(self, name, config, user_id):
            captured["upsert_user_id"] = user_id

        def get_pipeline_preset(self, name, user_id):
            captured["get_user_id"] = user_id
            return {"name": name, "config": {"retrieval": {"k": 3}}, "created_at": None, "updated_at": None}

        def list_pipeline_presets(self, limit, offset, user_id):
            captured["list_user_ids"].append(user_id)
            return ([{"name": "tenant-preset", "config": {"retrieval": {"k": 3}}, "created_at": None, "updated_at": None}], 1)

    class _Svc:
        def __init__(self):
            self.db = _DB()

    async def _get_user_override():
        return User(id="tenant-user", username="tenant-user", is_active=True)

    def _service_factory(user_id):
        captured["service_user_id"] = user_id
        return _Svc()

    monkeypatch.setattr(eval_pipeline, "get_unified_evaluation_service_for_user", _service_factory)
    app.dependency_overrides[eval_pipeline.get_eval_request_user] = _get_user_override

    with TestClient(app) as client:
        create_response = client.post(
            "/api/v1/evaluations/rag/pipeline/presets",
            json={"name": "tenant-preset", "config": {"retrieval": {"k": 3}}},
        )
        list_response = client.get("/api/v1/evaluations/rag/pipeline/presets")

    assert create_response.status_code == 200, create_response.text
    assert list_response.status_code == 200, list_response.text
    assert captured["service_user_id"] == "tenant-user"
    assert captured["upsert_user_id"] == "tenant-user"
    assert captured["get_user_id"] == "tenant-user"
    assert captured["list_user_ids"] == ["tenant-user"]
```

- [ ] **Step 2: Run the route-binding matrix and confirm the remaining gaps fail**

Run:
```bash
source .venv/bin/activate && python -m pytest -v tldw_Server_API/tests/Evaluations/unit/test_evaluations_identity_route_bindings.py
```

Expected: failures showing at least one of the remaining route modules still binds services on `current_user.id` or records webhook ownership/DB filters with the wrong identity shape.

- [ ] **Step 3: Finish the remaining route bindings and document the identity contract**

Update the remaining CRUD, RAG pipeline, and unified A/B call sites so they all use the route-facing identity wrapper:

```python
identity = get_evaluations_identity(current_user)
svc = get_unified_evaluation_service_for_user(identity.user_scope)
```

Specifically, ensure these three patterns are gone:

```python
get_unified_evaluation_service_for_user(current_user.id)
svc.db.get_abtest(test_id, created_by=user_ctx)
cleanup_abtest_resources(svc.db, str(current_user.id), test_id, created_by=user_ctx)
```

Replace them with:

```python
identity = get_evaluations_identity(current_user)
svc = get_unified_evaluation_service_for_user(identity.user_scope)
row = svc.db.get_abtest(test_id, created_by=identity.created_by)
cleanup_abtest_resources(
    svc.db,
    identity.user_scope,
    test_id,
    delete_db=True,
    delete_idempotency=True,
    created_by=identity.created_by,
)
```

Add this section to `tldw_Server_API/app/core/Evaluations/README.md`:

```markdown
## Identity Contract

- `verify_api_key()` authenticates requests but does not define storage ownership.
- Evaluations ownership comes from `current_user.id_str` or `str(current_user.id)`.
- The same canonical scope must be used for per-user DB binding, rate-limit accounting, `created_by`, idempotency ownership, and Jobs `owner_user_id`.
- Webhook owner ids are derived from that scope as `user_<scope>`.
```

Add this section to `tldw_Server_API/app/core/Evaluations/SECURITY.md`:

```markdown
## Identity Handling

- Never reuse raw API keys, bearer tokens, or `verify_api_key()` return values as `created_by`, per-user DB selectors, or rate-limit subjects.
- Use the canonical evaluations scope derived from the authenticated request user for all ownership and isolation checks.
- Jobs and webhooks must carry the canonical user scope, not an auth token surrogate.
```

- [ ] **Step 4: Run the final verification pack and Bandit on the touched scope**

Run:
```bash
source .venv/bin/activate && python -m pytest -v \
  tldw_Server_API/tests/Evaluations/unit/test_evals_identity.py \
  tldw_Server_API/tests/AuthNZ/unit/test_evaluations_auth_runtime_guards.py \
  tldw_Server_API/tests/Evaluations/test_evaluations_stage1_route_and_error_regressions.py \
  tldw_Server_API/tests/Evaluations/test_evaluations_stage4_auth_policy_and_dataset_permissions.py \
  tldw_Server_API/tests/Evaluations/test_evaluations_benchmarks_api.py \
  tldw_Server_API/tests/Evaluations/test_embeddings_abtest_jobs_worker.py \
  tldw_Server_API/tests/Evaluations/integration/test_webhook_multi_user_api.py \
  tldw_Server_API/tests/Evaluations/integration/test_embeddings_abtest_multi_user_api.py \
  tldw_Server_API/tests/Evaluations/unit/test_evaluations_identity_route_bindings.py
```

Expected: PASS for the full priority-1 regression pack.

Run:
```bash
source .venv/bin/activate && python -m bandit -r \
  tldw_Server_API/app/api/v1/endpoints/evaluations \
  tldw_Server_API/app/core/Evaluations \
  -f json -o /tmp/bandit_evals_identity.json
```

Expected: Bandit completes successfully and any new findings in the touched identity paths are fixed before the work is considered done.

- [ ] **Step 5: Commit the final route-matrix, docs, and verification-safe identity contract**

Run:
```bash
git add \
  tldw_Server_API/app/api/v1/endpoints/evaluations/evaluations_crud.py \
  tldw_Server_API/app/api/v1/endpoints/evaluations/evaluations_rag_pipeline.py \
  tldw_Server_API/app/api/v1/endpoints/evaluations/evaluations_unified.py \
  tldw_Server_API/app/core/Evaluations/README.md \
  tldw_Server_API/app/core/Evaluations/SECURITY.md \
  tldw_Server_API/tests/Evaluations/unit/test_evaluations_identity_route_bindings.py
git commit -m "docs: record evals identity contract"
```

Expected: one final commit capturing the remaining route bindings, docs, and final verification-ready state for remediation priority 1.
