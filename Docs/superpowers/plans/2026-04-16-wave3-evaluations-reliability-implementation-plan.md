# Wave 3 Evaluations Reliability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stabilize evaluations identity, run lifecycle, and batch/webhook contract behavior so multi-user ownership, cancellation, and terminal-state reporting stay consistent across the unified API, Jobs workers, and persistence layer.

**Architecture:** Execute Wave 3 in three coherent code batches. First add one canonical evaluations identity layer and move the core per-user factories to stable string scopes instead of route-local `int(...)` coercion. Then migrate the affected routes, Jobs workers, and ownership filters to that identity. Finally introduce one shared lifecycle/status helper so cancellation, idempotent replay, A/B terminal-state reporting, and batch strict-fail-fast behavior all depend on the same persisted contract.

**Tech Stack:** Python 3, FastAPI, asyncio, Pydantic, pytest, SQLite/PostgreSQL-aware DB path helpers, Loguru, Bandit, Markdown

---

## Current-Tree Evidence

- `2026-04-16` focused Wave 3 smoke slice:
  - Run: `source .venv/bin/activate && python -m pytest -v tldw_Server_API/tests/AuthNZ/unit/test_evaluations_auth_runtime_guards.py tldw_Server_API/tests/Evaluations/test_evaluations_stage1_route_and_error_regressions.py tldw_Server_API/tests/Evaluations/test_evaluations_stage3_batch_failfast_and_metrics_none.py tldw_Server_API/tests/Evaluations/test_evaluations_benchmarks_api.py tldw_Server_API/tests/Evaluations/test_embeddings_abtest_jobs_worker.py`
  - Result: `21 passed, 1 failed, 33 warnings in 17.74s`
  - Live failure:
    - `tldw_Server_API/tests/Evaluations/test_evaluations_stage3_batch_failfast_and_metrics_none.py::test_batch_parallel_strict_fail_fast_cancels_remaining`
    - Current error: `TypeError ... evaluate_geval() got an unexpected keyword argument 'webhook_user_id'`
- Live source seams confirmed before writing this plan:
  - `tldw_Server_API/app/core/Evaluations/unified_evaluation_service.py` still keeps `_service_instances_by_user` keyed by `int` and `get_unified_evaluation_service_for_user(user_id: int)` still coerces non-numeric ids to the single-user DB.
  - `tldw_Server_API/app/core/Evaluations/evaluation_manager.py` still coerces `user_id` to `int`, which breaks tenant-style string scopes.
  - `tldw_Server_API/app/core/Evaluations/evaluation_manager.py` also lets explicit `db_path` arguments bypass the containment rules applied to config-driven evaluation DB paths, and `tldw_Server_API/app/core/DB_Management/db_path_utils.py` currently compares trusted temp roots and candidate paths in different canonical forms during test-mode absolute-path validation.
  - `tldw_Server_API/app/core/Evaluations/embeddings_abtest_jobs_worker.py` still normalizes owner ids to `(str, int)` and binds services through the `int` value.
  - `tldw_Server_API/app/api/v1/schemas/evaluation_schemas_unified.py` uses `cancelled`, while `tldw_Server_API/app/api/v1/schemas/embeddings_abtest_schemas.py` and A/B SSE code still use `canceled`.
  - Several evaluations routes still bind services with `current_user.id` while separately using string `created_by` filters, which means ownership and DB binding are not driven by one canonical subject.

## Implementation File Map

**Create:**
- `tldw_Server_API/app/core/Evaluations/identity.py`: Canonical evaluations identity helpers and `EvaluationIdentity` dataclass.
- `tldw_Server_API/app/core/Evaluations/run_state.py`: Shared status normalization and lifecycle transition guards for runs and A/B status reporting.
- `tldw_Server_API/tests/Evaluations/unit/test_evaluations_identity.py`: Unit coverage for canonical user scope derivation, service cache keys, limiter keys, benchmark manager binding, and webhook owner derivation.
- `tldw_Server_API/tests/Evaluations/unit/test_eval_run_state.py`: Unit coverage for status normalization and cancellation transition guards.
- `tldw_Server_API/tests/Evaluations/unit/test_evaluations_identity_route_bindings.py`: Focused route-binding regressions for CRUD, datasets, webhooks, and A/B routes that must use canonical string scopes.
- `tldw_Server_API/tests/Evaluations/integration/test_embeddings_abtest_multi_user_api.py`: Integration coverage for tenant-style user ids across A/B create/run/read/delete flows.

**Modify:**
- `tldw_Server_API/app/api/v1/endpoints/evaluations/evaluations_auth.py`: Route-facing helper that exposes one canonical evaluations identity from `current_user`.
- `tldw_Server_API/app/api/v1/endpoints/evaluations/evaluations_crud.py`: Switch service binding, idempotency, `created_by`, and webhook ownership to canonical identity.
- `tldw_Server_API/app/api/v1/endpoints/evaluations/evaluations_datasets.py`: Switch service binding and idempotency ownership to canonical identity.
- `tldw_Server_API/app/api/v1/endpoints/evaluations/evaluations_benchmarks.py`: Stop numeric-only `EvaluationManager` binding and preserve canonical string user scopes.
- `tldw_Server_API/app/api/v1/endpoints/evaluations/evaluations_unified.py`: Use canonical identity in batch, propositions, OCR, A/B SSE/delete/export paths; normalize terminal-state handling; fix the strict fail-fast route/regression drift around explicit `webhook_user_id` forwarding.
- `tldw_Server_API/app/api/v1/endpoints/evaluations/evaluations_webhooks.py`: Bind webhook manager ownership through canonical identity.
- `tldw_Server_API/app/api/v1/endpoints/evaluations/evaluations_embeddings_abtest.py`: Use canonical identity for idempotency, ownership, and replay status payloads.
- `tldw_Server_API/app/core/Evaluations/unified_evaluation_service.py`: Use canonical string service-cache keys and shared lifecycle/status helpers.
- `tldw_Server_API/app/core/Evaluations/user_rate_limiter.py`: Use canonical string limiter keys instead of token- or int-shaped drift.
- `tldw_Server_API/app/core/Evaluations/evaluation_manager.py`: Accept `user_id: int | str | None`, preserve string user scopes when resolving per-user DBs, and prevent explicit `db_path` values from bypassing trusted-path validation.
- `tldw_Server_API/app/core/DB_Management/db_path_utils.py`: Canonicalize absolute trusted-path checks so symlinked temp aliases and trusted roots are compared symmetrically.
- `tldw_Server_API/app/core/Evaluations/webhook_identity.py`: Delegate to the canonical evaluations identity layer instead of maintaining a separate partial normalization rule.
- `tldw_Server_API/app/core/Evaluations/eval_runner.py`: Guard cancellation transitions with the shared lifecycle helper.
- `tldw_Server_API/app/core/Evaluations/embeddings_abtest_jobs_worker.py`: Preserve canonical string owner ids for service binding and media DB access.
- `tldw_Server_API/app/api/v1/schemas/evaluation_schemas_unified.py`: Keep the canonical run status vocabulary.
- `tldw_Server_API/app/api/v1/schemas/embeddings_abtest_schemas.py`: Align A/B API status vocabulary with canonical run status output.
- `tldw_Server_API/tests/AuthNZ/unit/test_evaluations_auth_runtime_guards.py`: Extend auth helper coverage to the canonical identity wrapper.
- `tldw_Server_API/tests/DB_Management/test_db_path_utils.py`: Cover trusted-path canonicalization for symlinked temp aliases.
- `tldw_Server_API/tests/Evaluations/unit/test_evaluation_manager.py`: Cover explicit-path fallback for evaluation manager trusted-path enforcement.
- `tldw_Server_API/tests/Evaluations/test_evaluations_stage1_route_and_error_regressions.py`: Extend route regressions to assert canonical service-binding and stable limiter/user subjects.
- `tldw_Server_API/tests/Evaluations/test_evaluations_stage3_batch_failfast_and_metrics_none.py`: Keep the current live `webhook_user_id` batch contract failure as a regression until fixed.
- `tldw_Server_API/tests/Evaluations/test_evaluations_benchmarks_api.py`: Add tenant-style string-id coverage.
- `tldw_Server_API/tests/Evaluations/test_evaluations_stage4_auth_policy_and_dataset_permissions.py`: Add dataset ownership/service-binding coverage for tenant-style string ids.
- `tldw_Server_API/tests/Evaluations/integration/test_webhook_multi_user_api.py`: Extend webhook multi-user coverage to canonical string scopes.
- `tldw_Server_API/tests/Evaluations/test_embeddings_abtest_jobs_worker.py`: Add owner-user string preservation coverage.
- `tldw_Server_API/tests/Evaluations/test_abtest_events_sse_stream.py`: Assert canonical terminal-state handling for A/B SSE.
- `tldw_Server_API/tests/Evaluations/test_embeddings_abtest_run_api.py`: Assert replay returns stored normalized status rather than a route-local placeholder.
- `tldw_Server_API/app/core/Evaluations/README.md`: Document the canonical evaluations identity and lifecycle contract.
- `tldw_Server_API/app/core/Evaluations/SECURITY.md`: Document that auth tokens must never become ownership or limiter subjects.

## Notes

- Wave 3 is intentionally limited to the auth/identity, lifecycle/state, and route/job contract seams named in the review and confirmed above. Because Task 1 already touches `evaluation_manager.py` and per-user DB binding, the explicit-manager-path escape and the coupled `db_path_utils.py` canonicalization fix needed to keep manager verification usable are in scope for this wave. Do not widen beyond those directly coupled storage-path fixes.
- `verify_api_key()` may keep returning an auth-context token for compatibility, but route/business logic must stop using that return value as a storage, limiter, or ownership subject.
- The canonical evaluations identity helper must fail closed. If no stable scope is present, raise `ValueError` unless the caller passes an explicit fallback for single-user compatibility.
- `RunStatus.CANCELLED` is the canonical vocabulary for unified APIs. If legacy rows or A/B internals still emit `canceled`, normalize them on the read path first, then remove the drift at the schema and route layer.
- The current batch strict-fail-fast failure is route/regression-scaffold drift around `webhook_user_id`, not a service-signature redesign problem. Keep it as a hard regression and fix it by aligning the route-facing contract and its strict doubles with the real service signature.

### Task 1: Introduce Canonical Evaluations Identity and Core Factory / Path Support

**Files:**
- Create: `tldw_Server_API/app/core/Evaluations/identity.py`
- Create: `tldw_Server_API/tests/Evaluations/unit/test_evaluations_identity.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/evaluations/evaluations_auth.py`
- Modify: `tldw_Server_API/app/core/DB_Management/db_path_utils.py`
- Modify: `tldw_Server_API/app/core/Evaluations/unified_evaluation_service.py`
- Modify: `tldw_Server_API/app/core/Evaluations/user_rate_limiter.py`
- Modify: `tldw_Server_API/app/core/Evaluations/evaluation_manager.py`
- Modify: `tldw_Server_API/app/core/Evaluations/webhook_identity.py`
- Modify: `tldw_Server_API/tests/AuthNZ/unit/test_evaluations_auth_runtime_guards.py`
- Modify: `tldw_Server_API/tests/DB_Management/test_db_path_utils.py`
- Modify: `tldw_Server_API/tests/Evaluations/unit/test_evaluation_manager.py`

- [x] **Step 1: Write the failing identity and manager-path unit tests**

```python
def test_canonical_evaluations_user_scope_preserves_tenant_string():
    user = User(id="tenant-user", username="x", email=None, is_active=True)
    identity = evaluations_identity_from_user(user)

    assert identity.user_scope == "tenant-user"
    assert identity.created_by == "tenant-user"
    assert identity.rate_limit_subject == "tenant-user"
    assert identity.webhook_user_id == "user_tenant-user"


def test_canonical_evaluations_user_scope_requires_explicit_fallback_when_missing():
    with pytest.raises(ValueError, match="Evaluations user scope is required"):
        canonical_evaluations_user_scope(User(id="", username="x", email=None, is_active=True))


def test_service_cache_uses_canonical_string_scope(monkeypatch):
    created = {}

    class _DummyService:
        def __init__(self, db_path: str, **_kwargs):
            created["db_path"] = db_path

    monkeypatch.setattr(service_module, "UnifiedEvaluationService", _DummyService)
    service_module._service_instances_by_user.clear()

    service = service_module.get_unified_evaluation_service_for_user("tenant-user")

    assert isinstance(service, _DummyService)
    assert created["db_path"] == str(DatabasePaths.get_evaluations_db_path("tenant-user"))
    assert "tenant-user" in service_module._service_instances_by_user


def test_evaluation_manager_preserves_string_user_scope(monkeypatch):
    monkeypatch.setattr(EvaluationManager, "_init_database", lambda self: None, raising=False)
    manager = EvaluationManager(user_id="tenant-user")
    assert manager.db_path == DatabasePaths.get_evaluations_db_path("tenant-user")


def test_explicit_db_path_outside_trusted_roots_falls_back_to_default(monkeypatch):
    monkeypatch.setattr(EvaluationManager, "_init_database", lambda self: None, raising=False)
    manager = EvaluationManager(db_path="/etc/escape-evals.db", user_id="tenant-user")
    assert manager.db_path == DatabasePaths.get_evaluations_db_path("tenant-user")
```

- [x] **Step 2: Run the identity slice to verify the current code fails**

Run:
```bash
source .venv/bin/activate && python -m pytest -v tldw_Server_API/tests/Evaluations/unit/test_evaluations_identity.py tldw_Server_API/tests/AuthNZ/unit/test_evaluations_auth_runtime_guards.py tldw_Server_API/tests/Evaluations/unit/test_evaluation_manager.py::TestEvaluationManagerInit::test_explicit_db_path_outside_trusted_roots_falls_back_to_default tldw_Server_API/tests/DB_Management/test_db_path_utils.py::test_resolve_trusted_database_path_accepts_symlink_alias_to_temp_root
```

Expected: FAIL because one or more current behaviors still coerce tenant-style ids to numeric/single-user state, fail open without an explicit fallback, or allow manager/trusted-path drift to escape the intended storage roots.

- [x] **Step 3: Implement the canonical identity helper and move the core factories to stable string scopes**

Create `tldw_Server_API/app/core/Evaluations/identity.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class EvaluationIdentity:
    user_scope: str
    created_by: str
    rate_limit_subject: str
    webhook_user_id: str


def canonical_evaluations_user_scope(user_or_id: Any, *, fallback: str | int | None = None) -> str:
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
        raise ValueError("Evaluations user scope is required")
    return raw.strip()


def evaluations_identity_from_user(user: Any, *, fallback: str | int | None = None) -> EvaluationIdentity:
    user_scope = canonical_evaluations_user_scope(user, fallback=fallback)
    return EvaluationIdentity(
        user_scope=user_scope,
        created_by=user_scope,
        rate_limit_subject=user_scope,
        webhook_user_id=f"user_{user_scope}" if not user_scope.startswith("user_") else user_scope,
    )
```

Update the core factories:

```python
def get_unified_evaluation_service_for_user(user_id: str | int) -> UnifiedEvaluationService:
    uid_key = canonical_evaluations_user_scope(user_id)
    ...
    db_path = str(DatabasePaths.get_evaluations_db_path(uid_key))
```

```python
def get_user_rate_limiter_for_user(user_id: str | int) -> UserRateLimiter:
    uid_key = canonical_evaluations_user_scope(user_id)
    ...
```

```python
class EvaluationManager:
    def __init__(self, db_path: str | Path | None = None, *, user_id: int | str | None = None):
        self._user_id = canonical_evaluations_user_scope(
            user_id,
            fallback=DatabasePaths.get_single_user_id(),
        )
        ...
```

Harden the manager/db-path seam while touching Task 1:

```python
candidate_path = resolve_trusted_database_path(
    candidate_str,
    label="evaluations_db_path",
    extra_roots=[base_resolved],
).resolve()
```

And canonicalize trusted absolute-path comparisons in `db_path_utils.py` so test-mode temp aliases and trusted roots are compared in the same resolved form.

Route-facing auth helper in `evaluations_auth.py`:

```python
def get_evaluation_identity(current_user: User = Depends(get_eval_request_user)) -> EvaluationIdentity:
    return evaluations_identity_from_user(current_user)
```

- [x] **Step 4: Re-run the identity slice**

Run:
```bash
source .venv/bin/activate && python -m pytest -v tldw_Server_API/tests/Evaluations/unit/test_evaluations_identity.py tldw_Server_API/tests/AuthNZ/unit/test_evaluations_auth_runtime_guards.py tldw_Server_API/tests/Evaluations/unit/test_evaluation_manager.py::TestEvaluationManagerInit::test_explicit_db_path_outside_trusted_roots_falls_back_to_default tldw_Server_API/tests/DB_Management/test_db_path_utils.py::test_resolve_trusted_database_path_accepts_symlink_alias_to_temp_root
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tldw_Server_API/app/core/Evaluations/identity.py tldw_Server_API/app/api/v1/endpoints/evaluations/evaluations_auth.py tldw_Server_API/app/core/DB_Management/db_path_utils.py tldw_Server_API/app/core/Evaluations/unified_evaluation_service.py tldw_Server_API/app/core/Evaluations/user_rate_limiter.py tldw_Server_API/app/core/Evaluations/evaluation_manager.py tldw_Server_API/app/core/Evaluations/webhook_identity.py tldw_Server_API/tests/Evaluations/unit/test_evaluations_identity.py tldw_Server_API/tests/AuthNZ/unit/test_evaluations_auth_runtime_guards.py tldw_Server_API/tests/DB_Management/test_db_path_utils.py tldw_Server_API/tests/Evaluations/unit/test_evaluation_manager.py
git commit -m "fix: canonicalize evaluations identity"
```

### Task 2: Migrate Route, Benchmark, Webhook, and Jobs Bindings to Canonical Identity

**Files:**
- Create: `tldw_Server_API/tests/Evaluations/unit/test_evaluations_identity_route_bindings.py`
- Create: `tldw_Server_API/tests/Evaluations/integration/test_embeddings_abtest_multi_user_api.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/evaluations/evaluations_crud.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/evaluations/evaluations_datasets.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/evaluations/evaluations_benchmarks.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/evaluations/evaluations_webhooks.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/evaluations/evaluations_embeddings_abtest.py`
- Modify: `tldw_Server_API/app/core/Evaluations/embeddings_abtest_jobs_worker.py`
- Modify: `tldw_Server_API/tests/Evaluations/test_evaluations_stage1_route_and_error_regressions.py`
- Modify: `tldw_Server_API/tests/Evaluations/test_evaluations_benchmarks_api.py`
- Modify: `tldw_Server_API/tests/Evaluations/test_evaluations_stage4_auth_policy_and_dataset_permissions.py`
- Modify: `tldw_Server_API/tests/Evaluations/integration/test_webhook_multi_user_api.py`
- Modify: `tldw_Server_API/tests/Evaluations/test_embeddings_abtest_jobs_worker.py`
- Modify: `tldw_Server_API/app/core/Evaluations/README.md`
- Modify: `tldw_Server_API/app/core/Evaluations/SECURITY.md`

- [x] **Step 1: Add failing route and Jobs regressions for canonical string scope binding**

```python
def test_benchmark_route_uses_canonical_string_scope(monkeypatch):
    captured = {}

    class _Manager:
        def __init__(self, *, user_id=None, **_kwargs):
            captured["user_id"] = user_id

    monkeypatch.setattr(benchmarks_module, "EvaluationManager", _Manager)
    user = User(id="tenant-user", username="x", email=None, is_active=True)

    benchmarks_module._get_evaluation_manager_for_user(user)

    assert captured["user_id"] == "tenant-user"


@pytest.mark.asyncio
async def test_embeddings_abtest_job_preserves_string_owner_scope(monkeypatch):
    seen = {}

    class _Svc:
        def __init__(self):
            self.db = object()

    monkeypatch.setattr(worker, "get_unified_evaluation_service_for_user", lambda uid: seen.setdefault("service_user", uid) or _Svc())
    ...
    await worker.handle_abtest_job({"owner_user_id": "tenant-user", ...})
    assert seen["service_user"] == "tenant-user"
```

Follow-on route coverage added after the first pass:
- CRUD create-run must bind the unified service, idempotency ownership, and webhook owner through `EvaluationIdentity`.
- Webhook registration must bind the per-user webhook manager through `EvaluationIdentity.user_scope`.

- [x] **Step 2: Run the route/job identity slice to verify the current code fails**

Run:
```bash
source .venv/bin/activate && python -m pytest -v tldw_Server_API/tests/Evaluations/unit/test_evaluations_identity_route_bindings.py tldw_Server_API/tests/Evaluations/test_evaluations_stage1_route_and_error_regressions.py tldw_Server_API/tests/Evaluations/test_evaluations_benchmarks_api.py tldw_Server_API/tests/Evaluations/test_evaluations_stage4_auth_policy_and_dataset_permissions.py tldw_Server_API/tests/Evaluations/integration/test_webhook_multi_user_api.py tldw_Server_API/tests/Evaluations/test_embeddings_abtest_jobs_worker.py tldw_Server_API/tests/Evaluations/integration/test_embeddings_abtest_multi_user_api.py
```

Expected: FAIL because one or more routes or Jobs paths still bind services through `current_user.id`, raw `user_ctx`, or numeric-only owner coercion.

Observed:
- The initial Task 2 route/job identity slice failed on numeric-only binding drift before the canonical-identity rollout was completed.
- The follow-on CRUD/webhook red slice failed with route-local numeric binding still in place:
  - `service_user == 7` instead of `"tenant-scope"` in the create-run route binding regression.
  - `manager_scope == 7` instead of `"tenant-scope"` in the webhook registration route binding regression.

- [x] **Step 3: Switch the affected routes and Jobs workers to canonical identity**

Apply the pattern below everywhere in this task:

```python
identity = get_evaluation_identity(current_user)
svc = get_unified_evaluation_service_for_user(identity.user_scope)

run = await svc.create_run(
    ...,
    created_by=identity.created_by,
    webhook_user_id=identity.webhook_user_id,
)

existing_id = svc.db.lookup_idempotency("run", idempotency_key, identity.created_by)
```

Update the A/B worker:

```python
owner_scope = canonical_evaluations_user_scope(job.get("owner_user_id") or payload.get("user_id"))
svc = get_unified_evaluation_service_for_user(owner_scope)
media_db = _build_media_db(owner_scope)
```

Update benchmarks:

```python
def _get_evaluation_manager_for_user(current_user: User) -> EvaluationManager:
    identity = evaluations_identity_from_user(current_user)
    return EvaluationManager(user_id=identity.user_scope)
```

Keep docs aligned:
- `README.md`: describe canonical user-scope routing.
- `SECURITY.md`: explicitly forbid raw auth tokens as limiter or ownership subjects.

- [x] **Step 4: Re-run the route/job identity slice**

Run:
```bash
source .venv/bin/activate && python -m pytest -v tldw_Server_API/tests/Evaluations/unit/test_evaluations_identity_route_bindings.py tldw_Server_API/tests/Evaluations/test_evaluations_stage1_route_and_error_regressions.py tldw_Server_API/tests/Evaluations/test_evaluations_benchmarks_api.py tldw_Server_API/tests/Evaluations/test_evaluations_stage4_auth_policy_and_dataset_permissions.py tldw_Server_API/tests/Evaluations/integration/test_webhook_multi_user_api.py tldw_Server_API/tests/Evaluations/test_embeddings_abtest_jobs_worker.py tldw_Server_API/tests/Evaluations/integration/test_embeddings_abtest_multi_user_api.py
```

Expected: PASS

Observed:
- Focused follow-on route binding verification passed:
  - `2 passed` for the new CRUD create-run and webhook manager regressions.
  - `11 passed` across the affected CRUD/webhook/route-binding surface.
- The later expanded Wave 3 reliability pack also stayed green with these route changes included:
  - `107 passed, 36 warnings in 40.56s`

- [ ] **Step 5: Commit**

```bash
git add tldw_Server_API/app/api/v1/endpoints/evaluations/evaluations_crud.py tldw_Server_API/app/api/v1/endpoints/evaluations/evaluations_datasets.py tldw_Server_API/app/api/v1/endpoints/evaluations/evaluations_benchmarks.py tldw_Server_API/app/api/v1/endpoints/evaluations/evaluations_webhooks.py tldw_Server_API/app/api/v1/endpoints/evaluations/evaluations_embeddings_abtest.py tldw_Server_API/app/core/Evaluations/embeddings_abtest_jobs_worker.py tldw_Server_API/tests/Evaluations/unit/test_evaluations_identity_route_bindings.py tldw_Server_API/tests/Evaluations/integration/test_embeddings_abtest_multi_user_api.py tldw_Server_API/tests/Evaluations/test_evaluations_stage1_route_and_error_regressions.py tldw_Server_API/tests/Evaluations/test_evaluations_benchmarks_api.py tldw_Server_API/tests/Evaluations/test_evaluations_stage4_auth_policy_and_dataset_permissions.py tldw_Server_API/tests/Evaluations/integration/test_webhook_multi_user_api.py tldw_Server_API/tests/Evaluations/test_embeddings_abtest_jobs_worker.py tldw_Server_API/app/core/Evaluations/README.md tldw_Server_API/app/core/Evaluations/SECURITY.md
git commit -m "fix: align evaluations routes and jobs with canonical identity"
```

### Task 3: Introduce Shared Run-State Helpers and Eliminate Status / Replay Drift

**Files:**
- Create: `tldw_Server_API/app/core/Evaluations/run_state.py`
- Create: `tldw_Server_API/tests/Evaluations/unit/test_eval_run_state.py`
- Modify: `tldw_Server_API/app/core/Evaluations/unified_evaluation_service.py`
- Modify: `tldw_Server_API/app/core/Evaluations/eval_runner.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/evaluations/evaluations_unified.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/evaluations/evaluations_crud.py`
- Modify: `tldw_Server_API/app/api/v1/schemas/evaluation_schemas_unified.py`
- Modify: `tldw_Server_API/app/api/v1/schemas/embeddings_abtest_schemas.py`
- Modify: `tldw_Server_API/tests/Evaluations/unit/test_eval_runner.py`
- Modify: `tldw_Server_API/tests/Evaluations/test_evaluations_stage3_batch_failfast_and_metrics_none.py`
- Modify: `tldw_Server_API/tests/Evaluations/test_abtest_events_sse_stream.py`
- Modify: `tldw_Server_API/tests/Evaluations/test_embeddings_abtest_run_api.py`

- [x] **Step 1: Add failing lifecycle and status-vocabulary regressions**

```python
def test_cancellation_cannot_overwrite_completed_status():
    assert can_transition_run_status("completed", "cancelled") is False


def test_normalize_abtest_status_maps_canceled_to_cancelled():
    assert normalize_run_status("canceled") == "cancelled"


def test_batch_parallel_strict_fail_fast_cancels_remaining(...):
    ...
    assert response.status_code == 200
```

Add route-level regressions:
- A/B SSE stream should terminate on normalized terminal status.
- A/B run idempotent replay should return the stored normalized status, not a route-local hard-coded `"running"`.

- [x] **Step 2: Run the lifecycle slice to verify the current code fails**

Run:
```bash
source .venv/bin/activate && python -m pytest -v tldw_Server_API/tests/Evaluations/unit/test_eval_run_state.py tldw_Server_API/tests/Evaluations/test_evaluations_stage3_batch_failfast_and_metrics_none.py tldw_Server_API/tests/Evaluations/test_abtest_events_sse_stream.py tldw_Server_API/tests/Evaluations/test_embeddings_abtest_run_api.py
```

Expected: FAIL on at least the currently reproduced strict fail-fast test and one or more lifecycle/status normalization gaps.

- [x] **Step 3: Implement shared lifecycle/status helpers and use them everywhere**

Create `tldw_Server_API/app/core/Evaluations/run_state.py`:

```python
from __future__ import annotations

TERMINAL_RUN_STATUSES = frozenset({"completed", "failed", "cancelled"})
ACTIVE_RUN_STATUSES = frozenset({"pending", "running"})


def normalize_run_status(value: str | None) -> str:
    raw = str(value or "").strip().lower()
    if raw == "canceled":
        return "cancelled"
    return raw or "pending"


def can_transition_run_status(current: str | None, target: str | None) -> bool:
    current_norm = normalize_run_status(current)
    target_norm = normalize_run_status(target)
    if current_norm in TERMINAL_RUN_STATUSES and target_norm != current_norm:
        return False
    return True
```

Use the helper in service/runner cancellation:

```python
current = self.db.get_run(run_id, created_by=created_by)
current_status = normalize_run_status(current.get("status") if current else None)
if not can_transition_run_status(current_status, "cancelled"):
    return False
```

Normalize A/B outward status:

```python
status = normalize_run_status(row.get("status"))
if status in TERMINAL_RUN_STATUSES:
    await stream.done()
```

Close the batch route/regression drift explicitly:
- Keep `webhook_user_id` as an intentional service argument for the real evaluation service methods.
- Update the strict batch regression doubles and any local compatibility shims so the route-to-service signature is one explicit contract rather than a silent drift point.

- [x] **Step 4: Re-run the lifecycle slice**

Run:
```bash
source .venv/bin/activate && python -m pytest -v tldw_Server_API/tests/Evaluations/unit/test_eval_run_state.py tldw_Server_API/tests/Evaluations/test_evaluations_stage3_batch_failfast_and_metrics_none.py tldw_Server_API/tests/Evaluations/test_abtest_events_sse_stream.py tldw_Server_API/tests/Evaluations/test_embeddings_abtest_run_api.py
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tldw_Server_API/app/core/Evaluations/run_state.py tldw_Server_API/app/core/Evaluations/unified_evaluation_service.py tldw_Server_API/app/core/Evaluations/eval_runner.py tldw_Server_API/app/api/v1/endpoints/evaluations/evaluations_unified.py tldw_Server_API/app/api/v1/endpoints/evaluations/evaluations_crud.py tldw_Server_API/app/api/v1/schemas/evaluation_schemas_unified.py tldw_Server_API/app/api/v1/schemas/embeddings_abtest_schemas.py tldw_Server_API/tests/Evaluations/unit/test_eval_run_state.py tldw_Server_API/tests/Evaluations/test_evaluations_stage3_batch_failfast_and_metrics_none.py tldw_Server_API/tests/Evaluations/test_abtest_events_sse_stream.py tldw_Server_API/tests/Evaluations/test_embeddings_abtest_run_api.py
git commit -m "fix: stabilize evaluations lifecycle and status contracts"
```

### Task 4: Final Verification and Security Gate

**Files:**
- Verify only

- [x] **Step 1: Re-run the Wave 3 focused reliability pack**

Run:
```bash
source .venv/bin/activate && python -m pytest -v tldw_Server_API/tests/AuthNZ/unit/test_evaluations_auth_runtime_guards.py tldw_Server_API/tests/DB_Management/test_db_path_utils.py tldw_Server_API/tests/Evaluations/unit/test_evaluations_identity.py tldw_Server_API/tests/Evaluations/unit/test_evaluation_manager.py tldw_Server_API/tests/Evaluations/unit/test_eval_runner.py tldw_Server_API/tests/Evaluations/unit/test_unified_evaluation_service_mapping.py tldw_Server_API/tests/Evaluations/unit/test_eval_run_state.py tldw_Server_API/tests/Evaluations/unit/test_evaluations_identity_route_bindings.py tldw_Server_API/tests/Evaluations/test_evaluations_stage1_route_and_error_regressions.py tldw_Server_API/tests/Evaluations/test_evaluations_stage3_batch_failfast_and_metrics_none.py tldw_Server_API/tests/Evaluations/test_evaluations_benchmarks_api.py tldw_Server_API/tests/Evaluations/test_evaluations_stage4_auth_policy_and_dataset_permissions.py tldw_Server_API/tests/Evaluations/integration/test_webhook_multi_user_api.py tldw_Server_API/tests/Evaluations/test_embeddings_abtest_jobs_worker.py tldw_Server_API/tests/Evaluations/test_abtest_events_sse_stream.py tldw_Server_API/tests/Evaluations/test_embeddings_abtest_run_api.py
```

Expected: PASS

Observed:
- Expanded reliability verification after the CRUD/webhook canonical-identity cleanup:
  - `107 passed, 36 warnings in 40.56s`

- [x] **Step 2: Re-run the current-tree smoke slice that initially exposed the live failure**

Run:
```bash
source .venv/bin/activate && python -m pytest -v tldw_Server_API/tests/AuthNZ/unit/test_evaluations_auth_runtime_guards.py tldw_Server_API/tests/Evaluations/test_evaluations_stage1_route_and_error_regressions.py tldw_Server_API/tests/Evaluations/test_evaluations_stage3_batch_failfast_and_metrics_none.py tldw_Server_API/tests/Evaluations/test_evaluations_benchmarks_api.py tldw_Server_API/tests/Evaluations/test_embeddings_abtest_jobs_worker.py
```

Expected: PASS

- [x] **Step 3: Run Bandit on the touched Wave 3 production scope**

Run:
```bash
source .venv/bin/activate && python -m bandit -r tldw_Server_API/app/api/v1/endpoints/evaluations/evaluations_auth.py tldw_Server_API/app/api/v1/endpoints/evaluations/evaluations_crud.py tldw_Server_API/app/api/v1/endpoints/evaluations/evaluations_datasets.py tldw_Server_API/app/api/v1/endpoints/evaluations/evaluations_benchmarks.py tldw_Server_API/app/api/v1/endpoints/evaluations/evaluations_unified.py tldw_Server_API/app/api/v1/endpoints/evaluations/evaluations_webhooks.py tldw_Server_API/app/api/v1/endpoints/evaluations/evaluations_embeddings_abtest.py tldw_Server_API/app/core/DB_Management/db_path_utils.py tldw_Server_API/app/core/Evaluations/identity.py tldw_Server_API/app/core/Evaluations/run_state.py tldw_Server_API/app/core/Evaluations/unified_evaluation_service.py tldw_Server_API/app/core/Evaluations/user_rate_limiter.py tldw_Server_API/app/core/Evaluations/evaluation_manager.py tldw_Server_API/app/core/Evaluations/webhook_identity.py tldw_Server_API/app/core/Evaluations/eval_runner.py tldw_Server_API/app/core/Evaluations/embeddings_abtest_jobs_worker.py -f json -o /tmp/bandit_wave3_evals_reliability.json
```

Result: `/tmp/bandit_wave3_evals_reliability.json`

Expected: no new High-severity findings in touched production files

Observed:
- 3 existing low-severity findings in `tldw_Server_API/app/core/Evaluations/eval_runner.py`
- `B311` on deterministic `random.sample` usage in the RAG pipeline search strategy
- `B105` false-positive hardcoded-password heuristics on numeric zero defaults
- no High-severity findings in the touched Wave 3 production scope
- Re-ran after the CRUD/webhook canonical-identity cleanup; findings were unchanged.

- [ ] **Step 4: Commit any verification-only doc adjustments if needed, then hand off for branch completion**

```bash
git status --short
```

Expected: only the intended Wave 3 changes remain. After this step, use `superpowers:finishing-a-development-branch` before merge/push/PR decisions.
