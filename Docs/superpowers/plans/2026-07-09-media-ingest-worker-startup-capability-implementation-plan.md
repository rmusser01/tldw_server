# Media Ingest Worker Startup Capability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix media ingest jobs so the normal in-process worker starts by default when the `media` route is enabled, and make docs-info report that normal worker capability accurately.

**Architecture:** Reuse the existing `should_start_inprocess_worker()` startup policy instead of introducing another flag parser. Extend that helper with an optional injected route checker so declarative lifecycle specs can use their `WorkerLifecycleContext`, while existing global-config callers keep current behavior. Add one local lifecycle-spec predicate in the content jobs poller provider and wire it only to the normal and heavy media ingest worker specs. Update docs-info to compute `hasMediaIngestWorker` from the same normal-worker policy with sidecar awareness.

**Tech Stack:** FastAPI service code, pytest unit tests, Loguru-backed startup services, Backlog.md task `TASK-12100`.

---

## Source Spec

- `Docs/superpowers/specs/2026-07-09-media-ingest-worker-startup-capability-design.md`

## Files

- Modify: `tldw_Server_API/app/services/worker_startup_policy.py`
  - Add optional injected `route_enabled` support while preserving current global-config behavior.
- Modify: `tldw_Server_API/app/services/startup_content_jobs_pollers.py`
  - Add a focused media ingest lifecycle predicate that delegates to `should_start_inprocess_worker()`.
  - Replace only the media ingest worker spec predicates.
- Modify: `tldw_Server_API/tests/MediaIngestion_NEW/unit/test_media_ingest_startup.py`
  - Add failing tests for the startup policy helper and the actual lifecycle spec boundary.
- Modify: `tldw_Server_API/app/api/v1/endpoints/config_info.py`
  - Compute `hasMediaIngestWorker` from normal worker policy plus `TLDW_WORKERS_SIDECAR_MODE`.
- Modify: `tldw_Server_API/tests/Config/test_docs_info_capabilities.py`
  - Update existing media ingest capability expectation and add sidecar/opt-out coverage.
- Modify: `Docs/API-related/Media_Ingest_Jobs_API.md`
- Modify: `Docs/Published/API-related/Media_Ingest_Jobs_API.md`
- Modify: `Docs/Deployment/Long_Term_Admin_Guide.md`
- Modify: `Docs/Published/Deployment/Long_Term_Admin_Guide.md`
  - Reconcile normal worker default wording.
- Update: `backlog/tasks/task-12100 - Fix-media-ingest-worker-startup-default-and-capability-reporting.md`
  - Keep status, notes, changed files, and verification results current.

---

### Task 1: Add Startup Policy And Lifecycle Spec Regression Tests

**Files:**
- Modify: `tldw_Server_API/tests/MediaIngestion_NEW/unit/test_media_ingest_startup.py`

- [ ] **Step 1: Add imports for the real spec provider and context**

Add:

```python
from tldw_Server_API.app.services.lifecycle_worker_specs import WorkerLifecycleContext
from tldw_Server_API.app.services.startup_content_jobs_pollers import provide_content_jobs_worker_specs
```

- [ ] **Step 2: Add a tiny context helper**

```python
def _worker_context(*, sidecar_mode: bool = False, route_allowed: bool = True) -> WorkerLifecycleContext:
    return WorkerLifecycleContext(
        app=object(),
        settings={},
        test_mode=False,
        route_enabled=lambda *_args, **_kwargs: route_allowed,
        logger=None,
        startup_guard_exceptions=(),
        import_exceptions=(),
        sidecar_mode=sidecar_mode,
    )
```

- [ ] **Step 3: Add a spec lookup helper**

```python
def _content_spec(name: str):
    specs = {spec.name: spec for spec in provide_content_jobs_worker_specs()}
    return specs[name]
```

- [ ] **Step 4: Add failing startup policy tests for injected route checks**

```python
def test_should_start_inprocess_worker_uses_injected_route_policy_when_flag_unset(
    monkeypatch,
):
    from tldw_Server_API.app.services.worker_startup_policy import should_start_inprocess_worker

    monkeypatch.delenv("MEDIA_INGEST_JOBS_WORKER_ENABLED", raising=False)

    assert should_start_inprocess_worker(
        "MEDIA_INGEST_JOBS_WORKER_ENABLED",
        "media",
        sidecar_mode=False,
        default_stable=True,
        test_mode=False,
        route_enabled=lambda route_key, **_kwargs: route_key == "media",
    )


def test_should_start_inprocess_worker_honors_injected_route_disabled_when_flag_unset(
    monkeypatch,
):
    from tldw_Server_API.app.services.worker_startup_policy import should_start_inprocess_worker

    monkeypatch.delenv("MEDIA_INGEST_JOBS_WORKER_ENABLED", raising=False)

    assert not should_start_inprocess_worker(
        "MEDIA_INGEST_JOBS_WORKER_ENABLED",
        "media",
        sidecar_mode=False,
        default_stable=True,
        test_mode=False,
        route_enabled=lambda *_args, **_kwargs: False,
    )
```

Expected before implementation: `TypeError` because `route_enabled` is not accepted yet.

- [ ] **Step 5: Add the failing normal-worker route-default lifecycle tests**

```python
def test_media_ingest_lifecycle_spec_uses_route_policy_when_flag_unset(monkeypatch):
    monkeypatch.delenv("MEDIA_INGEST_JOBS_WORKER_ENABLED", raising=False)

    spec = _content_spec("media_ingest_jobs_task")

    assert spec.enabled(_worker_context(route_allowed=True)) is True


def test_media_ingest_lifecycle_spec_disables_when_route_policy_disabled(monkeypatch):
    monkeypatch.delenv("MEDIA_INGEST_JOBS_WORKER_ENABLED", raising=False)

    spec = _content_spec("media_ingest_jobs_task")

    assert spec.enabled(_worker_context(route_allowed=False)) is False
```

Expected before implementation: `False`.

- [ ] **Step 6: Add explicit opt-out, sidecar, and heavy default tests**

```python
def test_media_ingest_lifecycle_spec_respects_explicit_false(monkeypatch):
    monkeypatch.setenv("MEDIA_INGEST_JOBS_WORKER_ENABLED", "false")

    spec = _content_spec("media_ingest_jobs_task")

    assert spec.enabled(_worker_context(route_allowed=True)) is False


def test_media_ingest_lifecycle_spec_skips_in_sidecar_mode(monkeypatch):
    monkeypatch.setenv("MEDIA_INGEST_JOBS_WORKER_ENABLED", "true")

    spec = _content_spec("media_ingest_jobs_task")

    assert spec.enabled(_worker_context(sidecar_mode=True, route_allowed=True)) is False


def test_media_ingest_heavy_lifecycle_spec_remains_disabled_by_default(monkeypatch):
    monkeypatch.delenv("MEDIA_INGEST_HEAVY_JOBS_WORKER_ENABLED", raising=False)

    spec = _content_spec("media_ingest_heavy_jobs_task")

    assert spec.enabled(_worker_context(route_allowed=False)) is False
```

- [ ] **Step 7: Run tests and confirm the intended failure**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/MediaIngestion_NEW/unit/test_media_ingest_startup.py -v
```

Expected: the injected-route startup policy tests fail before implementation, and the normal-worker route-default lifecycle spec test still fails until the media ingest spec is rewired.

---

### Task 2: Wire Media Ingest Specs To In-Process Worker Policy

**Files:**
- Modify: `tldw_Server_API/app/services/worker_startup_policy.py`
- Modify: `tldw_Server_API/app/services/startup_content_jobs_pollers.py`

- [ ] **Step 1: Extend worker startup policy with optional injected route checks**

In `worker_startup_policy.py`, import `Callable`:

```python
from collections.abc import Callable
```

Then extend the helper signatures:

```python
def worker_route_default(
    route_key: str,
    *,
    default_stable: bool = True,
    test_mode: bool = False,
    route_enabled: Callable[..., bool] | None = None,
) -> bool:
```

```python
def worker_path_enabled(
    flag_key: str,
    route_key: str,
    *,
    default_stable: bool = True,
    test_mode: bool = False,
    route_enabled: Callable[..., bool] | None = None,
) -> bool:
```

```python
def should_start_inprocess_worker(
    flag_key: str,
    route_key: str,
    *,
    sidecar_mode: bool,
    default_stable: bool = True,
    test_mode: bool = False,
    route_enabled: Callable[..., bool] | None = None,
) -> bool:
```

In `worker_route_default()`, keep the existing `test_mode` early return. After that, use the injected route checker when provided:

```python
    if route_enabled is not None:
        try:
            return bool(route_enabled(route_key, default_stable=default_stable))
        except _WORKER_POLICY_EXCEPTIONS as exc:
            logger.debug("Worker startup policy route check failed for {}: {}", route_key, exc)
            return bool(default_stable)
```

Thread `route_enabled=route_enabled` through `worker_path_enabled()` and `should_start_inprocess_worker()`.

- [ ] **Step 2: Import the existing policy helper**

Change imports near the lifecycle spec imports:

```python
from tldw_Server_API.app.services.worker_startup_policy import should_start_inprocess_worker
```

- [ ] **Step 3: Add the focused predicate helper**

Add near `provide_content_jobs_worker_specs()`:

```python
def media_ingest_worker_predicate(
    flag_key: str,
    route_key: str,
    *,
    default_stable: bool,
):
    """Return an in-process media ingest worker predicate."""

    def _enabled(context: WorkerLifecycleContext) -> bool:
        return should_start_inprocess_worker(
            flag_key,
            route_key,
            sidecar_mode=context.sidecar_mode,
            default_stable=default_stable,
            test_mode=context.test_mode,
            route_enabled=context.route_enabled,
        )

    return _enabled
```

Keep this helper local to the content jobs poller module unless another worker family needs the same semantics later.

- [ ] **Step 4: Replace only the media ingest predicates**

Change normal worker spec:

```python
enabled=media_ingest_worker_predicate(
    "MEDIA_INGEST_JOBS_WORKER_ENABLED",
    "media",
    default_stable=True,
),
```

Change heavy worker spec:

```python
enabled=media_ingest_worker_predicate(
    "MEDIA_INGEST_HEAVY_JOBS_WORKER_ENABLED",
    "media-ingest-heavy-jobs",
    default_stable=False,
),
```

Leave all other `route_enabled_predicate(...)` usages alone.

- [ ] **Step 5: Run the focused startup tests**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/MediaIngestion_NEW/unit/test_media_ingest_startup.py -v
```

Expected: all tests in that file pass.

- [ ] **Step 6: Run lifecycle catalog guard tests**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Services/test_lifecycle_worker_catalog.py tldw_Server_API/tests/Services/test_startup_worker_groups.py -v
```

Expected: pass, proving the provider graph still validates.

---

### Task 3: Fix Docs-Info Worker Capability Reporting

**Files:**
- Modify: `tldw_Server_API/tests/Config/test_docs_info_capabilities.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/config_info.py`

- [ ] **Step 1: Update the existing capability expectation to fail first**

In `test_docs_info_exposes_bulk_conference_ingest_capabilities`, clear the normal flag and sidecar mode:

```python
monkeypatch.delenv("MEDIA_INGEST_JOBS_WORKER_ENABLED", raising=False)
monkeypatch.delenv("TLDW_WORKERS_SIDECAR_MODE", raising=False)
```

Change:

```python
assert caps["hasMediaIngestWorker"] is True
```

Expected before implementation: this fails because the code still checks the heavy worker.

- [ ] **Step 2: Add explicit false and sidecar capability tests**

```python
def test_docs_info_media_ingest_worker_capability_respects_explicit_false(
    monkeypatch, tmp_path: Path
) -> None:
    config_path = tmp_path / "config.txt"
    _write_minimal_config(config_path)

    monkeypatch.setenv("TLDW_CONFIG_PATH", str(config_path))
    monkeypatch.setenv("MEDIA_INGEST_JOBS_WORKER_ENABLED", "false")
    monkeypatch.delenv("TLDW_WORKERS_SIDECAR_MODE", raising=False)
    config_mod._route_toggle_policy.cache_clear()

    safe_config = config_info.load_safe_config()

    assert safe_config["capabilities"]["hasMediaIngestWorker"] is False


def test_docs_info_media_ingest_worker_capability_is_false_in_sidecar_mode(
    monkeypatch, tmp_path: Path
) -> None:
    config_path = tmp_path / "config.txt"
    _write_minimal_config(config_path)

    monkeypatch.setenv("TLDW_CONFIG_PATH", str(config_path))
    monkeypatch.setenv("MEDIA_INGEST_JOBS_WORKER_ENABLED", "true")
    monkeypatch.setenv("TLDW_WORKERS_SIDECAR_MODE", "true")
    config_mod._route_toggle_policy.cache_clear()

    safe_config = config_info.load_safe_config()

    assert safe_config["capabilities"]["hasMediaIngestWorker"] is False
```

- [ ] **Step 3: Run docs-info tests and confirm failure**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Config/test_docs_info_capabilities.py -v
```

Expected before implementation: `hasMediaIngestWorker` route-default expectation fails.

- [ ] **Step 4: Import the correct helpers in config_info**

Change the worker startup policy import to include `should_start_inprocess_worker`; keep `worker_path_enabled` only if other code still uses it.

Also import `env_flag_enabled` if not already available:

```python
from tldw_Server_API.app.core.testing import env_flag_enabled
from tldw_Server_API.app.services.worker_startup_policy import should_start_inprocess_worker
```

- [ ] **Step 5: Compute capability from the normal worker policy**

Replace the current heavy-worker block with:

```python
caps["hasMediaIngestWorker"] = bool(
    should_start_inprocess_worker(
        "MEDIA_INGEST_JOBS_WORKER_ENABLED",
        "media",
        sidecar_mode=env_flag_enabled("TLDW_WORKERS_SIDECAR_MODE"),
        default_stable=True,
        test_mode=False,
    )
)
```

- [ ] **Step 6: Run docs-info tests**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Config/test_docs_info_capabilities.py -v
```

Expected: pass.

---

### Task 4: Reconcile Documentation Defaults

**Files:**
- Modify: `Docs/API-related/Media_Ingest_Jobs_API.md`
- Modify: `Docs/Published/API-related/Media_Ingest_Jobs_API.md`
- Modify: `Docs/Deployment/Long_Term_Admin_Guide.md`
- Modify: `Docs/Published/Deployment/Long_Term_Admin_Guide.md`

- [ ] **Step 1: Update API docs worker flag wording**

In both API docs files, replace:

```markdown
- `MEDIA_INGEST_JOBS_WORKER_ENABLED`: `true|false` (default false)
```

with:

```markdown
- `MEDIA_INGEST_JOBS_WORKER_ENABLED`: `true|false` (default follows the `media` route policy; set `false` to disable the in-process worker)
```

- [ ] **Step 2: Update deployment guide wording**

In both deployment guide files, replace the media ingest sentence with:

```markdown
- Background jobs: Chatbooks worker enabled by default (core backend). Control via `CHATBOOKS_CORE_WORKER_ENABLED`. Media ingest jobs worker follows the `media` route policy by default; control via `MEDIA_INGEST_JOBS_WORKER_ENABLED`, or use sidecar workers for multi-worker deployments.
```

- [ ] **Step 3: Check for stale contradictory docs**

Run:

```bash
rg -n "MEDIA_INGEST_JOBS_WORKER_ENABLED.*default false|Media ingest jobs worker is opt-in" Docs
```

Expected: no results.

---

### Task 5: Verification And Handoff

**Files:**
- Update: `backlog/tasks/task-12100 - Fix-media-ingest-worker-startup-default-and-capability-reporting.md`

- [ ] **Step 1: Run focused regression tests**

Run:

```bash
source .venv/bin/activate
python -m pytest \
  tldw_Server_API/tests/MediaIngestion_NEW/unit/test_media_ingest_startup.py \
  tldw_Server_API/tests/Config/test_docs_info_capabilities.py \
  tldw_Server_API/tests/Services/test_lifecycle_worker_catalog.py \
  tldw_Server_API/tests/Services/test_startup_worker_groups.py \
  -v
```

Expected: pass.

- [ ] **Step 2: Run Bandit on touched backend scopes**

Run:

```bash
source .venv/bin/activate
python -m bandit -r \
  tldw_Server_API/app/services/worker_startup_policy.py \
  tldw_Server_API/app/services/startup_content_jobs_pollers.py \
  tldw_Server_API/app/api/v1/endpoints/config_info.py \
  -f json -o /tmp/bandit_media_ingest_worker_startup.json
```

Expected: no new high or medium findings in touched code.

- [ ] **Step 3: Smoke-check startup policy with the real spec**

Run a small Python check:

```bash
source .venv/bin/activate
python - <<'PY'
from fastapi import FastAPI
from tldw_Server_API.app.services.lifecycle_worker_specs import WorkerLifecycleContext
from tldw_Server_API.app.services.startup_content_jobs_pollers import provide_content_jobs_worker_specs

ctx = WorkerLifecycleContext(
    app=FastAPI(),
    settings={},
    test_mode=False,
    route_enabled=lambda route_key, **kwargs: route_key == "media",
    logger=None,
    startup_guard_exceptions=(),
    import_exceptions=(),
    sidecar_mode=False,
)
specs = {spec.name: spec for spec in provide_content_jobs_worker_specs()}
print(specs["media_ingest_jobs_task"].enabled(ctx))
PY
```

Expected output: `True`.

- [ ] **Step 4: Run the end-to-end YouTube quick-ingest walkthrough**

Use the already established local WebUI/browser-extension walkthrough rather than stopping at unit tests:

- Start the backend and WebUI if they are not already running.
- Submit a YouTube quick-ingest job from the browser path that previously reproduced the 0% queue stall.
- Confirm the job status leaves `queued` and reaches `running`, `completed`, or a real ingestion failure with a specific error.
- Capture the relevant backend log line or API status payload in the final handoff.

- [ ] **Step 5: Update Backlog task with results**

Record:

- Files changed.
- Focused pytest command result.
- Bandit result.
- Startup smoke result.
- Browser walkthrough result.

- [ ] **Step 6: Self-review changed files**

Run:

```bash
git diff --check
git diff -- tldw_Server_API/app/services/startup_content_jobs_pollers.py \
  tldw_Server_API/app/services/worker_startup_policy.py \
  tldw_Server_API/app/api/v1/endpoints/config_info.py \
  tldw_Server_API/tests/MediaIngestion_NEW/unit/test_media_ingest_startup.py \
  tldw_Server_API/tests/Config/test_docs_info_capabilities.py \
  Docs/API-related/Media_Ingest_Jobs_API.md \
  Docs/Published/API-related/Media_Ingest_Jobs_API.md \
  Docs/Deployment/Long_Term_Admin_Guide.md \
  Docs/Published/Deployment/Long_Term_Admin_Guide.md
```

Expected: no whitespace errors; diff is limited to the planned files.

- [ ] **Step 7: Commit only this task's files**

Run:

```bash
git add \
  Docs/superpowers/specs/2026-07-09-media-ingest-worker-startup-capability-design.md \
  Docs/superpowers/plans/2026-07-09-media-ingest-worker-startup-capability-implementation-plan.md \
  tldw_Server_API/app/services/worker_startup_policy.py \
  tldw_Server_API/app/services/startup_content_jobs_pollers.py \
  tldw_Server_API/app/api/v1/endpoints/config_info.py \
  tldw_Server_API/tests/MediaIngestion_NEW/unit/test_media_ingest_startup.py \
  tldw_Server_API/tests/Config/test_docs_info_capabilities.py \
  Docs/API-related/Media_Ingest_Jobs_API.md \
  Docs/Published/API-related/Media_Ingest_Jobs_API.md \
  Docs/Deployment/Long_Term_Admin_Guide.md \
  Docs/Published/Deployment/Long_Term_Admin_Guide.md \
  "backlog/tasks/task-12100 - Fix-media-ingest-worker-startup-default-and-capability-reporting.md"
git commit -m "fix: start media ingest worker by route default"
```

Expected: commit succeeds without including unrelated dirty files.
