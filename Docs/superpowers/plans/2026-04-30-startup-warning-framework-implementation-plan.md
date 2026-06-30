# Startup Warning Framework Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a reusable current-process startup warning framework, wire `sandbox.vz_linux` reconciliation into it as the first producer, expose a generic admin surface, and block startup on helper protocol mismatch.

**Architecture:** Introduce a small startup-warning registry owned by app startup state, add a bounded sandbox startup-warning producer that translates helper/reconciliation truth into shared records, expose those records through a generic admin endpoint plus additive sandbox diagnostics summary, and enforce a fail-closed startup path only for helper protocol mismatch.

**Tech Stack:** Python 3, FastAPI, existing startup/lifespan services, Pydantic, pytest, Loguru, existing sandbox diagnostics/reconciliation modules.

---

## Source References

- Spec: `Docs/superpowers/specs/2026-04-30-startup-warning-framework-design.md`
- Doctrine: `Docs/Sandbox/sandbox-architecture-doctrine.md`
- Startup orchestration: `tldw_Server_API/app/services/lifespan_startup_sequence.py`
- Startup service tail: `tldw_Server_API/app/services/startup_service_tail.py`
- Sandbox diagnostics: `tldw_Server_API/app/core/Sandbox/macos_diagnostics.py`
- Sandbox reconciliation: `tldw_Server_API/app/core/Sandbox/vz_reconciliation.py`
- Sandbox admin endpoint: `tldw_Server_API/app/api/v1/endpoints/sandbox.py`

## File Structure

- Create: `tldw_Server_API/app/services/startup_warning_models.py`
  - Shared startup warning dataclasses/helpers.
- Create: `tldw_Server_API/app/services/startup_warning_registry.py`
  - Current-process registry owned through app startup state.
- Create: `tldw_Server_API/app/services/startup_warning_sandbox.py`
  - First producer translating sandbox helper/reconciliation truth into shared warning records.
- Modify: `tldw_Server_API/app/services/lifespan_startup_sequence.py`
  - Initialize registry and run producers at the correct startup point.
- Modify: `tldw_Server_API/app/services/startup_core_initialization.py` or the
  startup-owned seam that can safely expose sandbox/orchestrator dependencies
  to later startup producers.
- Modify: `tldw_Server_API/app/api/v1/schemas/admin_schemas.py`
  - Add generic admin startup-warning response models.
- Create or modify: `tldw_Server_API/app/api/v1/endpoints/admin/startup_warnings.py`
  - Admin-only current-process startup-warning endpoint.
- Modify: `tldw_Server_API/app/api/v1/endpoints/admin/__init__.py`
  - Include the new startup-warning admin router in the assembled admin surface.
- Modify: `tldw_Server_API/app/api/v1/router_groups/admin.py`
  - Ensure the admin router group exposes the new startup-warning route.
- Modify: `tldw_Server_API/app/core/Sandbox/macos_diagnostics.py`
  - Accept additive startup warning summary injection without depending on app state.
- Modify: `tldw_Server_API/app/core/Sandbox/service.py`
  - Project the startup warning summary into sandbox diagnostics through a startup-safe seam.
- Modify: `tldw_Server_API/app/api/v1/schemas/sandbox_schemas.py`
  - Add additive sandbox startup-warning summary schema.
- Modify tests:
  - `tldw_Server_API/tests/Services/test_startup_warning_registry.py`
  - `tldw_Server_API/tests/Services/test_startup_warning_sandbox.py`
  - `tldw_Server_API/tests/sandbox/test_macos_diagnostics.py`
  - `tldw_Server_API/tests/sandbox/test_admin_macos_diagnostics.py`
  - `tldw_Server_API/tests/api/test_admin_startup_warnings.py`
  - extend one startup/lifespan integration test file in `tldw_Server_API/tests/Services/`
- Modify docs:
  - `Docs/Sandbox/macos-runtime-operator-notes.md`
  - `tldw_Server_API/app/core/Sandbox/README.md`

## Task 1: Add Shared Startup Warning Models And Registry

**Files:**
- Create: `tldw_Server_API/app/services/startup_warning_models.py`
- Create: `tldw_Server_API/app/services/startup_warning_registry.py`
- Test: `tldw_Server_API/tests/Services/test_startup_warning_registry.py`

- [ ] **Step 1: Write the failing registry tests**

Add tests covering:

```python
def test_startup_warning_registry_starts_empty():
    registry = StartupWarningRegistry(startup_id="boot-1")
    assert registry.list_warnings() == []
    assert registry.summary()["total"] == 0
    assert registry.should_block_startup() is False


def test_startup_warning_registry_groups_and_detects_blockers():
    registry = StartupWarningRegistry(startup_id="boot-1")
    registry.add_warning(...)
    registry.add_warning(...)
    assert registry.summary()["by_component"]["sandbox.vz_linux"] == 2
    assert registry.should_block_startup() is True


def test_startup_warning_registry_clear_resets_state():
    ...
```

- [ ] **Step 2: Run the registry tests to verify failure**

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Services/test_startup_warning_registry.py -v
```

Expected: import or symbol failures.

- [ ] **Step 3: Implement the minimal models and registry**

Define:

- warning record model with:
  - `component`
  - `severity`
  - `startup_action`
  - `code`
  - `summary`
  - `remediation`
  - `details`
  - `detected_at`
- registry with:
  - `add_warning(record)`
  - `list_warnings()`
  - `summary()`
  - `clear()`
  - `should_block_startup()`

Implementation constraints:

- no persistence
- no globals required for correctness
- deterministic summary ordering for tests

- [ ] **Step 4: Run the registry tests to verify pass**

Run the Task 1 command again and expect PASS.

- [ ] **Step 5: Commit Task 1**

```bash
git add \
  tldw_Server_API/app/services/startup_warning_models.py \
  tldw_Server_API/app/services/startup_warning_registry.py \
  tldw_Server_API/tests/Services/test_startup_warning_registry.py
git commit -m "feat(startup): add startup warning registry"
```

## Task 2: Add Sandbox Startup Warning Producer

**Files:**
- Create: `tldw_Server_API/app/services/startup_warning_sandbox.py`
- Test: `tldw_Server_API/tests/Services/test_startup_warning_sandbox.py`

- [ ] **Step 1: Write the failing sandbox producer tests**

Cover translation rules:

```python
def test_sandbox_startup_producer_emits_warning_for_stale_and_orphaned_state():
    registry = StartupWarningRegistry(startup_id="boot-1")
    produce_sandbox_startup_warnings(..., registry=registry)
    codes = [item.code for item in registry.list_warnings()]
    assert "vz_stale_session_controls_detected" in codes
    assert "vz_orphaned_vms_detected" in codes


def test_sandbox_startup_producer_emits_blocker_for_protocol_mismatch():
    ...
    assert registry.should_block_startup() is True


def test_sandbox_startup_producer_helper_unavailable_is_warning_only():
    ...


def test_sandbox_startup_producer_does_not_mutate_runtime_state():
    ...
```

- [ ] **Step 2: Run the sandbox producer tests to verify failure**

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Services/test_startup_warning_sandbox.py -v
```

Expected: import or missing symbol failures.

- [ ] **Step 3: Implement the sandbox producer**

Implementation requirements:

- use bounded helper/reconciliation checks only
- do not call `collect_macos_diagnostics()`
- summarize counts rather than storing raw reconciliation items
- accept startup-owned dependencies explicitly; do not import or depend on
  `tldw_Server_API.app.api.v1.endpoints.sandbox._service`
- emit `warn` records for:
  - stale session controls
  - unhealthy session controls
  - orphaned VMs
  - skipped-active reconciliation items
  - helper unavailable
- emit `block_startup` for protocol mismatch
- log from the same record objects that are added to the registry

- [ ] **Step 4: Run the sandbox producer tests to verify pass**

Run the Task 2 command again and expect PASS.

- [ ] **Step 5: Commit Task 2**

```bash
git add \
  tldw_Server_API/app/services/startup_warning_sandbox.py \
  tldw_Server_API/tests/Services/test_startup_warning_sandbox.py
git commit -m "feat(startup): add sandbox startup warning producer"
```

## Task 3: Wire Registry Into Lifespan Startup

**Files:**
- Modify: `tldw_Server_API/app/services/lifespan_startup_sequence.py`
- Modify: the startup-owned handle/state module that owns app startup state
- Test: existing startup/lifespan test area under `tldw_Server_API/tests/Services/`

- [ ] **Step 1: Write the failing startup integration tests**

Cover:

```python
def test_startup_initializes_registry_and_runs_sandbox_producer(...):
    ...


def test_startup_blocks_on_protocol_mismatch_warning(...):
    ...
    assert "vz_helper_protocol_mismatch" in str(exc_info.value)


def test_startup_warning_registry_is_available_on_app_state(...):
    ...
```

- [ ] **Step 2: Run the startup integration tests to verify failure**

Run only the focused startup/lifespan tests you add.

- [ ] **Step 3: Implement startup wiring**

Requirements:

- initialize a registry per startup/boot
- attach it to `app.state`
- run the sandbox producer after startup-owned sandbox/orchestrator dependencies
  are available
- if `registry.should_block_startup()` is true, raise using the strongest
  blocking record
- keep startup non-mutating
- do not resolve sandbox startup dependencies by importing endpoint modules

- [ ] **Step 4: Run the startup integration tests to verify pass**

Re-run the Task 3 command and expect PASS.

- [ ] **Step 5: Commit Task 3**

```bash
git add <startup service files and tests>
git commit -m "feat(startup): wire startup warnings into lifespan"
```

## Task 4: Add Generic Admin Endpoint For Current-Process Startup Warnings

**Files:**
- Modify: `tldw_Server_API/app/api/v1/schemas/admin_schemas.py`
- Create or modify: `tldw_Server_API/app/api/v1/endpoints/admin/startup_warnings.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/admin/__init__.py`
- Modify: `tldw_Server_API/app/api/v1/router_groups/admin.py`
- Test: `tldw_Server_API/tests/api/test_admin_startup_warnings.py`

- [ ] **Step 1: Write the failing admin endpoint tests**

Cover:

```python
def test_admin_startup_warnings_returns_current_process_registry_summary(...):
    ...


def test_admin_startup_warnings_is_admin_only(...):
    ...


def test_admin_startup_warnings_reports_current_process_scope(...):
    ...
```

- [ ] **Step 2: Run the admin endpoint tests to verify failure**

Run the focused API test file and expect failure.

- [ ] **Step 3: Add schemas and endpoint**

Response must include:

- `startup_id`
- `scope="current_process"`
- `warnings_present`
- `blocking_present`
- grouped `summary`
- flat `items`

Implementation rules:

- read from `app.state.startup_warning_registry`
- if no registry is present, return an empty current-process response rather
  than 500
- keep endpoint admin-only

- [ ] **Step 4: Run the admin endpoint tests to verify pass**

Re-run the Task 4 command and expect PASS.

- [ ] **Step 5: Commit Task 4**

```bash
git add \
  tldw_Server_API/app/api/v1/schemas/admin_schemas.py \
  tldw_Server_API/app/api/v1/endpoints/admin/startup_warnings.py \
  tldw_Server_API/app/api/v1/endpoints/admin/__init__.py \
  tldw_Server_API/app/api/v1/router_groups/admin.py \
  tldw_Server_API/tests/api/test_admin_startup_warnings.py
git commit -m "feat(admin): expose startup warnings endpoint"
```

## Task 5: Surface Startup Warning Summary In Sandbox Diagnostics

**Files:**
- Modify: `tldw_Server_API/app/core/Sandbox/macos_diagnostics.py`
- Modify: `tldw_Server_API/app/core/Sandbox/service.py`
- Modify: `tldw_Server_API/app/api/v1/schemas/sandbox_schemas.py`
- Modify: `tldw_Server_API/tests/sandbox/test_macos_diagnostics.py`
- Modify: `tldw_Server_API/tests/sandbox/test_admin_macos_diagnostics.py`

- [ ] **Step 1: Write the failing diagnostics tests**

Add assertions like:

```python
assert data["startup_warning_summary"]["present"] is True
assert data["startup_warning_summary"]["blocking"] is False
assert "vz_stale_session_controls_detected" in data["startup_warning_summary"]["codes"]
```

- [ ] **Step 2: Run the diagnostics tests to verify failure**

Run the two focused sandbox diagnostics test files and expect failure.

- [ ] **Step 3: Implement the additive diagnostics summary**

Requirements:

- keep `macos_diagnostics.py` app-agnostic
- project warning records from the shared registry into a compact sandbox-local
  summary one layer above `collect_macos_diagnostics()`
- include only sandbox-related codes
- do not duplicate the full generic endpoint payload

- [ ] **Step 4: Run the diagnostics tests to verify pass**

Re-run the Task 5 command and expect PASS.

- [ ] **Step 5: Commit Task 5**

```bash
git add \
  tldw_Server_API/app/core/Sandbox/macos_diagnostics.py \
  tldw_Server_API/app/core/Sandbox/service.py \
  tldw_Server_API/app/api/v1/schemas/sandbox_schemas.py \
  tldw_Server_API/tests/sandbox/test_macos_diagnostics.py \
  tldw_Server_API/tests/sandbox/test_admin_macos_diagnostics.py
git commit -m "feat(sandbox): summarize startup warnings in diagnostics"
```

## Task 6: Update Operator Docs

**Files:**
- Modify: `Docs/Sandbox/macos-runtime-operator-notes.md`
- Modify: `tldw_Server_API/app/core/Sandbox/README.md`

- [ ] **Step 1: Update docs**

Add:

- startup warnings are current-process scoped in this PR
- startup remains non-mutating
- protocol mismatch blocks startup
- startup blockers are guaranteed in logs, not in the generic admin endpoint
- generic startup warnings are available at the new admin endpoint after
  successful boot

- [ ] **Step 2: Verify docs and diff hygiene**

Run:

```bash
git diff --check
```

Expected: no whitespace errors.

- [ ] **Step 3: Commit Task 6**

```bash
git add \
  Docs/Sandbox/macos-runtime-operator-notes.md \
  tldw_Server_API/app/core/Sandbox/README.md
git commit -m "docs(startup): describe startup warning framework"
```

## Task 7: Final Verification

**Files:**
- Verify touched startup, admin, and sandbox test files

- [ ] **Step 1: Run focused verification suite**

Run:

```bash
source .venv/bin/activate && python -m pytest \
  tldw_Server_API/tests/Services/test_startup_warning_registry.py \
  tldw_Server_API/tests/Services/test_startup_warning_sandbox.py \
  tldw_Server_API/tests/api/test_admin_startup_warnings.py \
  tldw_Server_API/tests/sandbox/test_macos_diagnostics.py \
  tldw_Server_API/tests/sandbox/test_admin_macos_diagnostics.py \
  <startup integration tests> \
  -v
```

- [ ] **Step 2: Run Bandit on touched production scope**

Run:

```bash
source .venv/bin/activate && python -m bandit -r \
  tldw_Server_API/app/services/startup_warning_models.py \
  tldw_Server_API/app/services/startup_warning_registry.py \
  tldw_Server_API/app/services/startup_warning_sandbox.py \
  tldw_Server_API/app/api/v1/endpoints/admin/startup_warnings.py \
  tldw_Server_API/app/core/Sandbox/macos_diagnostics.py \
  -f json -o /tmp/bandit_startup_warning_framework.json
```

- [ ] **Step 3: Run final diff hygiene**

Run:

```bash
git diff --check
```

- [ ] **Step 4: Prepare PR**

```bash
git status --short
git log --oneline dev..HEAD
```

- [ ] **Step 5: Final commit if needed**

```bash
git add <remaining files>
git commit -m "test(startup): finalize startup warning coverage"
```
