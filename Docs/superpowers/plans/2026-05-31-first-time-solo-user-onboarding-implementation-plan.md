# First-Time Solo User Onboarding Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the approved unified first-time solo-user onboarding flow: peer Docker/local solo setup paths, backend-authoritative setup state, WebUI progressive wizard, provider configuration without manual config edits, real first-chat completion, and source-ingest follow-up.

**Architecture:** Extend the existing backend `/api/v1/setup/*` surface instead of creating a second setup API. Add focused setup state/readiness services under `tldw_Server_API/app/core/Setup/`, expose typed setup endpoints, consume them through a new frontend setup-onboarding API domain, and replace the current WebUI first-run route with a focused progressive wizard. Keep backend `/setup` as operator/recovery and keep CLI/docs aligned with the same lifecycle.

**Tech Stack:** FastAPI, Pydantic, Loguru, config.txt/.env setup helpers, pytest, Next.js/React, TypeScript, Ant Design, Zustand/React Query where already used, Vitest, Playwright.

---

## Source Documents

- PRD: `Docs/superpowers/specs/2026-05-31-first-time-solo-user-onboarding-prd-design.md`
- Backlog planning task: `TASK-488`
- Related PRD task: `TASK-487`
- Existing onboarding docs: `Docs/Getting_Started/README.md`
- Existing setup API: `tldw_Server_API/app/api/v1/endpoints/setup.py`
- Existing setup manager: `tldw_Server_API/app/core/Setup/setup_manager.py`
- Existing WebUI setup route: `apps/packages/ui/src/routes/option-setup.tsx`
- Existing WebUI onboarding form: `apps/packages/ui/src/components/Option/Onboarding/OnboardingConnectForm.tsx`
- Existing audio setup panel/hook: `apps/packages/ui/src/components/Option/Setup/AudioInstallerPanel.tsx`, `apps/packages/ui/src/components/Option/Setup/hooks/useAudioInstaller.ts`

## Spec Traceability

- Peer Docker/local solo setup paths and multi-user exit: Tasks 7 and 9.
- Backend-authoritative setup state and restart-safe completion: Tasks 1, 2, 4, and 6.
- Setup access boundary, bundled auth metadata, and recovery diagnostics: Tasks 2 and 7.
- Provider/key/local endpoint setup without manual config editing: Tasks 3, 6, and 8.
- Ingest defaults, audio/STT/TTS defaults, and optional RAG/storage deferral: Tasks 5 and 8.
- Actual first-chat completion gate with required screen acknowledgements: Tasks 1, 4, 8, and 10.
- Focused setup shell hiding normal navigation until completion or skip: Tasks 7 and 10.
- Post-onboarding add-first-source milestone: Tasks 8, 9, and 10.
- Cleanup of conflicting docs, CLI, and user-facing setup surfaces: Task 9.
- Final backend/frontend/E2E/security verification: Task 10.

## Scope And Slicing

This is a broad product change. Implement it as staged PR slices, not one giant patch. Each task below should be independently testable and committed. The recommended order is backend state/API foundation first, then provider config, then first-chat completion, then WebUI wizard, then docs/CLI cleanup, then E2E hardening.

Do not migrate multi-user setup into the solo wizard. Multi-user is a documented exit ramp.

## File Map

### Backend Setup State And APIs

- Create: `tldw_Server_API/app/core/Setup/first_run_state.py`
  - Owns durable setup state file, state transitions, masking, and completion/skip semantics.
- Create: `tldw_Server_API/app/core/Setup/provider_catalog.py`
  - Backend-generated provider catalog for setup UI; maps provider keys to config/env fields, provider kind, default base URL, and validation capability.
- Create: `tldw_Server_API/app/core/Setup/provider_validation.py`
  - Hosted/local provider validation helpers, safe failure categories, and local endpoint checks.
- Create: `tldw_Server_API/app/core/Setup/first_chat_verifier.py`
  - Executes the first-chat verification using the existing chat completion path or a thin service wrapper.
- Modify: `tldw_Server_API/app/api/v1/schemas/setup_schemas.py`
  - Add first-run state, provider catalog, provider save/validate, ingest defaults, optional advanced, first-chat, and skip schemas.
- Modify: `tldw_Server_API/app/api/v1/endpoints/setup.py`
  - Add `/first-run/*` endpoints while keeping existing `/setup` and audio endpoints stable.
- Modify: `tldw_Server_API/app/api/v1/API_Deps/setup_deps.py`
  - Tighten setup-write trust-boundary helpers and expose testable reason codes.

### Backend Tests

- Create: `tldw_Server_API/tests/Setup/test_first_run_state.py`
- Create: `tldw_Server_API/tests/Setup/test_setup_provider_catalog.py`
- Create: `tldw_Server_API/tests/Setup/test_setup_provider_validation.py`
- Create: `tldw_Server_API/tests/Setup/test_setup_first_chat_completion.py`
- Modify: `tldw_Server_API/tests/Setup/test_setup_deps_remote_admin.py`
- Modify or create: `tldw_Server_API/tests/integration/test_unified_first_run_setup_api.py`

### Frontend API And State

- Create: `apps/packages/ui/src/types/setup-onboarding.ts`
- Create: `apps/packages/ui/src/services/tldw/domains/setup-onboarding.ts`
- Modify: `apps/packages/ui/src/services/tldw/domains/index.ts`
- Modify: `apps/packages/ui/src/services/tldw/TldwApiClient.ts`
- Modify: `apps/packages/ui/src/services/tldw/openapi-guard.ts`
- Create: `apps/packages/ui/src/hooks/useSetupOnboarding.ts`

### Frontend Wizard

- Create: `apps/packages/ui/src/components/Option/Onboarding/UnifiedSetupWizard.tsx`
- Create: `apps/packages/ui/src/components/Option/Onboarding/steps/SetupPathStep.tsx`
- Create: `apps/packages/ui/src/components/Option/Onboarding/steps/PrivacySecurityStep.tsx`
- Create: `apps/packages/ui/src/components/Option/Onboarding/steps/ProviderSetupStep.tsx`
- Create: `apps/packages/ui/src/components/Option/Onboarding/steps/IngestDefaultsStep.tsx`
- Create: `apps/packages/ui/src/components/Option/Onboarding/steps/AudioSetupStep.tsx`
- Create: `apps/packages/ui/src/components/Option/Onboarding/steps/OptionalAdvancedStep.tsx`
- Create: `apps/packages/ui/src/components/Option/Onboarding/steps/FirstChatStep.tsx`
- Create: `apps/packages/ui/src/components/Option/Onboarding/steps/MultiUserExitPanel.tsx`
- Create: `apps/packages/ui/src/components/Option/Onboarding/FirstSourceMilestonePrompt.tsx`
- Modify: `apps/packages/ui/src/routes/option-index.tsx`
- Modify: `apps/packages/ui/src/routes/option-setup.tsx`
- Modify: `apps/tldw-frontend/pages/setup.tsx` only if wrapper metadata needs changing.

### Frontend Tests

- Create: `apps/packages/ui/src/services/tldw/__tests__/setup-onboarding.test.ts`
- Create: `apps/packages/ui/src/hooks/__tests__/useSetupOnboarding.test.tsx`
- Create: `apps/packages/ui/src/components/Option/Onboarding/__tests__/UnifiedSetupWizard.test.tsx`
- Create: `apps/packages/ui/src/components/Option/Onboarding/__tests__/ProviderSetupStep.test.tsx`
- Create: `apps/packages/ui/src/components/Option/Onboarding/__tests__/FirstChatStep.test.tsx`
- Create: `apps/packages/ui/src/components/Option/Onboarding/__tests__/FirstSourceMilestonePrompt.test.tsx`
- Create: `apps/packages/ui/src/routes/__tests__/option-index.unified-setup.test.tsx`
- Modify: existing onboarding guards under `apps/packages/ui/src/components/Option/Onboarding/__tests__/` when they assert old copy/behavior.

### Docs, CLI, And Cleanup

- Modify: `Makefile`
- Modify: `tldw_Server_API/cli/wizard/cli.py`
- Modify: `tldw_Server_API/cli/wizard/profile_verify.py`
- Modify: `tldw_Server_API/cli/wizard/profiles.py` only if profile metadata needs WebUI peer choice labels.
- Modify: `Docs/Getting_Started/README.md`
- Modify: `Docs/Getting_Started/Profile_Docker_Single_User.md`
- Modify: `Docs/Getting_Started/Profile_Local_Single_User.md`
- Modify: `Docs/Getting_Started/Profile_Docker_Multi_User_Postgres.md`
- Modify: `Docs/Getting_Started/onboarding_manifest.yaml`
- Modify matching `Docs/Published/Getting_Started/*` files if this repo still expects published parity.
- Modify tests under `tldw_Server_API/tests/Docs/` and `tldw_Server_API/tests/Utils/`.

## Implementation Tasks

### Task 0: Branch, Backlog, And Baseline Guard

**Files:**
- Reference: `Docs/superpowers/plans/2026-05-31-first-time-solo-user-onboarding-implementation-plan.md`
- Backlog: create or update implementation task(s) before product code edits.

- [ ] **Step 1: Verify branch and dirty worktree**

Run:

```bash
git branch --show-current
git status --short
```

Expected: active branch is known. Existing unrelated dirty files are left untouched.

- [ ] **Step 2: Create implementation Backlog parent task**

Create a Backlog task named `Implement unified first-time solo user onboarding`.

Expected: task links this plan, the PRD, and all child task ids as they are created.

- [ ] **Step 3: Create child Backlog tasks for each implementation slice**

Create child tasks for:

- backend state/access foundation;
- provider catalog and validation;
- first-chat completion;
- WebUI setup client and wizard shell;
- wizard steps;
- docs/CLI cleanup;
- E2E verification.

Expected: each child task has a clear modified-file set before edits begin.

- [ ] **Step 4: Capture current baseline tests**

Run read-only baseline checks:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Setup/test_setup_deps_remote_admin.py -v
python -m pytest tldw_Server_API/tests/Config/test_config_providers_endpoints.py -v
```

From `apps/packages/ui`:

```bash
bunx vitest run src/components/Option/Onboarding/__tests__/OnboardingConnectForm.design-system.test.tsx
```

Expected: note current pass/fail in Backlog. Do not fix unrelated failures in this task.

- [ ] **Step 5: Commit planning/backlog-only updates if any**

```bash
git add backlog/tasks/<task-files>
git commit -m "chore: track unified onboarding implementation"
```

Expected: commit contains only Backlog/task metadata if any files changed.

### Task 1: Backend First-Run State Store

**Files:**
- Create: `tldw_Server_API/app/core/Setup/first_run_state.py`
- Modify: `tldw_Server_API/app/api/v1/schemas/setup_schemas.py`
- Test: `tldw_Server_API/tests/Setup/test_first_run_state.py`

- [ ] **Step 1: Write failing state-store tests**

Create `tldw_Server_API/tests/Setup/test_first_run_state.py`:

```python
from pathlib import Path

import pytest

from tldw_Server_API.app.core.Setup.first_run_state import (
    REQUIRED_FIRST_RUN_STEPS,
    FirstRunStateStore,
    FirstRunStatus,
    InvalidFirstRunTransition,
)


def test_new_store_defaults_to_not_started(tmp_path: Path):
    store = FirstRunStateStore(tmp_path / "first_run_state.json")

    state = store.load()

    assert state.status == FirstRunStatus.NOT_STARTED
    assert state.completed_at is None
    assert state.first_chat.completed is False


def test_records_step_and_persists_across_store_instances(tmp_path: Path):
    path = tmp_path / "first_run_state.json"
    store = FirstRunStateStore(path)

    store.update_step("providers", {"default_provider": "openai"})

    reloaded = FirstRunStateStore(path).load()
    assert reloaded.status == FirstRunStatus.IN_PROGRESS
    assert reloaded.current_step == "providers"
    assert reloaded.step_data["providers"]["default_provider"] == "openai"


def test_complete_requires_first_chat_success(tmp_path: Path):
    store = FirstRunStateStore(tmp_path / "first_run_state.json")

    with pytest.raises(InvalidFirstRunTransition):
        store.mark_completed()


def test_complete_requires_required_step_acknowledgements(tmp_path: Path):
    store = FirstRunStateStore(tmp_path / "first_run_state.json")

    for step in REQUIRED_FIRST_RUN_STEPS:
        store.update_step(step, {"acknowledged": False})
    store.record_first_chat_success(
        provider="openai",
        model="gpt-4.1-mini",
        response_id="chatcmpl-test",
    )

    with pytest.raises(InvalidFirstRunTransition) as excinfo:
        store.mark_completed()

    assert "required_steps_missing" in str(excinfo.value)


def test_skip_records_skipped_not_completed(tmp_path: Path):
    store = FirstRunStateStore(tmp_path / "first_run_state.json")

    state = store.mark_skipped(reason="user_skip")

    assert state.status == FirstRunStatus.SKIPPED
    assert state.completed_at is None
    assert state.skip_reason == "user_skip"


def test_first_chat_success_allows_completion(tmp_path: Path):
    store = FirstRunStateStore(tmp_path / "first_run_state.json")

    for step in REQUIRED_FIRST_RUN_STEPS:
        store.update_step(step, {"acknowledged": True})
    store.record_first_chat_success(
        provider="openai",
        model="gpt-4.1-mini",
        response_id="chatcmpl-test",
    )
    state = store.mark_completed()

    assert state.status == FirstRunStatus.COMPLETED
    assert state.completed_at is not None
    assert state.first_chat.completed is True
```

- [ ] **Step 2: Run the failing tests**

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Setup/test_first_run_state.py -v
```

Expected: FAIL because `first_run_state.py` does not exist.

- [ ] **Step 3: Add setup state schemas**

In `tldw_Server_API/app/api/v1/schemas/setup_schemas.py`, add:

```python
from datetime import datetime
from enum import Enum


class FirstRunStatus(str, Enum):
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    BLOCKED = "blocked"
    SKIPPED = "skipped"
    FIRST_CHAT_COMPLETE = "first_chat_complete"
    COMPLETED = "completed"


class FirstRunStepStatus(str, Enum):
    NOT_STARTED = "not_started"
    CURRENT = "current"
    COMPLETE = "complete"
    SKIPPED = "skipped"
    BLOCKED = "blocked"


class FirstRunChatResult(BaseModel):
    completed: bool = False
    provider: str | None = None
    model: str | None = None
    response_id: str | None = None
    completed_at: datetime | None = None


class FirstRunStateResponse(BaseModel):
    status: FirstRunStatus
    current_step: str | None = None
    completed_steps: list[str] = Field(default_factory=list)
    skipped_steps: list[str] = Field(default_factory=list)
    step_data: dict[str, dict[str, Any]] = Field(default_factory=dict)
    first_chat: FirstRunChatResult = Field(default_factory=FirstRunChatResult)
    acknowledged_steps: list[str] = Field(default_factory=list)
    skip_reason: str | None = None
    created_at: datetime
    updated_at: datetime
    completed_at: datetime | None = None
```

Keep imports sorted and avoid duplicating `BaseModel`, `Field`, or `Any` imports.

- [ ] **Step 4: Implement `first_run_state.py` minimally**

Create `tldw_Server_API/app/core/Setup/first_run_state.py`:

```python
"""Durable state for the unified first-run setup flow."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from tldw_Server_API.app.api.v1.schemas.setup_schemas import (
    FirstRunChatResult,
    FirstRunStateResponse,
    FirstRunStatus,
)


REQUIRED_FIRST_RUN_STEPS = (
    "setup_path",
    "privacy_security",
    "providers",
    "ingest_defaults",
    "audio_defaults",
    "optional_advanced",
)


class InvalidFirstRunTransition(ValueError):
    """Raised when a setup state transition would violate first-run rules."""


class FirstRunState(FirstRunStateResponse):
    """Internal state model persisted by FirstRunStateStore."""


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _default_state() -> FirstRunState:
    now = _now()
    return FirstRunState(
        status=FirstRunStatus.NOT_STARTED,
        current_step=None,
        created_at=now,
        updated_at=now,
    )


class FirstRunStateStore:
    """JSON-backed first-run setup state store."""

    def __init__(self, path: Path):
        self.path = path

    def load(self) -> FirstRunState:
        if not self.path.exists():
            return _default_state()
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        return FirstRunState.model_validate(payload)

    def save(self, state: FirstRunState) -> FirstRunState:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        state.updated_at = _now()
        self.path.write_text(
            state.model_dump_json(indent=2),
            encoding="utf-8",
        )
        return state

    def update_step(self, step: str, data: dict[str, Any] | None = None) -> FirstRunState:
        state = self.load()
        if state.status == FirstRunStatus.NOT_STARTED:
            state.status = FirstRunStatus.IN_PROGRESS
        state.current_step = step
        if data is not None:
            state.step_data[step] = data
            if data.get("acknowledged") is True and step not in state.acknowledged_steps:
                state.acknowledged_steps.append(step)
        if step not in state.completed_steps:
            state.completed_steps.append(step)
        return self.save(state)

    def record_first_chat_success(self, *, provider: str, model: str, response_id: str | None) -> FirstRunState:
        state = self.load()
        state.status = FirstRunStatus.FIRST_CHAT_COMPLETE
        state.first_chat = FirstRunChatResult(
            completed=True,
            provider=provider,
            model=model,
            response_id=response_id,
            completed_at=_now(),
        )
        if "first_chat" not in state.completed_steps:
            state.completed_steps.append("first_chat")
        return self.save(state)

    def mark_completed(self) -> FirstRunState:
        state = self.load()
        if not state.first_chat.completed:
            raise InvalidFirstRunTransition("first_chat_required")
        missing_steps = [
            step for step in REQUIRED_FIRST_RUN_STEPS
            if step not in state.acknowledged_steps
        ]
        if missing_steps:
            raise InvalidFirstRunTransition(
                "required_steps_missing:" + ",".join(missing_steps)
            )
        state.status = FirstRunStatus.COMPLETED
        state.completed_at = _now()
        return self.save(state)

    def mark_skipped(self, *, reason: str | None = None) -> FirstRunState:
        state = self.load()
        state.status = FirstRunStatus.SKIPPED
        state.skip_reason = reason
        return self.save(state)
```

If the repo still supports Pydantic v1 in this path, use the local compatibility helpers instead of `model_validate` and `model_dump_json`.

- [ ] **Step 5: Run state-store tests**

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Setup/test_first_run_state.py -v
```

Expected: PASS.

- [ ] **Step 6: Run setup schema smoke tests**

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Setup/test_setup_manager_masking.py -v
```

Expected: PASS or unrelated pre-existing failure recorded in Backlog.

- [ ] **Step 7: Commit backend state store**

```bash
git add tldw_Server_API/app/core/Setup/first_run_state.py tldw_Server_API/app/api/v1/schemas/setup_schemas.py tldw_Server_API/tests/Setup/test_first_run_state.py
git commit -m "feat: add first-run setup state store"
```

### Task 2: Setup Access Boundary And First-Run State Endpoints

**Files:**
- Modify: `tldw_Server_API/app/api/v1/API_Deps/setup_deps.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/setup.py`
- Modify: `tldw_Server_API/app/api/v1/schemas/setup_schemas.py`
- Test: `tldw_Server_API/tests/Setup/test_setup_deps_remote_admin.py`
- Create: `tldw_Server_API/tests/integration/test_unified_first_run_setup_api.py`

- [ ] **Step 1: Add failing setup access-boundary tests**

Append to `tldw_Server_API/tests/Setup/test_setup_deps_remote_admin.py`:

```python
def _make_remote_post_request(path="/api/v1/setup/first-run/state") -> Request:
    scope = {
        "type": "http",
        "method": "POST",
        "path": path,
        "headers": [(b"host", b"example.test")],
        "client": ("203.0.113.10", 4444),
    }
    return Request(scope)


@pytest.mark.asyncio
async def test_remote_setup_write_rejected_when_remote_setup_disabled(monkeypatch):
    monkeypatch.delenv("TLDW_SETUP_ALLOW_REMOTE", raising=False)
    monkeypatch.setattr(setup_deps, "_config_allows_remote", lambda: False)

    with pytest.raises(HTTPException) as excinfo:
        await setup_deps.require_local_setup_access(_make_remote_post_request())

    assert excinfo.value.status_code == 403
    assert "localhost" in str(excinfo.value.detail).lower()


@pytest.mark.asyncio
async def test_remote_setup_write_requires_admin_guard_when_remote_override_enabled(monkeypatch):
    called = {"value": False}

    async def fake_guard(_request):
        called["value"] = True

    monkeypatch.setenv("TLDW_SETUP_ALLOW_REMOTE", "1")
    monkeypatch.setattr(setup_deps, "_config_allows_remote", lambda: False)
    monkeypatch.setattr(setup_deps, "_require_admin_for_remote", fake_guard)

    await setup_deps.require_local_setup_access(_make_remote_post_request())

    assert called["value"] is True
```

- [ ] **Step 2: Add failing first-run API integration tests**

Create `tldw_Server_API/tests/integration/test_unified_first_run_setup_api.py`:

```python
from fastapi.testclient import TestClient

from tldw_Server_API.app.api.v1.endpoints import setup as setup_endpoint
from tldw_Server_API.app.core.Setup.first_run_state import (
    REQUIRED_FIRST_RUN_STEPS,
    FirstRunStateStore,
)
from tldw_Server_API.app.main import app


def _setup_needs_setup(monkeypatch):
    monkeypatch.setattr(
        setup_endpoint.setup_manager,
        "get_status_snapshot",
        lambda: {
            "enabled": True,
            "setup_completed": False,
            "needs_setup": True,
            "auth_mode": "single_user",
        },
    )


def test_first_run_state_endpoint_returns_backend_state(monkeypatch, tmp_path):
    state_path = tmp_path / "first_run_state.json"
    monkeypatch.setattr(setup_endpoint, "FIRST_RUN_STATE_PATH", state_path, raising=False)

    with TestClient(app) as client:
        response = client.get("/api/v1/setup/first-run/state")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "not_started"
    assert body["first_chat"]["completed"] is False


def test_first_run_skip_endpoint_records_skipped(monkeypatch, tmp_path):
    state_path = tmp_path / "first_run_state.json"
    monkeypatch.setattr(setup_endpoint, "FIRST_RUN_STATE_PATH", state_path, raising=False)
    _setup_needs_setup(monkeypatch)

    with TestClient(app) as client:
        response = client.post("/api/v1/setup/first-run/skip", json={"reason": "user_skip"})

    assert response.status_code == 200
    assert response.json()["status"] == "skipped"


def test_first_run_metadata_returns_auth_and_setup_path_guidance(monkeypatch, tmp_path):
    state_path = tmp_path / "first_run_state.json"
    monkeypatch.setattr(setup_endpoint, "FIRST_RUN_STATE_PATH", state_path, raising=False)

    with TestClient(app) as client:
        response = client.get("/api/v1/setup/first-run/metadata")

    assert response.status_code == 200
    body = response.json()
    assert "auth_mode" in body
    assert "manual_auth_required" in body
    assert "bundled_single_user_auth_available" in body
    assert "frontend_origin" in body["connection"]
    assert "api_origin" in body["connection"]
    assert body["connection"]["browser_access"] in {"local", "lan", "remote", "unknown"}
    assert {path["key"] for path in body["setup_paths"]} >= {"docker_single_user", "local_single_user", "multi_user"}
    assert body["multi_user_exit"]["guide_path"]


def test_completed_setup_rejects_first_run_writes(monkeypatch, tmp_path):
    state_path = tmp_path / "first_run_state.json"
    monkeypatch.setattr(setup_endpoint, "FIRST_RUN_STATE_PATH", state_path, raising=False)
    _setup_needs_setup(monkeypatch)
    store = FirstRunStateStore(state_path)
    for step in REQUIRED_FIRST_RUN_STEPS:
        store.update_step(step, {"acknowledged": True})
    store.record_first_chat_success(
        provider="openai",
        model="gpt-4.1-mini",
        response_id="chatcmpl-test",
    )
    store.mark_completed()

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/setup/first-run/state",
            json={"step": "providers", "data": {}},
        )

    assert response.status_code == 409
    assert response.json()["detail"] == "setup_already_completed"


def test_legacy_completed_setup_rejects_first_run_writes_without_state_file(monkeypatch, tmp_path):
    state_path = tmp_path / "first_run_state.json"
    monkeypatch.setattr(setup_endpoint, "FIRST_RUN_STATE_PATH", state_path, raising=False)
    monkeypatch.setattr(
        setup_endpoint.setup_manager,
        "get_status_snapshot",
        lambda: {
            "enabled": True,
            "setup_completed": True,
            "needs_setup": False,
            "auth_mode": "single_user",
        },
    )

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/setup/first-run/state",
            json={"step": "providers", "data": {}},
        )

    assert response.status_code == 409
    assert response.json()["detail"] == "setup_already_completed"
```

- [ ] **Step 3: Run tests to verify failure**

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Setup/test_setup_deps_remote_admin.py tldw_Server_API/tests/integration/test_unified_first_run_setup_api.py -v
```

Expected: first-run endpoint tests FAIL because endpoints do not exist yet. Boundary tests may pass if current behavior already satisfies them; record result.

- [ ] **Step 4: Add request/response schemas**

In `setup_schemas.py`, add:

```python
class FirstRunStepUpdateRequest(BaseModel):
    step: str = Field(..., min_length=1)
    data: dict[str, Any] = Field(default_factory=dict)


class FirstRunSkipRequest(BaseModel):
    reason: str | None = Field(None, max_length=120)


class FirstRunSetupPath(BaseModel):
    key: str
    label: str
    recommended: bool = False
    guide_path: str | None = None


class FirstRunMultiUserExit(BaseModel):
    guide_path: str
    checklist_path: str | None = None


class FirstRunConnectionDiagnostics(BaseModel):
    frontend_origin: str | None = None
    api_origin: str | None = None
    browser_access: str | None = None  # local, lan, remote, unknown


class FirstRunMetadataResponse(BaseModel):
    auth_mode: str
    bundled_single_user_auth_available: bool
    manual_auth_required: bool
    setup_required: bool
    setup_completed: bool
    remote_setup_enabled: bool
    connection: FirstRunConnectionDiagnostics
    setup_paths: list[FirstRunSetupPath]
    multi_user_exit: FirstRunMultiUserExit
```

- [ ] **Step 5: Add first-run state endpoints**

In `setup.py`, import `FirstRunStateStore`, `InvalidFirstRunTransition`, and the new schemas. Add a module constant near the top:

```python
FIRST_RUN_STATE_PATH = setup_manager.resolve_config_root() / "first_run_state.json"


def _first_run_store() -> FirstRunStateStore:
    return FirstRunStateStore(FIRST_RUN_STATE_PATH)


async def _require_first_run_write_access(request: Request) -> None:
    await require_local_setup_access(request)
    status_snapshot = setup_manager.get_status_snapshot()
    if not status_snapshot.get("enabled"):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="setup_disabled",
        )
    if (
        status_snapshot.get("setup_completed")
        or status_snapshot.get("completed")
        or not status_snapshot.get("needs_setup")
    ):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="setup_already_completed",
        )
    state = _first_run_store().load()
    if state.status == FirstRunStatus.COMPLETED:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="setup_already_completed",
        )
```

If `resolve_config_root` is not exposed by `setup_manager`, add a focused helper there or use the existing config root utility already imported by `setup_manager`.

Add endpoints:

```python
@router.get("/first-run/state", openapi_extra={"security": []}, response_model=FirstRunStateResponse)
async def get_first_run_state(_guard: None = Depends(require_local_setup_access)) -> FirstRunStateResponse:
    return _first_run_store().load()


@router.get("/first-run/metadata", openapi_extra={"security": []}, response_model=FirstRunMetadataResponse)
async def get_first_run_metadata(
    request: Request,
    _guard: None = Depends(require_local_setup_access),
) -> FirstRunMetadataResponse:
    return build_first_run_metadata(request)


@router.post("/first-run/state", openapi_extra={"security": []}, response_model=FirstRunStateResponse)
async def update_first_run_state(
    payload: FirstRunStepUpdateRequest,
    _guard: None = Depends(_require_first_run_write_access),
) -> FirstRunStateResponse:
    return _first_run_store().update_step(payload.step, payload.data)


@router.post("/first-run/skip", openapi_extra={"security": []}, response_model=FirstRunStateResponse)
async def skip_first_run(
    payload: FirstRunSkipRequest,
    _guard: None = Depends(_require_first_run_write_access),
) -> FirstRunStateResponse:
    return _first_run_store().mark_skipped(reason=payload.reason)
```

`build_first_run_metadata(request)` should derive, without exposing secrets:

- auth mode;
- whether bundled single-user WebUI/API auth can be handled automatically;
- whether manual API-key entry is required;
- whether setup is required or already completed;
- whether remote setup is enabled;
- frontend origin, API origin, and local/LAN/remote browser classification;
- Docker single-user, local single-user, and multi-user setup path metadata;
- multi-user guide/checklist links.

Every later first-run write endpoint in this plan must use `_require_first_run_write_access`, not `require_local_setup_access` directly. After setup completion, first-run setup write endpoints return `409 setup_already_completed`; post-completion config changes must go through authenticated admin/settings endpoints with system-configure permission.

- [ ] **Step 6: Keep access failures plain-language**

If boundary tests reveal vague setup guard detail, update `setup_deps.require_local_setup_access()` so remote disallowed writes fail with a detail like:

```python
raise HTTPException(
    status.HTTP_403_FORBIDDEN,
    detail="Setup writes are only available from localhost unless remote setup access is explicitly enabled.",
)
```

Do not expose client IPs or raw proxy internals in the default detail.

- [ ] **Step 7: Run backend setup tests**

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Setup/test_setup_deps_remote_admin.py tldw_Server_API/tests/integration/test_unified_first_run_setup_api.py -v
```

Expected: PASS.

- [ ] **Step 8: Commit access/state API slice**

```bash
git add tldw_Server_API/app/api/v1/API_Deps/setup_deps.py tldw_Server_API/app/api/v1/endpoints/setup.py tldw_Server_API/app/api/v1/schemas/setup_schemas.py tldw_Server_API/tests/Setup/test_setup_deps_remote_admin.py tldw_Server_API/tests/integration/test_unified_first_run_setup_api.py
git commit -m "feat: expose first-run setup state"
```

### Task 3: Provider Catalog, Config Writes, And Validation

**Files:**
- Create: `tldw_Server_API/app/core/Setup/provider_catalog.py`
- Create: `tldw_Server_API/app/core/Setup/provider_validation.py`
- Modify: `tldw_Server_API/app/api/v1/schemas/setup_schemas.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/setup.py`
- Test: `tldw_Server_API/tests/Setup/test_setup_provider_catalog.py`
- Test: `tldw_Server_API/tests/Setup/test_setup_provider_validation.py`
- Test: `tldw_Server_API/tests/integration/test_unified_first_run_setup_api.py`
- Modify: `tldw_Server_API/tests/Config/test_config_providers_endpoints.py` only if shared helpers move.

- [ ] **Step 1: Write failing provider catalog tests**

Create `tldw_Server_API/tests/Setup/test_setup_provider_catalog.py`:

```python
from tldw_Server_API.app.api.v1.schemas.setup_schemas import (
    SetupProviderSaveResponse,
    SetupProviderSaveStatus,
)
from tldw_Server_API.app.core.Setup.provider_catalog import (
    REQUIRED_SETUP_PROVIDER_KEYS,
    get_setup_provider_catalog,
    mask_secret,
)


def test_catalog_covers_required_prd_provider_keys():
    catalog = get_setup_provider_catalog()
    keys = {entry.provider_key for entry in catalog.providers}

    assert set(REQUIRED_SETUP_PROVIDER_KEYS) <= keys


def test_catalog_marks_local_providers_as_endpoint_based():
    catalog = get_setup_provider_catalog()
    providers = {entry.provider_key: entry for entry in catalog.providers}

    assert providers["ollama"].provider_type == "local_endpoint"
    assert providers["llamacpp"].provider_type == "local_endpoint"
    assert providers["custom_openai"].provider_type == "local_endpoint"


def test_mask_secret_never_returns_raw_value():
    assert mask_secret("sk-abcdefghijklmnopqrstuvwxyz") == "sk-...wxyz"
    assert mask_secret("tiny") == "****ny"
    assert mask_secret("") == ""


def test_provider_save_response_contract_masks_secret_and_uses_saved_status():
    response = SetupProviderSaveResponse(
        provider_key="openai",
        status=SetupProviderSaveStatus.SAVED,
        masked_api_key=mask_secret("sk-abcdefghijklmnopqrstuvwxyz"),
        make_default=True,
    )

    assert response.status == SetupProviderSaveStatus.SAVED
    assert response.masked_api_key == "sk-...wxyz"
```

- [ ] **Step 2: Write failing provider validation tests**

Create `tldw_Server_API/tests/Setup/test_setup_provider_validation.py`:

```python
import pytest

from tldw_Server_API.app.core.Setup.provider_validation import (
    LocalEndpointValidationRequest,
    validate_local_openai_endpoint,
)


@pytest.mark.asyncio
async def test_local_endpoint_validation_reports_unreachable(monkeypatch):
    async def fake_get(*_args, **_kwargs):
        raise TimeoutError("connect timed out")

    monkeypatch.setattr("httpx.AsyncClient.get", fake_get)

    result = await validate_local_openai_endpoint(
        LocalEndpointValidationRequest(
            provider_key="custom_openai",
            base_url="http://127.0.0.1:65534/v1",
            model="local-model",
        )
    )

    assert result.status == "failed"
    assert result.failure_category == "local_provider_unreachable"


@pytest.mark.asyncio
async def test_local_endpoint_validation_accepts_openai_models_shape(monkeypatch):
    class Response:
        status_code = 200

        def json(self):
            return {"data": [{"id": "local-model"}]}

    async def fake_get(*_args, **_kwargs):
        return Response()

    monkeypatch.setattr("httpx.AsyncClient.get", fake_get)

    result = await validate_local_openai_endpoint(
        LocalEndpointValidationRequest(
            provider_key="custom_openai",
            base_url="http://127.0.0.1:8001/v1",
            model="local-model",
        )
    )

    assert result.status == "ready"
    assert result.models == ["local-model"]
```

- [ ] **Step 3: Run tests to verify failure**

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Setup/test_setup_provider_catalog.py tldw_Server_API/tests/Setup/test_setup_provider_validation.py -v
```

Expected: FAIL because new modules do not exist.

- [ ] **Step 4: Add provider save endpoint contract test**

Append to `tldw_Server_API/tests/integration/test_unified_first_run_setup_api.py`:

```python
def test_provider_save_endpoint_masks_secret_and_returns_saved_status(monkeypatch, tmp_path):
    state_path = tmp_path / "first_run_state.json"
    monkeypatch.setattr(setup_endpoint, "FIRST_RUN_STATE_PATH", state_path, raising=False)
    _setup_needs_setup(monkeypatch)
    saved_updates = {}

    def fake_update_config(updates, **_kwargs):
        saved_updates.update(updates)

    monkeypatch.setattr(setup_endpoint.setup_manager, "update_config", fake_update_config)

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/setup/first-run/providers",
            json={
                "provider_key": "openai",
                "api_key": "sk-abcdefghijklmnopqrstuvwxyz",
                "model": "gpt-4.1-mini",
                "make_default": True,
            },
        )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "saved"
    assert body["masked_api_key"] == "sk-...wxyz"
    assert "sk-abcdefghijklmnopqrstuvwxyz" not in str(body)
    assert saved_updates
```

- [ ] **Step 5: Add provider schemas**

Add to `setup_schemas.py`:

```python
class SetupProviderType(str, Enum):
    HOSTED_API_KEY = "hosted_api_key"
    LOCAL_ENDPOINT = "local_endpoint"


class SetupProviderSaveStatus(str, Enum):
    SAVED = "saved"
    FAILED = "failed"


class SetupProviderCatalogEntry(BaseModel):
    provider_key: str
    label: str
    provider_type: SetupProviderType
    config_section: str
    api_key_field: str | None = None
    base_url_field: str | None = None
    model_field: str | None = None
    default_base_url: str | None = None
    supports_preflight: bool = False
    recommended_for_first_chat: bool = False


class SetupProviderCatalogResponse(BaseModel):
    providers: list[SetupProviderCatalogEntry]


class SetupProviderSaveRequest(BaseModel):
    provider_key: str
    api_key: str | None = None
    base_url: str | None = None
    model: str | None = None
    make_default: bool = False


class SetupProviderSaveResponse(BaseModel):
    provider_key: str
    status: SetupProviderSaveStatus
    masked_api_key: str | None = None
    base_url: str | None = None
    model: str | None = None
    make_default: bool = False
    requires_restart: bool = False
    failure_category: str | None = None
    message: str | None = None


class SetupProviderValidationResponse(BaseModel):
    provider_key: str
    status: str
    failure_category: str | None = None
    message: str | None = None
    models: list[str] = Field(default_factory=list)
```

- [ ] **Step 6: Implement provider catalog**

Create `provider_catalog.py` with a typed list. Use config keys that match current `config.txt` and `config_info.py` provider behavior. Include these keys at minimum:

```python
REQUIRED_SETUP_PROVIDER_KEYS = (
    "openai",
    "anthropic",
    "cohere",
    "deepseek",
    "google",
    "groq",
    "huggingface",
    "mistral",
    "openrouter",
    "qwen",
    "moonshot",
    "zai",
    "ollama",
    "llamacpp",
    "koboldcpp",
    "oobabooga",
    "tabbyapi",
    "vllm",
    "aphrodite",
    "custom_openai",
)
```

Prefer a backend catalog because the implementation-plan reviewer already called out avoiding duplicated UI config.

- [ ] **Step 7: Implement provider validation helpers**

Create `provider_validation.py`:

```python
from __future__ import annotations

from pydantic import BaseModel, Field
import httpx

from tldw_Server_API.app.api.v1.schemas.setup_schemas import SetupProviderValidationResponse


class LocalEndpointValidationRequest(BaseModel):
    provider_key: str
    base_url: str
    model: str | None = None
    api_key: str | None = None


async def validate_local_openai_endpoint(payload: LocalEndpointValidationRequest) -> SetupProviderValidationResponse:
    url = payload.base_url.rstrip("/") + "/models"
    headers = {"Authorization": f"Bearer {payload.api_key}"} if payload.api_key else None
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(url, headers=headers)
    except Exception:
        return SetupProviderValidationResponse(
            provider_key=payload.provider_key,
            status="failed",
            failure_category="local_provider_unreachable",
        )
    if response.status_code in {401, 403}:
        return SetupProviderValidationResponse(
            provider_key=payload.provider_key,
            status="failed",
            failure_category="auth_failed",
        )
    if response.status_code >= 400:
        return SetupProviderValidationResponse(
            provider_key=payload.provider_key,
            status="failed",
            failure_category="unsupported_api_shape",
        )
    try:
        body = response.json()
    except ValueError:
        return SetupProviderValidationResponse(
            provider_key=payload.provider_key,
            status="failed",
            failure_category="unsupported_api_shape",
        )
    models = [
        str(item.get("id"))
        for item in body.get("data", [])
        if isinstance(item, dict) and item.get("id")
    ]
    return SetupProviderValidationResponse(
        provider_key=payload.provider_key,
        status="ready",
        models=models,
    )
```

Add hosted validation later through existing provider validation helpers where available; keep first-chat verification decisive.

- [ ] **Step 8: Add provider endpoints**

In `setup.py`, add:

- `GET /api/v1/setup/first-run/providers/catalog`
- `POST /api/v1/setup/first-run/providers`
- `POST /api/v1/setup/first-run/providers/validate`

The `GET` catalog endpoint can use `require_local_setup_access`. Both `POST` endpoints must use `_require_first_run_write_access` from Task 2 so provider secrets and local endpoints cannot be written through the first-run trust path after setup completion.

Provider save endpoint should translate `SetupProviderSaveRequest` into `setup_manager.update_config()` updates and return `SetupProviderSaveResponse`; it must never return raw secrets.

Expected save behavior:

```python
updates = {
    entry.config_section: {
        entry.api_key_field: payload.api_key,
        entry.base_url_field: payload.base_url,
        entry.model_field: payload.model,
    }
}
```

Filter `None` values before writing.

- [ ] **Step 9: Run provider tests**

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Setup/test_setup_provider_catalog.py tldw_Server_API/tests/Setup/test_setup_provider_validation.py tldw_Server_API/tests/integration/test_unified_first_run_setup_api.py tldw_Server_API/tests/Config/test_config_providers_endpoints.py -v
```

Expected: PASS.

- [ ] **Step 10: Commit provider setup backend**

```bash
git add tldw_Server_API/app/core/Setup/provider_catalog.py tldw_Server_API/app/core/Setup/provider_validation.py tldw_Server_API/app/api/v1/schemas/setup_schemas.py tldw_Server_API/app/api/v1/endpoints/setup.py tldw_Server_API/tests/Setup/test_setup_provider_catalog.py tldw_Server_API/tests/Setup/test_setup_provider_validation.py tldw_Server_API/tests/integration/test_unified_first_run_setup_api.py tldw_Server_API/tests/Config/test_config_providers_endpoints.py
git commit -m "feat: add setup provider catalog and validation"
```

### Task 4: First-Chat Verification And Completion Gate

**Files:**
- Create: `tldw_Server_API/app/core/Setup/first_chat_verifier.py`
- Modify: `tldw_Server_API/app/api/v1/schemas/setup_schemas.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/setup.py`
- Test: `tldw_Server_API/tests/Setup/test_setup_first_chat_completion.py`

- [ ] **Step 1: Write failing first-chat completion tests**

Create `tldw_Server_API/tests/Setup/test_setup_first_chat_completion.py`:

```python
import pytest

from tldw_Server_API.app.core.Setup.first_chat_verifier import (
    FirstChatVerificationRequest,
    verify_first_chat,
)


@pytest.mark.asyncio
async def test_first_chat_verification_records_success(monkeypatch, tmp_path):
    async def fake_chat_completion(payload):
        return {"id": "chatcmpl-test", "choices": [{"message": {"content": "Hello."}}]}

    monkeypatch.setattr(
        "tldw_Server_API.app.core.Setup.first_chat_verifier._call_chat_completion",
        fake_chat_completion,
    )

    result = await verify_first_chat(
        FirstChatVerificationRequest(
            provider="openai",
            model="gpt-4.1-mini",
            prompt="Say hello",
        )
    )

    assert result.status == "ready"
    assert result.response_text == "Hello."
    assert result.response_id == "chatcmpl-test"


@pytest.mark.asyncio
async def test_first_chat_verification_maps_provider_failure(monkeypatch):
    async def fake_chat_completion(payload):
        raise RuntimeError("401 invalid api key at /private/config.txt")

    monkeypatch.setattr(
        "tldw_Server_API.app.core.Setup.first_chat_verifier._call_chat_completion",
        fake_chat_completion,
    )

    result = await verify_first_chat(
        FirstChatVerificationRequest(
            provider="openai",
            model="gpt-4.1-mini",
            prompt="Say hello",
        )
    )

    assert result.status == "failed"
    assert result.failure_category in {"provider_key_invalid", "provider_unavailable"}
    assert "/private" not in (result.message or "")
```

- [ ] **Step 2: Run tests to verify failure**

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Setup/test_setup_first_chat_completion.py -v
```

Expected: FAIL because verifier does not exist.

- [ ] **Step 3: Add first-chat schemas**

In `setup_schemas.py`, add:

```python
class FirstChatVerifyRequest(BaseModel):
    provider: str = Field(..., min_length=1)
    model: str = Field(..., min_length=1)
    prompt: str = Field("Say hello in one short sentence.", min_length=1, max_length=1000)


class FirstChatVerifyResponse(BaseModel):
    status: str
    provider: str
    model: str
    response_id: str | None = None
    response_text: str | None = None
    failure_category: str | None = None
    message: str | None = None


class FirstRunCompleteRequest(BaseModel):
    acknowledged_steps: list[str] = Field(default_factory=list)
```

- [ ] **Step 4: Implement first-chat verifier**

Create `first_chat_verifier.py`. Use a thin wrapper so tests can patch `_call_chat_completion`; then wire `_call_chat_completion` to the existing chat completion service or endpoint helper. If no clean service seam exists, add one in the smallest existing chat module rather than importing FastAPI endpoint internals directly.

Expose a concrete request model from this module so service tests and endpoints use one type:

```python
class FirstChatVerificationRequest(FirstChatVerifyRequest):
    """Internal first-chat verification request used by setup services."""
```

Minimum shape:

```python
async def verify_first_chat(payload: FirstChatVerificationRequest) -> FirstChatVerifyResponse:
    try:
        response = await _call_chat_completion(payload)
    except Exception:
        return FirstChatVerifyResponse(
            status="failed",
            provider=payload.provider,
            model=payload.model,
            failure_category="provider_unavailable",
            message="First chat failed. Check provider credentials and model availability.",
        )
    text = _extract_response_text(response)
    return FirstChatVerifyResponse(
        status="ready",
        provider=payload.provider,
        model=payload.model,
        response_id=str(response.get("id") or "") or None,
        response_text=text,
    )
```

- [ ] **Step 5: Add verify and complete endpoints**

In `setup.py`, add:

- `POST /api/v1/setup/first-run/first-chat`
- `POST /api/v1/setup/first-run/complete`

Both endpoints must use `_require_first_run_write_access` from Task 2.

The first-chat endpoint:

1. calls `verify_first_chat`;
2. if `status == "ready"`, calls `FirstRunStateStore.record_first_chat_success`;
3. returns `FirstChatVerifyResponse`.

The completion endpoint:

1. accepts `FirstRunCompleteRequest`;
2. records any acknowledged required steps from `payload.acknowledged_steps`;
3. calls `FirstRunStateStore.mark_completed`;
4. maps `InvalidFirstRunTransition("first_chat_required")` to HTTP 409;
5. maps `InvalidFirstRunTransition("required_steps_missing:...")` to HTTP 409 with the missing step ids;
6. optionally calls existing `setup_manager.mark_setup_completed(True)` only after first-chat success and required screen acknowledgements.

Minimum endpoint shape:

```python
@router.post("/first-run/complete", openapi_extra={"security": []}, response_model=FirstRunStateResponse)
async def complete_first_run(
    payload: FirstRunCompleteRequest,
    _guard: None = Depends(_require_first_run_write_access),
) -> FirstRunStateResponse:
    store = _first_run_store()
    for step in payload.acknowledged_steps:
        if step in REQUIRED_FIRST_RUN_STEPS:
            store.update_step(step, {"acknowledged": True})
    try:
        state = store.mark_completed()
    except InvalidFirstRunTransition as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    setup_manager.mark_setup_completed(True)
    return state
```

- [ ] **Step 6: Add endpoint tests for invalid completion**

Extend `test_unified_first_run_setup_api.py`:

```python
def test_complete_rejects_without_first_chat(monkeypatch, tmp_path):
    state_path = tmp_path / "first_run_state.json"
    monkeypatch.setattr(setup_endpoint, "FIRST_RUN_STATE_PATH", state_path, raising=False)
    _setup_needs_setup(monkeypatch)

    with TestClient(app) as client:
        response = client.post("/api/v1/setup/first-run/complete", json={})

    assert response.status_code == 409
    assert response.json()["detail"] == "first_chat_required"


def test_complete_rejects_without_required_screen_acknowledgements(monkeypatch, tmp_path):
    state_path = tmp_path / "first_run_state.json"
    monkeypatch.setattr(setup_endpoint, "FIRST_RUN_STATE_PATH", state_path, raising=False)
    _setup_needs_setup(monkeypatch)
    store = FirstRunStateStore(state_path)
    store.record_first_chat_success(
        provider="openai",
        model="gpt-4.1-mini",
        response_id="chatcmpl-test",
    )

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/setup/first-run/complete",
            json={"acknowledged_steps": ["setup_path"]},
        )

    assert response.status_code == 409
    assert response.json()["detail"].startswith("required_steps_missing")
```

- [ ] **Step 7: Run first-chat tests**

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Setup/test_setup_first_chat_completion.py tldw_Server_API/tests/integration/test_unified_first_run_setup_api.py -v
```

Expected: PASS.

- [ ] **Step 8: Commit first-chat completion gate**

```bash
git add tldw_Server_API/app/core/Setup/first_chat_verifier.py tldw_Server_API/app/api/v1/schemas/setup_schemas.py tldw_Server_API/app/api/v1/endpoints/setup.py tldw_Server_API/tests/Setup/test_setup_first_chat_completion.py tldw_Server_API/tests/integration/test_unified_first_run_setup_api.py
git commit -m "feat: require first chat for setup completion"
```

### Task 5: Ingest Defaults, Audio Defaults, And Optional Advanced API Shape

**Files:**
- Modify: `tldw_Server_API/app/api/v1/schemas/setup_schemas.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/setup.py`
- Modify: `tldw_Server_API/app/core/Setup/setup_manager.py` only for field hints/allowlist support.
- Test: `tldw_Server_API/tests/integration/test_unified_first_run_setup_api.py`
- Test: `tldw_Server_API/tests/Setup/test_setup_manager_user_db_base_dir_validation.py`

- [ ] **Step 1: Add failing tests for non-provider wizard saves**

Append tests:

```python
def test_ingest_defaults_save_returns_restart_state(monkeypatch, tmp_path):
    monkeypatch.setattr(setup_endpoint, "FIRST_RUN_STATE_PATH", tmp_path / "first_run_state.json", raising=False)
    _setup_needs_setup(monkeypatch)

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/setup/first-run/ingest-defaults",
            json={
                "allow_local_file_ingest": False,
                "chunking_profile": "balanced",
                "metadata_mode": "automatic",
            },
        )

    assert response.status_code == 200
    assert response.json()["step"] == "ingest_defaults"


def test_optional_advanced_can_be_deferred(monkeypatch, tmp_path):
    monkeypatch.setattr(setup_endpoint, "FIRST_RUN_STATE_PATH", tmp_path / "first_run_state.json", raising=False)
    _setup_needs_setup(monkeypatch)

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/setup/first-run/optional-advanced",
            json={"rag": "defer", "storage_paths": "defer"},
        )

    assert response.status_code == 200
    assert response.json()["status"] in {"in_progress", "not_started"}
```

- [ ] **Step 2: Run tests to verify failure**

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/integration/test_unified_first_run_setup_api.py -v
```

Expected: FAIL on missing endpoints.

- [ ] **Step 3: Add schemas for first-run settings**

Add schemas:

```python
class IngestDefaultsRequest(BaseModel):
    allow_local_file_ingest: bool = False
    chunking_profile: str = "balanced"
    metadata_mode: str = "automatic"
    allowed_local_roots: list[str] = Field(default_factory=list)


class AudioDefaultsRequest(BaseModel):
    mode: str = Field("skip", pattern="^(defaults|configure|skip)$")
    stt_provider: str | None = None
    tts_provider: str | None = None
    tts_voice: str | None = None


class OptionalAdvancedRequest(BaseModel):
    rag: str = Field("defer", pattern="^(configure|skip|defer)$")
    storage_paths: str = Field("defer", pattern="^(configure|skip|defer)$")
    values: dict[str, Any] = Field(default_factory=dict)


class FirstRunStepSaveResponse(BaseModel):
    status: FirstRunStatus
    step: str
    requires_restart: bool = False
```

- [ ] **Step 4: Add endpoints that save state and safe config only**

Add endpoints:

- `POST /api/v1/setup/first-run/ingest-defaults`
- `POST /api/v1/setup/first-run/audio-defaults`
- `POST /api/v1/setup/first-run/optional-advanced`

Implementation rule:

- Use `_require_first_run_write_access` from Task 2 for all three endpoints.
- Save all submitted choices to `FirstRunStateStore.update_step`.
- Write only config keys already supported and safe.
- If a field maps to a risky path setting, validate through existing setup-manager path validation helpers.
- Return `requires_restart=True` only when config writes happened.

- [ ] **Step 5: Reuse existing audio recommendations**

Do not duplicate audio recommendation logic. The frontend wizard should call existing:

- `GET /api/v1/setup/audio/recommendations`
- `GET /api/v1/setup/audio/readiness`
- `POST /api/v1/setup/audio/verify`

This task only adds lightweight defaults/defer state.

- [ ] **Step 6: Run setup API tests**

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/integration/test_unified_first_run_setup_api.py tldw_Server_API/tests/Setup/test_setup_manager_user_db_base_dir_validation.py -v
```

Expected: PASS.

- [ ] **Step 7: Commit non-provider wizard API shape**

```bash
git add tldw_Server_API/app/api/v1/schemas/setup_schemas.py tldw_Server_API/app/api/v1/endpoints/setup.py tldw_Server_API/app/core/Setup/setup_manager.py tldw_Server_API/tests/integration/test_unified_first_run_setup_api.py tldw_Server_API/tests/Setup/test_setup_manager_user_db_base_dir_validation.py
git commit -m "feat: add first-run settings endpoints"
```

### Task 6: Frontend Setup API Domain And Hook

**Files:**
- Create: `apps/packages/ui/src/types/setup-onboarding.ts`
- Create: `apps/packages/ui/src/services/tldw/domains/setup-onboarding.ts`
- Modify: `apps/packages/ui/src/services/tldw/domains/index.ts`
- Modify: `apps/packages/ui/src/services/tldw/TldwApiClient.ts`
- Modify: `apps/packages/ui/src/services/tldw/openapi-guard.ts`
- Create: `apps/packages/ui/src/hooks/useSetupOnboarding.ts`
- Test: `apps/packages/ui/src/services/tldw/__tests__/setup-onboarding.test.ts`
- Test: `apps/packages/ui/src/hooks/__tests__/useSetupOnboarding.test.tsx`

- [x] **Step 1: Add failing setup API client tests**

Create `apps/packages/ui/src/services/tldw/__tests__/setup-onboarding.test.ts`:

```ts
import { describe, expect, it, vi } from "vitest"

vi.mock("@/services/background-proxy", () => ({
  bgRequest: vi.fn()
}))

describe("setup onboarding API domain", () => {
  it("fetches first-run state from setup endpoint", async () => {
    const { bgRequest } = await import("@/services/background-proxy")
    vi.mocked(bgRequest).mockResolvedValueOnce({ status: "not_started" })
    const { setupOnboardingMethods } = await import("../domains/setup-onboarding")

    const result = await setupOnboardingMethods.getFirstRunState.call({})

    expect(result.status).toBe("not_started")
    expect(bgRequest).toHaveBeenCalledWith({
      path: "/api/v1/setup/first-run/state",
      method: "GET"
    })
  })

  it("fetches setup metadata for auth and setup path decisions", async () => {
    const { bgRequest } = await import("@/services/background-proxy")
    vi.mocked(bgRequest).mockResolvedValueOnce({
      auth_mode: "single_user",
      bundled_single_user_auth_available: true,
      manual_auth_required: false,
      setup_required: true,
      setup_completed: false,
      remote_setup_enabled: false,
      connection: {
        frontend_origin: "http://127.0.0.1:3000",
        api_origin: "http://127.0.0.1:8000",
        browser_access: "local"
      },
      setup_paths: [],
      multi_user_exit: { guide_path: "/docs/multi-user" }
    })
    const { setupOnboardingMethods } = await import("../domains/setup-onboarding")

    const result = await setupOnboardingMethods.getFirstRunMetadata.call({})

    expect(result.manual_auth_required).toBe(false)
    expect(bgRequest).toHaveBeenCalledWith({
      path: "/api/v1/setup/first-run/metadata",
      method: "GET"
    })
  })

  it("saves provider setup without leaking raw secret into return shape", async () => {
    const { bgRequest } = await import("@/services/background-proxy")
    vi.mocked(bgRequest).mockResolvedValueOnce({
      provider_key: "openai",
      status: "saved",
      masked_api_key: "sk-...abcd"
    })
    const { setupOnboardingMethods } = await import("../domains/setup-onboarding")

    const result = await setupOnboardingMethods.saveSetupProvider.call({}, {
      provider_key: "openai",
      api_key: "sk-secret",
      make_default: true
    })

    expect(result.masked_api_key).toBe("sk-...abcd")
  })
})
```

- [x] **Step 2: Run tests to verify failure**

From `apps/packages/ui`:

```bash
bunx vitest run src/services/tldw/__tests__/setup-onboarding.test.ts
```

Expected: FAIL because the domain file does not exist.

- [x] **Step 3: Add setup onboarding types**

Create `apps/packages/ui/src/types/setup-onboarding.ts` with types matching backend schemas:

```ts
export type FirstRunStatus =
  | "not_started"
  | "in_progress"
  | "blocked"
  | "skipped"
  | "first_chat_complete"
  | "completed"

export type FirstRunState = {
  status: FirstRunStatus
  current_step?: string | null
  completed_steps: string[]
  skipped_steps: string[]
  step_data: Record<string, Record<string, unknown>>
  first_chat: {
    completed: boolean
    provider?: string | null
    model?: string | null
    response_id?: string | null
    completed_at?: string | null
  }
  acknowledged_steps: string[]
  skip_reason?: string | null
}

export type SetupProviderCatalogEntry = {
  provider_key: string
  label: string
  provider_type: "hosted_api_key" | "local_endpoint"
  default_base_url?: string | null
  supports_preflight: boolean
  recommended_for_first_chat: boolean
}

export type SetupProviderSaveResponse = {
  provider_key: string
  status: "saved" | "failed"
  masked_api_key?: string | null
  base_url?: string | null
  model?: string | null
  make_default?: boolean
  requires_restart?: boolean
  failure_category?: string | null
  message?: string | null
}

export type FirstRunMetadata = {
  auth_mode: string
  bundled_single_user_auth_available: boolean
  manual_auth_required: boolean
  setup_required: boolean
  setup_completed: boolean
  remote_setup_enabled: boolean
  connection: {
    frontend_origin?: string | null
    api_origin?: string | null
    browser_access?: "local" | "lan" | "remote" | "unknown" | null
  }
  setup_paths: Array<{ key: string; label: string; recommended: boolean; guide_path?: string | null }>
  multi_user_exit: { guide_path: string; checklist_path?: string | null }
}
```

Add request/response types for provider save, validation, ingest defaults, audio defaults, optional advanced, and first chat.

- [x] **Step 4: Add setup onboarding API domain**

Create `apps/packages/ui/src/services/tldw/domains/setup-onboarding.ts`:

```ts
import { bgRequest } from "@/services/background-proxy"
import type {
  FirstRunState,
  SetupProviderCatalogEntry,
  SetupProviderSaveRequest,
  SetupProviderSaveResponse
} from "@/types/setup-onboarding"

export const setupOnboardingMethods = {
  async getFirstRunState(): Promise<FirstRunState> {
    return await bgRequest({
      path: "/api/v1/setup/first-run/state",
      method: "GET"
    })
  },

  async getFirstRunMetadata(): Promise<FirstRunMetadata> {
    return await bgRequest({
      path: "/api/v1/setup/first-run/metadata",
      method: "GET"
    })
  },

  async getSetupProviderCatalog(): Promise<{ providers: SetupProviderCatalogEntry[] }> {
    return await bgRequest({
      path: "/api/v1/setup/first-run/providers/catalog",
      method: "GET"
    })
  },

  async saveSetupProvider(payload: SetupProviderSaveRequest): Promise<SetupProviderSaveResponse> {
    return await bgRequest({
      path: "/api/v1/setup/first-run/providers",
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: payload
    })
  }
}

export type SetupOnboardingMethods = typeof setupOnboardingMethods
```

Add all required methods before finishing this task.

- [x] **Step 5: Wire domain into `TldwApiClient`**

Modify:

- `apps/packages/ui/src/services/tldw/domains/index.ts`
- `apps/packages/ui/src/services/tldw/TldwApiClient.ts`

Add `setupOnboardingMethods` to imports, interface extension, and `Object.assign`.

- [x] **Step 6: Add setup paths to OpenAPI guard**

Modify `apps/packages/ui/src/services/tldw/openapi-guard.ts` and add:

```ts
  | "/api/v1/setup/first-run/state"
  | "/api/v1/setup/first-run/metadata"
  | "/api/v1/setup/first-run/skip"
  | "/api/v1/setup/first-run/providers/catalog"
  | "/api/v1/setup/first-run/providers"
  | "/api/v1/setup/first-run/providers/validate"
  | "/api/v1/setup/first-run/ingest-defaults"
  | "/api/v1/setup/first-run/audio-defaults"
  | "/api/v1/setup/first-run/optional-advanced"
  | "/api/v1/setup/first-run/first-chat"
  | "/api/v1/setup/first-run/complete"
```

- [x] **Step 7: Add hook tests**

Create `apps/packages/ui/src/hooks/__tests__/useSetupOnboarding.test.tsx`:

```tsx
// @vitest-environment jsdom
import { renderHook, waitFor } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"

vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: {
    getFirstRunState: vi.fn().mockResolvedValue({
      status: "not_started",
      completed_steps: [],
      skipped_steps: [],
      step_data: {},
      acknowledged_steps: [],
      first_chat: { completed: false }
    })
  }
}))

describe("useSetupOnboarding", () => {
  it("loads backend first-run state", async () => {
    const { useSetupOnboarding } = await import("../useSetupOnboarding")

    const { result } = renderHook(() => useSetupOnboarding())

    await waitFor(() => expect(result.current.state?.status).toBe("not_started"))
  })
})
```

- [x] **Step 8: Implement `useSetupOnboarding`**

Create `apps/packages/ui/src/hooks/useSetupOnboarding.ts`. Use local React state first; use React Query only if the surrounding setup code already has a provider in the tested route. Expose:

- `state`
- `metadata`
- `loading`
- `error`
- `refresh`
- `saveStep`
- `skip`
- `saveProvider`
- `validateProvider`
- `verifyFirstChat`
- `complete`

- [x] **Step 9: Run frontend API/hook tests**

From `apps/packages/ui`:

```bash
bunx vitest run src/services/tldw/__tests__/setup-onboarding.test.ts src/hooks/__tests__/useSetupOnboarding.test.tsx
```

Expected: PASS.

- [x] **Step 10: Commit frontend setup API domain**

```bash
git add apps/packages/ui/src/types/setup-onboarding.ts apps/packages/ui/src/services/tldw/domains/setup-onboarding.ts apps/packages/ui/src/services/tldw/domains/index.ts apps/packages/ui/src/services/tldw/TldwApiClient.ts apps/packages/ui/src/services/tldw/openapi-guard.ts apps/packages/ui/src/hooks/useSetupOnboarding.ts apps/packages/ui/src/services/tldw/__tests__/setup-onboarding.test.ts apps/packages/ui/src/hooks/__tests__/useSetupOnboarding.test.tsx
git commit -m "feat: add setup onboarding frontend client"
```

### Task 7: Focused WebUI Setup Shell And Wizard Skeleton

**Files:**
- Create: `apps/packages/ui/src/components/Option/Onboarding/UnifiedSetupWizard.tsx`
- Create: `apps/packages/ui/src/components/Option/Onboarding/steps/SetupPathStep.tsx`
- Create: `apps/packages/ui/src/components/Option/Onboarding/steps/PrivacySecurityStep.tsx`
- Create: `apps/packages/ui/src/components/Option/Onboarding/steps/MultiUserExitPanel.tsx`
- Modify: `apps/packages/ui/src/routes/option-index.tsx`
- Modify: `apps/packages/ui/src/routes/option-setup.tsx`
- Test: `apps/packages/ui/src/components/Option/Onboarding/__tests__/UnifiedSetupWizard.test.tsx`
- Test: `apps/packages/ui/src/routes/__tests__/option-index.unified-setup.test.tsx`

- [x] **Step 1: Write failing wizard skeleton test**

Create `apps/packages/ui/src/components/Option/Onboarding/__tests__/UnifiedSetupWizard.test.tsx`:

```tsx
// @vitest-environment jsdom
import React from "react"
import { fireEvent, render, screen } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"

vi.mock("@/hooks/useSetupOnboarding", () => ({
  useSetupOnboarding: () => ({
    state: {
      status: "not_started",
      completed_steps: [],
      skipped_steps: [],
      step_data: {},
      acknowledged_steps: [],
      first_chat: { completed: false }
    },
    metadata: {
      auth_mode: "single_user",
      bundled_single_user_auth_available: true,
      manual_auth_required: false,
      setup_required: true,
      setup_completed: false,
      remote_setup_enabled: false,
      connection: {
        frontend_origin: "http://127.0.0.1:3000",
        api_origin: "http://127.0.0.1:8000",
        browser_access: "local"
      },
      setup_paths: [],
      multi_user_exit: { guide_path: "/docs/multi-user" }
    },
    loading: false,
    error: null,
    saveStep: vi.fn(),
    skip: vi.fn()
  })
}))

describe("UnifiedSetupWizard", () => {
  it("renders a focused first-run heading and setup path choices", async () => {
    const { UnifiedSetupWizard } = await import("../UnifiedSetupWizard")

    render(<UnifiedSetupWizard />)

    expect(screen.getByRole("heading", { name: /first-time setup/i })).toBeInTheDocument()
    expect(screen.getByRole("button", { name: /solo, docker/i })).toBeInTheDocument()
    expect(screen.getByRole("button", { name: /solo, local/i })).toBeInTheDocument()
    expect(screen.getByRole("button", { name: /multi-user/i })).toBeInTheDocument()
  })

  it("shows multi-user exit guidance instead of continuing solo wizard", async () => {
    const { UnifiedSetupWizard } = await import("../UnifiedSetupWizard")

    render(<UnifiedSetupWizard />)
    fireEvent.click(screen.getByRole("button", { name: /multi-user/i }))

    expect(screen.getByText(/multi-user setup guide/i)).toBeInTheDocument()
  })
})
```

- [x] **Step 2: Write failing route-shell test**

Create `apps/packages/ui/src/routes/__tests__/option-index.unified-setup.test.tsx`:

```tsx
// @vitest-environment jsdom
import React from "react"
import { MemoryRouter } from "react-router-dom"
import { render, screen } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"

vi.mock("~/components/Layouts/Layout", () => ({
  default: ({ children, hideHeader, hideSidebar }: any) => (
    <main data-hide-header={String(Boolean(hideHeader))} data-hide-sidebar={String(Boolean(hideSidebar))}>
      {children}
    </main>
  )
}))

vi.mock("@/hooks/useSetupOnboarding", () => ({
  useSetupOnboarding: () => ({
    state: { status: "not_started", completed_steps: [], skipped_steps: [], step_data: {}, acknowledged_steps: [], first_chat: { completed: false } },
    metadata: {
      auth_mode: "single_user",
      bundled_single_user_auth_available: true,
      manual_auth_required: false,
      setup_required: true,
      setup_completed: false,
      remote_setup_enabled: false,
      connection: { browser_access: "local" },
      setup_paths: [],
      multi_user_exit: { guide_path: "/docs/multi-user" }
    },
    loading: false,
    error: null
  })
}))

describe("OptionIndex unified setup resolver", () => {
  it("renders setup in focused shell when backend state is not complete", async () => {
    const { default: OptionIndex } = await import("../option-index")

    render(
      <MemoryRouter>
        <OptionIndex />
      </MemoryRouter>
    )

    expect(screen.getByRole("main")).toHaveAttribute("data-hide-header", "true")
    expect(screen.getByRole("main")).toHaveAttribute("data-hide-sidebar", "true")
  })
})
```

- [x] **Step 3: Run tests to verify failure**

From `apps/packages/ui`:

```bash
bunx vitest run src/components/Option/Onboarding/__tests__/UnifiedSetupWizard.test.tsx src/routes/__tests__/option-index.unified-setup.test.tsx
```

Expected: FAIL because wizard does not exist or route still uses old local first-run state.

- [x] **Step 4: Implement `SetupPathStep` and `MultiUserExitPanel`**

Create small focused components. Use buttons, not decorative cards-only click targets:

```tsx
export function SetupPathStep({ onSelect }: { onSelect: (path: "docker" | "local" | "multi_user") => void }) {
  return (
    <section aria-labelledby="setup-path-title">
      <h2 id="setup-path-title">Choose your setup path</h2>
      <div className="grid gap-3 md:grid-cols-3">
        <button type="button" onClick={() => onSelect("docker")}>Solo, Docker</button>
        <button type="button" onClick={() => onSelect("local")}>Solo, local install</button>
        <button type="button" onClick={() => onSelect("multi_user")}>Multi-user or shared server</button>
      </div>
    </section>
  )
}
```

Style with existing design-system classes used in onboarding; keep text compact and non-marketing.

- [x] **Step 5: Implement `PrivacySecurityStep`**

Create `apps/packages/ui/src/components/Option/Onboarding/steps/PrivacySecurityStep.tsx`.

Requirements:

- consume `FirstRunMetadata` from `useSetupOnboarding`;
- show current auth mode and solo single-user assumption;
- show whether bundled single-user auth can be handled automatically or manual API-key entry is required;
- show local/LAN/remote browser access classification from backend metadata when available;
- explain that provider secrets are stored by the backend and returned masked;
- require acknowledging local/remote access and secret-storage notices before continuing;
- warn when remote setup is enabled or browser access is non-local;
- call `onContinue` only after acknowledgement;
- show docs/recovery links without raw stack traces, filesystem internals, or secrets.

Add a focused assertion to `UnifiedSetupWizard.test.tsx`:

```tsx
it("requires privacy and security acknowledgement before provider setup", async () => {
  const { UnifiedSetupWizard } = await import("../UnifiedSetupWizard")

  render(<UnifiedSetupWizard />)
  fireEvent.click(screen.getByRole("button", { name: /solo, docker/i }))

  expect(screen.getByRole("heading", { name: /privacy and security/i })).toBeInTheDocument()
  expect(screen.getByRole("button", { name: /continue/i })).toBeDisabled()

  fireEvent.click(screen.getByLabelText(/i understand/i))
  expect(screen.getByRole("button", { name: /continue/i })).toBeEnabled()
})
```

- [x] **Step 6: Implement wizard skeleton**

`UnifiedSetupWizard` should:

- load backend first-run state through `useSetupOnboarding`;
- load first-run metadata through `useSetupOnboarding`;
- render a single `h1`;
- track current step locally until backend state drives it;
- show `SetupPathStep`;
- show `PrivacySecurityStep` after Docker/local selection;
- show `MultiUserExitPanel` after multi-user selection;
- expose a skip button that calls backend skip state.

- [x] **Step 7: Replace first-run route usage**

Modify `option-index.tsx`:

- remove dependence on `__tldw_first_run_complete` as authoritative completion;
- use backend first-run state from `useSetupOnboarding`;
- render `UnifiedSetupWizard` in `OptionLayout hideHeader hideSidebar` when status is `not_started`, `in_progress`, `blocked`, or `first_chat_complete`;
- render normal app when status is `completed` or `skipped`.

Keep existing connection `checkOnce()` behavior only for backend reachability.

Modify `option-setup.tsx`:

- use the same `UnifiedSetupWizard` for setup-required states;
- label `/setup` as operator/recovery in copy;
- keep `hideHeader hideSidebar`.

- [x] **Step 8: Run wizard and route tests**

```bash
bunx vitest run src/components/Option/Onboarding/__tests__/UnifiedSetupWizard.test.tsx src/routes/__tests__/option-index.unified-setup.test.tsx
```

Expected: PASS.

- [x] **Step 9: Run existing onboarding guard tests**

```bash
bunx vitest run src/components/Option/Onboarding/__tests__/OnboardingConnectForm.design-system.test.tsx src/components/Option/Onboarding/__tests__/OnboardingConnectForm.success-screen.guard.test.tsx
```

Expected: PASS or update tests only where they intentionally asserted replaced first-run behavior.

- [x] **Step 10: Commit wizard shell**

```bash
git add apps/packages/ui/src/components/Option/Onboarding/UnifiedSetupWizard.tsx apps/packages/ui/src/components/Option/Onboarding/steps/SetupPathStep.tsx apps/packages/ui/src/components/Option/Onboarding/steps/PrivacySecurityStep.tsx apps/packages/ui/src/components/Option/Onboarding/steps/MultiUserExitPanel.tsx apps/packages/ui/src/routes/option-index.tsx apps/packages/ui/src/routes/option-setup.tsx apps/packages/ui/src/components/Option/Onboarding/__tests__/UnifiedSetupWizard.test.tsx apps/packages/ui/src/routes/__tests__/option-index.unified-setup.test.tsx
git commit -m "feat: add focused first-run setup shell"
```

### Task 8: Provider, Ingest, Audio, Advanced, And First-Chat Wizard Steps

**Files:**
- Create: `apps/packages/ui/src/components/Option/Onboarding/steps/ProviderSetupStep.tsx`
- Create: `apps/packages/ui/src/components/Option/Onboarding/steps/IngestDefaultsStep.tsx`
- Create: `apps/packages/ui/src/components/Option/Onboarding/steps/AudioSetupStep.tsx`
- Create: `apps/packages/ui/src/components/Option/Onboarding/steps/OptionalAdvancedStep.tsx`
- Create: `apps/packages/ui/src/components/Option/Onboarding/steps/FirstChatStep.tsx`
- Create: `apps/packages/ui/src/components/Option/Onboarding/FirstSourceMilestonePrompt.tsx`
- Modify: `apps/packages/ui/src/components/Option/Onboarding/UnifiedSetupWizard.tsx`
- Modify: `tldw_Server_API/app/api/v1/endpoints/setup.py`
- Test: `apps/packages/ui/src/components/Option/Onboarding/__tests__/ProviderSetupStep.test.tsx`
- Test: `apps/packages/ui/src/components/Option/Onboarding/__tests__/FirstChatStep.test.tsx`
- Test: `apps/packages/ui/src/components/Option/Onboarding/__tests__/FirstSourceMilestonePrompt.test.tsx`
- Test: `tldw_Server_API/tests/integration/test_unified_first_run_setup_api.py`

- [x] **Step 1: Write provider step tests**

Create `ProviderSetupStep.test.tsx`:

```tsx
// @vitest-environment jsdom
import React from "react"
import { fireEvent, render, screen, waitFor } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"

describe("ProviderSetupStep", () => {
  it("lets users select multiple providers and save one default", async () => {
    const saveProvider = vi.fn().mockResolvedValue({ status: "saved" })
    const { ProviderSetupStep } = await import("../steps/ProviderSetupStep")

    render(
      <ProviderSetupStep
        providers={[
          { provider_key: "openai", label: "OpenAI", provider_type: "hosted_api_key", supports_preflight: true, recommended_for_first_chat: true },
          { provider_key: "ollama", label: "Ollama", provider_type: "local_endpoint", default_base_url: "http://127.0.0.1:11434/v1", supports_preflight: true, recommended_for_first_chat: false }
        ]}
        onSaveProvider={saveProvider}
        onContinue={vi.fn()}
      />
    )

    fireEvent.click(screen.getByLabelText(/openai/i))
    fireEvent.change(screen.getByLabelText(/openai api key/i), { target: { value: "sk-test" } })
    fireEvent.click(screen.getByRole("button", { name: /save provider/i }))

    await waitFor(() => expect(saveProvider).toHaveBeenCalled())
  })
})
```

- [x] **Step 2: Write first-chat step tests**

Create `FirstChatStep.test.tsx`:

```tsx
// @vitest-environment jsdom
import React from "react"
import { fireEvent, render, screen, waitFor } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"

describe("FirstChatStep", () => {
  it("requires a successful first chat before calling complete", async () => {
    const verifyFirstChat = vi.fn().mockResolvedValue({
      status: "ready",
      provider: "openai",
      model: "gpt-4.1-mini",
      response_text: "Hello."
    })
    const complete = vi.fn().mockResolvedValue({ status: "completed" })
    const { FirstChatStep } = await import("../steps/FirstChatStep")

    render(
      <FirstChatStep
        provider="openai"
        model="gpt-4.1-mini"
        verifyFirstChat={verifyFirstChat}
        complete={complete}
      />
    )

    fireEvent.click(screen.getByRole("button", { name: /send test chat/i }))

    await screen.findByText("Hello.")
    await waitFor(() => expect(complete).toHaveBeenCalled())
  })
})
```

- [x] **Step 3: Run tests to verify failure**

```bash
bunx vitest run src/components/Option/Onboarding/__tests__/ProviderSetupStep.test.tsx src/components/Option/Onboarding/__tests__/FirstChatStep.test.tsx
```

Expected: FAIL because step components do not exist.

- [x] **Step 4: Write first-source milestone prompt test**

Create `FirstSourceMilestonePrompt.test.tsx`:

```tsx
// @vitest-environment jsdom
import React from "react"
import { fireEvent, render, screen } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"

describe("FirstSourceMilestonePrompt", () => {
  it("offers adding a first source immediately after onboarding completion", async () => {
    const onAddSource = vi.fn()
    const { FirstSourceMilestonePrompt } = await import("../FirstSourceMilestonePrompt")

    render(<FirstSourceMilestonePrompt onAddSource={onAddSource} onDismiss={vi.fn()} />)

    expect(screen.getByRole("heading", { name: /add your first source/i })).toBeInTheDocument()
    fireEvent.click(screen.getByRole("button", { name: /add source/i }))

    expect(onAddSource).toHaveBeenCalled()
  })
})
```

- [x] **Step 5: Run tests to verify failure**

```bash
bunx vitest run src/components/Option/Onboarding/__tests__/ProviderSetupStep.test.tsx src/components/Option/Onboarding/__tests__/FirstChatStep.test.tsx src/components/Option/Onboarding/__tests__/FirstSourceMilestonePrompt.test.tsx
```

Expected: FAIL because step components and first-source prompt do not exist.

- [x] **Step 6: Implement `ProviderSetupStep`**

Requirements:

- render backend catalog sorted with recommended hosted providers first;
- support selecting multiple providers;
- hosted providers show masked/savable API-key fields;
- local providers show base URL, model, and optional token;
- call `onSaveProvider` per provider;
- choose one default provider/model before continuing;
- never display raw saved secret after save.

Keep the component focused. Do not call `tldwClient` directly inside it; pass callbacks from `UnifiedSetupWizard` or hook.

- [x] **Step 7: Implement `IngestDefaultsStep`**

Fields:

- local file ingest toggle;
- allowed roots text area or disabled state;
- chunking profile segmented control: `simple`, `balanced`, `advanced`;
- metadata mode segmented control: `automatic`, `ask`, `minimal`.

Primary action calls `saveIngestDefaults`.

- [x] **Step 8: Implement `AudioSetupStep`**

Reuse existing audio setup logic where possible:

- call `GET /api/v1/setup/audio/recommendations` through the setup onboarding domain or reuse `useAudioInstaller` after adding a setup-mode path variant;
- show configure now, use defaults, skip;
- do not require provisioning to continue;
- call `saveAudioDefaults`.

Avoid copying the full admin `AudioInstallerPanel` UI into first-run. The first-run step should be simpler and link to deeper setup.

- [x] **Step 9: Implement `OptionalAdvancedStep`**

Show RAG/embeddings and storage paths as optional:

- configure now;
- defer;
- skip.

For V1, allow deferring without exposing every advanced field. If configure now is selected, deep-link to the relevant settings page after completion unless safe backend fields already exist.

- [x] **Step 10: Implement `FirstChatStep`**

Requirements:

- uses selected default provider/model;
- shows editable first prompt with default "Say hello in one short sentence.";
- calls backend first-chat verification;
- displays actual response text;
- calls completion only after `status === "ready"`;
- shows provider failure categories with retry and back-to-provider actions.

- [x] **Step 11: Implement `FirstSourceMilestonePrompt`**

Create `apps/packages/ui/src/components/Option/Onboarding/FirstSourceMilestonePrompt.tsx`.

Requirements:

- render only after backend state is `completed`;
- provide one primary `Add source` action that navigates to the existing source/ingest entrypoint or opens the existing Quick Ingest flow;
- provide a dismiss action persisted only in frontend local state because dismissed tips are allowed frontend-local state;
- do not block normal app navigation;
- use copy that frames this as the next milestone after first chat, not as part of setup completion.

- [x] **Step 12: Wire steps into `UnifiedSetupWizard`**

Wizard order:

1. setup path;
2. privacy/security;
3. providers;
4. ingest defaults;
5. audio/STT/TTS;
6. optional advanced;
7. first chat.

Persist step completion to backend after every successful step.

- [x] **Step 13: Wire first-source prompt into the normal app handoff**

Modify `option-index.tsx` or the existing post-onboarding shell entrypoint so that:

- when setup state is `completed` and the frontend local dismissed-tip state is false, `FirstSourceMilestonePrompt` appears in the normal app shell;
- clicking `Add source` navigates to the existing source/ingest entrypoint;
- dismissing the prompt writes only local UI state, not backend setup completion.

- [x] **Step 14: Run wizard step tests**

```bash
bunx vitest run src/components/Option/Onboarding/__tests__/UnifiedSetupWizard.test.tsx src/components/Option/Onboarding/__tests__/ProviderSetupStep.test.tsx src/components/Option/Onboarding/__tests__/FirstChatStep.test.tsx src/components/Option/Onboarding/__tests__/FirstSourceMilestonePrompt.test.tsx
```

Expected: PASS.

- [x] **Step 15: Commit wizard steps**

```bash
git add apps/packages/ui/src/components/Option/Onboarding/UnifiedSetupWizard.tsx apps/packages/ui/src/components/Option/Onboarding/steps apps/packages/ui/src/components/Option/Onboarding/FirstSourceMilestonePrompt.tsx apps/packages/ui/src/components/Option/Onboarding/__tests__/ProviderSetupStep.test.tsx apps/packages/ui/src/components/Option/Onboarding/__tests__/FirstChatStep.test.tsx apps/packages/ui/src/components/Option/Onboarding/__tests__/FirstSourceMilestonePrompt.test.tsx apps/packages/ui/src/routes/option-index.tsx
git commit -m "feat: add progressive first-run wizard steps"
```

### Task 9: Docs, CLI, Makefile, And Onboarding Cleanup

**Files:**
- Modify: `Makefile`
- Modify: `tldw_Server_API/cli/wizard/profile_verify.py`
- Modify: `Docs/Getting_Started/README.md`
- Modify: `Docs/Getting_Started/Profile_Docker_Single_User.md`
- Modify: `Docs/Getting_Started/Profile_Local_Single_User.md`
- Modify: `Docs/Getting_Started/Profile_Docker_Multi_User_Postgres.md`
- Modify: `Docs/Getting_Started/onboarding_manifest.yaml`
- Modify: `Docs/Published/Getting_Started/README.md`
- Modify: `Docs/Published/Getting_Started/Profile_Docker_Single_User.md`
- Modify: `Docs/Published/Getting_Started/Profile_Local_Single_User.md`
- Modify: `Docs/Published/Getting_Started/Profile_Docker_Multi_User_Postgres.md`
- Modify: `Docs/Published/Getting_Started/onboarding_manifest.yaml`
- Test: `tldw_Server_API/tests/Docs/test_onboarding_guides_structure.py`
- Test: `tldw_Server_API/tests/Docs/test_onboarding_command_boundaries.py`
- Test: `tldw_Server_API/tests/Utils/test_makefile_onboarding_profiles.py`
- Test: `tldw_Server_API/tests/Utils/test_makefile_quickstart_default.py`
- Test: `tldw_Server_API/tests/wizard/test_cli_verify_profiles.py`

- [x] **Step 1: Add failing docs/Makefile tests**

Update docs tests so they assert:

- Docker single-user and local single-user are peer solo choices;
- multi-user routes to guide/checklist;
- docs mention WebUI first-chat completion;
- docs identify adding a first source as the immediate post-onboarding milestone;
- docs do not say normal setup requires editing `.env`/`config.txt`;
- `make quickstart` still exists and points to WebUI path;
- local setup has equivalent prepare/start/verify semantics.

Example assertion in docs tests:

```python
def test_getting_started_presents_docker_and_local_as_peer_solo_paths():
    text = Path("Docs/Getting_Started/README.md").read_text(encoding="utf-8")
    docker_idx = text.index("Docker single-user")
    local_idx = text.index("Local single-user")
    assert abs(docker_idx - local_idx) < 2000
    assert "first successful chat" in text
```

- [x] **Step 2: Run docs/Makefile tests to verify failure**

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Docs/test_onboarding_guides_structure.py tldw_Server_API/tests/Docs/test_onboarding_command_boundaries.py tldw_Server_API/tests/Utils/test_makefile_onboarding_profiles.py tldw_Server_API/tests/Utils/test_makefile_quickstart_default.py -v
```

Expected: FAIL until docs are updated.

- [x] **Step 3: Update Getting Started index**

`Docs/Getting_Started/README.md` should start with:

- Solo setup chooser;
- Docker single-user and local single-user as peer paths;
- multi-user as "shared server/operator path";
- lifecycle: prepare, start, verify, open WebUI, complete first chat;
- adding first source as next milestone.

- [x] **Step 4: Update profile docs**

Update Docker and local single-user docs:

- same lifecycle headings;
- WebUI onboarding handoff;
- first chat completion target;
- no required manual config editing for normal setup;
- API-key recovery in troubleshooting only.

Update multi-user doc:

- clearly says solo wizard is not the multi-user path;
- points to admin bootstrap/operator checklist.

- [x] **Step 5: Update CLI/Makefile messaging**

Make output should print WebUI URL and next action:

```text
Next: open http://127.0.0.1:8080 and complete first-time setup in the WebUI.
```

CLI verification can report:

- install/runtime ready;
- WebUI reachable;
- first chat not complete yet;
- open WebUI next.

Do not make CLI claim first-run complete without backend first-chat state.

- [x] **Step 6: Update onboarding manifest and published parity**

Update source and published onboarding manifests as existing tests require. Do not manually move generated published files unless existing repo workflow expects committed parity.

- [x] **Step 7: Run docs/Makefile tests**

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Docs/test_onboarding_guides_structure.py tldw_Server_API/tests/Docs/test_onboarding_command_boundaries.py tldw_Server_API/tests/Docs/test_published_onboarding_parity.py tldw_Server_API/tests/Utils/test_makefile_onboarding_profiles.py tldw_Server_API/tests/Utils/test_makefile_quickstart_default.py -v
```

Expected: PASS.

- [x] **Step 8: Commit docs/CLI cleanup**

```bash
git add Makefile tldw_Server_API/cli/wizard/cli.py tldw_Server_API/cli/wizard/profile_verify.py Docs/Getting_Started Docs/Published/Getting_Started tldw_Server_API/tests/Docs tldw_Server_API/tests/Utils
git commit -m "docs: align onboarding around first chat"
```

### Task 10: End-To-End Verification, Security, And Release Gate

**Files:**
- Create or modify: `apps/tldw-frontend/e2e/workflows/unified-first-run-onboarding.spec.ts`
- Modify: `apps/tldw-frontend/e2e/smoke/page-inventory.ts` only if route expectations change.
- Modify: `tldw_Server_API/tests/frontend_e2e/test_onboarding_workflow.py` if backend-driven E2E hooks need updating.
- Backlog: update all child tasks with verification.

- [x] **Step 1: Add Playwright E2E for setup shell**

Create `apps/tldw-frontend/e2e/workflows/unified-first-run-onboarding.spec.ts`:

```ts
import { expect, test } from "@playwright/test"

test("first-run setup hides global navigation until skipped", async ({ page }) => {
  await page.addInitScript(() => {
    localStorage.clear()
  })
  await page.goto("/")

  await expect(page.getByRole("heading", { name: /first-time setup/i })).toBeVisible()
  await expect(page.getByRole("navigation")).toHaveCount(0)

  await page.getByRole("button", { name: /skip/i }).click()

  await expect(page.getByRole("navigation")).toBeVisible()
})
```

Adjust selectors to match actual app shell once implemented.

- [x] **Step 2: Add E2E for completed setup first-source milestone**

Add a Playwright test that starts with backend setup state mocked or pre-seeded as `completed` and verifies the normal app shell shows the first-source milestone prompt with an `Add source` action.

Expected: the prompt is visible after completion, does not hide normal navigation, and can be dismissed without changing backend setup state.

- [x] **Step 3: Add E2E for hosted-key happy path with mock backend**

If live provider calls are not available in CI, use a mocked backend route or test mode endpoint so first chat returns a deterministic response. The E2E should still verify the UI waits for a real backend success response shape, not just local validation.

- [x] **Step 4: Run focused backend tests**

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Setup tldw_Server_API/tests/integration/test_unified_first_run_setup_api.py tldw_Server_API/tests/Config/test_config_providers_endpoints.py -v
```

Expected: PASS.

Result: `test_first_run_state.py`, `test_unified_first_run_setup_api.py`, and
`test_config_providers_endpoints.py` passed: 135 passed, 2 warnings. A later
release-gate cleanup resolved the broader setup audio failures and the full
setup/config command passed: 324 passed, 4 warnings.

- [x] **Step 5: Run focused frontend tests**

From `apps/packages/ui`:

```bash
bunx vitest run src/services/tldw/__tests__/setup-onboarding.test.ts src/hooks/__tests__/useSetupOnboarding.test.tsx src/components/Option/Onboarding/__tests__/UnifiedSetupWizard.test.tsx src/components/Option/Onboarding/__tests__/ProviderSetupStep.test.tsx src/components/Option/Onboarding/__tests__/FirstChatStep.test.tsx src/components/Option/Onboarding/__tests__/FirstSourceMilestonePrompt.test.tsx src/routes/__tests__/option-index.unified-setup.test.tsx
```

Expected: PASS.

Result: passed: 7 files, 23 tests. Existing warnings observed for Vitest
localStorage and a missing test i18next instance in the option index test.

- [x] **Step 6: Run Playwright E2E**

From `apps/tldw-frontend`:

```bash
bunx playwright test e2e/workflows/unified-first-run-onboarding.spec.ts --reporter=line
```

Expected: PASS. If the app requires a running server, document exact server command and URL in Backlog.

Result: passed: 3 Chromium tests.

- [x] **Step 7: Run Bandit on touched backend scope**

```bash
source .venv/bin/activate
python -m bandit -r tldw_Server_API/app/core/Setup tldw_Server_API/app/api/v1/endpoints/setup.py tldw_Server_API/app/api/v1/API_Deps/setup_deps.py -f json -o /tmp/bandit_unified_first_run_onboarding.json
```

Expected: no new high/medium findings in touched code. Fix new findings before completion.

Result: no findings; report written to
`/tmp/bandit_unified_first_run_onboarding.json`.

- [x] **Step 8: Run diff whitespace check**

```bash
git diff --check
```

Expected: no whitespace errors.

Result: passed.

- [x] **Step 9: Update Backlog final summaries**

For each child task and parent task, record:

- files changed;
- tests run;
- Bandit result path;
- known skips/blockers;
- follow-up items.

- [x] **Step 10: Final commit if verification-only files changed**

```bash
git add apps/tldw-frontend/e2e/workflows/unified-first-run-onboarding.spec.ts apps/tldw-frontend/e2e/smoke/page-inventory.ts tldw_Server_API/tests/frontend_e2e/test_onboarding_workflow.py backlog/tasks
git commit -m "test: verify unified first-run onboarding"
```

## Final Verification Checklist

Before opening a PR or declaring implementation complete:

- [x] `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Setup tldw_Server_API/tests/integration/test_unified_first_run_setup_api.py tldw_Server_API/tests/Config/test_config_providers_endpoints.py -v` - passed: 324 passed, 4 warnings.
- [x] `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Docs tldw_Server_API/tests/Utils/test_makefile_onboarding_profiles.py tldw_Server_API/tests/Utils/test_makefile_quickstart_default.py -v`
- [x] `cd apps/packages/ui && bunx vitest run src/services/tldw/__tests__/setup-onboarding.test.ts src/hooks/__tests__/useSetupOnboarding.test.tsx src/components/Option/Onboarding/__tests__/UnifiedSetupWizard.test.tsx src/components/Option/Onboarding/__tests__/ProviderSetupStep.test.tsx src/components/Option/Onboarding/__tests__/FirstChatStep.test.tsx src/components/Option/Onboarding/__tests__/FirstSourceMilestonePrompt.test.tsx src/routes/__tests__/option-index.unified-setup.test.tsx`
- [x] `cd apps/tldw-frontend && bunx playwright test e2e/workflows/unified-first-run-onboarding.spec.ts --reporter=line`
- [x] `source .venv/bin/activate && python -m bandit -r tldw_Server_API/app/core/Setup tldw_Server_API/app/api/v1/endpoints/setup.py tldw_Server_API/app/api/v1/API_Deps/setup_deps.py -f json -o /tmp/bandit_unified_first_run_onboarding.json`
- [x] `git diff --check`

## Implementation Notes

- Prefer generated/backend provider catalog over frontend hardcoded provider lists.
- Keep first-run setup state backend-authoritative. Do not reintroduce localStorage as the completion source of truth.
- Keep setup secret writes inside `require_local_setup_access` before completion and authenticated admin/system-configure permissions after completion.
- Preserve backend `/setup` as recovery/operator surface; the WebUI wizard is the primary solo-user surface.
- Use existing audio recommendation/provision/verify endpoints instead of duplicating audio setup logic.
- Keep multi-user as an exit ramp to docs/checklist.
- Avoid broad app-shell redesign. Only hide global navigation during setup-required states.
- The first-source milestone is post-onboarding guidance only. It must not become part of setup completion or block normal navigation.
