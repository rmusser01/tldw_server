# First-Time Readiness Setup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the backend-owned first-time model readiness setup flow so new users can choose, preview, provision, verify, skip, and later administer chat, embeddings/RAG, and speech readiness from backend `/setup` and the native WebUI.

**Architecture:** Add a small setup readiness service under the existing `core/Setup` package, expose it through `/api/v1/setup/readiness/*`, and reuse existing config writers, install plans, audio bundle recommendations, audio readiness, and setup guards. The WebUI gets a native `/setup` readiness screen backed by the same endpoints, while backend `/setup` remains the recovery fallback.

**Tech Stack:** FastAPI, Pydantic, pytest, existing setup/install manager helpers, React, Ant Design, Vitest, shared `bgRequest` WebUI client, existing WebUI setup/onboarding route.

---

## Source Artifacts

- Spec: `Docs/superpowers/specs/2026-05-18-first-time-model-readiness-setup-design.md`
- Planning task: `TASK-427`
- Design task: `TASK-426`
- Worktree: `.worktrees/first-time-readiness-setup-contract`

## Implementation Boundaries

- Preserve backend `/setup` as authoritative.
- Do not create a second WebUI-only setup/config system.
- Do not start downloads, package installs, config writes, hosted calls, or expensive local model loads during profile selection or preview.
- Keep `restart_required`, `requires_admin`, and `remote_setup_blocked` as overlays, not lane statuses.
- Treat TTS as secondary metadata inside the canonical `speech` lane.
- Never echo submitted provider secrets in preview/status responses.
- Put trusted/custom Hugging Face models behind advanced acknowledgement.

## File Structure

Backend files:

- Create `tldw_Server_API/app/core/Setup/readiness_models.py`
  - Own canonical lane/status/overlay literals and Pydantic records used by the readiness service.
- Create `tldw_Server_API/app/core/Setup/readiness_profiles.py`
  - Build curated profile recommendations from setup status, config snapshot, audio recommendations, and conservative chat/embedding defaults.
- Create `tldw_Server_API/app/core/Setup/readiness_service.py`
  - Own preview/status/provision/verify orchestration while delegating config writes and installer work to existing helpers.
- Create `tldw_Server_API/app/core/Setup/readiness_store.py`
  - Persist lane status, selected profile, last preview/provision operation, warnings, overlays, and verification snapshots using the same JSON-store pattern as `audio_readiness_store.py`.
- Modify `tldw_Server_API/app/api/v1/schemas/setup_schemas.py`
  - Add request/response models for profiles, preview, provision, status, and verification.
- Modify `tldw_Server_API/app/api/v1/endpoints/setup.py`
  - Add `/readiness/profiles`, `/readiness/preview`, `/readiness/status`, `/readiness/provision`, `/readiness/verify`, and admin equivalents.
- Modify `tldw_Server_API/app/api/v1/API_Deps/setup_deps.py` only if implementation needs a shared dependency that switches first-run local access vs post-setup admin access.

Backend tests:

- Create `tldw_Server_API/tests/Setup/test_setup_readiness_profiles.py`
- Create `tldw_Server_API/tests/Setup/test_setup_readiness_preview.py`
- Create `tldw_Server_API/tests/Setup/test_setup_readiness_api.py`
- Create `tldw_Server_API/tests/Setup/test_setup_readiness_store.py`

Frontend files:

- Create `apps/packages/ui/src/services/tldw/setup-readiness.ts`
  - Shared typed client for first-run and admin setup readiness endpoints.
- Create `apps/packages/ui/src/components/Option/Setup/ReadinessSetupScreen.tsx`
  - Native readiness dashboard for profile picker, lanes, preview, provision, verify, completion, and fallback link.
- Create `apps/packages/ui/src/components/Option/Setup/hooks/useSetupReadiness.ts`
  - State machine and polling hook for readiness profiles/status/provisioning.
- Modify `apps/packages/ui/src/routes/option-setup.tsx`
  - Render native readiness setup when connected backend reports setup-required; keep current connection onboarding as fallback when no backend is configured.
- Modify `apps/packages/ui/src/components/Option/Setup/AudioInstallerPanel.tsx`
  - Reuse styling/status helpers where useful, but do not couple first-run readiness to the admin audio-only panel.
- Modify `apps/packages/ui/src/services/tldw/openapi-guard.ts`
  - Add new setup readiness endpoints once they exist in the backend OpenAPI.

Frontend tests:

- Create `apps/packages/ui/src/components/Option/Setup/__tests__/ReadinessSetupScreen.test.tsx`
- Create `apps/packages/ui/src/components/Option/Setup/hooks/__tests__/useSetupReadiness.test.tsx`
- Extend `apps/packages/ui/src/components/Option/Setup/__tests__/AudioInstallerPanel.test.tsx` only if shared helpers are extracted.

Docs:

- Update `Docs/Development/Setup.md` or the nearest current setup doc if implementation exposes user-visible behavior.

---

### Task 1: Backend Readiness Models And Profile Builder

**Files:**
- Create: `tldw_Server_API/app/core/Setup/readiness_models.py`
- Create: `tldw_Server_API/app/core/Setup/readiness_profiles.py`
- Modify: `tldw_Server_API/app/api/v1/schemas/setup_schemas.py`
- Test: `tldw_Server_API/tests/Setup/test_setup_readiness_profiles.py`

- [ ] **Step 1: Write failing tests for canonical lane/status/overlay semantics**

```python
from tldw_Server_API.app.core.Setup.readiness_profiles import build_readiness_profiles


def test_profiles_return_canonical_lanes_and_restart_overlay(sample_setup_snapshot):
    response = build_readiness_profiles(
        setup_status=sample_setup_snapshot,
        config_snapshot={"sections": []},
        audio_recommendations={"recommendations": [], "catalog": [], "machine_profile": {}},
    )

    assert [lane["lane_id"] for lane in response["lanes"]] == ["chat", "embeddings_rag", "speech"]
    assert "restart_required" in response["supported_overlays"]
    assert all(lane["status"] != "restart_required" for lane in response["lanes"])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/Setup/test_setup_readiness_profiles.py -q`

Expected: FAIL with `ModuleNotFoundError` or missing `build_readiness_profiles`.

- [ ] **Step 3: Add minimal model and profile builder**

Implementation outline:

```python
LANE_IDS = ("chat", "embeddings_rag", "speech")
LANE_STATUSES = (
    "not_configured",
    "previewed",
    "provisioning",
    "ready",
    "ready_with_warnings",
    "failed",
    "blocked",
    "skipped",
)
OVERLAYS = (
    "restart_required",
    "requires_admin",
    "remote_setup_blocked",
    "network_unavailable",
    "downloads_disabled",
    "package_installs_disabled",
)
```

`build_readiness_profiles()` should return:

- `lanes`: three canonical lane summaries.
- `profiles`: curated choices: `local_light`, `local_balanced`, `local_performance`, `hosted_plus_local_speech`, `advanced_custom`.
- `recommended_profile_id`: conservative default based on audio recommendations and config placeholders.
- `supported_overlays`: known overlay IDs.
- `setup_access`: first-run/admin availability metadata.

- [ ] **Step 4: Run test to verify it passes**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/Setup/test_setup_readiness_profiles.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tldw_Server_API/app/core/Setup/readiness_models.py \
  tldw_Server_API/app/core/Setup/readiness_profiles.py \
  tldw_Server_API/app/api/v1/schemas/setup_schemas.py \
  tldw_Server_API/tests/Setup/test_setup_readiness_profiles.py
git commit -m "feat: add setup readiness profile models"
```

---

### Task 2: Backend Preview Contract Without Mutations

**Files:**
- Create: `tldw_Server_API/app/core/Setup/readiness_service.py`
- Modify: `tldw_Server_API/app/api/v1/schemas/setup_schemas.py`
- Test: `tldw_Server_API/tests/Setup/test_setup_readiness_preview.py`

- [ ] **Step 1: Write failing tests for preview safety**

```python
from tldw_Server_API.app.core.Setup.readiness_service import preview_readiness_selection


def test_preview_returns_config_updates_and_install_plan_without_writing(monkeypatch):
    called = False

    def fail_if_called(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("preview must not write config")

    monkeypatch.setattr("tldw_Server_API.app.core.Setup.setup_manager.update_config", fail_if_called)

    preview = preview_readiness_selection(
        {
            "profile_id": "local_balanced",
            "lanes": {
                "chat": {"mode": "skip"},
                "embeddings_rag": {"provider": "huggingface", "model": "Qwen/Qwen3-Embedding-0.6B"},
                "speech": {"bundle_id": "cpu_local", "resource_profile": "balanced"},
            },
        }
    )

    assert called is False
    assert preview["operation_required"] is True
    assert preview["config_updates"]["Embeddings"]["embedding_provider"] == "huggingface"
    assert "Qwen/Qwen3-Embedding-0.6B" in preview["install_plan"]["embeddings"]["huggingface"]
```

- [ ] **Step 2: Write failing tests for secret and trusted-model rules**

```python
def test_preview_never_echoes_hosted_provider_secret():
    preview = preview_readiness_selection({
        "profile_id": "advanced_custom",
        "lanes": {
            "chat": {
                "mode": "hosted",
                "provider": "openai",
                "api_key": "sk-sensitive",
                "model": "gpt-4.1-mini",
            }
        },
    })

    assert "sk-sensitive" not in str(preview)
    assert preview["secret_fields"][0]["state"] == "submitted"
```

```python
def test_trusted_custom_hf_requires_acknowledgement():
    preview = preview_readiness_selection({
        "profile_id": "advanced_custom",
        "lanes": {
            "embeddings_rag": {
                "provider": "huggingface",
                "model": "custom/requires-trust",
                "trusted_custom_model": True,
                "trusted_custom_model_acknowledged": False,
            }
        },
    })

    assert preview["lanes"]["embeddings_rag"]["status"] == "blocked"
```

- [ ] **Step 3: Run test to verify it fails**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/Setup/test_setup_readiness_preview.py -q`

Expected: FAIL with missing service/schema.

- [ ] **Step 4: Implement preview service**

Implementation responsibilities:

- Normalize profile and lane overrides.
- Return config updates only for known keys already accepted by `setup_manager.update_config`.
- Build `InstallPlan` using existing `InstallPlan`, `STTInstall`, `TTSInstall`, and `EmbeddingsInstall`.
- Redact hosted-provider secrets in all returned fields.
- Mark trusted custom HF entries blocked unless acknowledged.
- Mark `restart_required` overlay whenever config updates are non-empty.
- Include skip consequences for skipped chat or embeddings.

- [ ] **Step 5: Run tests to verify pass**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/Setup/test_setup_readiness_preview.py -q`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add tldw_Server_API/app/core/Setup/readiness_service.py \
  tldw_Server_API/app/api/v1/schemas/setup_schemas.py \
  tldw_Server_API/tests/Setup/test_setup_readiness_preview.py
git commit -m "feat: add setup readiness preview contract"
```

---

### Task 3: Read-Only Setup Readiness API

**Files:**
- Modify: `tldw_Server_API/app/api/v1/endpoints/setup.py`
- Modify: `tldw_Server_API/app/api/v1/schemas/setup_schemas.py`
- Test: `tldw_Server_API/tests/Setup/test_setup_readiness_api.py`

- [ ] **Step 1: Write failing endpoint tests**

```python
def test_readiness_profiles_available_during_local_first_run(client, monkeypatch):
    monkeypatch.setattr(setup_endpoint.setup_manager, "get_status_snapshot", lambda: {
        "enabled": True,
        "setup_completed": False,
        "needs_setup": True,
        "placeholder_fields": [],
    })

    response = client.get("/api/v1/setup/readiness/profiles", headers={"host": "localhost"})

    assert response.status_code == 200
    assert [lane["lane_id"] for lane in response.json()["lanes"]] == ["chat", "embeddings_rag", "speech"]
```

```python
def test_readiness_status_reports_overlays_separately(client, monkeypatch):
    response = client.get("/api/v1/setup/readiness/status", headers={"host": "localhost"})

    assert response.status_code == 200
    payload = response.json()
    assert isinstance(payload["overlays"], list)
    assert all(lane["status"] != "restart_required" for lane in payload["lanes"])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/Setup/test_setup_readiness_api.py -q`

Expected: FAIL with 404 for new endpoints.

- [ ] **Step 3: Add read-only endpoints**

Add to `setup.py`:

- `GET /api/v1/setup/readiness/profiles`
- `POST /api/v1/setup/readiness/preview`
- `GET /api/v1/setup/readiness/status`

Use `Depends(require_local_setup_access)` for first-run routes. Keep `openapi_extra={"security": []}` consistent with existing local setup endpoints.

- [ ] **Step 4: Run targeted API tests**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/Setup/test_setup_readiness_api.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tldw_Server_API/app/api/v1/endpoints/setup.py \
  tldw_Server_API/app/api/v1/schemas/setup_schemas.py \
  tldw_Server_API/tests/Setup/test_setup_readiness_api.py
git commit -m "feat: expose setup readiness read APIs"
```

---

### Task 4: Pollable Provision Operation And Readiness Store

**Files:**
- Create: `tldw_Server_API/app/core/Setup/readiness_store.py`
- Modify: `tldw_Server_API/app/core/Setup/readiness_service.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/setup.py`
- Test: `tldw_Server_API/tests/Setup/test_setup_readiness_store.py`
- Test: `tldw_Server_API/tests/Setup/test_setup_readiness_api.py`

- [ ] **Step 1: Write failing store tests**

```python
from tldw_Server_API.app.core.Setup.readiness_store import SetupReadinessStore


def test_readiness_store_round_trips_lane_status(tmp_path):
    store = SetupReadinessStore(tmp_path / "setup_readiness.json")
    saved = store.save({
        "lanes": [{"lane_id": "chat", "status": "skipped"}],
        "overlays": ["restart_required"],
    })

    assert saved["lanes"][0]["status"] == "skipped"
    assert store.load()["overlays"] == ["restart_required"]
```

- [ ] **Step 2: Write failing provision API tests**

```python
def test_provision_returns_pollable_status_without_waiting_for_download(client, monkeypatch):
    monkeypatch.setattr(setup_endpoint.setup_manager, "update_config", lambda *args, **kwargs: None)
    monkeypatch.setattr(setup_endpoint.install_manager, "get_install_status_snapshot", lambda: {"status": "queued"})

    response = client.post(
        "/api/v1/setup/readiness/provision",
        headers={"host": "localhost"},
        json={"preview_id": "preview-1", "confirmed": True},
    )

    assert response.status_code == 202
    assert response.json()["status_url"] == "/api/v1/setup/readiness/status"
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/Setup/test_setup_readiness_store.py tldw_Server_API/tests/Setup/test_setup_readiness_api.py -q`

Expected: FAIL with missing store/provision endpoint.

- [ ] **Step 4: Implement store and provision endpoint**

Implementation responsibilities:

- Use JSON persistence similar to `audio_readiness_store.py`.
- Save last preview/provision selection and lane statuses.
- Require `confirmed=True`.
- Apply config updates through `setup_manager.update_config`.
- Submit install work as a background task or reuse existing install manager status without blocking for downloads.
- Return `202 Accepted` for queued/running work and a pollable `status_url`.
- Preserve `restart_required` overlay until status/verification clears it after restart.

- [ ] **Step 5: Run targeted tests**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/Setup/test_setup_readiness_store.py tldw_Server_API/tests/Setup/test_setup_readiness_api.py -q`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add tldw_Server_API/app/core/Setup/readiness_store.py \
  tldw_Server_API/app/core/Setup/readiness_service.py \
  tldw_Server_API/app/api/v1/endpoints/setup.py \
  tldw_Server_API/tests/Setup/test_setup_readiness_store.py \
  tldw_Server_API/tests/Setup/test_setup_readiness_api.py
git commit -m "feat: add pollable setup readiness provisioning"
```

---

### Task 5: Verification Helpers For Chat, Embeddings, And Speech

**Files:**
- Modify: `tldw_Server_API/app/core/Setup/readiness_service.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/setup.py`
- Test: `tldw_Server_API/tests/Setup/test_setup_readiness_api.py`
- Test: `tldw_Server_API/tests/Setup/test_setup_readiness_preview.py`

- [ ] **Step 1: Write failing verification tests**

```python
def test_verify_skip_does_not_call_hosted_provider(monkeypatch):
    called = False

    async def fail_hosted(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("skip verification must not call hosted provider")

    result = await verify_readiness_lanes({"lanes": {"chat": {"mode": "skip"}}})

    assert called is False
    assert result["lanes"]["chat"]["status"] == "skipped"
```

```python
def test_verify_speech_reuses_audio_bundle_verification(monkeypatch):
    async def fake_verify(bundle_id, resource_profile, tts_choice=None):
        return {"status": "ready", "stt_health": {"status": "ready"}, "tts_health": {"status": "failed"}}

    monkeypatch.setattr(install_manager, "verify_audio_bundle_async", fake_verify)

    result = await verify_readiness_lanes({"lanes": {"speech": {"bundle_id": "cpu_local", "resource_profile": "balanced"}}})

    assert result["lanes"]["speech"]["status"] == "ready_with_warnings"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/Setup/test_setup_readiness_api.py -q`

Expected: FAIL with missing verification behavior.

- [ ] **Step 3: Implement verification boundaries**

Implementation responsibilities:

- Chat:
  - Skip returns `skipped`.
  - Local endpoint validation may be reachability/model-name only in V1.
  - Hosted test requires explicit `verify` action and sanitized errors.
- Embeddings:
  - Use existing embeddings health/provider helpers where cheap.
  - Avoid model downloads or expensive loads unless the user explicitly verifies.
- Speech:
  - Delegate to `install_manager.verify_audio_bundle_async`.
  - Map STT ready + TTS failed into `ready_with_warnings`.

- [ ] **Step 4: Run targeted backend tests**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/Setup/test_setup_readiness_api.py tldw_Server_API/tests/Setup/test_setup_public_error_sanitization.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tldw_Server_API/app/core/Setup/readiness_service.py \
  tldw_Server_API/app/api/v1/endpoints/setup.py \
  tldw_Server_API/tests/Setup/test_setup_readiness_api.py
git commit -m "feat: verify setup readiness lanes"
```

---

### Task 6: WebUI Setup Readiness Client And Hook

**Files:**
- Create: `apps/packages/ui/src/services/tldw/setup-readiness.ts`
- Create: `apps/packages/ui/src/components/Option/Setup/hooks/useSetupReadiness.ts`
- Create: `apps/packages/ui/src/components/Option/Setup/hooks/__tests__/useSetupReadiness.test.tsx`
- Modify: `apps/packages/ui/src/services/tldw/openapi-guard.ts`

- [ ] **Step 1: Write failing hook tests**

```tsx
it("loads readiness profiles and status without provisioning", async () => {
  const requests: string[] = []
  mockBgRequest((init) => {
    requests.push(String(init.path))
    if (String(init.path).endsWith("/profiles")) return profilesFixture
    if (String(init.path).endsWith("/status")) return statusFixture
    return {}
  })

  renderHook(() => useSetupReadiness())

  await waitFor(() => expect(requests).toContain("/api/v1/setup/readiness/profiles"))
  expect(requests).not.toContain("/api/v1/setup/readiness/provision")
})
```

```tsx
it("maps 403 setup guard failures to remote setup blocked fallback", async () => {
  mockBgRequestFailure(403, "Setup access is restricted to local requests.")

  const { result } = renderHook(() => useSetupReadiness())

  await waitFor(() => expect(result.current.guard).toBe("remote_setup_blocked"))
  expect(result.current.fallbackUrl).toBe("/setup")
})
```

- [ ] **Step 2: Run hook tests to verify failure**

Run: `bunx vitest run apps/packages/ui/src/components/Option/Setup/hooks/__tests__/useSetupReadiness.test.tsx`

Expected: FAIL with missing hook/client.

- [ ] **Step 3: Implement client and hook**

Implementation responsibilities:

- Wrap `/api/v1/setup/readiness/profiles`, `/preview`, `/status`, `/provision`, and `/verify`.
- Poll while provisioning status is `queued`, `running`, or `provisioning`.
- Keep `Provision now` as a separate action.
- Map 401/403/404 setup guard cases into display states.
- Preserve fallback link to backend `/setup`.

- [ ] **Step 4: Run hook tests**

Run: `bunx vitest run apps/packages/ui/src/components/Option/Setup/hooks/__tests__/useSetupReadiness.test.tsx`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add apps/packages/ui/src/services/tldw/setup-readiness.ts \
  apps/packages/ui/src/components/Option/Setup/hooks/useSetupReadiness.ts \
  apps/packages/ui/src/components/Option/Setup/hooks/__tests__/useSetupReadiness.test.tsx \
  apps/packages/ui/src/services/tldw/openapi-guard.ts
git commit -m "feat: add WebUI setup readiness client"
```

---

### Task 7: Native WebUI First-Run Readiness Screen

**Files:**
- Create: `apps/packages/ui/src/components/Option/Setup/ReadinessSetupScreen.tsx`
- Create: `apps/packages/ui/src/components/Option/Setup/__tests__/ReadinessSetupScreen.test.tsx`
- Modify: `apps/packages/ui/src/routes/option-setup.tsx`

- [ ] **Step 1: Write failing component tests**

```tsx
it("renders profile picker, canonical lanes, and secondary TTS copy", async () => {
  render(<ReadinessSetupScreen />)

  expect(await screen.findByText("Local Balanced")).toBeInTheDocument()
  expect(screen.getByText("Chat")).toBeInTheDocument()
  expect(screen.getByText("Embeddings/RAG")).toBeInTheDocument()
  expect(screen.getByText("Speech")).toBeInTheDocument()
  expect(screen.getByText(/TTS/i)).toHaveClass("secondary")
})
```

```tsx
it("does not provision until Provision now is clicked", async () => {
  render(<ReadinessSetupScreen />)

  await userEvent.click(await screen.findByRole("radio", { name: /Local Balanced/i }))

  expect(mockProvision).not.toHaveBeenCalled()

  await userEvent.click(screen.getByRole("button", { name: /Provision now/i }))
  expect(mockProvision).toHaveBeenCalledTimes(1)
})
```

- [ ] **Step 2: Run tests to verify failure**

Run: `bunx vitest run apps/packages/ui/src/components/Option/Setup/__tests__/ReadinessSetupScreen.test.tsx`

Expected: FAIL with missing component.

- [ ] **Step 3: Implement screen**

Implementation responsibilities:

- Compact readiness dashboard, not a marketing landing page.
- Profile picker first, advanced controls behind disclosure.
- Lane cards for Chat, Embeddings/RAG, and Speech.
- TTS visible but secondary in the Speech lane.
- Preview panel with config changes, install plan, warnings, blockers, and restart overlay.
- Separate `Provision now` button.
- Fallback `/setup` link always visible.
- Completion area with skipped-lane consequences.

- [ ] **Step 4: Wire `/setup` route**

In `option-setup.tsx`:

- Render native readiness screen when a backend is configured and setup readiness APIs load.
- Keep existing `OnboardingWizard` when the server URL is missing or connection setup is still needed.
- Show remote setup blocked state when hook reports setup guard failure.

- [ ] **Step 5: Run component tests**

Run: `bunx vitest run apps/packages/ui/src/components/Option/Setup/__tests__/ReadinessSetupScreen.test.tsx`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add apps/packages/ui/src/components/Option/Setup/ReadinessSetupScreen.tsx \
  apps/packages/ui/src/components/Option/Setup/__tests__/ReadinessSetupScreen.test.tsx \
  apps/packages/ui/src/routes/option-setup.tsx
git commit -m "feat: add native first-run readiness setup screen"
```

---

### Task 8: Admin Post-Setup Entry And Shared Permission States

**Files:**
- Modify: `tldw_Server_API/app/api/v1/endpoints/setup.py`
- Modify: `apps/packages/ui/src/components/Option/Setup/ReadinessSetupScreen.tsx`
- Modify: `apps/packages/ui/src/components/Option/Setup/hooks/useSetupReadiness.ts`
- Test: `tldw_Server_API/tests/Setup/test_setup_readiness_api.py`
- Test: `apps/packages/ui/src/components/Option/Setup/__tests__/ReadinessSetupScreen.test.tsx`

- [ ] **Step 1: Write failing admin endpoint tests**

```python
def test_admin_readiness_available_after_setup_completed(admin_client, monkeypatch):
    monkeypatch.setattr(setup_endpoint.setup_manager, "get_status_snapshot", lambda: {
        "enabled": False,
        "setup_completed": True,
        "needs_setup": False,
    })

    response = admin_client.get("/api/v1/setup/admin/readiness/status")

    assert response.status_code == 200
```

```python
def test_non_admin_post_setup_cannot_provision(non_admin_client):
    response = non_admin_client.post("/api/v1/setup/admin/readiness/provision", json={})

    assert response.status_code in {401, 403}
```

- [ ] **Step 2: Run tests to verify failure**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/Setup/test_setup_readiness_api.py -q`

Expected: FAIL with missing admin readiness paths.

- [ ] **Step 3: Add admin readiness endpoints**

Endpoint behavior:

- First-run paths stay local setup guarded.
- Post-setup paths use `require_shared_audio_installer_access` or a stricter admin/system-configure dependency.
- Regular users can see admin-required UI state but cannot provision server-wide models.

- [ ] **Step 4: Add frontend admin state tests**

Run: `bunx vitest run apps/packages/ui/src/components/Option/Setup/__tests__/ReadinessSetupScreen.test.tsx`

Expected: PASS after admin state implementation.

- [ ] **Step 5: Commit**

```bash
git add tldw_Server_API/app/api/v1/endpoints/setup.py \
  tldw_Server_API/tests/Setup/test_setup_readiness_api.py \
  apps/packages/ui/src/components/Option/Setup/ReadinessSetupScreen.tsx \
  apps/packages/ui/src/components/Option/Setup/hooks/useSetupReadiness.ts \
  apps/packages/ui/src/components/Option/Setup/__tests__/ReadinessSetupScreen.test.tsx
git commit -m "feat: add admin setup readiness controls"
```

---

### Task 9: Contract Verification, Docs, And Browser QA

**Files:**
- Modify: `Docs/Development/Setup.md` or nearest current setup doc.
- Modify: `apps/packages/ui/src/services/tldw/openapi-guard.ts` if not already updated.
- Modify: `backlog/tasks/task-427 - Plan-first-time-readiness-setup-implementation.md`

- [ ] **Step 1: Run backend setup tests**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/Setup/test_setup_readiness_profiles.py tldw_Server_API/tests/Setup/test_setup_readiness_preview.py tldw_Server_API/tests/Setup/test_setup_readiness_store.py tldw_Server_API/tests/Setup/test_setup_readiness_api.py tldw_Server_API/tests/Setup/test_setup_audio_installer_lifecycle_api.py tldw_Server_API/tests/Setup/test_setup_manager_masking.py -q`

Expected: PASS.

- [ ] **Step 2: Run frontend setup tests**

Run: `bunx vitest run apps/packages/ui/src/components/Option/Setup/__tests__/ReadinessSetupScreen.test.tsx apps/packages/ui/src/components/Option/Setup/hooks/__tests__/useSetupReadiness.test.tsx apps/packages/ui/src/components/Option/Setup/__tests__/AudioInstallerPanel.test.tsx`

Expected: PASS.

- [ ] **Step 3: Run OpenAPI verification**

Run from `apps/packages/ui`: `bun run verify:openapi`

Expected: PASS, or update `openapi-guard.ts`/generated spec according to the verifier output.

- [ ] **Step 4: Run Bandit on touched backend code**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m bandit -r tldw_Server_API/app/core/Setup/readiness_models.py tldw_Server_API/app/core/Setup/readiness_profiles.py tldw_Server_API/app/core/Setup/readiness_service.py tldw_Server_API/app/core/Setup/readiness_store.py tldw_Server_API/app/api/v1/endpoints/setup.py -f json -o /tmp/bandit_first_time_readiness_setup.json`

Expected: no new findings in touched code.

- [ ] **Step 5: Browser QA**

Start the backend and WebUI with the repo's normal local commands, then verify:

- Setup-required backend shows native WebUI readiness screen.
- Fallback `/setup` link is visible.
- Profile selection does not provision.
- `Provision now` triggers pollable status.
- Remote/proxy guard failure shows fallback instead of weakening setup protections.
- Admin post-setup screen shows admin-required state for non-admin and controls for admin.

- [ ] **Step 6: Update docs and Backlog final summaries**

Update setup docs with:

- first-run readiness lanes
- explicit provisioning behavior
- skip consequences
- post-setup admin-only controls
- backend `/setup` fallback

- [ ] **Step 7: Commit final verification/docs**

```bash
git add Docs/Development/Setup.md \
  apps/packages/ui/src/services/tldw/openapi-guard.ts \
  "backlog/tasks/task-427 - Plan-first-time-readiness-setup-implementation.md"
git commit -m "docs: document first-time readiness setup flow"
```

---

## Completion Checklist

- [ ] Backend readiness profiles expose canonical lanes and overlay semantics.
- [ ] Preview returns config/install plan without mutation.
- [ ] Provisioning requires `Provision now` and returns pollable status.
- [ ] Secrets are write-only and never echoed.
- [ ] Trusted custom HF models require explicit acknowledgement.
- [ ] Verification is explicit for hosted calls and expensive local checks.
- [ ] WebUI first-run setup is native when setup guard permits it.
- [ ] Backend `/setup` fallback remains visible.
- [ ] Post-setup provisioning is admin-only.
- [ ] Backend tests pass.
- [ ] Frontend tests pass.
- [ ] OpenAPI guard passes.
- [ ] Bandit reports no new findings in touched backend code.
- [ ] Browser QA covers first-run and post-setup paths.
