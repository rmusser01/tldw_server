# Onboarding Confidence Flow Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make first-time solo onboarding reliable enough for a new user to configure a provider, validate it, complete a real first chat, and immediately add one useful source.

**Architecture:** Build on the merged unified onboarding flow and the existing setup/readiness APIs. Backend remains authoritative for first-run state, provider persistence, readiness, first-chat verification, skip, and completion; frontend state remains limited to unsaved form values, transient validation cache, readiness disclosure state, recovery UI state, and dismissed tips.

**Tech Stack:** FastAPI, Pydantic, pytest, Next.js, React, TypeScript, Vitest, Playwright, Backlog.md.

---

## Source Documents

- Spec: `Docs/superpowers/specs/2026-06-01-onboarding-confidence-flow-design.md`
- Backlog: `TASK-583`
- Existing backend provider validation: `tldw_Server_API/app/core/Setup/provider_validation.py`
- Existing backend schemas: `tldw_Server_API/app/api/v1/schemas/setup_schemas.py`
- Existing frontend onboarding client: `apps/packages/ui/src/services/tldw/domains/setup-onboarding.ts`
- Existing frontend readiness client: `apps/packages/ui/src/services/tldw/setup-readiness.ts`
- Existing post-onboarding readiness hook: `apps/packages/ui/src/hooks/usePostOnboardingMediaReadiness.ts`
- Existing first-source quick ingest opener: `apps/packages/ui/src/utils/quick-ingest-open.ts`

## File Map

Backend provider validation:

- Modify `tldw_Server_API/app/api/v1/schemas/setup_schemas.py`
  - Add optional non-secret validation metadata to `SetupProviderValidationResponse`: `validation_level` and `can_gate_first_chat`.
- Modify `tldw_Server_API/app/core/Setup/provider_validation.py`
  - Populate the new response fields for hosted accepted checks, local OpenAI-compatible checks, native Kobold checks, and failures.
- Modify `tldw_Server_API/tests/Setup/test_setup_provider_validation.py`
  - Lock accepted hosted validation, ready local validation, failed validation, discovered models, and secret-safety semantics.

Frontend provider validation:

- Modify `apps/packages/ui/src/types/setup-onboarding.ts`
  - Mirror the additive validation response fields.
- Modify `apps/packages/ui/src/components/Option/Onboarding/steps/ProviderSetupStep.tsx`
  - Add manual `Validate` actions, validation cache, model discovery display, invalidation on edit, and the default-provider gate.
- Modify `apps/packages/ui/src/components/Option/Onboarding/UnifiedSetupWizard.tsx`
  - Pass `validateProvider` to `ProviderSetupStep` and refresh setup readiness after provider validation/save.
- Modify `apps/packages/ui/src/components/Option/Onboarding/__tests__/ProviderSetupStep.test.tsx`
  - Cover default-provider validation gate, non-default unverified save, edit invalidation, accepted vs ready copy, and model discovery fallback.
- Modify `apps/packages/ui/src/components/Option/Onboarding/__tests__/UnifiedSetupWizard.test.tsx`
  - Cover validation plumbing and wizard advance only after provider save plus validation.

Readiness panel:

- Create `apps/packages/ui/src/hooks/useSetupReadinessSummary.ts`
  - Thin first-run wrapper around `getSetupReadinessStatus({ mode: "first-run" })`.
- Create `apps/packages/ui/src/components/Option/Onboarding/SetupReadinessPanel.tsx`
  - Compact lane summary for Chat, Embeddings/RAG, and Speech using existing readiness response shapes.
- Create `apps/packages/ui/src/components/Option/Onboarding/__tests__/SetupReadinessPanel.test.tsx`
  - Cover lane statuses, blockers, warnings, overlays, retry, and collapsed details.
- Modify `apps/packages/ui/src/components/Option/Onboarding/UnifiedSetupWizard.tsx`
  - Render the panel in the focused shell and refresh it after setup-changing actions.
- Modify `apps/packages/ui/src/hooks/__tests__/useSetupReadinessSummary.test.tsx`
  - Cover successful status load and request failure.

First-chat recovery:

- Modify `apps/packages/ui/src/components/Option/Onboarding/steps/FirstChatStep.tsx`
  - Add inline recovery actions: retry, edit provider, switch provider, skip setup, and check endpoint.
- Modify `apps/packages/ui/src/components/Option/Onboarding/UnifiedSetupWizard.tsx`
  - Wire recovery callbacks to provider setup, skip, and local endpoint/provider validation paths.
- Modify `apps/packages/ui/src/components/Option/Onboarding/__tests__/FirstChatStep.test.tsx`
  - Cover categorized failure copy and all recovery buttons.
- Modify `apps/packages/ui/src/components/Option/Onboarding/__tests__/UnifiedSetupWizard.test.tsx`
  - Cover first-chat recovery navigation and skip propagation.

First-source milestone:

- Modify `apps/packages/ui/src/components/Option/Onboarding/FirstSourceMilestonePrompt.tsx`
  - Upgrade from a single CTA to a three-choice picker: Web URL, File upload, Paste text.
- Modify `apps/packages/ui/src/utils/quick-ingest-open.ts`
  - Add typed first-source kind metadata while preserving the existing first-source session seed behavior.
- Modify `apps/packages/ui/src/routes/option-index.tsx`
  - Pass the selected source kind to quick ingest, watch quick ingest run summary, and only offer grounded chat after an ingest success has a queryable media id or backend media readiness is confirmed.
- Modify `apps/packages/ui/src/components/Option/Onboarding/__tests__/FirstSourceMilestonePrompt.test.tsx`
  - Cover default Web URL choice, file/paste choices, progress/error/success states, dismiss, retry, and ask-about-source visibility.
- Modify `apps/packages/ui/src/routes/__tests__/option-index.unified-setup.test.tsx`
  - Cover selected source kind propagation and readiness-gated grounded chat CTA.
- Modify `apps/tldw-frontend/e2e/workflows/unified-first-run-onboarding.spec.ts`
  - Extend the mocked unified first-run journey to validate provider before continue, show readiness, recover from one first-chat failure, and open first-source ingest after completion.

## Commit Plan

Ship one PR with four staged commits:

1. `feat: validate onboarding providers before first chat`
2. `feat: show setup readiness in onboarding wizard`
3. `fix: add first-chat onboarding recovery actions`
4. `feat: guide first source after onboarding`

Backlog task updates, plan status updates, and verification notes should be committed with the related staged commit unless the user asks for different staging.

## Review Adjustments Before Execution

This plan deliberately avoids two implementation traps found during local review:

- Provider validation should use a validate-first happy path. The gate is "validated and saved"; saving may clear the raw API key from the UI, so validation must survive that safe key clearing when a masked saved key is present.
- First-source "Ask a question about this source" must not overclaim grounded readiness. If implementation cannot find an existing backend source/queryable readiness signal, keep the ask action hidden and show a safe source/view action instead of presenting grounded chat as ready.

---

## Task 1: Provider Validation Gate

**Goal:** The default first-chat provider cannot continue until the user manually validates it and saves it. Non-default providers can still be saved unverified.

**Files:**

- Modify `tldw_Server_API/app/api/v1/schemas/setup_schemas.py`
- Modify `tldw_Server_API/app/core/Setup/provider_validation.py`
- Modify `tldw_Server_API/tests/Setup/test_setup_provider_validation.py`
- Modify `apps/packages/ui/src/types/setup-onboarding.ts`
- Modify `apps/packages/ui/src/components/Option/Onboarding/steps/ProviderSetupStep.tsx`
- Modify `apps/packages/ui/src/components/Option/Onboarding/UnifiedSetupWizard.tsx`
- Modify `apps/packages/ui/src/components/Option/Onboarding/__tests__/ProviderSetupStep.test.tsx`
- Modify `apps/packages/ui/src/components/Option/Onboarding/__tests__/UnifiedSetupWizard.test.tsx`
- Update `backlog/tasks/task-583 - Design-onboarding-confidence-flow-follow-up.md`

### Backend Contract

Add fields:

```python
class SetupProviderValidationResponse(BaseModel):
    """Provider validation result safe to return to unauthenticated setup clients."""

    provider_key: str
    status: str
    failure_category: str | None = None
    message: str | None = None
    models: list[str] = Field(default_factory=list)
    validation_level: str | None = None
    can_gate_first_chat: bool = False
```

Response semantics:

- `ready`, `validation_level="live_non_generative"`, `can_gate_first_chat=True`: local OpenAI-compatible `/models` validation succeeded.
- `ready`, `validation_level="live_endpoint_shape"`, `can_gate_first_chat=True`: existing native Kobold endpoint shape check succeeded. Do not expand this into new generative checks.
- `accepted`, `validation_level="local_syntax"`, `can_gate_first_chat=True`: hosted key passed local syntax/presence checks, but first chat remains the live verification.
- `failed`, `can_gate_first_chat=False`: auth, required field, endpoint, model, API shape, or unknown validation failure.

### Frontend Contract

Add fields:

```ts
export type SetupProviderValidationResponse = {
  provider_key: string
  status: string
  failure_category?: string | null
  message?: string | null
  models: string[]
  validation_level?: string | null
  can_gate_first_chat?: boolean
}
```

Add local validation helpers near `ProviderSetupStep`:

```ts
type ProviderValidationViewState = {
  fingerprint: string
  response: SetupProviderValidationResponse
}

const validationCanGate = (
  response: SetupProviderValidationResponse | null | undefined
) =>
  Boolean(
    response?.can_gate_first_chat ??
      (response?.status === "ready" || response?.status === "accepted")
  )
```

Provider fingerprint must never include raw secrets:

```ts
const providerValidationFingerprint = (
  provider: SetupProviderCatalogEntry,
  values: ProviderFormValues,
  saved: SetupProviderSaveResponse | undefined,
  model: string
) =>
  [
    provider.provider_key,
    provider.provider_type === "local_endpoint"
      ? values.baseUrl.trim() || provider.default_base_url || ""
      : "",
    model.trim(),
    values.apiKey.trim() || saved?.masked_api_key ? "secret-present" : "no-secret"
  ].join("|")
```

When a user edits `apiKey`, `baseUrl`, or `model`, clear that provider's validation result immediately. Do not clear validation when the component clears the raw API key after a successful save and a masked key is present.

### Steps

- [ ] **Step 1: Add backend failing tests for response metadata**

Add assertions to `test_hosted_provider_validation_accepts_plausible_openai_key_without_echo`, `test_openai_models_shape_maps_to_ready`, and one failed-validation test:

```python
assert response.validation_level == "local_syntax"
assert response.can_gate_first_chat is True
```

For ready local OpenAI-compatible validation:

```python
assert response.validation_level == "live_non_generative"
assert response.can_gate_first_chat is True
```

For failed validation:

```python
assert response.can_gate_first_chat is False
```

- [ ] **Step 2: Run backend RED**

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Setup/test_setup_provider_validation.py -q
```

Expected: FAIL because `validation_level` and `can_gate_first_chat` do not exist yet.

- [ ] **Step 3: Implement backend metadata**

Update `SetupProviderValidationResponse` and populate fields in `provider_validation.py`. Prefer tiny response helpers:

```python
def _ready_response(
    provider_key: str,
    *,
    validation_level: str,
    models: list[str] | None = None,
) -> SetupProviderValidationResponse:
    return SetupProviderValidationResponse(
        provider_key=provider_key,
        status=VALIDATION_STATUS_READY,
        models=models or [],
        validation_level=validation_level,
        can_gate_first_chat=True,
    )
```

Use the same pattern for accepted and failed responses.

- [ ] **Step 4: Run backend GREEN**

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Setup/test_setup_provider_validation.py -q
```

Expected: PASS.

- [ ] **Step 5: Add frontend failing tests for provider validation**

In `ProviderSetupStep.test.tsx`, add tests that assert:

- Continue is disabled until the default provider is both validated and saved.
- The happy path validates first, saves second, and the validation result remains valid after the saved provider clears the raw API key while retaining a masked key.
- `accepted` validation enables continue but shows copy like "First chat verifies this provider."
- Failed validation keeps continue disabled and shows a categorized message.
- Non-default selected provider can be saved without validation.
- Discovered models are visible while manual model entry remains available.
- Editing the default model invalidates the validation result.

Representative test shape:

```tsx
const validateProvider = vi.fn().mockResolvedValue({
  provider_key: "openai",
  status: "accepted",
  validation_level: "local_syntax",
  can_gate_first_chat: true,
  models: []
})

render(
  <ProviderSetupStep
    providers={[openAiProvider]}
    onSaveProvider={saveProvider}
    onValidateProvider={validateProvider}
    onContinue={onContinue}
  />
)
```

- [ ] **Step 6: Run frontend RED**

Run:

```bash
bunx vitest run apps/packages/ui/src/components/Option/Onboarding/__tests__/ProviderSetupStep.test.tsx apps/packages/ui/src/components/Option/Onboarding/__tests__/UnifiedSetupWizard.test.tsx
```

Expected: FAIL because the component has no validation prop or validation UI.

- [ ] **Step 7: Implement ProviderSetupStep validation UI**

Add prop:

```ts
onValidateProvider: (
  payload: SetupProviderSaveRequest
) => Promise<SetupProviderValidationResponse>
```

Add per-provider `Validate` button, visible states, model discovery, and `canContinue`:

```ts
const defaultValidation =
  validationByProvider[defaultProvider]?.fingerprint === currentFingerprint
    ? validationByProvider[defaultProvider]?.response
    : null

const canContinue = Boolean(
  defaultProvider &&
    selectedDefaultModel &&
    savedProviders[defaultProvider] &&
    validationCanGate(defaultValidation)
)
```

Validation request uses the same sanitized save payload shape as persistence. Validation failures must never render raw keys or raw endpoint exception text.

- [ ] **Step 8: Wire validation through UnifiedSetupWizard**

Destructure `validateProvider` from `useSetupOnboarding()`.

Pass to `ProviderSetupStep`:

```tsx
onValidateProvider={async (payload) => {
  const response = await validateProvider(payload)
  await refreshReadiness?.()
  return response
}}
```

If the readiness hook is added in Task 2, use a temporary no-op here and replace it in Task 2.

- [ ] **Step 9: Run frontend GREEN**

Run:

```bash
bunx vitest run apps/packages/ui/src/components/Option/Onboarding/__tests__/ProviderSetupStep.test.tsx apps/packages/ui/src/components/Option/Onboarding/__tests__/UnifiedSetupWizard.test.tsx
```

Expected: PASS.

- [ ] **Step 10: Commit Task 1**

Run:

```bash
git add tldw_Server_API/app/api/v1/schemas/setup_schemas.py tldw_Server_API/app/core/Setup/provider_validation.py tldw_Server_API/tests/Setup/test_setup_provider_validation.py apps/packages/ui/src/types/setup-onboarding.ts apps/packages/ui/src/components/Option/Onboarding/steps/ProviderSetupStep.tsx apps/packages/ui/src/components/Option/Onboarding/UnifiedSetupWizard.tsx apps/packages/ui/src/components/Option/Onboarding/__tests__/ProviderSetupStep.test.tsx apps/packages/ui/src/components/Option/Onboarding/__tests__/UnifiedSetupWizard.test.tsx "backlog/tasks/task-583 - Design-onboarding-confidence-flow-follow-up.md"
git diff --cached --check
git commit -m "feat: validate onboarding providers before first chat"
```

---

## Task 2: Setup Readiness Panel

**Goal:** Show a compact, always-visible readiness summary in the focused wizard shell without making optional readiness lanes mandatory.

**Files:**

- Create `apps/packages/ui/src/hooks/useSetupReadinessSummary.ts`
- Create `apps/packages/ui/src/hooks/__tests__/useSetupReadinessSummary.test.tsx`
- Create `apps/packages/ui/src/components/Option/Onboarding/SetupReadinessPanel.tsx`
- Create `apps/packages/ui/src/components/Option/Onboarding/__tests__/SetupReadinessPanel.test.tsx`
- Modify `apps/packages/ui/src/components/Option/Onboarding/UnifiedSetupWizard.tsx`
- Update `backlog/tasks/task-583 - Design-onboarding-confidence-flow-follow-up.md`

### Component Contract

`SetupReadinessPanel` should accept already-loaded backend data and no-op gracefully when readiness is unavailable:

```ts
type SetupReadinessPanelProps = {
  status: SetupReadinessStatusResponse | null
  loading?: boolean
  error?: string | null
  onRetry?: () => void
}
```

Status labels:

- Chat lane: blocking when backend says `failed` or `blocked`; ready when `ready` or `ready_with_warnings`.
- Embeddings/RAG lane: optional if not configured, skipped, or blocked by packages/downloads unless user opted in.
- Speech lane: optional if not configured, skipped, or blocked by packages/downloads unless user opted in.

Use backend overlays directly. Expected overlay labels:

- `restart_required`
- `admin_required`
- `remote_setup_blocked`
- `downloads_disabled`
- `package_installs_disabled`
- `network_unavailable`

### Steps

- [ ] **Step 1: Write hook tests**

In `useSetupReadinessSummary.test.tsx`, mock `getSetupReadinessStatus` and assert:

- Hook loads status on mount.
- Hook exposes `refresh`.
- Hook exposes sanitized error state on failure.

- [ ] **Step 2: Write panel tests**

In `SetupReadinessPanel.test.tsx`, render sample lanes:

```ts
const status = {
  active_overlays: ["downloads_disabled"],
  lanes: [
    { lane_id: "chat", label: "Chat", status: "ready" },
    {
      lane_id: "embeddings_rag",
      label: "Embeddings/RAG",
      status: "ready_with_warnings",
      warnings: ["Embedding provider not configured."]
    },
    {
      lane_id: "speech",
      label: "Speech",
      status: "blocked",
      blockers: ["Audio downloads disabled."]
    }
  ]
}
```

Assert lane labels, warning/blocker disclosure, optional lane language, and retry button.

- [ ] **Step 3: Run RED**

Run:

```bash
bunx vitest run apps/packages/ui/src/hooks/__tests__/useSetupReadinessSummary.test.tsx apps/packages/ui/src/components/Option/Onboarding/__tests__/SetupReadinessPanel.test.tsx
```

Expected: FAIL because the hook and panel do not exist.

- [ ] **Step 4: Implement useSetupReadinessSummary**

Use the existing client only:

```ts
import { getSetupReadinessStatus } from "@/services/tldw/setup-readiness"
```

Do not duplicate readiness types.

- [ ] **Step 5: Implement SetupReadinessPanel**

Use compact layout, expandable lane details, and stable button dimensions. Keep it outside the main step card so it reads as shell state, not another wizard step.

- [ ] **Step 6: Wire the panel into UnifiedSetupWizard**

Render below the shell header and above the active step. Refresh after provider save, provider validation, ingest defaults save, audio defaults save, optional advanced save, first-chat success, setup completion, and skip failure retry.

- [ ] **Step 7: Run GREEN**

Run:

```bash
bunx vitest run apps/packages/ui/src/hooks/__tests__/useSetupReadinessSummary.test.tsx apps/packages/ui/src/components/Option/Onboarding/__tests__/SetupReadinessPanel.test.tsx apps/packages/ui/src/components/Option/Onboarding/__tests__/UnifiedSetupWizard.test.tsx
```

Expected: PASS.

- [ ] **Step 8: Commit Task 2**

Run:

```bash
git add apps/packages/ui/src/hooks/useSetupReadinessSummary.ts apps/packages/ui/src/hooks/__tests__/useSetupReadinessSummary.test.tsx apps/packages/ui/src/components/Option/Onboarding/SetupReadinessPanel.tsx apps/packages/ui/src/components/Option/Onboarding/__tests__/SetupReadinessPanel.test.tsx apps/packages/ui/src/components/Option/Onboarding/UnifiedSetupWizard.tsx "backlog/tasks/task-583 - Design-onboarding-confidence-flow-follow-up.md"
git diff --cached --check
git commit -m "feat: show setup readiness in onboarding wizard"
```

---

## Task 3: First-Chat Recovery Actions

**Goal:** A failed first-chat attempt stays visible inline and gives the user explicit recovery buttons instead of a generic failure state.

**Files:**

- Modify `apps/packages/ui/src/components/Option/Onboarding/steps/FirstChatStep.tsx`
- Modify `apps/packages/ui/src/components/Option/Onboarding/UnifiedSetupWizard.tsx`
- Modify `apps/packages/ui/src/components/Option/Onboarding/__tests__/FirstChatStep.test.tsx`
- Modify `apps/packages/ui/src/components/Option/Onboarding/__tests__/UnifiedSetupWizard.test.tsx`
- Update `backlog/tasks/task-583 - Design-onboarding-confidence-flow-follow-up.md`

### Recovery Mapping

Normalize backend categories before display:

```ts
type FirstChatRecoveryCategory =
  | "auth_failed"
  | "quota_or_rate_limit"
  | "endpoint_unreachable"
  | "unsupported_api_shape"
  | "model_unavailable"
  | "config_write_failed"
  | "provider_unvalidated"
  | "unknown"
```

Map aliases:

- `auth`, `authentication_failed`, `provider_api_key_invalid` -> `auth_failed`
- `rate_limit`, `quota`, `quota_exceeded` -> `quota_or_rate_limit`
- `local_provider_unreachable`, `endpoint_unreachable` -> `endpoint_unreachable`
- `unsupported_api_shape` -> `unsupported_api_shape`
- `model_not_found`, `model_unavailable` -> `model_unavailable`
- `provider_unvalidated` -> `provider_unvalidated`
- otherwise `unknown`

### Steps

- [ ] **Step 1: Write FirstChatStep recovery tests**

Add tests for:

- Failed auth response shows credential copy and buttons: Retry, Edit provider, Switch provider, Skip setup.
- Endpoint failure shows Check endpoint.
- Retry calls `verifyFirstChat` again and keeps the failed attempt visible until a new result exists.
- Completion failure after successful chat still uses the separate completion error copy.

- [ ] **Step 2: Run RED**

Run:

```bash
bunx vitest run apps/packages/ui/src/components/Option/Onboarding/__tests__/FirstChatStep.test.tsx
```

Expected: FAIL because recovery buttons and category mapping do not exist.

- [ ] **Step 3: Implement FirstChatStep recovery props**

Add props:

```ts
onEditProvider: () => void
onSwitchProvider: () => void
onSkip: () => void
onCheckEndpoint?: () => void
```

Keep `onBack` as a plain back action if existing tests or users rely on it, but route visible recovery buttons through the explicit callbacks.

- [ ] **Step 4: Implement categorized recovery UI**

Use stable, accessible buttons. Do not auto-navigate. Render endpoint-specific guidance only for endpoint/API-shape failures.

- [ ] **Step 5: Wire callbacks in UnifiedSetupWizard**

- `Edit provider`: set step to `provider_setup`.
- `Switch provider`: set step to `provider_setup`.
- `Check endpoint`: set step to `provider_setup`; keep any current provider selection.
- `Skip setup`: call existing `handleSkip`.

- [ ] **Step 6: Run GREEN**

Run:

```bash
bunx vitest run apps/packages/ui/src/components/Option/Onboarding/__tests__/FirstChatStep.test.tsx apps/packages/ui/src/components/Option/Onboarding/__tests__/UnifiedSetupWizard.test.tsx
```

Expected: PASS.

- [ ] **Step 7: Commit Task 3**

Run:

```bash
git add apps/packages/ui/src/components/Option/Onboarding/steps/FirstChatStep.tsx apps/packages/ui/src/components/Option/Onboarding/UnifiedSetupWizard.tsx apps/packages/ui/src/components/Option/Onboarding/__tests__/FirstChatStep.test.tsx apps/packages/ui/src/components/Option/Onboarding/__tests__/UnifiedSetupWizard.test.tsx "backlog/tasks/task-583 - Design-onboarding-confidence-flow-follow-up.md"
git diff --cached --check
git commit -m "fix: add first-chat onboarding recovery actions"
```

---

## Task 4: First-Source Guided Milestone

**Goal:** After first chat completes, immediately offer one guided first-source flow with Web URL selected by default, while only offering grounded chat after ingest readiness is verified.

**Files:**

- Modify `apps/packages/ui/src/components/Option/Onboarding/FirstSourceMilestonePrompt.tsx`
- Modify `apps/packages/ui/src/components/Option/Onboarding/__tests__/FirstSourceMilestonePrompt.test.tsx`
- Modify `apps/packages/ui/src/utils/quick-ingest-open.ts`
- Modify `apps/packages/ui/src/routes/option-index.tsx`
- Modify `apps/packages/ui/src/routes/__tests__/option-index.unified-setup.test.tsx`
- Modify `apps/tldw-frontend/e2e/workflows/unified-first-run-onboarding.spec.ts`
- Update `backlog/tasks/task-583 - Design-onboarding-confidence-flow-follow-up.md`

### Prompt Contract

Add source kind:

```ts
export type FirstSourceKind = "web_url" | "file_upload" | "paste_text"
```

Prompt props:

```ts
type FirstSourceMilestonePromptProps = {
  readinessStatus: "idle" | "processing" | "ready" | "error"
  lastSourceLabel?: string | null
  errorMessage?: string | null
  onAddSource: (kind: FirstSourceKind) => void
  onAskAboutSource?: () => void
  onRetry?: () => void
  onDismiss: () => void
}
```

Default selected kind is `web_url`.

### Quick Ingest Contract

Extend first-source quick ingest detail without breaking existing seed behavior:

```ts
type FirstSourceQuickIngestKind = "web_url" | "file_upload" | "paste_text"
```

Add optional metadata to the existing `source: "first_source_milestone"` detail:

```ts
firstSourceKind?: FirstSourceQuickIngestKind
```

The current `createQuickIngestSessionSeedFromOpenDetail` should still return a first-source seed when `source === "first_source_milestone"` or `firstSource === true`.

### Readiness Gate

Use existing quick ingest and media readiness state:

- Show picker before ingest starts.
- Show processing copy while the quick ingest session is running.
- Show retry/edit actions on quick ingest failure.
- Show `Ask a question about this source` only when:
  - `useQuickIngestStore().lastRunSummary.status === "success"`, and
  - `lastRunSummary.firstMediaId` is present, and
  - media readiness is `ready`.
- Dispatch `tldw:discuss-media` with `{ mediaId, title, mode: "rag_media" }` for the ask-about-source action.

If the backend cannot confirm media readiness, keep the user on a safe "view or add another source" action and do not offer grounded chat.

### Steps

- [ ] **Step 1: Write prompt tests**

Cover:

- Web URL is selected by default.
- File upload and Paste text can be selected.
- `onAddSource("web_url")`, `onAddSource("file_upload")`, and `onAddSource("paste_text")` fire correctly.
- Processing state hides the picker and shows progress copy.
- Error state shows retry.
- Ready state shows Ask about this source only when provided.

- [ ] **Step 2: Write route tests**

In `option-index.unified-setup.test.tsx`, cover:

- First-source detail includes `firstSourceKind: "web_url"` by default.
- Selecting file/paste changes the quick ingest detail.
- Ask-about-source is not shown before media readiness.
- Ask-about-source dispatches `tldw:discuss-media` after quick ingest success and readiness.

- [ ] **Step 3: Run RED**

Run:

```bash
bunx vitest run apps/packages/ui/src/components/Option/Onboarding/__tests__/FirstSourceMilestonePrompt.test.tsx apps/packages/ui/src/routes/__tests__/option-index.unified-setup.test.tsx
```

Expected: FAIL because the picker and readiness-gated ask action do not exist.

- [ ] **Step 4: Implement FirstSourceMilestonePrompt picker**

Use radio or segmented buttons with stable dimensions. Keep buttons concise:

- Web URL
- File
- Paste

Use `FilePlus2`, `Link`, and `ClipboardPaste` lucide icons where available.

- [ ] **Step 5: Extend quick-ingest detail typing**

Update `QuickIngestOpenDetail` to include `firstSourceKind`. Keep all existing branches compatible.

- [ ] **Step 6: Wire OptionIndex**

Use:

```ts
requestQuickIngestOpen(
  {
    source: "first_source_milestone",
    preferredPreset: "quick",
    firstSource: true,
    firstSourceKind: selectedKind
  },
  { focusTrigger: true }
)
```

Subscribe to `useQuickIngestStore` for `lastRunSummary`; derive prompt readiness state from `lastRunSummary` and `usePostOnboardingMediaReadiness`.

- [ ] **Step 7: Run frontend GREEN**

Run:

```bash
bunx vitest run apps/packages/ui/src/components/Option/Onboarding/__tests__/FirstSourceMilestonePrompt.test.tsx apps/packages/ui/src/routes/__tests__/option-index.unified-setup.test.tsx apps/packages/ui/src/hooks/__tests__/usePostOnboardingMediaReadiness.test.tsx
```

Expected: PASS.

- [ ] **Step 8: Update E2E mock and test**

In `unified-first-run-onboarding.spec.ts`, add route handling for provider validation:

```ts
if (path === '/api/v1/setup/first-run/providers/validate' && method === 'POST') {
  const body = requestJson(route);
  await json(route, {
    provider_key: body.provider_key,
    status: 'accepted',
    validation_level: 'local_syntax',
    can_gate_first_chat: true,
    models: []
  });
  return;
}
```

Extend the happy path to click `Validate` before save/continue and assert the first-source picker opens quick ingest with `firstSourceKind: "web_url"`.

- [ ] **Step 9: Run E2E focused test**

Run with working directory `apps/tldw-frontend`:

```bash
bun run e2e:pw e2e/workflows/unified-first-run-onboarding.spec.ts --project=chromium --reporter=line
```

Expected: PASS.

- [ ] **Step 10: Commit Task 4**

Run:

```bash
git add apps/packages/ui/src/components/Option/Onboarding/FirstSourceMilestonePrompt.tsx apps/packages/ui/src/components/Option/Onboarding/__tests__/FirstSourceMilestonePrompt.test.tsx apps/packages/ui/src/utils/quick-ingest-open.ts apps/packages/ui/src/routes/option-index.tsx apps/packages/ui/src/routes/__tests__/option-index.unified-setup.test.tsx apps/tldw-frontend/e2e/workflows/unified-first-run-onboarding.spec.ts "backlog/tasks/task-583 - Design-onboarding-confidence-flow-follow-up.md"
git diff --cached --check
git commit -m "feat: guide first source after onboarding"
```

---

## Final Verification

Run these from the repo root unless noted.

Backend:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Setup/test_setup_provider_validation.py tldw_Server_API/tests/Setup/test_setup_first_chat_completion.py -q
```

Frontend unit/integration:

```bash
bunx vitest run apps/packages/ui/src/components/Option/Onboarding/__tests__/ProviderSetupStep.test.tsx apps/packages/ui/src/components/Option/Onboarding/__tests__/UnifiedSetupWizard.test.tsx apps/packages/ui/src/components/Option/Onboarding/__tests__/FirstChatStep.test.tsx apps/packages/ui/src/components/Option/Onboarding/__tests__/FirstSourceMilestonePrompt.test.tsx apps/packages/ui/src/components/Option/Onboarding/__tests__/SetupReadinessPanel.test.tsx apps/packages/ui/src/hooks/__tests__/useSetupReadinessSummary.test.tsx apps/packages/ui/src/hooks/__tests__/usePostOnboardingMediaReadiness.test.tsx apps/packages/ui/src/routes/__tests__/option-index.unified-setup.test.tsx
```

Frontend E2E:

```bash
bun run e2e:pw e2e/workflows/unified-first-run-onboarding.spec.ts --project=chromium --reporter=line
```

Bandit for touched backend setup scope:

```bash
source .venv/bin/activate && python -m bandit -r tldw_Server_API/app/api/v1/schemas/setup_schemas.py tldw_Server_API/app/core/Setup/provider_validation.py -f json -o /tmp/bandit_onboarding_confidence_flow.json
```

Diff checks:

```bash
git diff --check
git status --short --branch
```

Manual UAT after implementation:

1. Start a clean server/database install in a temporary copy or test worktree.
2. Use the existing project `.env` OpenAI key.
3. Select OpenAI as default, validate it, save providers, and continue.
4. Set TTS to `pocket-tts`.
5. Set STT to `onnx-parakeet`.
6. Complete privacy/security, ingest defaults, audio defaults, optional advanced, and first chat.
7. Confirm first chat only completes after a real successful response.
8. Confirm the first-source picker appears, defaults to Web URL, opens quick ingest, and only offers grounded chat after ingest readiness is confirmed.
9. Repeat a failure path: invalid provider key, unreachable local endpoint, and first-chat failure. Confirm inline recovery buttons work.

## Cleanup Requirements

- Keep `.worktrees/unified-solo-onboarding` untouched unless the user explicitly asks to inspect it as a reference.
- Do not rebase or cherry-pick broad stale worktree changes.
- Remove any temporary screenshots, Playwright artifacts, mock server logs, or generated UAT files before final staging unless they are intentionally part of test output.
- Update `TASK-583` notes after each staged commit with tests run and known skips.
- If a test cannot run because of missing local dependencies, record the exact command, error, and reason in the Backlog task and final response.
