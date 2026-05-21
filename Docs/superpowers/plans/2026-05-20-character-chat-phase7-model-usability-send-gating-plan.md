# Character Chat Phase 7 Model Usability And Send Gating Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Character Chat use one truthful model-usability contract so readiness panels, status surfaces, model labels, and SEND behavior agree, and invalid character sends never hit `/complete-v2`.

**Architecture:** Add a pure model-usability classifier beside the existing character-chat readiness utilities, then thread that result into the readiness panel, status strip, runtime/composition summaries, model selector copy, and composer send controls. Keep user drafts and character/session state intact while converting invalid SEND actions into explicit setup/recovery actions.

**Tech Stack:** Next.js/React, TypeScript, Zustand-backed playground state, Ant Design controls, Vitest/Testing Library, Playwright against the real FastAPI backend.

---

## Context

This plan implements Phase 7 of `Docs/Product/WebUI/Character_Chat_Roleplay_First_Class_PRD_2026_05_18.md`.

The post-Phase-6 browser review found a high-trust defect: Character Chat can show contradictory model readiness states, for example a readiness panel requiring a chat model while the selector/status area reports a healthy `OpenAI / gpt-4o`, and SEND can still call the real backend `/api/v1/chats/{id}/complete-v2` path and return a provider/configuration `503`.

Phase 7 is intentionally narrower than the full PRD. It does not redesign sessions, history, character import/create, extension parity, or mobile IA. It fixes the model-readiness truth source and SEND gate first because users must be able to trust whether a role-play session is actually ready to generate.

## Current Code Map

Modify these files:

- `apps/packages/ui/src/utils/chat-model-availability.ts`
  - Add the model-usability contract and keep existing model ID normalization helpers in one place.
  - Make `buildCharacterChatReadiness` consume the new contract instead of reimplementing model availability with less detail.
- `apps/packages/ui/src/utils/__tests__/chat-model-availability.test.ts`
  - Add pure unit coverage for usability statuses and readiness mapping.
- `apps/packages/ui/src/components/Option/Playground/Playground.tsx`
  - Build one `characterChatModelUsability` result and pass it to every visible Character Chat status surface.
- `apps/packages/ui/src/components/Option/Playground/PlaygroundStatusStrip.tsx`
  - Replace the boolean-only `modelUnavailable` surface with explicit model-usability status/copy.
- `apps/packages/ui/src/components/Option/Playground/playground-composition-preview.ts`
  - Show the same model-readiness copy in composition summaries.
- `apps/packages/ui/src/components/Option/Playground/PlaygroundRuntimeInspector.tsx`
  - Use the same model-usability status/detail in the runtime inspector.
- `apps/packages/ui/src/components/Option/Playground/PlaygroundForm.tsx`
  - Add a guarded submit path for Character Chat and pass a send blocker to send controls and next-generation composers.
- `apps/packages/ui/src/components/Option/Playground/PlaygroundSendControl.tsx`
  - Convert blocked Character Chat sends into setup actions and ensure the submit handler is not invoked.
- `apps/packages/ui/src/hooks/playground/useModelSelector.tsx`
  - Stop labeling an unusable selected model as healthy; expose neutral/blocked model-label copy when Character Chat has an unusable model.
- `apps/packages/ui/src/components/Option/Playground/ChatModelSelectorDropdown.tsx`
  - Accept and render model-usability label/title if the selected model is unusable.

Likely test files to modify or create:

- `apps/packages/ui/src/components/Option/Playground/__tests__/PlaygroundStatusStrip.first-slice.test.tsx`
- `apps/packages/ui/src/components/Option/Playground/__tests__/playground-composition-preview.test.ts`
- `apps/packages/ui/src/components/Option/Playground/__tests__/PlaygroundForm.role-play-starter.integration.test.tsx`
- `apps/packages/ui/src/components/Option/Playground/__tests__/PlaygroundSendControl.character-gating.test.tsx`
- `apps/tldw-frontend/tests/e2e/workflows/journeys/character-chat-phase7-readiness.spec.ts`

Do not modify unrelated Character Chat phases in this PR. If implementation reveals a larger session/history defect, record it for the next phase instead of widening this slice.

## Model Usability Contract

Add these exported types in `apps/packages/ui/src/utils/chat-model-availability.ts`:

```ts
export type ChatModelUsabilityStatus =
  | "loading"
  | "no_server"
  | "no_selection"
  | "no_models"
  | "selected_missing"
  | "provider_unconfigured"
  | "model_unavailable"
  | "degraded"
  | "ready"

export type ChatModelUsability = {
  status: ChatModelUsabilityStatus
  canSend: boolean
  selectedModelId: string | null
  providerQualifiedModelId: string | null
  matchedModelId: string | null
  matchedProvider: string | null
  recommendedAction: CharacterChatReadinessAction | null
  detailReason: string | null
}
```

Classifier rules:

- `loading`: model/provider catalog is not loaded yet.
- `no_server`: server is not connected.
- `no_selection`: no selected model.
- `no_models`: loaded catalog has zero callable models and the selected model does not match a descriptor with a more specific blocker.
- `selected_missing`: selected model is not present in the loaded catalog by qualified ID or base ID.
- `provider_unconfigured`: selected model is known, but any provider/configured flag says the provider/model is not configured.
- `model_unavailable`: selected model is known but not callable, for example `catalog_only: true`.
- `degraded`: selected model is callable but server/model health is degraded and sends are explicitly allowed.
- `ready`: selected model is callable.

Only `ready` and intentionally allowed `degraded` return `canSend: true`.

When a selected model matches a descriptor, return the descriptor-specific status before falling back to generic `no_models`. This keeps a known `openai:gpt-4o` with missing provider configuration from collapsing into a vague "no models" state.

## Task 0: Preflight And Baseline

**Files:**
- Read: `Docs/Product/WebUI/Character_Chat_Roleplay_First_Class_PRD_2026_05_18.md`
- Read: `apps/packages/ui/src/utils/chat-model-availability.ts`
- Read: `apps/packages/ui/src/components/Option/Playground/Playground.tsx`
- Read: `apps/packages/ui/src/components/Option/Playground/PlaygroundForm.tsx`
- Update: existing Backlog task for the implementation slice

- [x] **Step 1: Confirm branch and task scope**

Run:

```bash
git status --short --branch
```

Expected: implementation branch is based on latest `dev`; unrelated untracked files are left untouched.

- [x] **Step 2: Create or update the implementation Backlog task**

Use Backlog MCP or CLI. Reference this plan and the PRD. The task scope is production implementation, not this planning task.

- [x] **Step 3: Run focused baseline tests**

Run from `apps/tldw-frontend`:

```bash
bunx vitest run ../packages/ui/src/utils/__tests__/chat-model-availability.test.ts ../packages/ui/src/components/Option/Playground/__tests__/CharacterChatReadinessPanel.test.tsx ../packages/ui/src/components/Option/Playground/__tests__/PlaygroundStatusStrip.first-slice.test.tsx --reporter=verbose
```

Expected: current dev baseline passes. If it fails, stop and classify the failure before editing.

- [ ] **Step 4: Capture current browser failure with the real backend when practical**

Run the real backend and WebUI using the project's existing dev commands. Navigate to `/chat?mode=character`, select a character, leave the selected model in an unusable provider/config state, type a draft, and click SEND.

Expected current behavior to verify before implementation: a contradictory model state can appear and/or SEND can attempt `/api/v1/chats/{id}/complete-v2`. If the local environment cannot reproduce because a real callable provider is configured, document that and continue with tests.

## Task 1: Add Pure Model Usability Classification

**Files:**
- Modify: `apps/packages/ui/src/utils/chat-model-availability.ts`
- Test: `apps/packages/ui/src/utils/__tests__/chat-model-availability.test.ts`

- [x] **Step 1: Write failing tests for every usability status**

Add a new `describe("chat model usability", () => { ... })` block:

```ts
it("reports provider_unconfigured for a known selected model whose provider flags are false", () => {
  expect(
    buildChatModelUsability({
      isServerConnected: true,
      selectedModel: "openai:gpt-4o",
      availableModels: [
        {
          id: "gpt-4o",
          model: "tldw:gpt-4o",
          provider: "openai",
          is_configured: false,
          provider_is_configured: false,
          catalog_only: false,
        } as any,
      ],
    }),
  ).toMatchObject({
    status: "provider_unconfigured",
    canSend: false,
    matchedModelId: "gpt-4o",
    matchedProvider: "openai",
    recommendedAction: "open-model-settings",
  })
})
```

Also add tests for:

- `loading` when `modelsLoading: true`.
- `no_server` when `isServerConnected: false`.
- `no_selection` when selected model is null or blank.
- `no_models` when the loaded catalog has no callable models.
- `selected_missing` when neither qualified nor base selected model exists in the catalog.
- `model_unavailable` when the matching descriptor is catalog-only.
- `ready` when the selected qualified or base model is callable.
- Provider-qualified fallback: `openai:gpt-4o` matches a descriptor with base `gpt-4o` and provider `openai`.
- Unknown provider-qualified fallback: `local:gpt-4o` can match base `gpt-4o` only when no provider-specific descriptor conflicts.

- [x] **Step 2: Run the focused utility test and verify red**

Run from `apps/tldw-frontend`:

```bash
bunx vitest run ../packages/ui/src/utils/__tests__/chat-model-availability.test.ts --reporter=verbose
```

Expected: fails because `buildChatModelUsability` is not implemented.

- [x] **Step 3: Implement descriptor matching helpers**

In `chat-model-availability.ts`, keep the existing normalization helpers and add small private helpers:

```ts
function descriptorCandidateIds(model: ModelDescriptor): Set<string> {
  const ids = new Set<string>()
  const modelId = normalizeAvailableModelId(model)
  const provider = normalizeProviderKey(model)
  const providerQualified = normalizeProviderQualifiedChatModelId(modelId)
  const baseModelId = normalizeBaseChatModelId(modelId)

  for (const candidate of [modelId, providerQualified, baseModelId]) {
    if (candidate) ids.add(candidate)
  }
  if (provider && baseModelId) ids.add(`${provider}:${baseModelId}`)
  return ids
}

function selectedModelCandidateIds(selectedModel: string | null | undefined): Set<string> {
  const normalized = normalizeChatModelId(selectedModel)
  const providerQualified = normalizeProviderQualifiedChatModelId(selectedModel)
  const baseModelId = normalizeBaseChatModelId(selectedModel)
  return new Set([normalized, providerQualified, baseModelId].filter(Boolean) as string[])
}
```

Keep these helpers private unless tests need a public contract. Do not duplicate ID parsing in React components.

- [x] **Step 4: Implement `buildChatModelUsability`**

Add an exported function with this input shape:

```ts
export type ChatModelUsabilityInput = {
  isServerConnected?: boolean
  selectedModel?: string | null
  availableModels?: ModelDescriptor[] | null
  modelsLoading?: boolean
  allowDegradedSend?: boolean
  serverDegraded?: boolean
}
```

Implementation notes:

- Return `loading` before `no_models` when the catalog is still hydrating.
- Return `no_server` before evaluating models.
- Use existing `isUsableChatModelDescriptor` to identify callable models, but inspect descriptor flags directly to distinguish `provider_unconfigured` from `model_unavailable`.
- Preserve `AUTO_CHAT_MODEL_ID` semantics. If the selected model is `auto`, return `ready` when at least one callable model exists; return `no_models` when none exists.
- The function must never make network calls.

- [x] **Step 5: Run utility tests and verify green**

Run:

```bash
bunx vitest run ../packages/ui/src/utils/__tests__/chat-model-availability.test.ts --reporter=verbose
```

Expected: all tests in this file pass.

- [x] **Step 6: Commit Task 1**

```bash
git add apps/packages/ui/src/utils/chat-model-availability.ts apps/packages/ui/src/utils/__tests__/chat-model-availability.test.ts
git commit -m "feat: classify character chat model usability"
```

## Task 2: Make Character Readiness Consume Model Usability

**Files:**
- Modify: `apps/packages/ui/src/utils/chat-model-availability.ts`
- Test: `apps/packages/ui/src/utils/__tests__/chat-model-availability.test.ts`
- Test: `apps/packages/ui/src/components/Option/Playground/__tests__/CharacterChatReadinessPanel.test.tsx`

- [x] **Step 1: Write failing readiness mapping tests**

Add tests proving `buildCharacterChatReadiness` maps model usability to precise blocked reasons:

```ts
it("blocks character chat with provider-unconfigured copy when the selected model provider lacks setup", () => {
  const readiness = buildCharacterChatReadiness({
    isServerConnected: true,
    selectedCharacter: { id: 1, name: "Ada" },
    selectedModel: "openai:gpt-4o",
    availableModels: [
      {
        id: "gpt-4o",
        model: "tldw:gpt-4o",
        provider: "openai",
        is_configured: false,
        provider_is_configured: false,
      } as any,
    ],
  })

  expect(readiness).toMatchObject({
    status: "blocked",
    missingRequirement: "chat-model",
    reason: "provider-unconfigured",
    recommendedAction: "open-model-settings",
  })
})
```

Also test `models-loading`, `selected-model-missing`, `model-unavailable`, `no-models-available`, and ready.

- [x] **Step 2: Run tests and verify red**

Run:

```bash
bunx vitest run ../packages/ui/src/utils/__tests__/chat-model-availability.test.ts ../packages/ui/src/components/Option/Playground/__tests__/CharacterChatReadinessPanel.test.tsx --reporter=verbose
```

Expected: fails on new reason/copy expectations.

- [x] **Step 3: Extend readiness reason types without breaking old callers**

Update `CharacterChatReadinessReason`:

```ts
export type CharacterChatReadinessReason =
  | "server-unavailable"
  | "missing-character"
  | "models-loading"
  | "no-selected-model"
  | "no-models-available"
  | "selected-model-missing"
  | "provider-unconfigured"
  | "model-unavailable"
  | "selected-model-unavailable"
  | "send-disabled"
```

Keep `selected-model-unavailable` temporarily for compatibility with tests/components that have not moved yet. New code should use the more specific reasons.

- [x] **Step 4: Refactor `buildCharacterChatReadiness` to call `buildChatModelUsability`**

The readiness builder should:

- Check server and missing character first.
- Call `buildChatModelUsability`.
- Map usability statuses to existing `CharacterChatReadinessMissingRequirement` and `CharacterChatReadinessAction`.
- Only evaluate `isSendBlocked` after model usability returns `ready` or allowed `degraded`.

- [x] **Step 5: Add copy for precise model failures**

Extend `getCharacterChatReadinessCopy` so first-time and power users see the actual blocker:

- `models-loading`: "Checking chat model readiness"
- `provider-unconfigured`: "Configure the selected model provider before chatting as {{characterName}}"
- `selected-model-missing`: "Choose an available chat model before chatting as {{characterName}}"
- `model-unavailable`: "The selected chat model is not callable right now"
- `no-models-available`: "Configure a chat model before chatting as {{characterName}}"

Each description must preserve character context and say drafts/selections are kept.

- [x] **Step 6: Run readiness and panel tests**

Run:

```bash
bunx vitest run ../packages/ui/src/utils/__tests__/chat-model-availability.test.ts ../packages/ui/src/components/Option/Playground/__tests__/CharacterChatReadinessPanel.test.tsx --reporter=verbose
```

Expected: pass.

- [x] **Step 7: Commit Task 2**

```bash
git add apps/packages/ui/src/utils/chat-model-availability.ts apps/packages/ui/src/utils/__tests__/chat-model-availability.test.ts apps/packages/ui/src/components/Option/Playground/__tests__/CharacterChatReadinessPanel.test.tsx
git commit -m "feat: map character readiness to model usability"
```

## Task 3: Align Readiness Panel, Status Strip, Composition Preview, And Runtime Inspector

**Files:**
- Modify: `apps/packages/ui/src/components/Option/Playground/Playground.tsx`
- Modify: `apps/packages/ui/src/components/Option/Playground/PlaygroundStatusStrip.tsx`
- Modify: `apps/packages/ui/src/components/Option/Playground/playground-composition-preview.ts`
- Modify: `apps/packages/ui/src/components/Option/Playground/PlaygroundRuntimeInspector.tsx`
- Test: `apps/packages/ui/src/components/Option/Playground/__tests__/PlaygroundStatusStrip.first-slice.test.tsx`
- Test: `apps/packages/ui/src/components/Option/Playground/__tests__/playground-composition-preview.test.ts`

- [x] **Step 1: Write failing status strip tests**

In `PlaygroundStatusStrip.first-slice.test.tsx`, add tests for:

- `modelUsabilityStatus="provider_unconfigured"` shows unavailable/setup language, not `Ready` or `Healthy`.
- `modelUsabilityStatus="model_unavailable"` shows not-callable language.
- `modelUsabilityStatus="loading"` shows checking language.
- Non-character chat still preserves existing status behavior.

- [x] **Step 2: Run tests and verify red**

Run:

```bash
bunx vitest run ../packages/ui/src/components/Option/Playground/__tests__/PlaygroundStatusStrip.first-slice.test.tsx --reporter=verbose
```

Expected: fails because the component only accepts `modelUnavailable`.

- [x] **Step 3: Build one model usability result in `Playground.tsx`**

Import `buildChatModelUsability` and compute next to `characterChatReadiness`:

```ts
const characterChatModelUsability = React.useMemo(
  () =>
    buildChatModelUsability({
      isServerConnected: serverReadinessState !== "blocked",
      selectedModel: providerRouteSummary.selectedModel,
      availableModels: characterChatAvailableModels,
      modelsLoading: !Array.isArray(characterChatAvailableModels),
      serverDegraded: serverReadinessState === "degraded",
      allowDegradedSend: false,
    }),
  [
    characterChatAvailableModels,
    providerRouteSummary.selectedModel,
    serverReadinessState,
  ],
)
```

Then pass `modelUsability: characterWorkflowActive ? characterChatModelUsability : null` to status, inspector, and composition preview surfaces.

- [x] **Step 4: Update `PlaygroundStatusStrip` props**

Replace the boolean-only API with explicit props while keeping backward compatibility during migration:

```ts
modelUsabilityStatus?: ChatModelUsabilityStatus | null
modelUsabilityMessage?: string | null
modelUnavailable?: boolean
modelUnavailableMessage?: string | null
```

Status priority should be:

1. explicit error
2. server blocked
3. streaming/loading
4. model usability blocked/loading
5. selected model missing
6. degraded
7. ready

If `modelUsabilityStatus` is present and not `ready`/allowed `degraded`, do not render `Ready`, `Healthy`, or equivalent positive copy.

- [x] **Step 5: Update composition preview and runtime inspector**

Replace derived `modelUnavailable` booleans with the same usability status/copy. The preview and inspector do not need a new visual design; they only need to stop contradicting the readiness panel.

- [x] **Step 6: Run focused status tests**

Run:

```bash
bunx vitest run ../packages/ui/src/components/Option/Playground/__tests__/PlaygroundStatusStrip.first-slice.test.tsx ../packages/ui/src/components/Option/Playground/__tests__/playground-composition-preview.test.ts --reporter=verbose
```

Expected: pass.

- [x] **Step 7: Commit Task 3**

```bash
git add apps/packages/ui/src/components/Option/Playground/Playground.tsx apps/packages/ui/src/components/Option/Playground/PlaygroundStatusStrip.tsx apps/packages/ui/src/components/Option/Playground/playground-composition-preview.ts apps/packages/ui/src/components/Option/Playground/PlaygroundRuntimeInspector.tsx apps/packages/ui/src/components/Option/Playground/__tests__/PlaygroundStatusStrip.first-slice.test.tsx apps/packages/ui/src/components/Option/Playground/__tests__/playground-composition-preview.test.ts
git commit -m "feat: align character chat model readiness surfaces"
```

## Task 4: Gate SEND Without Losing Drafts Or Character State

**Files:**
- Modify: `apps/packages/ui/src/components/Option/Playground/Playground.tsx`
- Modify: `apps/packages/ui/src/components/Option/Playground/PlaygroundForm.tsx`
- Modify: `apps/packages/ui/src/components/Option/Playground/PlaygroundSendControl.tsx`
- Test: `apps/packages/ui/src/components/Option/Playground/__tests__/PlaygroundSendControl.character-gating.test.tsx`
- Test: `apps/packages/ui/src/components/Option/Playground/__tests__/PlaygroundForm.role-play-starter.integration.test.tsx`

- [x] **Step 1: Add failing `PlaygroundSendControl` tests**

Create `PlaygroundSendControl.character-gating.test.tsx`:

```tsx
it("turns blocked character SEND into a setup action without submitting", async () => {
  const user = userEvent.setup()
  const onSubmitForm = vi.fn()
  const onAction = vi.fn()

  render(
    <PlaygroundSendControl
      isProMode={false}
      isMobileViewport={false}
      isSending={false}
      isConnectionReady
      sendWhenEnter
      onSendWhenEnterChange={vi.fn()}
      sendLabel="SEND"
      compareNeedsMoreModels={false}
      onStopStreaming={vi.fn()}
      onStopListening={vi.fn()}
      onSubmitForm={onSubmitForm}
      sendMenuOpen={false}
      onSendMenuChange={vi.fn()}
      characterChatSendBlocker={{
        active: true,
        title: "Configure the selected model provider before chatting as Ada",
        actionLabel: "Open model settings",
        onAction,
      }}
      t={(_, fallback) => fallback}
    />,
  )

  await user.click(screen.getByRole("button", { name: /open model settings/i }))

  expect(onAction).toHaveBeenCalledTimes(1)
  expect(onSubmitForm).not.toHaveBeenCalled()
})
```

- [x] **Step 2: Run the send-control test and verify red**

Run:

```bash
bunx vitest run ../packages/ui/src/components/Option/Playground/__tests__/PlaygroundSendControl.character-gating.test.tsx --reporter=verbose
```

Expected: fails because `characterChatSendBlocker` does not exist.

- [x] **Step 3: Add send blocker prop to `PlaygroundSendControl`**

Add:

```ts
export type PlaygroundSendBlocker = {
  active: boolean
  title: string
  actionLabel: string
  onAction: () => void
}
```

Behavior:

- If `characterChatSendBlocker.active` is true, the primary button is `htmlType="button"`.
- The primary button label is the blocker `actionLabel`, not SEND.
- `onClick` calls `onStopListening()` and `characterChatSendBlocker.onAction()`.
- It must not call `onSubmitForm`.
- The textarea/draft remains untouched.
- Existing compare-mode, queue, streaming, and offline behavior remains unchanged for non-character blockers.

- [x] **Step 4: Thread send blocker from `Playground.tsx` into `PlaygroundForm.tsx`**

Extend `PlaygroundForm` props:

```ts
characterChatSendBlocker?: PlaygroundSendBlocker | null
```

In `Playground.tsx`, create it from readiness copy:

```ts
const characterChatSendBlocker = React.useMemo(
  () =>
    characterWorkflowActive &&
    characterChatReadiness.status === "blocked" &&
    characterChatReadiness.missingRequirement === "chat-model" &&
    characterChatReadinessCopy
      ? {
          active: true,
          title: characterChatReadinessCopy.title,
          actionLabel: characterChatReadinessCopy.actionLabel,
          onAction: () =>
            handleCharacterChatReadinessAction(
              characterChatReadiness.recommendedAction ?? "open-model-settings",
            ),
        }
      : null,
  [
    characterChatReadiness,
    characterChatReadinessCopy,
    characterWorkflowActive,
    handleCharacterChatReadinessAction,
  ],
)
```

Place this memo after `handleCharacterChatReadinessAction` is declared, or move that action callback above the memo first. Then pass the blocker to `PlaygroundForm`.

- [x] **Step 5: Guard every submit entry point in `PlaygroundForm.tsx`**

Create one guarded handler:

```ts
const handleComposerSend = React.useCallback(() => {
  if (characterChatSendBlocker?.active) {
    stopListening()
    characterChatSendBlocker.onAction()
    return
  }
  submitForm()
}, [characterChatSendBlocker, stopListening, submitForm])
```

Use it for:

- `PlaygroundSendControl.onSubmitForm`
- `ChatComposer` variants v1, v3, and v5 `onSend`
- any local keyboard shortcut path that directly calls `submitForm()` for composer SEND

Do not alter `usePlaygroundSubmit` for this phase unless tests prove a hidden submit path still bypasses the gate. The gate belongs near the UI action because users must keep typing and editing drafts.

- [x] **Step 6: Add integration coverage that blocked character SEND does not submit**

Extend `PlaygroundForm.role-play-starter.integration.test.tsx` or create a focused sibling test. Mock `onSubmit` and render Character Chat with:

- selected character present
- selected model present but usability blocked
- draft text present

Click SEND and assert:

- setup action is invoked or settings surface opens
- `onSubmit` is not called
- draft text is still present
- selected character text is still present

- [x] **Step 7: Run focused composer tests**

Run:

```bash
bunx vitest run ../packages/ui/src/components/Option/Playground/__tests__/PlaygroundSendControl.character-gating.test.tsx ../packages/ui/src/components/Option/Playground/__tests__/PlaygroundForm.role-play-starter.integration.test.tsx --reporter=verbose
```

Expected: pass.

- [x] **Step 8: Commit Task 4**

```bash
git add apps/packages/ui/src/components/Option/Playground/Playground.tsx apps/packages/ui/src/components/Option/Playground/PlaygroundForm.tsx apps/packages/ui/src/components/Option/Playground/PlaygroundSendControl.tsx apps/packages/ui/src/components/Option/Playground/__tests__/PlaygroundSendControl.character-gating.test.tsx apps/packages/ui/src/components/Option/Playground/__tests__/PlaygroundForm.role-play-starter.integration.test.tsx
git commit -m "feat: gate invalid character chat sends"
```

## Task 5: Fix Model Selector Health Copy In Character Chat

**Files:**
- Modify: `apps/packages/ui/src/hooks/playground/useModelSelector.tsx`
- Modify: `apps/packages/ui/src/components/Option/Playground/ChatModelSelectorDropdown.tsx`
- Modify: `apps/packages/ui/src/components/Option/Playground/PlaygroundForm.tsx`
- Test: add or update the closest model selector/dropdown test

- [x] **Step 1: Write failing tests for unusable model label copy**

Add coverage proving an unusable Character Chat model does not render `Healthy`, `Ready`, or a positive provider label.

Expected user-visible examples:

- `OpenAI / gpt-4o - Provider setup needed`
- `gpt-4o - Not callable`
- `Checking model readiness`

Do not use the word "healthy" for a blocked or loading usability status.

- [x] **Step 2: Run model selector tests and verify red**

Run the nearest selector test file. If no targeted test exists, create one under:

```text
apps/packages/ui/src/components/Option/Playground/__tests__/ChatModelSelectorDropdown.character-usability.test.tsx
```

- [x] **Step 3: Pass model usability label into the selector surface**

Extend `PlaygroundForm` props with:

```ts
characterChatModelUsability?: ChatModelUsability | null
characterChatModelUsabilityLabel?: string | null
```

In `Playground.tsx`, derive the label from the same readiness copy used by the panel. Pass it down.

In `useModelSelector.tsx`, keep generic model selection behavior unchanged. Prefer adding optional override props at render sites over making this hook character-chat aware unless the hook already owns the selector button copy.

- [x] **Step 4: Render blocked usability copy in `ChatModelSelectorDropdown`**

When an override label/title is provided:

- The button title and aria-label must name the blocker.
- The visible compact label must include a concise blocker state.
- Existing provider/model text remains available so power users know which model is affected.

- [x] **Step 5: Run selector tests**

Run:

```bash
bunx vitest run ../packages/ui/src/components/Option/Playground/__tests__/ChatModelSelectorDropdown.character-usability.test.tsx --reporter=verbose
```

Expected: pass.

- [x] **Step 6: Commit Task 5**

```bash
git add apps/packages/ui/src/hooks/playground/useModelSelector.tsx apps/packages/ui/src/components/Option/Playground/ChatModelSelectorDropdown.tsx apps/packages/ui/src/components/Option/Playground/PlaygroundForm.tsx apps/packages/ui/src/components/Option/Playground/__tests__/ChatModelSelectorDropdown.character-usability.test.tsx
git commit -m "feat: show truthful character chat model labels"
```

## Task 6: Map Provider/Model Failures To Recovery Copy

**Files:**
- Modify: `apps/packages/ui/src/components/Option/Playground/Playground.tsx`
- Modify: `apps/packages/ui/src/hooks/chat/useCharacterChatMode.ts`
- Test: `apps/packages/ui/src/hooks/chat/__tests__/useCharacterChatMode.contract.test.ts`
- Test: nearest Playground error/recovery component test

- [x] **Step 1: Inspect current character failure path**

Read:

```bash
sed -n '1,220p' apps/packages/ui/src/hooks/chat/useCharacterChatMode.ts
sed -n '220,560p' apps/packages/ui/src/hooks/chat/useCharacterChatMode.ts
```

Find where provider/configuration failures from `/complete-v2` are caught and surfaced. Do not add a duplicate error system if the hook already returns a typed result.

- [x] **Step 2: Add failing tests for provider setup failures**

Add or extend tests so a backend `503` or structured provider/config error maps to:

- missing/unconfigured provider/model copy
- `open-model-settings` recovery action
- preserved character and draft state
- no generic retry-only guidance when the failure is clearly configuration-related

- [x] **Step 3: Implement a small classifier**

If no suitable helper exists, add a local pure helper in the hook file or a nearby utility:

```ts
type CharacterChatFailureRecovery =
  | { kind: "provider_unconfigured"; action: "open-model-settings"; message: string }
  | { kind: "model_unavailable"; action: "open-model-settings"; message: string }
  | { kind: "transient"; action: "retry"; message: string }
```

Classification should use structured backend fields first, then conservative message matching for known provider/config phrases. Do not classify arbitrary `503` errors as provider setup if the response lacks evidence.

- [x] **Step 4: Surface recovery through existing error UI**

Use the existing Playground notice/error surface. Do not introduce a modal. The message should keep the user in the Character Chat workflow and offer model settings when recovery is local.

- [x] **Step 5: Run failure-recovery tests**

Run:

```bash
bunx vitest run ../packages/ui/src/hooks/chat/__tests__/useCharacterChatMode.contract.test.ts --reporter=verbose
```

Expected: pass.

- [x] **Step 6: Commit Task 6**

```bash
git add apps/packages/ui/src/hooks/chat/useCharacterChatMode.ts apps/packages/ui/src/hooks/chat/__tests__/useCharacterChatMode.contract.test.ts apps/packages/ui/src/components/Option/Playground/Playground.tsx
git commit -m "feat: recover character chat model failures"
```

## Task 7: Real-Backend E2E Verification

**Files:**
- Create: `apps/tldw-frontend/e2e/workflows/journeys/character-chat-phase7-readiness.spec.ts`
- Modify: `apps/packages/ui/src/services/tldw/domains/models-audio.ts`
- Modify: `apps/packages/ui/src/services/tldw/TldwModels.ts`
- Test: `apps/packages/ui/src/services/__tests__/tldw-api-client.models-normalization.test.ts`
- Test: `apps/packages/ui/src/services/tldw/__tests__/TldwModels.test.ts`

- [x] **Step 1: Write a no-provider/send-gating Playwright test**

The test must run against a real backend URL and real WebUI. It may intercept requests only to observe that `/complete-v2` was not called; it must not simulate a successful character response.

Test flow:

1. Navigate to `/chat?mode=character`.
2. Select or use a real server-created character.
3. Ensure selected model is known but unusable, or use the no-provider fixture/state exposed by the real backend.
4. Type a draft.
5. Click the primary send/setup action.
6. Assert:
   - readiness panel, status strip, model selector, and SEND action all show compatible blocked/setup language
   - draft remains in the composer
   - selected character remains active
   - no `/api/v1/chats/*/complete-v2` request was made

- [x] **Step 2: Write a provider-failure recovery test when the environment supports it**

Use a real backend provider path that returns a real provider/configuration failure. Do not replace backend responses in the browser.

If the environment lacks a controlled provider-failure setup, mark this scenario skipped with a precise reason and keep the no-provider/send-gating test active.

- [x] **Step 3: Write successful-send test only when a real callable provider is configured**

If a real local or external provider is configured, add a successful character message test that proves `/complete-v2` returns a character response.

If no callable provider is available, explicitly mark successful-send signoff blocked in the test output or verification notes. Do not use frontend interception as proof.

- [x] **Step 4: Run the Playwright test against real backend**

Use the project's existing environment variables:

```bash
TLDW_SERVER_URL=http://127.0.0.1:8000 TLDW_E2E_SERVER_URL=http://127.0.0.1:8000 TLDW_API_KEY=THIS-IS-A-SECURE-KEY-123-FAKE-KEY TLDW_E2E_API_KEY=THIS-IS-A-SECURE-KEY-123-FAKE-KEY bunx playwright test e2e/workflows/journeys/character-chat-phase7-readiness.spec.ts --project=journeys --reporter=line
```

Expected:

- No-provider/send-gating passes.
- Provider-failure passes or is skipped with an environment reason.
- Successful-send passes only when a real callable provider is configured; otherwise it is marked blocked, not faked.

Actual verification on 2026-05-21 against the real FastAPI backend at `http://127.0.0.1:8000`:

- Backend health returned `status: ok` with `auth_mode: single_user`.
- Focused Vitest suite passed: `6` files, `110` tests.
- Real-backend Playwright suite passed with `1 passed, 2 skipped`.
- The active no-provider/send-gating scenario used the backend-advertised OpenAI `gpt-4o` catalog-only/unconfigured model and verified blocked readiness, preserved draft/character state, and no `/api/v1/chats/*/complete-v2` call.
- Provider-failure and successful-send scenarios skipped because this environment exposes no explicit forced provider-failure model and no non-local/non-custom callable model. Local/custom-risk providers were intentionally not used as proof.

Implementation note: Real-browser verification exposed that the domain `models-audio` client mixin was dropping `is_configured`, `provider_is_configured`, and `catalog_only` before `TldwModels` cached the catalog. Task 7 now preserves those flags and bumps the model cache schema to evict stale readiness caches.

- [x] **Step 5: Commit Task 7**

```bash
git add apps/tldw-frontend/e2e/workflows/journeys/character-chat-phase7-readiness.spec.ts apps/packages/ui/src/services/tldw/domains/models-audio.ts apps/packages/ui/src/services/tldw/TldwModels.ts apps/packages/ui/src/services/__tests__/tldw-api-client.models-normalization.test.ts apps/packages/ui/src/services/tldw/__tests__/TldwModels.test.ts
git commit -m "test: cover character chat readiness gating e2e"
```

## Task 8: Final Verification And Documentation

**Files:**
- Modify: implementation Backlog task
- Optional modify: `Docs/Product/WebUI/Character_Chat_Roleplay_First_Class_PRD_2026_05_18.md` only if acceptance wording needs a factual status update

- [ ] **Step 1: Run focused unit/component suite**

Run from `apps/tldw-frontend`:

```bash
bunx vitest run ../packages/ui/src/utils/__tests__/chat-model-availability.test.ts ../packages/ui/src/components/Option/Playground/__tests__/CharacterChatReadinessPanel.test.tsx ../packages/ui/src/components/Option/Playground/__tests__/PlaygroundStatusStrip.first-slice.test.tsx ../packages/ui/src/components/Option/Playground/__tests__/playground-composition-preview.test.ts ../packages/ui/src/components/Option/Playground/__tests__/PlaygroundSendControl.character-gating.test.tsx ../packages/ui/src/components/Option/Playground/__tests__/PlaygroundForm.role-play-starter.integration.test.tsx ../packages/ui/src/hooks/chat/__tests__/useCharacterChatMode.contract.test.ts --reporter=verbose
```

Expected: pass.

- [ ] **Step 2: Run TypeScript check for the frontend package**

Use the existing frontend command. If the package has known inherited TypeScript debt, record which failures are inherited and which are touched-scope regressions.

- [ ] **Step 3: Run real-backend Playwright verification**

Run the Phase 7 E2E command from Task 7. Capture the exact pass/skip/blocker state for no-provider, provider-failure, and successful-send scenarios.

- [ ] **Step 4: Run Bandit only if backend Python was touched**

Phase 7 should be frontend-only. If no Python files changed, record "Bandit skipped: documentation/frontend-only changes." If Python files were changed unexpectedly, run Bandit against the touched backend scope before finishing.

- [ ] **Step 5: Update Backlog task final summary**

Record:

- files changed
- tests run and results
- real-backend E2E status
- whether successful-send signoff used a real configured provider or was blocked by environment
- any deferred non-Phase-7 work

- [ ] **Step 6: Final diff hygiene**

Run:

```bash
git diff --check
git status --short
```

Expected: no whitespace errors; only intended files are modified.

- [ ] **Step 7: Commit final docs/task updates**

```bash
git add <task-file> <optional-doc-file>
git commit -m "docs: record character chat phase7 verification"
```

## Acceptance Tests

Phase 7 is complete only when all of these are true:

- No-provider or provider-unconfigured Character Chat state never shows `Healthy`, `Ready`, or equivalent positive copy on any character-chat status surface.
- Readiness panel, status strip, model selector, runtime inspector, composition preview, and SEND action all derive from the same model-usability result.
- Character selected plus no usable model blocks SEND or converts SEND into setup action.
- A blocked SEND does not invoke the submit handler and does not create a `/complete-v2` backend request.
- User draft, selected character, and current session context remain intact after a blocked SEND.
- Provider/model configuration failures show model/provider recovery copy, not generic retry-only guidance.
- Real-backend Playwright verification proves no-provider/send-gating without simulated frontend responses.
- Successful-send is verified through a real callable backend provider, or explicitly marked blocked because the environment lacks one.

## Non-Goals

- No new model/provider configuration wizard.
- No session naming or history reuse redesign.
- No character import/create IA changes.
- No extension sidepanel redesign beyond keeping shared status/copy contracts compatible.
- No backend provider implementation changes unless a frontend blocker reveals an existing API contract bug.

## Rollback Plan

If the new usability contract causes unexpected regressions:

1. Revert the commits from Tasks 3-6 first; the pure utility from Task 1 can remain if unused.
2. Keep the new utility tests if they still pass and document the UI rollback.
3. Do not re-enable invalid SEND behavior as a partial fix. If SEND gating causes problems, disable the new visible setup action while preserving the no-submit guard for unusable model states.
