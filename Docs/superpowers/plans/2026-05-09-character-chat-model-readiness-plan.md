# Character Chat Model Readiness And In-Context Blockers Plan

> For implementation agents: use the repository superpowers workflow before editing code.

**Goal:** Provide one shared readiness contract for character chat so missing model/provider state is shown locally and consistently across Characters, Chat, and character creation.

**Primary evidence:** The audit saw three different blocker messages: `No models available` in character creation, `No AI models available` in Chat, and `Configure an LLM provider` after row-level chat redirected to Home.

**Likely surfaces:**
- `apps/packages/ui/src/hooks/useChatModelsSelect.ts`
- `apps/packages/ui/src/utils/chat-model-validation.ts`
- `apps/packages/ui/src/components/Sidepanel/Chat/empty.tsx`
- `apps/packages/ui/src/components/Option/Characters/GenerateCharacterPanel.tsx`
- `apps/packages/ui/src/components/Option/Characters/hooks/useCharacterQuickChat.tsx`
- `apps/packages/ui/src/components/Option/Playground/PlaygroundChatErrorBanner.tsx`
- `apps/packages/ui/src/services/model-settings.ts`
- `apps/tldw-frontend/pages/settings/model.tsx`

## Stage 1: Define Readiness Contract

**Goal:** Establish one small API for "can character chat start?"

**Success Criteria:**
- Readiness distinguishes server connection, character selected, model/provider configured, and chat send capability.
- Existing model-loading hooks are reused.
- A failing test captures inconsistent blockers.

**Tests:** Unit tests for readiness states.

**Status:** Complete

Steps:

- Inventory existing model availability checks.
- Define a typed readiness result with status, missing requirement, and recommended action.
- Decide whether AI-generation model readiness is separate from chat model readiness.

Outcome:

- Added shared `buildCharacterChatReadiness` and `getCharacterChatReadinessCopy` helpers in `chat-model-availability.ts`.
- Readiness now distinguishes server connection, selected character, chat model selection/catalog state, and send-disabled state.
- Kept AI character generation model copy separate from character-chat model readiness.

## Stage 2: Replace Fragmented Blockers

**Goal:** Use the shared contract in character-chat surfaces.

**Success Criteria:**
- Characters row action shows a local blocker when model readiness fails.
- Chat empty state uses the same readiness language.
- Character creation/generation distinguishes optional AI-generation models from required chat models.

**Tests:** Component tests for each missing requirement.

**Status:** Complete

Steps:

- Add a compact readiness panel or inline callout.
- Include local actions such as `Open Model Settings`, `Retry`, and `Back to character`.
- Preserve selected character context when taking setup actions.

Outcome:

- Added an in-context quick-chat blocker in the Characters modal when the selected character exists but no chat model is ready.
- Reused the readiness copy in the chat no-model banner when a selected character is active.
- Updated generation no-model copy so saved characters are not framed as unavailable.

## Stage 3: Verify Positive And Negative Paths

**Goal:** Prove users can tell what is missing and proceed once fixed.

**Success Criteria:**
- No-model state blocks send clearly without route displacement.
- Model-present state allows chat bootstrap.
- The same message does not appear in unrelated non-character flows unless applicable.

**Tests:** Focused mocked component tests and existing chat model utility tests; full browser screenshot re-walkthrough remains in the post-package audit plan.

**Status:** Complete

Steps:

- Add a mocked-model happy path if existing E2E harness supports it.
- Add screenshots for no-model and model-ready states.
- Re-run affected chat and Characters tests.

Outcome:

- Verified utility readiness states, quick-chat no-model blocker, and generation-model copy with focused Vitest coverage.
- Ran package TypeScript verification; it remains blocked by existing broad UI type debt outside this package, while the new helper-specific diagnostics were resolved.
- Deferred full Puppeteer/Chrome screenshot validation to the explicit post-implementation re-walkthrough package.

## Risks

- Model readiness can be provider-specific and may not map cleanly to a boolean.
- Sharing one contract too early can overfit character chat and regress other chat modes.

## Handoff Notes

This package is a dependency for intent preservation, route-aware onboarding, and character-mode sequencing.
