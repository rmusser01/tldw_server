# Character Chat Intent Preservation Plan

> For implementation agents: use the repository superpowers workflow before editing code.

**Goal:** Make every `Chat as character` entry point preserve the selected character and either open the intended chat or show an in-context blocker that can return to the same character after model setup.

**Primary evidence:** The audit clicked a row-level `Chat as...` action from Characters and landed on `/` Companion Home with a generic model setup message.

**Likely surfaces:**
- `apps/packages/ui/src/components/Option/Characters/hooks/useCharacterQuickChat.tsx`
- `apps/packages/ui/src/components/Option/Characters/CharacterListContent.tsx`
- `apps/packages/ui/src/components/Option/Characters/CharacterGalleryCard.tsx`
- `apps/packages/ui/src/utils/characters-route.ts`
- `apps/packages/ui/src/hooks/useSelectedCharacter.ts`
- `apps/packages/ui/src/hooks/chat/useCharacterChatMode.ts`
- `apps/packages/ui/src/store/chat-surface-coordinator.ts`
- `apps/tldw-frontend/e2e/workflows/journeys/character-chat.spec.ts`

## Stage 1: Reproduce And Define The Contract

**Goal:** Lock the current failed flow in a test before changing it.

**Success Criteria:**
- A component or E2E test reproduces that row-level chat loses character context when no model is available.
- A written contract defines where selected character intent is stored while setup is incomplete.

**Tests:** Targeted Vitest around quick-chat hook and, if feasible, Puppeteer/Playwright E2E for the row action.

**Status:** Complete

Steps:

- Trace the row action from Characters to navigation target and state mutation.
- Identify whether selected character should be held in URL state, shared store, or chat session bootstrap state.
- Write the failing test around the current redirect/context-loss behavior.

Outcome:

- Confirmed the row-level `Chat as ...` handler stored the selected character and immediately navigated to `/` without checking model readiness.
- Chose the existing selected-character storage handoff as the durable intent carrier, with a local blocker as the incomplete-setup UI.
- Added a failing regression test for the no-model row action before changing production code.

## Stage 2: Preserve Selected Character Through Blockers

**Goal:** Keep the user in the character-chat task context when chat cannot start immediately.

**Success Criteria:**
- `Chat as [character]` records the intended character before any navigation.
- Missing model state shows a local blocker naming the selected character.
- The blocker has actions for model setup, retry, and return to character.
- After setup, the selected character is still available to the chat flow.

**Tests:** Unit/component tests for state preservation and blocker rendering.

**Status:** Complete

Steps:

- Add or reuse a typed selected-character handoff object.
- Preserve character id, name, and source route.
- Avoid a generic `/` fallback unless no better local route exists.
- Ensure no stale character persists after cancellation or explicit clear.

Outcome:

- The row-level chat action now stores the selected character before readiness checks.
- Missing chat-model readiness opens a local character-chat setup blocker that names the selected character and offers model settings, retry, and return actions.
- Explicit return/close clears the selected-character handoff to avoid stale intent leakage.

## Stage 3: Wire Chat Creation/Resume Behavior

**Goal:** Make the preserved intent lead to a real character-chat session once prerequisites are satisfied.

**Success Criteria:**
- With a configured model, the row action opens or creates a character chat.
- With no model, the user can configure one and return to the same character.
- Existing regular chat and persona flows are not regressed.

**Tests:** Character chat journey E2E with mocked or test provider state; existing chat action tests.

**Status:** Complete

Steps:

- Resolve whether the row action should create an empty session immediately or defer until first send.
- Preserve selected character across model setup navigation.
- Add regression coverage for switching between two characters.

Outcome:

- Preserved the existing model-ready behavior: row chat sets the selected character, navigates to the chat surface, and focuses the composer.
- Deferred empty server-chat creation until first send, matching the existing chat flow and avoiding empty-history pollution.
- Added regression coverage for the no-model intent blocker and kept the existing model-ready handoff test green with an explicit mocked model.

## Risks

- Persisting selected character globally can leak stale intent into unrelated chats.
- URL-only state may be brittle if setup uses multiple routes.
- Creating empty conversations too early may pollute returning-user history.

## Handoff Notes

Coordinate with the model-readiness package. Intent preservation should consume the readiness contract instead of inventing separate model checks.
