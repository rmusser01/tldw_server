# Character Chat Phase 4 Sessions Continuity Implementation Plan

> **For agentic workers:** Follow this plan task-by-task with TDD. Keep the slice scoped to `/chat` Character Chat session continuity and reuse existing server chat history/loading paths.

**Goal:** Make returning Character Chat users able to find and resume recent character conversations from the `/chat` Character Chat workflow without confusing conversations with saved role-play setups.

**Architecture:** Add a compact Character Chat sessions panel inside the existing Playground cockpit context rail when Character Chat mode is active. Back it with the existing `useServerChatHistory(filterMode: "character")` and `useSelectServerChat` hooks. Keep generic `ChatSidebar` behavior intact; this is a mode-local continuity affordance, not a replacement for the global server chat sidebar.

**Tech Stack:** React, TypeScript, Testing Library, Vitest, existing Playground cockpit rail components, existing tldw server chat history APIs.

---

## Source Context

- PRD: `Docs/Product/WebUI/Character_Chat_Roleplay_First_Class_PRD_2026_05_18.md`
- Backlog: `TASK-449`
- Existing history hook: `apps/packages/ui/src/hooks/useServerChatHistory.ts`
- Existing resume hook: `apps/packages/ui/src/hooks/chat/useSelectServerChat.ts`
- Existing cockpit rail: `apps/packages/ui/src/components/Option/Playground/PlaygroundContextRail.tsx`
- Existing generic history list: `apps/packages/ui/src/components/Common/ChatSidebar/ServerChatList.tsx`

## Scope

- In scope:
  - Character Chat mode-local recent sessions panel on `/chat`.
  - Server-backed character-scope history query.
  - Resume buttons and active-chat state.
  - Current-character prioritization when the active character id is known.
  - Focused tests for history query contract, rendering states, and resume behavior.
  - Backlog closeout notes and browser/real-backend verification attempt.
- Out of scope:
  - Extension sidepanel parity.
  - Command palette shortcuts.
  - Full session archive/restore/delete redesign.
  - Backend endpoint changes.
  - Replacing the global ChatSidebar.
  - Saved role-play setup storage changes.

## Stage 1: Character Session Panel Contract

**Goal:** Define and test the component contract for a Character Chat sessions panel.

**Success Criteria:**
- Panel fetches only character-scoped server chats.
- Empty/loading/error states are local and do not mention saved setups.
- Recent sessions render with title, state/topic/update context, and active state.
- Resume action calls the existing server chat selection hook with the selected chat.

**Tests:**
- `apps/packages/ui/src/components/Option/Playground/__tests__/CharacterChatSessionsPanel.test.tsx`

**Status:** Complete

## Stage 2: Current-Character Prioritization And Safe Resume

**Goal:** Make the panel useful for returning role-play users by prioritizing sessions for the currently selected character and preventing no-op active-chat reloads.

**Success Criteria:**
- Sessions whose `character_id` matches the selected character appear first under a clear current-character label.
- Other character chats remain visible under a separate label.
- The active server chat is labelled `Current` and its resume button is disabled.
- Selecting another character chat clears stale next-send context through `useSelectServerChat`.

**Tests:**
- Extend `CharacterChatSessionsPanel.test.tsx`.
- Extend `apps/packages/ui/src/hooks/chat/__tests__/useSelectServerChat.context-reset.test.tsx` only if a new stale-state reset is required.

**Status:** Complete

## Stage 3: Wire Into Playground Character Chat Mode

**Goal:** Show the panel in `/chat` only when Character Chat mode is active.

**Success Criteria:**
- `Playground` passes the active selected character id/name and active server chat id into the panel.
- `PlaygroundContextRail` accepts an optional Character Chat sessions panel node and renders it near the conversation session section.
- Standard chat mode does not render the Character Chat sessions panel.
- Existing cockpit and readiness tests continue to pass.

**Tests:**
- Extend `apps/packages/ui/src/components/Option/Playground/__tests__/Playground.cockpit-shell.test.tsx`.
- Run the new panel tests.

**Status:** Complete

## Stage 4: Verification And Closeout

**Goal:** Verify focused behavior, attempt browser/real-backend evidence, and close the Backlog task.

**Success Criteria:**
- Focused Vitest suite passes.
- `git diff --check` passes.
- TypeScript is run or documented with existing baseline failures separated from touched-file errors.
- Real backend/browser verification is attempted against the running WebUI if feasible.
- Bandit is skipped only if no Python files are touched.
- `TASK-449` records touched files, verification, skips, and final summary.

**Tests:**
```bash
cd apps/tldw-frontend
bunx vitest run \
  ../packages/ui/src/components/Option/Playground/__tests__/CharacterChatSessionsPanel.test.tsx \
  ../packages/ui/src/components/Option/Playground/__tests__/Playground.cockpit-shell.test.tsx \
  ../packages/ui/src/hooks/chat/__tests__/useSelectServerChat.context-reset.test.tsx \
  --reporter=verbose
```

**Status:** Complete

## Implementation Notes

- Use `useServerChatHistory("", { mode: "overview", filterMode: "character", limit: 5 })`; do not client-side filter generic chats as the primary contract.
- Use `useSelectServerChat()` for resume so server chat metadata restoration and selected assistant synchronization stay centralized.
- Do not render saved role-play setup bundles in this panel. Those belong in `RolePlaySetupDrawer` / `SavedRolePlaySetupsPanel`.
- Keep copy concise and cockpit-density appropriate; this is an operational rail, not a marketing section.
- Preserve the global `ChatSidebar` server-history filter. The new panel is an in-context shortcut for Character Chat mode.

## Verification Notes

- `bunx vitest run ../packages/ui/src/components/Option/Playground/__tests__/CharacterChatSessionsPanel.test.tsx ../packages/ui/src/components/Option/Playground/__tests__/PlaygroundContextRail.first-slice.test.tsx ../packages/ui/src/components/Option/Playground/__tests__/Playground.cockpit-shell.test.tsx ../packages/ui/src/hooks/chat/__tests__/useSelectServerChat.context-reset.test.tsx --reporter=verbose` passed: 4 files, 49 tests.
- `git diff --check` passed.
- `bunx tsc --noEmit --pretty false` still fails on inherited frontend baseline errors outside the touched files: `MediaReadAlongPopover.tsx`, `EmbeddingsModelSelectionConfig.tsx`, `StudioPane/index.tsx`, `useShortcutConfig.ts`, and `e2e/workflows/tier-4-admin/admin-llamacpp.spec.ts`.
- Targeted ESLint from `apps/tldw-frontend` only reports shared UI files as outside the frontend base path; repo-root ESLint has no config at the root, so no useful ESLint signal was available for this touched shared package slice.
- Real backend/browser verification used FastAPI on `127.0.0.1:8000` and Next on `localhost:3000`. Browser inspection confirmed `Character chat sessions` appears on `/chat?mode=character`, and `/chat?mode=character&characterId=2` shows `Recent sessions for Helpful AI Assistant` with `Resume` and disabled `Current` actions.
- Bandit skipped: frontend-only TypeScript/React scope.
