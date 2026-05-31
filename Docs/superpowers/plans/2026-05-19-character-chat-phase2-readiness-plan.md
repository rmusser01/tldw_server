# Character Chat Phase 2 Readiness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement Phase 2 of the first-class Character Chat PRD by making incomplete setup, loading, provider/model failures, selector catalog failures, missing restored characters, and persistence state visible and actionable inside `/chat`.

**Architecture:** Reuse the existing chat cockpit and shared selector surfaces instead of adding a parallel role-play UI. Character Chat readiness remains derived from `buildCharacterChatReadiness(...)`; the UI layer renders that derived state with local recovery actions and live-region semantics.

**Tech Stack:** React, TypeScript, Vitest, Testing Library, existing tldw shared UI package, real `tldw_server` smoke verification when local backend/frontend can run.

**Primary Files:**
- `apps/packages/ui/src/utils/chat-model-availability.ts`
- `apps/packages/ui/src/components/Option/Playground/Playground.tsx`
- `apps/packages/ui/src/components/Option/Playground/PlaygroundStatusStrip.tsx`
- `apps/packages/ui/src/components/Common/AssistantSelect.tsx`
- `apps/packages/ui/src/components/Common/PromptSelect.tsx`
- Existing focused tests under `apps/packages/ui/src/components/**/__tests__/`

---

## Stage 1: Readiness Panel Contract

**Goal:** Add a compact Character Chat readiness panel that renders the existing readiness model with accessible status semantics and recovery actions.

**Success Criteria:**
- Missing server, missing character, missing model, no models available, unavailable selected model, and send-blocked states produce distinct visible copy.
- The panel uses `role="status"` or an equivalent polite live-region contract for non-destructive setup changes.
- Recovery actions map to existing flows: server settings, character selector, model settings, retry.
- Ready state is non-intrusive and does not add duplicate clutter to Standard Chat.

**Tests:**
- Unit tests for the panel's visible title/body/action per readiness reason.
- Unit test that status changes are exposed with `aria-live="polite"`.

**Status:** Complete

## Stage 2: Wire Readiness Into Character Chat

**Goal:** Render the readiness panel only when Character Chat mode is active and preserve selected character intent while the user fixes model/server setup.

**Success Criteria:**
- `/chat?mode=character&characterId=...` displays local setup guidance before the first send when setup is incomplete.
- Clicking model/server recovery does not clear `selectedCharacter` or reset Character Chat mode.
- Clicking choose-character opens the existing assistant selector on the Characters tab and restores focus to the Character Chat control after close.
- Missing/deleted restored characters get a visible recovery state with choose-character and retry affordances.

**Tests:**
- `Playground.cockpit-shell` or equivalent integration test for missing character/model/server readiness.
- Integration test that selected character copy survives opening model settings.
- Integration test for a restored route character that cannot be loaded.

**Status:** Complete

## Stage 3: Prompt Selector Loading/Error/Empty States

**Goal:** Keep `PromptSelect` present and understandable when the prompt library is loading, empty, or failed.

**Success Criteria:**
- The prompt trigger remains visible during loading/error instead of disappearing.
- Loading state is announced and prevents accidental empty-selection interpretation.
- Error state has visible local copy and a retry affordance through React Query.
- Empty state distinguishes "no saved prompts" from "no matches" and still allows editing the current system prompt.

**Tests:**
- Prompt selector tests for loading trigger, query error content, empty prompt library content, and existing custom prompt recovery.

**Status:** Complete

## Stage 4: Assistant Selector Loading/Error/Empty States

**Goal:** Make character/persona catalog state explicit without blocking partial success.

**Success Criteria:**
- Loading characters/personas is visible in the dropdown.
- Character catalog failure is visible on the Characters tab and does not silently become "No characters available."
- Persona catalog failure is isolated to the Personas tab when characters load successfully.
- Empty state copy remains searchable and actionable, with a route/event path to manage actor settings where available.

**Tests:**
- Assistant selector behavior tests for loading, character catalog failure, persona-only failure, and empty character catalog.

**Status:** Complete

## Stage 5: Persistence State Messaging

**Goal:** Make Character Chat persistence state legible in the status strip and local setup surfaces.

**Success Criteria:**
- Temporary Character Chat is labeled as temporary.
- Saved/server-backed Character Chat is labeled as saved or linked to chat history.
- Local draft/no-server state is labeled distinctly from saved history.
- Copy avoids implying persistence when server readiness is blocked.

**Tests:**
- Status strip tests for temporary, saved/server-backed, and local draft Character Chat persistence labels.

**Status:** Complete

## Stage 6: Real Backend Smoke And Closeout

**Goal:** Verify the implemented workflow against the real backend where possible and close the Backlog task with evidence.

**Success Criteria:**
- Focused Vitest suites pass.
- Browser smoke uses the running WebUI and real backend when available; if unavailable, record exact blocker.
- Bandit is run for touched Python scope or explicitly skipped because this slice touches only frontend/docs/task files.
- Backlog task records touched files, verification, residual risks, and PR link.

**Tests:**
- Focused Vitest command covering changed components.
- Real-backend browser smoke of Character Chat missing-model/character-selector flow.

**Status:** Complete
