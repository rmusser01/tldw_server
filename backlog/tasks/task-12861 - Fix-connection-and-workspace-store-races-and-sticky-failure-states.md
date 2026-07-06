---
id: TASK-12861
title: Fix connection and workspace store races and sticky failure states
status: Done
labels:
- bug
- high
- state
- zustand
- packages-ui
documentation:
- apps/FRONTEND_AUDIT.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
**Severity: High (intermittent wrong UI state).** From the 2026-07-02 frontend audit (findings H7, H8, H11). Paths relative to `apps/packages/ui/src/store/`.

1. **Connection store stale-snapshot clobber (H7).** `connection.tsx` `checkOnce` captures `currentState` at `:595`, runs a health check for up to 20s, then every terminal `set()` does `{...currentState, ...}` (`:949-1014`) — reverting any `setConfigPartial`/`markFirstRunComplete`/`setUserPersona`/`beginOnboarding` that fired meanwhile (onboarding jumps back a step; `hasCompletedFirstRun` flips back). The overlap guard reads `isChecking` at `:598` but sets it at `:698` after five awaits, so concurrent callers both run and the last stale finisher wins — UI flips to "disconnected" right after a good check.

2. **Workspace store silent-mutation rehydrate (H8).** `workspace.ts:3875-3970` `onRehydrateStorage` mutates the hydrated state in place (`Object.assign`, `state.storeHydrated = true`) with no `set()`, so subscribers are never notified. Hydration is async and components are already mounted → intermittent empty workspace / eternal loading gate. Compounded by `workspace-slices/workspace-list-slice.ts:1208` `reset` leaving `storeHydrated:false`.

3. **Sticky persisted failure (H11).** `folder.tsx`: any 404 (or a message containing "404", `:409`) sets `folderApiAvailable:false`, `partialize` persists it (`:843`), `refreshFromServer` hard-returns when false (`:314`), and the only reset to true (`:401`) is in the skipped path. One transient 404 disables folder sync across every future session until localStorage is cleared.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 `connection.tsx` uses functional `set((s) => ...)` in terminal health-check writes so concurrent state changes are not clobbered.
- [ ] #2 The health-check overlap guard is set synchronously before the first await (or uses a request token so only the latest result wins).
- [ ] #3 `workspace.ts` rehydration applies hydrated data via `set()` (subscribers notified); `storeHydrated` is set through `set()`.
- [ ] #4 `folder.tsx` does not permanently disable sync on a transient 404 (don't persist the flag, or add a reset/retry path).
- [ ] #5 Tests cover: a slow health check does not revert a concurrent config edit; a fresh reload renders workspace contents without an unrelated `set()`; a single 404 does not permanently kill folder sync.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
