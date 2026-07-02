---
id: TASK-12100
title: Fix web-build browser storage shims (wxt-browser clear/isolation, plasmo change propagation)
status: Done
labels:
- bug
- high
- frontend
- shims
documentation:
- apps/FRONTEND_AUDIT.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
**Severity: High (data loss + settings that don't apply).** From the 2026-07-02 frontend audit (findings H9, H10). These shims fake `chrome.storage`/`browser.*` over `localStorage` in the web build and are **live** (aliased in `next.config.js`), consumed by shared `packages/ui` code.

1. **`wxt-browser.ts` `clear()` wipes the whole origin (H9).** `apps/tldw-frontend/extension/shims/wxt-browser.ts:189-214` — `clear()` for any area calls `backend.clear()`, deleting the entire origin's localStorage (all areas, theme/feature-flag keys, plasmo-prefixed keys, `tldw-api-host`). Live caller: `packages/ui/src/components/Option/Settings/system-settings.tsx:62`.

2. **No per-area isolation (H9).** `:108-215` — `local`, `sync`, and `session` areas read/write the same unprefixed keys, so `sync.set` clobbers `local`, `session` (should be memory-only) persists to disk, and the same logical key lives at two physical keys depending on whether the wxt shim or the plasmo shim wrote it.

3. **Plasmo `useStorage`/`watch()` never propagate changes (H10).** `plasmo-storage.ts:143-185` keeps watch callbacks per-instance; `plasmo-storage-hook.tsx:37-59` `useStorage` never subscribes at all. Two components on the same key desync, and a value written elsewhere doesn't apply until a full page reload (e.g. `stickyChatInput` toggle at `ChatSettings.tsx:727` not reflected in `WebLayout.tsx:161`; `ReviewPage.tsx:262` config watch never fires).

Also fold in (Medium) the `react-router-dom` shim's `useSearchParams` setter (`extension/shims/react-router-dom.tsx:192-213`) silently failing on dynamic routes (rebuilds URL from the `[bracket]` pattern), plus unstable `useNavigate` identity and `useLocation` source-mixing.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 `wxt-browser` `clear()` for an area clears only that area's keys, not the whole origin localStorage.
- [ ] #2 The three storage areas are isolated (prefixed) and consistent with the plasmo shim's key scheme; `session` is not persisted to disk.
- [ ] #3 Quota/serialization failures in `set()` do not emit a phantom `onChanged` nor resolve as success.
- [ ] #4 Plasmo `useStorage`/`watch()` propagate changes across instances/tabs (subscribe to a real change signal), so settings apply without a reload.
- [ ] #5 `useSearchParams` setter works on dynamic routes (uses `router.asPath`/actual path, not the bracket pattern).
- [ ] #6 Tests cover: area-scoped `clear()`; a settings toggle reflected without reload; `setSearchParams` on a `/sources/:id`-style route.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
