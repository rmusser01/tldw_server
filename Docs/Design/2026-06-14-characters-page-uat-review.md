# `/characters` Page — Senior UX UAT Review

**Date:** 2026-06-14
**Reviewer role:** Senior UX specialist (User Acceptance Test)
**Build under test:** dev branch (`origin/dev` @ `62e3ba3298`, includes the merged `/chat` fixes), Next.js frontend on `:8080`, FastAPI backend on `:8000` (single-user). Seeded ~12 disposable `uat-test-char-*` characters (16 total) to exercise list density/filter/pagination.
**Method:** Live browser, driven through real Chromium via Playwright (`apps/tldw-frontend/scripts/chars-uat-driver.mjs` + focused probes). Auth via the env-driven smoke config (`seedAuth`). Personas: **first-time** and **power user**.
**Artifacts:** screenshots + `observations.json` in `./assets/characters-uat-2026-06-14/`.

---

## Summary

**First-time user:** Good. The page leads with clear intent ("Create reusable characters you can pick from the chat header and reuse across conversations"), a prominent "New character" button, and an "Upload character" affordance. Creating a character opens a right-side drawer that is genuinely helpful: five starter **templates** (Writing Assistant, Patient Teacher, Research Helper, Code Reviewer, Creative Partner), an **AI generate-from-concept** panel with a model selector, and the manual form. Keyboard shortcut hints are surfaced inline. The empty/loading/error/offline/demo states are all handled.

**Power user:** Strong and dense. Table and gallery views, search, multi-facet filters (tags, folder, creator, dates) with active-filter chips, active/Trash scope, per-row quick actions (quick-chat, edit, delete, favorite, more), bulk operations, pagination, and a documented shortcut set (N new, / focus search, G T / G G view, Esc close). The app was stable — 0 page errors. The one real defect is that **Escape does not close the New character drawer** despite the page's own "Esc Close modal" hint (Finding #1) — and the root cause is app-wide.

---

## What works well (keep)
- **Onboarding drawer**: templates + AI generation + manual form is an excellent first-run path.
- **Clear first-time copy** and inline keyboard-shortcut hints.
- **Density without clutter**: table/gallery toggle, faceted filters with removable chips, per-row quick actions.
- **State coverage**: loading skeletons, error+retry, empty, offline, demo — all present.
- **Stability**: 0 uncaught page errors across all flows; lazy-loaded dialogs/editor.

---

## Findings

| # | Feature / Area | Persona | Issue | Severity | Status |
|---|----------------|---------|-------|----------|--------|
| 1 | Keyboard / modals | Both | **Escape does not close the New character drawer**, despite the page documenting "Esc Close modal". Rigorously confirmed: drawer stays open after Escape in every focus state; only mask-click / Cancel close it. **Root cause is app-wide:** the globally-mounted `CommandPalette` registers an Escape `useShortcut` with **no `enabled` gate**; `useShortcut`'s capture-phase handler calls `stopPropagation()` on every match, so an always-on Escape shortcut swallows Escape before antd Drawer/Modal/Select can handle it — everywhere. (Same root cause as the `/chat` shortcuts-panel Escape issue worked around in #2350.) | **High** | **Fixed** |
| 2 | Form wiring | — | Console warning on load: *"Instance created by `useForm` is not connected to any Form element. Forget to pass `form` prop?"* — an antd `Form.useForm()` instance created but not bound to a `<Form>`. Harmless at runtime but indicates a wiring bug. | Low | Open (deferred) |
| 3 | Redundant fetch on load | Both (perf) | Passive load fires `/notifications` ×2, `/notifications/unread-count` ×2, `/persona/profiles` ×2. These are **app-shell** reads shared across pages (also seen on `/chat`); `/persona/profiles` uses `fetchWithAuth` (single-read Response, can't be coalesced). | Low | Open (cross-page follow-up) |

Note: `/characters/query` fires once on passive load (the ×3 seen in the interactive driver was caused by the driver's own search/toggle refetches, not redundant load fetches).

---

## Fix shipped (Finding #1) — app-wide Escape-to-close

`apps/packages/ui/src/components/Common/CommandPalette.tsx`: gate the Escape shortcut on `enabled: open` so it is only active while the palette is open. When the palette is closed, Escape is no longer swallowed and propagates normally to antd Drawers/Modals/Selects across the whole app.

This is a one-line root-cause fix that restores Escape-to-close on `/characters` (verified live) and app-wide (chat, media, settings, …). Tests:
- `apps/tldw-frontend/e2e/characters-drawer-escape.spec.ts` — Escape closes the New character drawer.
- `apps/packages/ui/src/components/Common/__tests__/CommandPalette.shortcuts.test.tsx` — Escape still closes the palette while open; Escape is **not** swallowed (default not prevented) while the palette is closed.

---

## Coverage & limitations
- **Covered live:** initial load (table view, 16 chars), search, table/gallery toggle, active/Trash scope, New character drawer (templates/AI/form), Escape behavior (deep, multi-focus-state), keyboard Tab order, mobile (390px), console/page-error + `/api` request-count capture.
- **Not exhausted:** import dropzone end-to-end, AI generation round-trip, bulk-tag/export operations, version history, quick-chat completion — the page is feature-rich; this pass prioritized the core list/create flows and the Escape regression.
- **Data note:** seeded `uat-test-char-*` characters were created via `POST /api/v1/characters/` to exercise list density; they are removed during cleanup.

## Reproduction
```bash
# Backend on :8000 (single-user). Frontend (dev-branch worktree):
cd apps/tldw-frontend
NEXT_PUBLIC_API_URL=http://127.0.0.1:8000 bun run dev -- -p 8080
node scripts/chars-uat-driver.mjs   # full walkthrough + screenshots
# Escape regression:
TLDW_WEB_AUTOSTART=false npx playwright test e2e/characters-drawer-escape.spec.ts --project=chromium
```
