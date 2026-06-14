# Characters Page UAT Remediation Plan

**Created:** 2026-06-14
**Source review:** `Docs/Design/2026-06-14-characters-page-uat-review.md`
**Scope:** `/characters` (CharactersWorkspace → Manager) + the app-wide Escape root cause in `Common/CommandPalette.tsx`.

The `/characters` page is well-built; the one concrete bug found is app-wide and is fixed in this PR. The rest are minor/deferred.

---

## Stage 1: App-wide Escape-to-close — **DONE**
**Finding:** #1. Escape did not close the New character drawer (or any antd Drawer/Modal/Select app-wide), despite the documented "Esc Close modal".
**Root cause:** `Common/CommandPalette.tsx` registered an Escape `useShortcut` with no `enabled` gate. `useShortcut` (`hooks/useKeyboardShortcuts.ts`) installs a **capture-phase** keydown listener on `window` and `document` and calls `preventDefault()` + `stopPropagation()` on a match — so an always-on Escape shortcut swallowed Escape before antd's built-in handlers.
**Fix:** add `enabled: open` to the Escape `useShortcut` so it is active only while the palette is open.
**Tests:** `e2e/characters-drawer-escape.spec.ts`; `CommandPalette.shortcuts.test.tsx` (closes-on-Escape-when-open; not-swallowed-when-closed).
**Status:** Completed. (Note: this also resolves the `/chat` shortcuts-panel Escape issue that #2350 worked around per-component; that workaround is now redundant but harmless and is left in place.)

### Follow-up (broader hardening, not done)
`useShortcut`'s `stopPropagation()` in a capture-phase window+document listener is aggressive for *every* key, and it registers on both `window` and `document` (redundant). A future change could (a) drop the `document` registration, and (b) avoid `stopPropagation()` for keys that overlays commonly own (Escape), so a missing `enabled` gate can never again swallow modal Escape globally. Out of scope here to keep the blast radius small.

---

## Stage 2: antd `useForm` not connected to a Form — **Open (deferred)**
**Finding:** #2. Console warning *"Instance created by `useForm` is not connected to any Form element"* on `/characters` load.
**Approach:** locate the `Form.useForm()` instance whose `form` is not passed to a mounted `<Form>` (likely a lazily-rendered editor/dialog form created eagerly). Either pass `form` to the rendered `<Form>` or create the instance lazily when the form mounts.
**Why deferred:** harmless at runtime; low priority; needs tracing the specific disconnected form among several.
**Status:** Not Started.

---

## Stage 3: App-shell redundant fetches — **Open (cross-page follow-up)**
**Finding:** #3. `/notifications` ×2, `/notifications/unread-count` ×2, `/persona/profiles` ×2 on passive load — shared app-shell reads (also on `/chat`). `/persona/profiles` uses `fetchWithAuth` (single-read Response, can't be coalesced via the merged `bgRequest` coalescing).
**Approach:** consolidate the duplicated app-shell hooks (notifications, persona) into shared React Query keys, and/or give `fetchWithAuth` consumers a cloned/parsed shared result. Same follow-up noted in the `/chat` plan (#2350) — best done once, app-wide, not per page.
**Status:** Not Started (deferred to a dedicated app-shell pass).
