# Chat Page UAT Remediation Plan

**Created:** 2026-06-13
**Source review:** `Docs/Design/2026-06-13-chat-page-uat-review.md`
**Scope:** `/chat` (Playground) — `apps/packages/ui/src/components/Option/Playground/*` and `apps/packages/ui/src/hooks/playground/*`
**Goal:** Make `/chat` usable on first run (unblock sending), remove contradictory state, and reduce first-time cognitive load — without sacrificing power-user depth.

Work is sequenced so the highest-impact issue is addressed first. Each stage is independently shippable and testable.

> **Diagnosis correction (2026-06-13):** The original Stage 1 claimed the chat page sourced models from the wrong endpoint and prescribed deriving models from configured providers. That is **wrong**. The chat page already sources models from `/api/v1/llm/models/metadata` (backend serves it correctly). The real issue is an **intermittent mount-time race** that returns an empty model list with no network call and no recovery. Stage 1 is rewritten accordingly.

---

## Stage 0 (prerequisite): Reliably reproduce — **DONE**
**Outcome:** Built looped Playwright repros (fresh context per load, with `seedAuth`). Findings that revised the diagnosis:
- `/api/v1/llm/models/metadata` **is** requested on essentially every fresh load (~600 ms after mount) and returns 843 KB of models — there is **no fetch-skipping race**, and the catalog source is correct (earlier "never called" observations were observation-window artifacts: the endpoint is slow, ~7 s server-side).
- The real defect: during that ~7 s fetch the picker has **no loading state**, so it shows the terminal-sounding "No models available. Connect your server in Settings." and blocks sends — which looks like a permanent failure. Models do load and a model auto-selects once the fetch returns.
**Exit criteria met:** Deterministic behavioral repro — `e2e/chat-model-loading.spec.ts` asserts the connect-server error must not appear while the fetch is in flight (red before fix).
**Status:** Completed

## Stage 1: Show a loading state instead of a false "connect your server" error — **DONE**
**Goal:** While the catalog is loading, the picker shows a loading affordance, never the terminal connect-server error against a reachable server.
**Findings addressed:** #1 (root cause corrected: missing loading state, not a config-readiness race).
**What was implemented:**
- `apps/packages/ui/src/components/Option/Playground/PlaygroundForm.tsx` — derive `composerModelsLoading` from the `playground:chatModels` query's `isFetching` (true only while fetching with no models yet); pass into `useModelSelector`.
- `apps/packages/ui/src/hooks/playground/useModelSelector.tsx` — new `modelsLoading` param; when the catalog is empty **and** loading, the dropdown shows a "Loading models…" item (`data-testid="model-loading"`) instead of the "No models available. Connect your server in Settings." + "Open model settings" items. The connect-server message now only appears once the fetch has genuinely completed with zero models.
**Tests:** `e2e/chat-model-loading.spec.ts` (green); unit coverage in `useModelSelector.capabilities.test.tsx` (loading-shows-affordance / loaded-empty-shows-connect-error).
**Status:** Completed

### Follow-ups not done (lower priority)
- Backend `/api/v1/llm/models/metadata` is slow (~7 s warm) — a backend perf item, separate scope.
- `refreshCharacterChatModels` in `Playground.tsx` calls `fetchChatModels({ forceRefresh: true })` imperatively in parallel with the `PlaygroundForm` React Query; consider unifying to one source (also relates to Stage 5 dedupe).

<details><summary>Original Stage 1 approach (superseded — kept for context)</summary>

1. Make the mount-time fetch await config readiness, **or** re-run the fetch on the existing `tldw:config-updated` event (dispatched by `updateConfig`, `TldwApiClient.ts:1695`).
2. Do **not** cache/treat the "config not ready" path as a real empty result, and do not let `FORCE_REFRESH_COOLDOWN` suppress the first *successful* fetch.
3. Replace the silent empty with a truthful transient state ("Connecting…") and a retry affordance; only show "No models available. Connect your server in Settings" once config is confirmed ready and the fetch genuinely returned nothing.

**Success criteria:**
- Stage 0 repro loop: picker populates on 100% of loads (vs. the measured baseline hit rate).
- When config is genuinely absent, the UI shows a connecting/not-connected state with a working retry — not a silent empty + "Healthy" badge.

**Tests:**
- Unit: `getModels` does not cache an empty result produced by the not-configured gate; re-fetches once config is present.
- E2E: looped `seedAuth` → `/chat` asserts `/llm/models/metadata` is called and the picker is non-empty every iteration; assert a message sends and streams a reply.
**Status:** Superseded (see corrected diagnosis above — the fetch is reliable; the fix was the loading state).
</details>

---

## Stage 2: Reconcile readiness state — **DONE**
**Goal:** No contradictory "Healthy" badge when you cannot chat yet.
**Findings addressed:** #3 (and, via the Stage 1 loading fix, #2 and #8 — see note).
**What was implemented:** `ChatModelSelectorDropdown.tsx` — when no model is selected and there is no explicit `modelUsabilityLabel`, the status badge shows "No model selected" in warn styling instead of falling back to the connection status ("Healthy"/"Connected"). Once a model is selected the connection badge is shown as before. Verified live: badge reads "No model selected" pre-load and "Healthy" after a model auto-selects.
**Tests:** `ChatModelSelectorDropdown.character-usability.test.tsx` (no connection badge with no model; restored once selected).
**Status:** Completed

### #2 (failure-state recovery) and #8 (empty-state CTA) — resolved by Stage 1
With the loading state in place, the empty picker only appears once the fetch genuinely returns zero models, and the model **auto-selects** after the catalog loads (verified live: `tldw:gpt-4o` becomes active without user action). The Playground model dropdown's recovery link is "Open model settings" → `/settings/tldw` (a real route); the dead "Open current chat settings" string is in the **Sidepanel** chat (`components/Sidepanel/Chat/form.tsx`), not `/chat`. The empty-state "Start chatting" CTA dispatches a starter and focuses the composer, which is the correct action. No separate `/chat` change required; closing #2 and #8.

---

## Stage 3: Overlay dismissal (Escape) — **DONE**
**Goal:** The shortcuts help panel closes on Escape and returns focus to the trigger.
**Findings addressed:** #4.
**Root cause found:** An existing bubble-phase Escape handler (`Playground.tsx:1971`) never fired because a **global capture-phase keydown listener calls `preventDefault()` + `stopPropagation()` on Escape** (confirmed via event-phase tracing: only the window capture phase saw the event, already `defaultPrevented`, and it never bubbled). No `stopImmediatePropagation` exists in the codebase, so a sibling capture-phase listener still receives the event.
**What was implemented:** `Playground.tsx` — added a `useEffect` (active only while `shortcutsHelpOpen`) that registers a **capture-phase** `window` keydown listener to close the panel on Escape and restore focus to the trigger.
**Tests:** `e2e/chat-shortcuts-escape.spec.ts` — opens the panel, presses Escape, asserts hidden + focus returned to trigger (green).
**Follow-up (not done):** audit other dismissible Playground overlays (artifacts sheet, modals) for the same swallowed-Escape issue; consider a shared `useEscapeToClose(capture)` helper. Click-outside dismissal not addressed.
**Status:** Completed

---

## Stage 4: First-run cognitive load & terminology — **Medium**
**Goal:** Composer-first empty state; cockpit available but not overwhelming; plainer labels.
**Findings addressed:** #5, #7, #9.
**Key files:** `PlaygroundEmpty.tsx`, `PlaygroundCockpitShell.tsx`, `PlaygroundContextRail.tsx`, `playground-cockpit-state.ts`, mobile layout (`mobile-composer-layout.ts`, `useMobileComposerViewport.ts`).

**What was implemented (#5 — rail default):** Defaulted the Context/Runtime cockpit rails **collapsed** for users without a saved preference (`Playground.tsx`: `playgroundChatContextRailVisible`/`playgroundChatRuntimeRailVisible` defaults `true`→`false`). First-run now leads with the composer and a clean "Start a new chat" card; both rails are one click away via the edge restore tabs (`playground-cockpit-left/right-rail-restore`) and the Context/Runtime header toggles. Restoring persists; returning users who set a preference are unaffected (the `!== false` normalization keeps explicit `true`). Verified live (screenshot `assets/chat-uat-2026-06-13/30-collapsed-firstrun.png`).
**Tests:** `e2e/chat-cockpit-rail-default.spec.ts` (collapsed by default with restore affordances; restore shows the rail and persists across reload).

**#7 (terminology) and #9 (mobile) — not done (need design direction):**
- #7: "cockpit" is deeply embedded (i18n namespace `cockpit.*`, CSS vars, data-attributes, component names, plus user-facing labels "Chat cockpit"/"Composition"/"Runtime"/"sidechannel"). Renaming the **display strings** is feasible but changes the product's established voice — a product/voice decision, not a bug.
- #9: mobile already defaults to a `"focus"` layout (not full cockpit); remaining work is refinement of the focus-mode IA. Lower priority.

**Status:** #5 Completed; #7 and #9 deferred (design decisions)

---

## Stage 5: Redundant fetching on load — **Medium**
**Goal:** Each load-time read fires once. Measured on a single `/chat` load: `/users/me/profile` ×5, `/persona/profiles` ×4, `/config/providers` ×3, `/characters/` ×2, `/notifications` ×2, `/notifications/unread-count` ×2, `/persona/catalog` ×2.
**Findings addressed:** #6.
**Key files:** the query hooks behind profile, persona, providers, characters, and notifications reads feeding the composer/cockpit (search for each path's consumers).

**What was implemented (in-flight coalescing):** Investigation showed the duplicates are **identical concurrent GETs** firing in the same tick on load (many components mounting, each fetching the same resource). Rather than a risky per-hook React Query migration, added **in-flight request coalescing** to `bgRequest` (`apps/packages/ui/src/services/background-proxy.ts`): concurrent GETs with the same path/headers/auth share a single underlying request; the entry is dropped once it settles, so sequential-call caching/staleness is unchanged. Only idempotent GETs (no body/abort/stream/`preferDirect`) are coalesced.

**Result on `/chat` load:** `/users/me/profile` 5→1, `/config/providers` 3→1, `/characters/` 2→1, `/persona/catalog` 2→1.

**Out of scope / follow-up:** Endpoints fetched via `tldwClient.fetchWithAuth` (e.g. `/persona/profiles` ×4, `/notifications` ×2) return a raw `Response` whose body is single-read and **cannot be safely shared** across callers — these need a different approach (consolidate callers into a shared React Query hook, or have `fetchWithAuth` return cloned/parsed data). Also still worth doing: unify `Playground.tsx` `refreshCharacterChatModels`'s imperative `getProvidersStatus()`/`fetchChatModels()` with `PlaygroundChat`'s React Query so they share one source.

**Tests:** `e2e/chat-request-dedupe.spec.ts` (network counts ≤1 for the coalesced endpoints on `/chat` load); `background-proxy.test.ts` (coalesces concurrent identical GETs; POST not coalesced).
**Status:** Completed (bgRequest endpoints); fetchWithAuth endpoints deferred

---

## Validation harness
Reuse the UAT drivers as regression checks:
- `apps/tldw-frontend/scripts/chat-uat-driver.mjs` — full walkthrough + screenshots + observations.json
- `apps/tldw-frontend/scripts/chat-uat-driver2.mjs` — model-select + send + Escape-dismiss
- `apps/tldw-frontend/scripts/netprobe.mjs` — provider/config fetch counting

Run after each stage; Stage 1 is the gate for re-reviewing downstream flows (character chat, role-play, voice, compare, branching, artifacts, image-gen) that were blocked during this UAT.
