# UAT126 / TASK13260.66 — bounded repair design notes

Read-only source/design investigation during the frozen cycle5 run. No production/test/repository edits, new tests, browser interaction, inference or runtime changes. Task remains To Do pending the controller's implementation window. This is a proposed bounded design, not a reproduced automated RED or implementation claim.

## Confirmed source cause

`apps/packages/ui/src/components/Flashcards/FlashcardsManager.tsx:32–56` parses tab aliases from `location.search`; lines83–86 seed `activeTab`, and lines190–213 apply later route intent. `handleTabChange` at337–358 confirms a dirty Scheduler exit and changes only local state. Thus `/flashcards?tab=importExport` survives an accepted Study click; a fresh mount correctly follows that stale URL back to Import / Export. This matches the native observation recorded in task66 and the current tracker. Saved review/session data need no repair.

## Nearby patterns inspected

1. **Prompt Studio route/state reconciliation:** `components/Option/Prompt/Studio/StudioTabContainer.tsx:223–244` clones existing search parameters, changes only its `subtab`, uses replace, and distinguishes incoming URL navigation from local tab state. `__tests__/StudioTabContainer.stage6-navigation.test.tsx` uses a real MemoryRouter/location probe and has a StrictMode initial-deep-link preservation control. Reuse the principles, not a new global synchronization abstraction. The parent Prompt page's `setSearchParams({tab: ...})` branch discards unrelated parameters and is unsuitable here.
2. **Saved Character route promotion:** `components/Option/Playground/Playground.tsx:206–250,1120–1151` updates the relevant parameter set and calls `navigate({pathname,search,hash}, {replace:true})`, preserving the other route portions and treating real embedded-hash routes deliberately. Existing coordinator/session tests cover incoming routes, remount and ownership. This is the preferred transport shape for a Flashcards tab replacement.
3. **Private Flashcards handoff consumer:** `components/Flashcards/hooks/useFlashcardsGenerateHandoff.ts` and `services/tldw/flashcards-generate-handoff.ts:191–210` consume under the existing authority, retain accepted source in memory, and replace the URL to remove private/token parameters. `FlashcardsManager.private-handoff.integration.test.tsx` already mounts the actual producer/router/consumer, verifies source-unmount delivery, token removal, edited text, StrictMode and owner changes. Extend this boundary for126.
4. **Quiz dirty navigation comparison:** `components/Quiz/QuizPlayground.tsx:263–286` confirms before setting its accepted active tab. It does not itself keep tabs in the URL, so copy only the guard ordering, not its route behavior.

## Recommended minimal contract

- After an **accepted user tab selection**, replace the current router entry with a canonical `tab` value using the existing `useNavigate` from `react-router-dom`. Clone the current query and preserve pathname, unrelated parameters and fragment. Use the existing internal keys `review`, `cards`, `importExport`, `templates`, `scheduler`; continue accepting present aliases on entry. Writing explicit `tab=review` is necessary: deleting it would permit a still-retained Generate/Study Pack intent to select Import / Export again.
- Keep the existing Scheduler confirmation first. Cancel changes neither active tab, route/history, nor dirty draft. Accept uses the existing discard signal, then updates visible tab and canonical route. Do not navigate before confirmation or add a second confirmation.
- Use **replace**, not push: tab switches currently add no browser-history entries, and private handoff cleanup already replaces. Back should still return to the preceding workspace/source rather than step through every tab or resurrect a consumed token.
- Keep the existing mounted Generate owner, scope and `generationKey`. Tab routing must not remount the manager/Generate subtree, serialize draft text or provenance, republish a token, replay consumption, or introduce persistence of private drafts. Switching away/back should retain the current in-memory edits as it does now. Reload only promises the selected tab; durable unsaved Generate-draft recovery is outside126.
- Preserve `quiz_id`, `attempt_id`, `deck_id`, `study_source`, `include_workspace_items` and any unrelated benign parameters. A tab-only replacement must not reinterpret a consumed entry handoff as a fresh deck selection. Keep explicit incoming deck/quiz links effective.
- Keep routing inside the installed router. The WebUI shim supports object navigation through Next `replace`; shared options use a real HashRouter, and the sidepanel may use hash-aware MemoryRouter. Do not use direct `window.history.replaceState`, change the shim globally, write to `window.location`, or hardcode the browser `/flashcards` URL from inside an extension route.

## Traps requiring permanent controls

**A URL write is not safely a one-line change by itself.** The existing combined route effect at190–213 depends on `currentTab` and calls `applyReviewDeckChange(currentStudyIntent.deckId)`, which clears all one-shot deck handoffs. If a user entered deck9, chose/cleared another live deck, then changed tabs, new route updates would reapply deck9. It can also clear a fresh Transfer task/source-review handoff. Separate incoming non-tab intent application from tab-only navigation, or equivalently use a narrow semantic intent comparison. Avoid comparing the entire location object or newly parsed Study Pack object, both of which change when only `tab` changes. Do not add a general route-state controller. The regression must establish that a tab-only echo preserves live state while a genuinely new external deck/quiz intent still applies.

`useFlashcardsGenerateHandoff`'s consume effect depends on `route.cleanRoute`. A tab update while a token is still being resolved/consumed changes that dependency. Its abort/remove awaits make this an important race control: no lost accepted source, duplicate consumption, stale redirect back to Import / Export, or private token resurrection. **This is a static risk, not a confirmed current126 failure.** Begin with the existing hook unchanged; if the new router-boundary RED exposes it, allow only a narrowly coordinated hook correction using its existing latest-route/lifetime ownership. No handoff storage/auth redesign.

Several internal CTAs call `setActiveTab` directly (review/create/manage/scheduler/import/generate/export/source-review). Scope should be explicit before implementation: the confirmed defect and task AC refer to top-level tab selection. The smallest initial repair changes that accepted-tab owner plus route-intent replay handling. If canonical URL behavior is also required for those CTAs, route their existing accepted transitions through the same tiny tab commit function; do not independently add URL effects to each child. Preserve their deck/task/provenance payloads. Their route echoes must not clear their just-installed handoffs.

The real HashRouter exposes its routed query as `location.search`; the current parser's unused `hash` type does not itself prove an extension defect. Test the actual configured router before adding legacy embedded-hash parsing. Hash-aware MemoryRouter seeds from the browser hash once and does not make internal navigation durable in browser history. Preserve that platform contract; do not claim popup/sidebar destruction-and-reopen persistence without native evidence.

## Proposed ownership and tests

**Expected production owner:** `apps/packages/ui/src/components/Flashcards/FlashcardsManager.tsx` only. A small local route builder/semantic-intent helper may remain in that file. No backend, review/session storage, auth, private producer, or shared router changes. `useFlashcardsGenerateHandoff.ts` is conditional scope only if the pending-token test proves the new transition would violate its existing contract; coordinate before broadening.

**Permanent RED first:** extend `apps/packages/ui/src/components/Flashcards/__tests__/FlashcardsManager.private-handoff.integration.test.tsx` using its actual transfer creator/storage, actual manager/consumer/Generate panel and actual MemoryRouter. Complete consumption, edit private source, click Study, assert `tab=review`, then remount from the returned route and assert Study. Before the fix the route remains Import / Export. Also switch back before remount and assert the exact edited draft/provenance remains, with no private text/title/token in the URL. Do not replace the consumer/route writer with a mock.

**Focused real-router controls** (new `FlashcardsManager.navigation.test.tsx` is reasonable if needed to keep the existing broad mock suite stable):

1. All five canonical tab selections restore on remount; current accepted aliases/invalid-default entry remain compatible.
2. Initial route deck9/quiz/attempt/workspace visibility survives query update; selecting deck12 or clearing the live deck then changing tabs does not resurrect9. A later external deck21 link applies normally. Verify the Quiz CTA context too.
3. Dirty Scheduler Cancel preserves exact URL, selection and draft; Accept changes URL once and discards once. Normal same-tab click does not create extra history or reset a handoff. Keep the zero-deck preview behavior.
4. Private handoff pending resolution/consumption plus tab click, completed handoff plus later edit, StrictMode, and existing account A→B→A tests. No second consumer/navigation overwrites the selected tab; no new private persistence.
5. Actual replace semantics: start with preceding source entry, perform multiple tab switches, then Back returns to that source once. Preserve a normal fragment and unknown harmless query field.
6. Actual shared HashRouter and hash-aware MemoryRouter route shapes; exercise the WebUI Next shim separately using the existing navigation test harness if shared-only tests cannot establish its behavior. Do not certify WebUI from a mocked `navigate` assertion alone.

`FlashcardsManager.consistency.test.tsx` is valuable for existing deck/task/dirty-state controls, but its `useNavigate` is a spy and `useLocation` reads manually replaced window state. It cannot prove the new URL→rerender→reload boundary. Preserve it and supplement it with actual router tests. Existing `ImportExportTab`/Generate/private-handoff, quiz-flashcards handoff, and shared router guard suites are the appropriate nearby regression set; no whole-suite expansion unless a new failure justifies it.

## Verification sequence after the freeze is released

1. Official task66 In Progress and approved bounded design; retain the real-boundary RED above.
2. Minimal owner change, then GREEN primary and negative controls. If a newly exposed route/handoff race cannot be fixed within three attempts, record evidence and reassess scope before further edits.
3. Scoped ESLint with the repository's frontend config and exact baseline comparison; compare any requested combined typecheck against the existing90-diagnostic baseline. Bandit does not apply to a TS/TSX-only patch; record that rather than claiming a Python scan.
4. Freeze exact owned hashes and independent review. Root owns native recheck: real Note→Generate, Study selection, URL capture, normal reload, retained completed review/session and cards; Scheduler Cancel/Accept and a benign deck/quiz link negative. Do not generate or mutate new model data merely to test tab routing.

No implementation has begun. The current full native matrices and their evidence remain authoritative and unchanged.
