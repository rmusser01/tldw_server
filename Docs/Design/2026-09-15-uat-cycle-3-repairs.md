# UAT cycle 3 repair design

Tracking: repair children under TASK-13260. Evidence: [running tracker](../Reviews/FRESH_INSTALL_SINGLE_MULTI_UAT_TRACKER_2026_09_14.md) and [single-user captures](../../output/playwright/cycle3-full-uat-2026-09-15/single/README.md). Product tested: `d40e17dc81`.

This design covers UAT056–077 and reopened UAT055 Manage scope, including evidence-review discovery076 and fresh multi-user discovery077. Fresh multi-user UAT is still running. Finish that frozen matrix and reconcile its findings before changing application code or tests. The user has authorized the continuing UAT → review → fix loop; these are repairs to existing behavior within that scope.

## Shared contracts

- Preserve server ownership, roles, source classification, remote access restrictions and existing source-grounding verification.
- Keep real local inference on the configured llama.cpp endpoint. A fake response or seeded artifact cannot establish live acceptance.
- Correct the source of invalid state. Do not hide errors, disable tests, suppress all console output, delete user history, or classify a workaround as a passing original flow.
- Use existing React state, settings, database and rendering abstractions. Add no dependency for these repairs.
- Protect pending work at server/account boundaries. Checking a character or conversation ID without its verified account is insufficient.
- Test the observed interaction across participating components. A test that mocks away the competing writer or hydration effect cannot prove the race fixed.
- Every repair needs a failing behavior regression, focused verification, independent review and targeted live evidence. The next full UAT uses new configuration, databases and browser profiles only after all repairs are ready.

## 1. Private Notes generation transfer — TASK-13260.13 / UAT064

The Notes action passes text and provenance into `buildFlashcardsGenerateRoute`, which serializes them into a URL. A source-ID-only fetch would avoid plaintext URLs but lose the current unsaved editor text. Keep that text in a short-lived, consume-once transfer and navigate using an opaque token.

Follow the existing `source-review-handoff.ts` storage/expiry pattern, adding a verified server/account binding. Capture the current authority before storing or consuming; unavailable authority must not be guessed. Remove consumed/expired records and clear these transfers at logout/server/account changes. Storage failure must show an actionable failure without falling back to a text-bearing URL. Keep the current 12,000-character bound explicit.

Once the Flashcards page has consumed the transfer, its editor and generated drafts remain private. Clear them and invalidate pending generation/save completions when the authority changes, including A→B→A. Bind generation and persistence to the captured authority; clearing only session storage is insufficient.

The shared route builder also has extension/new-tab callers. Do not replace their transport with same-tab session storage without testing the actual platform transition. Scope the Notes producer first; preserve existing callers explicitly and record any remaining text-bearing route surface rather than claiming a global guarantee.

Acceptance includes exact unsaved source text, origin IDs and real generation in the intended account; no source text/title/IDs in its navigation URL; expiry, consumed token, unavailable storage and delayed account-switch controls.

## 2. One owner for a saved Standard Chat — TASK-13260.14 / UAT062

The normal pipeline publishes a completed history before its awaited local persistence finishes. Autosave observes populated history without a server ID and creates a second server conversation. The completion's implicit default character also changes a Standard workspace into Character mode.

Extend `ensureWorkspaceServerChatForTurn` to create an owned neutral conversation for the first saved global Standard turn. Establish its server ID and validated linked local-history ID before invoking inference, and carry those IDs through completion and persistence. Preserve explicit persona/character and temporary paths. Autosave still supports genuine local-to-server promotion, but must recheck captured account/conversation state after asynchronous initialization before creating anything.

Exercise the real normal pipeline and autosave together against shared reactive state. Cover no queued turn, a queued second turn, delayed history linking, temporary-to-saved promotion, creation failure/abort and A→B during each awaited boundary. Assert one canonical conversation and one copy of each actual turn, as well as visible mode and request identity.

## 3. Selection and persisted Chat reconciliation — TASK-13260.15 / UAT067/068/070

`useSelectedAssistant` owns canonical selection and legacy migration. `useCharacterGreeting` must hydrate the current selection without independently replacing it from stale legacy settings. Remove that competing selection authority; invalidate pending server/profile loads after clear, replacement or account change. Opening Create must not write to an unmounted Edit form; retain correct edit initialization when Edit actually opens.

Persist `serverMessageId` in server-to-Dexie mirrors. On loading a pre-fix mirror, an exact message ID present in the fetched, owned conversation identifies a confirmed server row even if its cached `serverMessageId` is absent. Reconcile missing server messages into the existing mirror; preserve genuine unsynced rows, active streaming content and newer local results. An incomplete mirror must not permanently suppress a fetched final answer.

Use one validated linked local history. Dexie message IDs are global primary keys; writing the same IDs into another history can move rows away from the prior mirror. Test concurrent bootstrap/loader linking and reuse the existing `useServerChatHistoryId` boundary. Do not delete or merge the five historical empty mirrors wholesale as cleanup.

Recommended implementation order: canonical selection, first-saved neutral bootstrap, then mirror reconciliation. Cover actual picker interaction, deferred legacy/profile reads, the observed two-row old mirror versus three server rows, repeated reload, unsynced draft plus fetched reply, older snapshots during streaming, and identical IDs in different accounts.

## 4. Truthful ingestion — TASK-13260.16 / UAT056/057/061/065

Validate required analysis provider configuration before advancing from Configure to a Ready review. Keep final submission validation for stale state, with the same actionable field focus. Analysis-disabled presets remain valid without a model.

Byte-based estimates cannot predict model inference time. Preserve defensible estimates for measured work, but label unmeasured model duration honestly. Remove the timer that invents increasing percentages and Analyze/Store stages. Display confirmed server progress when available; otherwise show indeterminate activity and elapsed time. Terminal success remains tied to actual persisted results.

Retain safe extraction failure categories through the backend service and frontend result classifier. Distinguish policy/access denial, empty extraction, timeout and unknown failure; do not mark an unknown failure retryable by default. Never render raw upstream HTML or secrets as guidance. Keep denied article persistence blocked. The exact Wikipedia fixture may remain externally blocked after correct reporting; that outcome is not a green article workflow.

## 5. Media presentation and recovery — TASK-13260.17 / UAT060/066/071/072/073

Render analysis through the existing safe Markdown component while preserving its original text for copy/edit/export. Cover headings, emphasis, lists, plain text and hostile links/HTML. The source card and preview must use the same fallback type when the response omits an explicit type, while preserving explicit source types.

Keep the existing Trash navigation visible in the empty-library return path. Test deleting the sole item, refreshed empty results, cleared selected URL, visible Trash navigation and restore. Preserve ordinary-user delete denial.

Use the supported notification `actions` property and the existing context-backed message hook in deletion. Verify a real AntD App context, Undo exactly once, dismissal and failed restore instead of directly invoking a mocked deprecated `btn` property.

Reading progress uses percentage zoom units: send100 for the unzoomed viewer, including its signature fallback. Keep API validation25–400. Selection cleanup is expected to flush the old item's progress; test the old/new identities and restored position rather than treating any old-item request as wrong routing.

## 6. Notes status and usable results — TASK-13260.18 / UAT063/069

A successful server-detail load with saved version/time should announce saved state. Do not mark a new, dirty, offline-only or stale-account draft saved. The current inconsistency is in an accessibility live region; the visible footer already has the correct server metadata.

Expanded Views, Filters and Recent Notes must not consume all sidebar height. Bound/scroll the controls and reserve usable results space within the existing layout. Merely adding `min-h-0` or defaulting the controls closed does not prove the expanded state usable. Test1280×720 with at least five notes and three recent entries, actual pointer and keyboard access, resize, reload, offline state and mobile layout.

## 7. Study eligibility and supported lists — TASK-13260.19 / UAT055/074

Backend analytics categories overlap: `due` includes expired learning/relearning/review cards; `learning` includes future learning too. Dashboard readiness is `due + new`. Keep Learning as a descriptive count and preserve the actual queue's existing disjoint filters. Test future learning, expired learning, mixed states and a controlled due-time transition; capping the incorrect sum at total is insufficient.

Replace both active and pending-deletion AntD Lists in Manage with native supported markup. Preserve loading and empty states, compact/expanded rows, action menus, selection, keyboard focus, pagination, pending deletion and Undo. Existing RecentStudySessions is an adjacent working pattern, not proof that Manage is repaired.

## 8. Route and setup guidance — TASK-13260.20 / UAT058/059/075

Use the existing Next page-title pattern for fresh Home/setup, Media, Knowledge, Flashcards and tested Settings wrappers. Titles must follow the route and signed-out boundary without restoring old Chat/account metadata. Keep browser-dependent shared routes SSR-disabled.

Reading Queue should use the same prerequisite ordering as other personalized cards: unavailable capability, disabled profile, actual degraded fetch, then empty/success. Do not add requests merely to make an unconfigured feature look tested.

The server guide action must open maintained server setup documentation. The [server self-hosting profile index](https://github.com/rmusser01/tldw_server/blob/main/Docs/Getting_Started/README.md) resolves and describes the single/multi/local profile choices. Use that target and account for both the `serverOverview.docsUrl` override and onboarding fallback translations. Keep legitimate browser-extension links elsewhere unchanged. In-app keyboard Help already loads successfully and is not part of this destination repair.

## 9. Study run identity — TASK-13260.21 / UAT076

The review request has no run scope/session field. The endpoint chooses a session from each card's deck, and ReviewTab replaces its active ID with each response. Thus one all-decks run splits into multiple sessions and ends only the last. Existing database validation also requires exact session/card deck equality, so a global session cannot currently include deck-owned cards.

Add optional explicit review context and an acknowledged session ID to the request. Distinguish an explicit all-decks choice from omitted legacy context and derive the canonical scope key server-side. Resolve the first rating's session from that context, then retain/send its acknowledged ID throughout the run. Permit mixed decks only in a validated global session; keep deck-specific matching and unknown/inactive/context-mismatch rejection before any scheduling/history mutation. The existing session/review linkage supports this without a table migration.

Legacy context-free requests retain current per-card behavior and30-minute reuse/abandonment rules. Preserve practice-only Cram's no-scheduling-write path. Bind delayed review/end acknowledgements to their originating account and UI scope. Complete exactly the intended session on confirmed exhaustion or intentional scope change, never on a transient queue loading gap. Do not alter historical split sessions or close unrelated active sessions.

Backend controls: two undecked plus five deck-owned ratings in one explicit global run yield seven linked reviews and a completed seven-card rollup; unrelated deck sessions remain active; foreign/unknown/completed/mismatched sessions and out-of-deck cards fail without side effects. Frontend controls: stable ID across mixed cards, one completion, delayed rating/scope/account changes and transient loading.

## 10. Provider Keys loading failure — TASK-13260.22 / UAT077

The loading label must use the scalar `common:loading.title` translation. Reproduce the route with the actual i18n adapter and locale resources, because a mocked translator returning its input masks the failure. Verify pending, empty success and403/unavailable outcomes without a route crash. The actual key-list403 confirms BYOK is disabled, so preserve that accurate deployment guidance; a generic403 must not automatically imply the same reason.

Six static related label callers exist in PersonaGarden Scopes/Policies/Commands/Connections. Correct those scalar keys with pending/action translation coverage without representing them as live-confirmed UAT failures. For the shared ICU compatibility guard, replace placeholders only when the resource is a string, then delegate all resources unchanged otherwise to `super.parse`. Installed upstream accepts syntax-tree arrays and configurable parse-error handlers. Do not stringify objects, return non-strings early or globally change object-handling options. Test actual syntax-tree formatting and object error-handler delegation as well as existing interpolation/plural behavior. The guard alone cannot make React render an object-valued label.

## Design review

Independent read-only review found no material omission in the private-transfer or Chat ownership/selection/mirror contracts. The existing loader/session Vitest fixtures mock persistence, so their green results alone cannot establish the real Dexie round trip; use the existing browser harness for that regression and settled reload acceptance. Multi-user findings remain subject to reconciliation before implementation.

## Completion boundary

After both frozen matrices finish, incorporate multi-user-only findings into these units or separate reviewable tasks. Review and verify each repair, then the combined touched scope, including Bandit for touched Python code and existing TypeScript/lint baseline comparisons where applicable. Run fresh single/multi UAT again with real inference and an explicit result for every workflow. A completed test execution with findings or an externally blocked required journey cannot satisfy the issue-free active goal.
