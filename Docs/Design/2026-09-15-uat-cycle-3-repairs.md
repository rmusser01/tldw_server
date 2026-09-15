# UAT cycle 3 repair design

Tracking: repair children under TASK-13260. Evidence: [running tracker](../Reviews/FRESH_INSTALL_SINGLE_MULTI_UAT_TRACKER_2026_09_14.md) and [single-user captures](../../output/playwright/cycle3-full-uat-2026-09-15/single/README.md). Product tested: `d40e17dc81`.

Environment-only exception during the frozen multi-user run: repeated disk exhaustion required disabling Next's development filesystem cache when the existing `TLDW_NEXT_DIST_DIR` selects an isolated UAT build. The four-line configuration change preserves default development behavior and all application logic. Both browser profiles, API and databases survive the frontend restart. Actual normalized configuration comparison and independent review passed; record the transition in the evidence rather than claiming identical build configuration throughout.

This design covers UAT056–079 and reopened UAT055 Manage scope, including evidence-review discovery076 and fresh multi-user discoveries077–079. Fresh multi-user UAT is still running. Finish that frozen matrix and reconcile its findings before changing application code or tests. The user has authorized the continuing UAT → review → fix loop; these are repairs to existing behavior within that scope.

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

Follow the existing `source-review-handoff.ts` storage/expiry pattern, adding a verified server/account binding. Capture authority and its generation at the initiating action, before asynchronous selection or media loading; revalidate after acquisition, storage and before navigation. Unavailable authority must not be guessed. Remove consumed/expired records and clear these transfers at logout/server/account changes. Storage failure must show an actionable failure without falling back to a text-bearing URL. Use trimming only to detect empty content; preserve original whitespace within the12,000-character bound and visibly identify any truncated prefix. Do not claim exact transfer of a longer source.

Once the Flashcards page has consumed the transfer, its editor and generated drafts remain private. Clear them and invalidate pending generation/save completions when the authority changes, including A→B→A. Bind generation and persistence to the captured authority; clearing only session storage is insufficient.

Repair all five known producers of this same text-bearing route: Notes, Media Review, sidepanel Chat, sidepanel Flashcards selection and Quiz's generated-material fallback. Notes is the live-confirmed UAT finding; the other four are statically confirmed callers of the same serializer. Preserve their source text/provenance and intended same-tab or new-tab transition. Await successful storage before navigation and retain editable source content on failure; do not create a transfer prematurely when Quiz only needs a fallback after another action fails.

Use a transport shared by the actual source and target contexts. `sidepanel-chat-handoff.ts` already demonstrates opaque routes with the existing local storage abstraction, bounded payloads, expiry and write verification. Reuse those conventions without copying its unrelated package schema or its missing account binding. Same-tab session storage cannot establish extension-to-options/new-tab delivery. The WebUI Plasmo shim can silently fall back to memory when browser storage is unavailable, so same-instance readback alone cannot prove durable cross-document delivery. Cover the actual storage/backend boundary and fail the action visibly when its required transport is unavailable; do not globally change unrelated storage fallback behavior.

Verify navigation success at the actual platform boundary. The WebUI `tabs.create` shim resolves even when `window.open` is blocked, and awaiting selection/storage can lose popup activation. Use a task-local opener that distinguishes a real extension runtime from the WebUI shim; reserve a blank WebUI target during the initiating click when needed, navigate only after a valid transfer exists, and close the blank target/remove abandoned transfers on failure. Keep the source draft until opening is confirmed. A new tab may lack session-only authentication; leave the token unconsumed until that target verifies authority, without putting credentials into the payload.

Consume-once requires a supported atomic claim across target contexts; separate asynchronous read/remove calls do not establish it. Verify two target contexts, StrictMode replay, delayed storage and authority changes between read/claim/apply. URL cleanup must neither erase the imported text nor overwrite subsequent edits. Reject and remove legacy plaintext generation/provenance query/hash parameters with recoverable guidance to reopen from the source; importing them into whichever account opens an old URL would retain an unbound path. Plain text-free Transfer navigation remains valid.

Acceptance includes exact unsaved source text, origin IDs and real generation in the intended account; no source text/title/IDs in its navigation URL; expiry, consumed token, unavailable storage and delayed account-switch controls.

## 2. One owner for a saved Standard Chat — TASK-13260.14 / UAT062

The normal pipeline publishes a completed history before its awaited local persistence finishes. Autosave observes populated history without a server ID and creates a second server conversation. The completion's implicit default character also changes a Standard workspace into Character mode.

Extend `ensureWorkspaceServerChatForTurn` to create an owned neutral conversation for the first saved global Standard turn. Establish its server ID and validated linked local-history ID before invoking inference, and carry those IDs through completion and persistence. Preserve explicit persona/character and temporary paths. Autosave still supports genuine local-to-server promotion, but must recheck captured account/conversation state after asynchronous initialization before creating anything.

Exercise the real normal pipeline and autosave together against shared reactive state. Cover no queued turn, a queued second turn, delayed history linking, temporary-to-saved promotion, creation failure/abort and A→B during each awaited boundary. Assert one canonical conversation and one copy of each actual turn, as well as visible mode and request identity.

The multi-user Media→Chat sequence additionally reaches a participant-mismatch persistence400, then a successful fallback201, while a blocking development overlay appears. The source already catches and logs the initial error before fallback; the snapshot alone cannot distinguish a separately unhandled promise from development error interception. Verify the actual persistence/fallback boundary and browser error events, correct participant identity and handle asynchronous failure. A successful fallback must leave a usable UI; failed persistence must retain a truthful recoverable state. Do not merely suppress console errors or relabel a malformed request as expected.

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

## 6. Notes status and usable results — TASK-13260.18 / UAT063/069/079

A successful server-detail load with saved version/time should announce saved state. Do not mark a new, dirty, offline-only or stale-account draft saved. The current inconsistency is in an accessibility live region; the visible footer already has the correct server metadata.

Expanded Views, Filters and Recent Notes must not consume all sidebar height. Bound/scroll the controls and reserve usable results space within the existing layout. Merely adding `min-h-0` or defaulting the controls closed does not prove the expanded state usable. Test1280×720 with at least five notes and three recent entries, actual pointer and keyboard access, resize, reload, offline state and mobile layout.

Ordinary successful saves must not eagerly read `monitoring/alerts`, whose router requires `system.logs`. Preserve optional feedback for users who actually have that permission, including custom roles. Use TASK-13260.23's narrow current-user capabilities; deployment/OpenAPI flags and an `admin` role-name check are not authoritative entitlement checks. Capture the save's authority generation before its first await and carry it into monitoring, using the existing guarded Notes transport and the verified owner's `user_id` filter. Check the originating account and selected Note before dispatch and before publishing any alert. Permission lookup failure must not turn a successful Note save into an error or reveal a prior account's notice. Test ordinary denial, authorized feedback, unavailable lookup and delayed save/alert A→B→A or note replacement.

## 7. Study eligibility and supported lists — TASK-13260.19 / UAT055/074

Backend analytics categories overlap: `due` includes expired learning/relearning/review cards; `learning` includes future learning too. Dashboard readiness is `due + new`. Keep Learning as a descriptive count and preserve the actual queue's existing disjoint filters. Test future learning, expired learning, mixed states and a controlled due-time transition; capping the incorrect sum at total is insufficient.

Replace both active and pending-deletion AntD Lists in Manage with native supported markup. Preserve loading and empty states, compact/expanded rows, action menus, selection, keyboard focus, pagination, pending deletion and Undo. Existing RecentStudySessions is an adjacent working pattern, not proof that Manage is repaired.

## 8. Route and setup guidance — TASK-13260.20 / UAT058/059/075/078

Use the existing Next page-title pattern for fresh Home/setup, Media, Knowledge, Flashcards and tested Settings wrappers. Titles must follow the route and signed-out boundary without restoring old Chat/account metadata. Keep browser-dependent shared routes SSR-disabled.

Reading Queue should use the same prerequisite ordering as other personalized cards: unavailable capability, disabled profile, actual degraded fetch, then empty/success. Do not add requests merely to make an unconfigured feature look tested.

Automation Inbox must distinguish `tasks.read` denial from a temporary service failure. Gate reads on authoritative account capabilities when available, and retain accurate denied/unsupported/error states when older servers cannot supply those capabilities. Keep independently available notifications or results visible rather than discarding every source because one is denied. Clear previous-account items and ignore delayed results after disable/account transitions. Do not infer user entitlement from the current OpenAPI-derived capability flags. Test all-denied, mixed-source success/denial, transient failure, refresh after a permission change and A→B with a pending request.

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

Follow-up review of the expanded five-producer transfer found popup-success, early authority capture, simultaneous consumption, whitespace/length and legacy-reader gaps. The section1 contracts now include those controls. Static related surfaces are distinguished from live UAT failures.

## 11. Caller capability discovery — TASK-13260.23 / prerequisite for078/079

Add only `can_read_scheduled_tasks`, `can_read_notifications` and `can_read_monitoring_alerts` to an authenticated `/users/me/capabilities` endpoint using `get_auth_principal`. Evaluate each through the exact `RequirePermission` guard used by its protected endpoint, following `media/capabilities.py`; return no arbitrary-user lookup, full permission catalog or protected content. Mark responses `Cache-Control: no-store`. Preserve unexpected guard failures rather than relabeling them as denied.

The existing self-profile requires an active verified stored user, while Home/Notes permission eligibility does not imply that profile contract. Bob's profile200 is confirmed, but another retained profile403 has no captured body. A separate narrow capability endpoint preserves profile verification and avoids extending deprecated, optionally410 `/auth/me`. Do not assign the uncaptured403 a reason or equate a discovery403 with three known denied permissions.

Keep frontend permission discovery separate from deployment capabilities. Bind results to verified server/account/org plus authority generation; immediately mask them on disconnect/account change and reject delayed A→B→A results. Distinguish allowed, denied, unknown and unsupported. Refresh on reconnect, explicit refresh and a protected request's403; actual endpoint authorization remains decisive. Do not persist these decisions with preferences or reuse them indefinitely from login state. Home and Notes consume this shared bounded contract in their own repair units.

## Completion boundary

After both frozen matrices finish, incorporate multi-user-only findings into these units or separate reviewable tasks. Review and verify each repair, then the combined touched scope, including Bandit for touched Python code and existing TypeScript/lint baseline comparisons where applicable. Run fresh single/multi UAT again with real inference and an explicit result for every workflow. A completed test execution with findings or an externally blocked required journey cannot satisfy the issue-free active goal.
