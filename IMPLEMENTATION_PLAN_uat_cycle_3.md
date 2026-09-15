# UAT cycle 3 implementation plan

> For agentic workers: execute bounded tasks with systematic-debugging and test-driven-development, use dispatching-parallel-agents only for independent file ownership, and obtain independent review before integration.

**Goal:** Repair all findings from the frozen cycle3 fresh single/multi UAT, then repeat both full workflow matrices.
**Architecture:** Preserve the current shared UI/API/store boundaries. Establish ownership before asynchronous work, reconcile saved state without dropping drafts, and report confirmed processing/status results.
**Tech stack:** Next.js, shared React/Zustand/Dexie, Vitest/Playwright, FastAPI/pytest/SQLite and real llama.cpp.
**Spec:** [Cycle3 repair design](Docs/Design/2026-09-15-uat-cycle-3-repairs.md).

## Global constraints

- Application behavior remains frozen at `d40e17dc81` until both current matrices have actual outcomes or explicit blocks. No application/test edits during that run. One documented environment exception disables Next development filesystem caching only for existing isolated UAT builds; retain before/after configuration evidence and runtime restart provenance.
- Existing TASK-13260 covers UAT/evidence; its repair children cover the units below. Update their notes, verification and status through Backlog MCP or CLI.
- Preserve role/ownership/source classification and remote restrictions. No mocked response, skipped step or seeded artifact is a live acceptance pass.
- Reuse current dependencies and test harnesses. Use the project virtual environment before Python/Bandit. Never expose runtime credentials.
- Commit reviewed, passing units with their Backlog records. Do not stage unrelated work or delete another plan/cache/profile.

## Stage 1: Finish and review frozen UAT
**Goal:** Complete both workflow matrices and retain auditable evidence.
**Success criteria:** Every named journey/shared workflow has a result; all findings are tracked with expected/actual behavior and evidence limits.
**Tests:** Actual UI workflows, real provider requests, independent owned/foreign API controls, evidence hash/JSON/credential checks.
**Status:** Complete

- [x] Complete single-user execution and preserve144 captures through09:26:30UTC; initially record20new findings and reopened055. Independent evidence review additionally confirms076session accounting; retain its later read separately.
- [x] Start fresh multi-user API18201/WebUI18281/browser; retain the existing documented admin bootstrap.
- [x] Complete admin/Alice/Bob UI workflows, public/confidential QA controls, cross-account metadata/content/draft isolation and offline two-tab recovery. Browser cutoff2026-09-15T11:14:21Z; Wikipedia and deliberate Cedar card/study remain blocked. Multi mixed deck plus undecked076 is explicitly unexecuted, not a pass, and remains required for repair verification/full rerun.
- [x] Reconcile the combined tracker and retained multi evidence, independently review claims, and retain the frozen-run checkpoint. Parent and independent reviewer verified309 main captures/report hashes and41JSON; all4PNGs inspected. Parent scanned552 retained files against14known runtime credentials plus JWT/private-key patterns; zero matches. Review found no material claim/evidence contradiction.
- [x] Incorporate multi-user-only findings into the repair tasks before beginning Stage2. Open total32:14P2/18P3, including confirmed cross-account Note plaintext064 and QA metadata086.

## Stage 2: Repair private transfers and Chat ownership
**Goal:** Protect Notes generation content and make Chat selection/history reliable.
**Success criteria:** No note plaintext URL; one canonical saved Chat; explicit character selection and all persisted replies survive reload without losing drafts or crossing accounts.
**Tests:** Real interacting hooks/stores, deferred async boundaries and old Dexie cache round trips.
**Status:** In Progress

Repair release: reviewed frozen evidence committed `4815d44d4a`. Authentication backend/frontend and Chat selection have independent owners; ingestion Stage3 begins concurrently with separate file ownership. All findings remain open until targeted live verification.

### TASK-13260.24 — UAT080 session refresh

**Files:** AuthNZ refresh endpoint/dependencies/session manager/repository as required by the diagnosis; frontend refresh/proxy/connection and session-query lifecycle; existing real SQLite AuthNZ and authority regressions.
- [ ] Reproduce the real request's outer `BEGIN IMMEDIATE` blocking the session service's second transaction; retain the observed wrong401 before implementation.
- [x] Use the existing non-locking request connection pattern while retaining the inner atomic refresh/CAS transaction, rotation and replay/revocation protections. Reviewed backend checkpoint `3751292380`.
- [x] Distinguish retryable service failures from invalid sessions and verify rollback/repeated/concurrent refresh behavior with actual SQLite connections. Backend regression56passed/3fixture skips; additional transaction controls89passed; independent parent40passed/3fixture skips. No new Bandit findings.
- [ ] Verify retryable failures retain credentials; invalid sessions clear only their authority and stop cross-tab private queries; refresh cancellation cannot open an unrelated connection modal or discard saved/scoped data.
- [ ] Independently review, run focused backend/frontend checks and scoped Bandit, then perform targeted real expiry/recovery before the next full UAT.

### TASK-13260.13 — UAT064

**Files:** `apps/packages/ui/src/services/tldw/flashcards-generate-handoff.ts`, its five Notes/Media/sidepanel/Quiz producers, Flashcards page/GeneratePanel consumers, existing authority/logout cleanup; adjacent handoff and Notes/Flashcards/platform integration suites.

- [x] Add regressions for exact unsaved source transfer, URL privacy, expiry/consume/storage failure and A→B→A with delayed generation/save; observe the original plaintext-route failure.
- [x] Cover authority changes during source acquisition, blocked/new-tab opening, unresolved target authentication, simultaneous consumption and StrictMode replay, whitespace/length bounds, legacy plaintext rejection and URL cleanup without overwriting subsequent edits.
- [x] Implement the account-bound opaque transfer for all five known producers and consumer invalidation from the design. Preserve actual same-tab and extension/new-tab delivery; retain source drafts on storage/navigation failure.
- [ ] Run the existing handoff suite and new interacting consumer/authority/platform tests, including unavailable shared storage despite a memory fallback; review browser URLs, provenance and actual intended-account generation in live targeted UAT.
- Reviewed implementation checkpoint `aae6b72d05`: parent63 WebUI and12 shared storage controls pass after correcting a WebUI test spy leak; broader172 controls and independent review clear. Actual Bob Note handoff preserves exact text/provenance in a clean URL, and real generation/save creates two owned cards. Other producers and cross-account native acceptance remain pending.

### TASK-13260.15 — UAT067/068 selection portion, then TASK-13260.14 — UAT062

**Files:** `useCharacterGreeting.ts`, `useSelectedAssistant.ts`, `useCharacterData.tsx`, `hooks/chat/useChatActions.ts`, `hooks/chat/useServerChatLoader.ts`, `components/Option/Playground/hooks/usePlaygroundPersistence.tsx`; adjacent assistant, greeting, persona-integration and persistence tests.

- [x] Build the actual picker/canonical hook/greeting/store regression with deferred legacy/profile hydration; reproduce replacement reverting and unmounted Edit form warning.
- [x] Remove competing legacy selection hydration, guard cleared/replaced/account-changed loads and limit Edit form writes to its mounted lifecycle.
- [x] Add a combined normal pipeline/autosave regression with no queue, a queued second turn and delayed linked history; reproduce the duplicate conversation and mode change.
- [x] Extend neutral saved-chat bootstrap, carry established IDs into inference/persistence, and recheck autosave ownership after awaited work.
- [x] Verify temporary promotion, explicit persona/character, failed/aborted creation and delayed A→B results. Review and commit selection and canonical-creation changes separately.
- [x] Retain incomplete promotion across same-owner disconnect/reconnect, record an acknowledged created conversation before pausing, and block inference until explicit recovery succeeds. Verify zero/one copied rows, ambiguous writes and multi-page read-back. Use a backwards-compatible raw-content listing option to compare exact stored messages without placeholder expansion; retain normal display defaults and authorization.
- Code checkpoint `68906b148b`: parent306 frontend/9 backend tests pass, independent recovery probes pass, unchanged lint/type baselines and clean Bandit. Actual saved-Chat browser acceptance remains pending.

### TASK-13260.15 — UAT070 mirror and UAT085 backlink portion

**Files:** `hooks/chat/useServerChatLoader.ts`, NotesManager linked-Chat action, existing server-history linking/Dexie and session persistence utilities; `hooks/__tests__/useServerChatLoader.test.ts`, `usePlaygroundSessionPersistence.test.tsx`, and an adjacent integrated round-trip suite. Coordinate NotesManager edits with .13/.25.

- [ ] Reproduce the exact pre-fix greeting/user mirror lacking serverMessageId with a fetched third reply, using real formatters/Dexie/session restoration.
- [ ] Preserve server IDs and reconcile the owned existing mirror while retaining genuine unsynced/streaming/newer local rows.
- [ ] Exercise concurrent bootstrap/loader mapping, account changes during transactions and repeated reload. Assert rows are not moved to a competing linked history. Use the existing browser harness for real Dexie; the named mocked Vitest fixtures alone do not prove this round trip.
- [ ] Reproduce Cedar Note backlink retaining Robot selection and missing saved-message actions; restore owned conversation/character/message identities together with dirty/account/stale-source guards.
- [ ] Independently review and perform the real tracked Chat→Note→backlink→settled reload sequence before committing.

## Stage 3: Repair ingest and Media feedback/recovery
**Goal:** Make progress, extraction errors, analysis and recovery controls accurate and usable.
**Success criteria:** Required provider validation precedes Ready; no invented progress/estimate; source/analysis presentation is consistent; last-item Trash and valid reading-progress persistence work.
**Tests:** Wizard transition/job-state integration, scraper error boundary, actual Media empty/deletion/Undo controls, rendering and progress identity tests.
**Status:** In Progress

### TASK-13260.16 — UAT056/057/061/065

**Files:** `components/Common/QuickIngestWizardModal.tsx`, `QuickIngest/{ReviewStep,WizardConfigureStep,WizardResultsStep,timeEstimation}`, extraction-error propagation in `enhanced_web_scraping_service.py`; existing wizard integration/time-estimation/results and backend scraping tests.

- [x] Reproduce blank analysis provider advancing to Ready; unknown inference cost; server20% versus synthetic UI progress; lost denial/empty/timeout error distinctions.
- [x] Validate Configure transitions, use confirmed or indeterminate progress and preserve safe structured failure categories.
- [x] Verify analysis-disabled presets, terminal partial/failure/success, cancellation and unavailable progress. Run touched backend tests/Bandit and frontend lint/regressions.
- [ ] Review and exercise real small-source analysis plus the exact Wikipedia attempt without bypassing its refusal.

### TASK-13260.26 — UAT082 private ingest state

**Files:** `store/quick-ingest-session.ts`, `store/quick-ingest.tsx`, Quick Ingest wizard/button, DocumentPicker, `services/tldw/quick-ingest-session-reattach.ts`, `quick-ingest-batch.ts`, auth cleanup and result actions; existing close/resume and authority regressions.
- [x] Reproduce completed Bob results visible after normal logout/admin login, plus delayed completion and persisted-state hydration after A→B→A.
- [x] Immediately mask/reset private state at verified authority changes and reject stale async work/actions without cancelling another account's server jobs.
- [x] Reject unowned persisted sessions/recent-document metadata; prevent late upserts recreating cleared state and old serial batch entries using new credentials.
- [x] Preserve same-account close/resume, reconnect and valid source actions; verify different servers and colliding numeric IDs.
- [ ] Independently review, run scoped frontend checks, and exercise the exact same-browser account transition before the next full UAT.

- Reviewed checkpoint `af1e7bb08b`: parent61/5 and implementer480/32 tests pass; independent re-review clears all three expiry/marker probes. Lint0 errors/878 unchanged warnings; compiler90 exact baseline. Native account-switch and resume acceptance remains pending. Existing MV3 foreground preference remains; worker-only controls do not certify popup destruction.

### TASK-13260.27 — UAT086 QA account history

**Files:** active KnowledgeQAProvider, history persistence/selection and existing account/transport helpers; Provider history/persistence/streaming and actual Recent UI tests. Coordinate source presentation with .17.
- [x] Reproduce Alice questions and cited-result metadata visible to Bob from global local storage; keep the correct foreign-conversation404 as a negative control.
- [x] Scope history, active results, sources and pending work to verified account/server authority; reject legacy unowned entries and delayed A→B→A writes/restores.
- [ ] Preserve same-account reload/server history and bounded storage behavior; verify actual Recent actions plus delayed stream/restore/delete/share/export boundaries.
- [ ] Independently review, run focused frontend checks, and verify exact same-browser isolation before the next full UAT.
- Reviewed checkpoint `d05c13ecc0`: parent123 WebUI controls, broader746/69, unchanged90 compiler baseline and unchanged7 existing lint errors/938 warnings. Original stale-expiry probe and independent lifetime/changed-principal controls pass. Native account-history checks remain.

### TASK-13260.17 — UAT060/066/071/072/073

**Files:** `components/Media/ContentViewer.tsx`, Knowledge source card/preview type normalization, `components/Review/ViewMediaPage.tsx`, `hooks/useUndoNotification.tsx`, `components/Media/hooks/useContentEditState.tsx`, `hooks/useMediaReadingProgress.ts`; associated behavior suites.

- [x] Reproduce Markdown presented literally, mismatched missing source type, inaccessible Trash after last deletion, actual AntD warnings and invalid zoom1 payload.
- [x] Reuse safe Markdown/type fallback, expose Trash in the empty return, use context notifications/actions and percentage zoom100.
- [x] Verify original copy/edit text, unsafe rendering controls, actual App-context Undo once/dismiss/failure, selection cleanup old/new IDs and restored reading position.
- [ ] Independently review and run targeted Media analysis/reload/delete/Trash/restore UI controls with API corroboration.
- Reviewed code checkpoint `9c21e08f0c`: independent89 and parent48 WebUI tests pass; broader shared110 pass. The related pre-existing empty-library stale-selection notice was reproduced on HEAD and repaired. Live acceptance remains.

## Stage 4: Repair Notes, Study and setup presentation
**Goal:** Keep core route status, navigation and study counts useful and truthful.
**Success criteria:** Saved Notes have consistent announcements and accessible results; eligibility counts match the queue; Manage emits no deprecated List warning; route/setup guidance is accurate.
**Tests:** Notes state/layout, real Manage controls, dashboard mixed-state clock fixtures, route/auth titles and prerequisite state branches.
**Status:** In Progress

### TASK-13260.23 — narrow caller capabilities for UAT078/079

**Files:** existing users endpoint/schema area, a small shared caller-capability service/hook, and adjacent AuthNZ/current-user/authority regressions. This is a prerequisite for the permission portions of TASK-13260.18 and .20.
- [ ] Reproduce missing authoritative discovery for the three optional reads; add caller-only booleans evaluated by canonical permission guards with `no-store` responses.
- [ ] Preserve existing profile verification and deprecated auth/me contracts. Verify ordinary/custom-grant/admin decisions, unauthenticated access and unexpected guard failures against protected endpoints.
- [ ] Add account/generation-scoped discovery that separates denied/unknown/unsupported, masks stale data, rejects A→B→A completions and refreshes on reconnect/explicit refresh/protected403.
- [ ] Independently review, run focused backend/frontend checks and scoped Bandit, then integrate the Home/Notes consumers in their owning units.

### TASK-13260.18 — UAT063/069/079

**Files:** Notes `hooks/useNotesEditorState.tsx`, `NotesSidebar.tsx`, `NotesListPanel` and related layout helpers/tests.
- [ ] Reproduce a loaded versioned Note announcing no save status and a zero-height list with expanded controls at1280×720.
- [ ] Correct successful authority-scoped hydration status and bound/scroll controls with reserved results space.
- [ ] Reproduce the successful ordinary Note save followed by `system.logs`403. Resolve a narrow authoritative entitlement contract, gate optional monitoring reads and discard delayed notices after account/note changes without weakening permissions or losing authorized feedback.
- [ ] Verify real pointer/keyboard use with five notes/three recents, resize/reload/mobile, plus new/dirty/offline/stale-account status protections; review and commit.
- Reviewed Notes checkpoint `af14ac1258`: actual canonical/owner/editor89 independent tests and65 parent WebUI controls pass. Same-authority continuity requires valid refresh source lineage; initial, failed or changed authority masks. Live geometry/save verification remains pending.

- [x] UAT091 live follow-up: let heading/status and desktop action groups wrap at constrained widths while preserving compact controls and mobile targets; verify actual1280×720/390px layout. Reviewed/live-verified checkpoint `062e8b7cb2` also passes1024px, parent14tests and independent7tests.

### TASK-13260.25 — UAT081 Flashcard source links

**Files:** `Flashcards/utils/source-reference.ts`, Notes route/editor hydration and canonical Media/Chat route helpers; source action and actual destination integration tests. Coordinate Notes edits with .13 and Chat changes with .14/.15.
- [x] Reproduce the actual Note source click leaving a blank editor while an owned source exists; compare the Media/message builder branches with their actual route consumers.
- [x] Use canonical destinations and owned loading, preserving dirty drafts and giving truthful missing/deleted/foreign source feedback.
- [ ] Test click-to-loaded-source and reload for all three types, target changes, account generations and unavailable message conversation identity; URL-only tests are insufficient.
- [ ] Independently review and confirm actual linked Note/Media/Chat content in targeted live checks.
- Reviewed checkpoint `16485e90d4`: confirmation-save review findings (newer typing loss and new-draft navigation cancellation) reproduced and corrected; original confirmation2/guard2 probes, parent47 Notes and39 consumer controls pass. Lint144 unchanged warnings; compiler90 baseline. Actual saved Biology card → Note and exact reload content pass in source-guide-round4 evidence; native Media/Chat and negative source checks remain.

### TASK-13260.19 — UAT055/074/083/084

**Files:** `components/Flashcards/components/DeckStudyDashboard.tsx`, `ReviewProgress.tsx`, `tabs/ManageTab.tsx`, `ReviewTab.tsx`, next-due query/locale copy and existing dashboard/ReviewTab/Manage tests.
- [x] Reproduce five expired learning cards displaying ten ready and actual Manage List deprecations.
- [x] Use `due + new` and native active/pending-deletion lists preserving all controls/states.
- [x] Reproduce shrinking due queue minus cumulative reviews reporting zero too soon; use consistent remaining semantics while preserving fixed Cram queues, refetch/failure behavior and newly due cards.
- [x] Label the next-due one-hour count accurately; verify staggered timestamps, boundary inclusion, capped uncertainty and actual translations.
- [ ] Verify future/mixed/due-time states, row selection/edit/keyboard/pagination/Undo and actual queue agreement; review and live-check Study/Manage.
- Reviewed code checkpoint `9d9f5222f5`:70 broad tests independently passed,22 final affected controls passed, lint44 unchanged warnings and TypeScript90 unchanged signatures. Live Study/Manage checks remain.

### TASK-13260.20 — UAT058/059/075/078

**Files:** tested Next page wrappers, `components/Option/CompanionHome/CompanionHomePage.tsx`, `components/Common/ServerOverviewHint.tsx`, relevant locale targets and associated route/Home tests.
- [x] Verify empty titles, Reading Queue prerequisite classification and both effective setup-guide URL keys against the retained evidence.
- [x] Apply existing title ownership, prerequisite ordering and a verified maintained server documentation target.
- [x] Reproduce Automation Inbox's temporary-outage copy for `tasks.read`403. Separate denial from service failure, preserve independently available sources, use authoritative account gating when supported, and clear/guard results across disable/account changes.
- [ ] Verify route changes/logout cannot retain Chat metadata; capability-disabled/profile-disabled/fetch-failure/empty-success branches; localization overrides and actual guide navigation. Review and commit.
- Code checkpoint `545a7a59d2`:9 actual Next Head transitions,17 Home controls,20 guide locale/fallback controls and56 existing app/route checks pass; lint0/0. Automation checkpoint `f45f7749c1` independently reviewed with87 parent WebUI/141shared controls; no new lint/type signatures. Targeted core titles and actual maintained guide navigation pass in the retained source-guide-round4 bundle. Ordinary-user Automation restriction and Reading setup state pass; remaining roles and full fresh acceptance remain.

### TASK-13260.21 — UAT076

**Files:** `tldw_Server_API/app/api/v1/schemas/flashcards.py`, `endpoints/flashcards.py`, `core/DB_Management/ChaChaNotes_DB.py`; frontend `services/flashcards.ts`, `useReviewFlashcardMutation` and `tabs/ReviewTab.tsx`; backend `tests/StudySuggestions/test_flashcard_review_sessions.py` and frontend `tabs/__tests__/ReviewTab.study-suggestions.test.tsx`.
- [x] Reproduce the exact mixed seven-card global run split into two server sessions using real request/schema/database behavior and nonconstant returned session IDs in the UI test.
- [x] Add optional explicit scope/session context, validate and retain one acknowledged run ID, support mixed cards only in validated global scope, and end exactly that session.
- [x] Verify legacy requests, wrong/foreign/inactive sessions, no-write failure, unrelated active sessions, pending account/scope changes, transient queue gaps, practice-only Cram and Undo in regressions. Run backend tests/Bandit and frontend regressions; review and live-check a mixed-deck session and reload rollup.
- Reviewed backend `982b03a940` and frontend `536ad461e9`: backend35 controls/1 official PostgreSQL skip and Bandit0; parent82 WebUI controls, lint285 unchanged warnings and TypeScript90 baseline. Original canonical expiry/principal review probes now pass unchanged. The restarted multi API and actual browser now pass five decked plus two undecked cards → one new global session2/count7 → automatic End200 → reload retaining completed7. The20-file mixed-study-round6 bundle retains source/row/session evidence and the terminal-wait limitation. Explicit early End/Undo/practice native checks and full fresh matrices remain separate.

### TASK-13260.22 — UAT077 Provider Keys loading

**Files:** `components/Option/Settings/ProviderKeysSettings.tsx`, its actual-i18n behavior test, static related PersonaGarden Scopes/Policies/Commands/Connections label callers, `i18n/icu-format.ts` and `i18n/__tests__/icu-format.test.ts`.
- [x] Reproduce the object-valued `common:loading` label with real English resources and ICU; observe the route exception before edits.
- [x] Use scalar loading titles and verify pending/empty/success/403 states without a crash. Preserve the confirmed BYOK-disabled guidance and permission boundary; test static related PersonaGarden loading/test/delete states separately.
- [x] Guard only string placeholder transformation, then delegate all other input unchanged to upstream ICU; verify syntax-tree arrays and custom object parse-error handling alongside repeated interpolation/plural tests.
- [x] Review and run the same fresh-admin visible Provider Keys route; record any separately encountered issue before expanding scope. Committed `7e48f29cb1`; real disabled-deployment guidance rendered without the original crash. Existing title finding058 remains separate.

### TASK-13260.28 — UAT087 notification rotation follow-up

- [x] Record natural refresh200 followed by unread-count401 and stale sign-in state in both tabs.
- [x] Reproduce with actual rotation storage and notification request/lifecycle boundaries; reuse effective credential selection and generation-scoped rotation events.
- [x] Preserve revoked401, permission403, cross-account isolation, cancellation and no automatic mutation replay; review and live-check both tabs. Natural expiry16:22:26 refreshed16:22:28; both tabs resumed notification200. Revocation stopped private polling for71seconds. Full fresh run remains pending.

### TASK-13260.29 — UAT088 Settings login synchronization

- [x] Record successful normal Notes login with stale Login Required in the already-open Settings tab.
- [x] Reproduce with the actual mounted Settings owner and effective credential/storage events; update auth presentation without replacing unsaved form fields.
- [x] Reject delayed and A→B→A reads; cover logout, exact-pair invalidation, unrelated server changes and valid offline credentials. Independently review and repeat the actual two-tab sequence. Checkpoint `a3542ea385`,82 tests; normal cross-tab login and real mode-change Cancel pass. Includes UAT090 context-backed confirmation and native login autofill hints.

### TASK-13260.30 — UAT089 application shell login synchronization

- Reproduce the existing Settings tab's hidden header after cross-tab login using the actual app auth owner and canonical storage.
- Resolve current effective authentication without stale cached credentials; preserve invalidation and delayed-read guards.
- Verify same-account login, revoked sessions, account replacement, and Settings draft preservation; independently review and repeat the live two-tab control.
- Code and targeted verification complete at `af725e4330`: parent61 App tests and broader179 tests pass with unchanged baseline diagnostics. Second owned-session revocation followed by normal login restores existing Settings header, Logged In and active notifications; both tabs subsequently return notification200. Full fresh run remains pending.

### TASK-13260.31 — UAT092 required generation deck

**Files:** `Flashcards/tabs/ImportExport/GeneratePanel.tsx`; existing generation/deck controls.
- [x] Reproduce Clear immediately restoring the selected required deck; remove only the misleading `allowClear` prop.
- [x] Run existing deck creation, generation gating and decomposition regressions; independent review.
- [x] Verify the actual current native selector and existing/new deck choices after the isolated frontend rebuild; retain evidence. Reviewed/live-verified checkpoint `f9b0dd2f53`,24/3 regressions, flashcard-handoff-round5.

## Stage 5: Verify integration and repeat fresh UAT
**Goal:** Establish complete acceptance on the repaired application.
**Success criteria:** Every required single/multi workflow completes with no encountered product issue; explicit blocks remain visible and cannot qualify as success.
**Tests:** Combined touched frontend/backend regressions, scoped lint/Bandit, TypeScript baseline comparison, independent review, targeted live controls and fresh full workflow matrices.
**Status:** Not Started

- [ ] Resolve independent review findings and run combined relevant checks once all changed interfaces settle; record baseline failures separately.
- [ ] Keep the tracker current for each repaired, verified or blocked finding, with commit and evidence links.
- [ ] Freeze the reviewed product; create new empty configuration/data/browser profiles and run the authoritative named frontend journeys/shared workflows in both modes with real inference.
- [ ] Repeat review/fix/UAT for new findings. Close the active goal and remove only this plan when a complete issue-free pass actually exists.
