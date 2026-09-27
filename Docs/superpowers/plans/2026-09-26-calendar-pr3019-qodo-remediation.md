# Calendar PR #3019 Qodo Remediation

**Goal:** Address all original and follow-up Qodo findings, verify the rebased Calendar module, and merge PR #3019 only when its required gates are satisfied.
**Architecture:** Retain the SQLite repository, existing calendar permission model, read-only CalDAV adapter, bounded Jobs polling, and thin frontend. Share credential resolution and offload blocking provider operations. Preserve recurrence masters and exception identities rather than collapsing a series into one row.
**Tech Stack:** FastAPI, SQLite, HTTPX, AnyIO, icalendar, python-dateutil, pytest, Next.js, Vitest.
**Spec:** `Docs/superpowers/specs/2026-06-05-calendar-module-prd-design.md`.
**Tracking:** TASK-13356; https://github.com/rmusser01/tldw_server/pull/3019.

## Global Constraints

- Work only in `.worktrees/calendar-dev-pr`; preserve unrelated user changes.
- Write regression tests and observe their failures before implementation.
- Keep CalDAV imports read-only, same-origin, HTTPS-only, and credential scoped.
- Never log credentials or raw provider exception messages. Log safe operation context and traceback frame locations.
- Keep database SQL in `Calendar_DB.py`; make item/recurrence writes atomic.
- Do not invent a human Change summary or bypass required checks or merge policy.
- Reply to each Qodo thread with the verified disposition and commit; obtain a fresh review after fixes.

## Stage 1: Provider Safety and Worker Reliability
**Goal:** Fix findings 6, 7, 10, 11, 12, 15, and 19.
**Success Criteria:** Streamed responses stop at a byte limit; XML/ICS are bounded before parsing; synchronous provider calls run off the event loop; API and worker reuse scoped credential resolution; a deleted account or failed scan does not stop later polling; diagnostics contain safe context and stack locations without secrets.
**Tests:** Add oversized Content-Length/chunk/ICS tests to `tests/Calendar/unit/test_calendar_caldav_provider.py`; add event-loop responsiveness, safe logging, missing-account continuation, and scan-retry tests to `test_calendar_sync_worker.py`; add verify/discover off-thread and credential precedence/scope tests to `integration/test_calendar_api.py`.
**Files:** `core/Calendar/providers/caldav.py`, new scoped `core/Calendar/provider_operations.py`, `core/Calendar/calendar_sync_worker.py`, `services/calendar_sync_scheduler.py`, `api/v1/endpoints/calendar.py`, and the tests above.
**Steps:** Write and run regressions (expect failures); implement the smallest shared operations and bounded transport; rerun affected tests (expect pass); run Bandit and pre-commit; commit with TASK-13356.
**Status:** Complete

## Stage 2: Temporal and Recurrence Integrity
**Goal:** Fix findings 1, 2, 3, 5, 13, 14, and 16.
**Success Criteria:** Provider UID plus recurrence identity is stable; masters, exclusions, additions, and detached exceptions survive repeated imports; all-day dates and exclusive ends survive; local RDATE/EXDATE work; malformed times and reversed intervals are rejected before persistence; explicit null removes recurrence while omission preserves it; failed recurrence writes roll back item changes; afternoon windows contain all-day events.
**Tests:** Extend provider parsing/import tests, `test_calendar_recurrence.py`, `test_calendar_service.py`, `test_calendar_db.py`, and API tests with recurrence-only dates, exclusions, detached exceptions, date-only DTSTART/DTEND, invalid updates, all-day noon queries, null/omitted recurrence, and injected write failures. Extend recurrence property coverage where applicable.
**Files:** `core/Calendar/recurrence.py`, `view_service.py`, `calendar_service.py`, CalDAV provider/worker, `core/DB_Management/Calendar_DB.py`, schemas/endpoints, and corresponding tests.
**Steps:** Write regressions and observe failures; use dateutil recurrence sets and bounded iteration; preserve provider metadata and instance identities; implement nested repository transactions and deletion; validate merged item state; run Calendar unit/integration/property suite; run Bandit and pre-commit; commit with TASK-13356.
**Status:** Complete

## Stage 3: Permissions, Links, and Review Hygiene
**Goal:** Fix findings 4, 8, 9, 17, 18, 20, and 21.
**Success Criteria:** Active AuthNZ organization roles are resolved request-locally with org/tenant boundaries; permission tests have type hints and unit markers; existing item calendar selection cannot imply an unsupported move; persisted links load after refresh and can be removed through authorized APIs; centralized calendar exception exports retain compatibility; module/dependency/endpoint functions have concise meaningful docstrings.
**Tests:** API role member/nonmember/wrong-org and revoked membership tests; authorized link list/delete and refresh tests; drawer calendar-selector and persisted-link tests; exception export identity and endpoint docstring checks.
**Files:** Calendar API/schemas, permission tests, shared exception modules, `apps/packages/ui/src/services/calendar.ts`, Calendar drawer/types/tests, and backend integration tests.
**Steps:** Write failing tests; wire existing AuthNZ membership APIs; add link GET/DELETE and UI retrieval; disable only the unsupported move control; centralize exception definitions using existing lightweight export pattern; document API functions; run backend/frontend tests and typecheck; run security/format checks; commit with TASK-13356.
**Status:** Complete

## Stage 4: Re-review and Integration
**Goal:** Publish verified fixes, address follow-up review, and satisfy merge gates.
**Success Criteria:** Each Qodo finding has a verified disposition in its thread; fresh Qodo review and required CI checks pass; branch is current with dev; PR is merged, or an exact remaining external/policy gate is recorded without claiming completion.
**Tests:** Full Calendar pytest suite; focused frontend Vitest tests from frontend and shared-UI working directories; frontend TypeScript; touched-scope Bandit; pre-commit; shard coverage guard; git diff checks. Inspect baseline failures separately rather than masking them.
**Steps:** Self-review complete diff; verify and push; reply to Qodo threads; request fresh review; inspect CI and latest dev; rebase/reverify if required; merge only when allowed; update Backlog with PR/verification/final state.
**Status:** In Progress

### Follow-Up Review Fixes

- Bound provider recurrence before entering dateutil, including rules that never yield; report occurrence truncation as partial rather than complete.
- Preserve master timestamps when editing occurrence text and preserve offsets on unchanged temporal fields; disable unsupported occurrence time editing.
- Add authorized native-item soft deletion using the existing service/repository operation.
- Apply item timezone to nonrecurring overlap and serialize timed views with offsets.
- Treat date-only all-day values as civil dates in agenda/week rendering.
- Derive exclusive provider ends from valid VEVENT DURATION when DTEND is absent.
- Add failing regressions first, run focused/full checks, and obtain a scoped independent re-review before publishing.
- Ownership: controller owns backend changes; a frontend implementer owns only Calendar drawer/agenda/week and their frontend regression tests. No overlapping edits or concurrent commits.

### Exact-Head Review on 38c29d84bd

Review 5325803386 adds nine findings; the previous 26 remain resolved. Qodo clarification 5846051515 confirms the findings index has only these nine active entries and every other indexed entry is implemented. Keep the approved feature scope and merge gates unchanged.

1. **Authorization and destructive actions (Complete):** Add failing service/API regressions rejecting organization creation without active authenticated membership and drawer regressions proving recurrence occurrences cannot delete the shared master. Enforce organization authorization in CalendarService with a request-scoped membership resolver; hide ordinary occurrence deletion. Frontend worker owns only the drawer and its tests; controller owns service/API and their tests.
2. **Worker event-loop responsiveness (Complete):** Reproduce blocking initialization, lookup, persistence, and failure bookkeeping with deterministic off-thread regressions. Offload complete synchronous phases, keeping each transaction on one worker thread and preserving bounded provider dispatch, atomic imports, and scoped credentials. Worker agent owns calendar_sync_worker.py and its tests, including missing docstrings there. Independent review additionally reproduced native asyncio cancellation abandoning a live database phase and AnyIO scope cancellation spinning the drain. Drain started work and failure bookkeeping before propagating cancellation; shield subsequent drain attempts from level-triggered cancellation while retaining repeated native-cancel handling. Six native and two scope regressions failed before their fixes. Final narrow re-review is clear, with one drain retry in each of four 200ms probes and no orphan work or transaction-context leakage.
3. **URL and view correctness/performance (Complete):** Write red tests for malformed base/collection ports and end-exclusive scheduled projections. Validate URL ports before account persistence and convert parse errors to domain validation errors. Replace per-row item refetches with authorization of loaded rows, calendar-context caching and batch provider ownership lookup; prove query count is independent of candidate count while retaining privacy and tenant checks. Controller owns provider, view/service, DB helper and corresponding tests. Independent review found and verified an additional collection-authority urljoin error; four regressions failed before translation, six focused URL/API cases now pass. Narrow re-review reports no remaining findings.
4. **Compliance and publish (In Progress):** Annotate/classify DB tests and document new schema/DB symbols with concise meaningful docstrings; verify contracts. Run full Calendar/backend/frontend, TypeScript, Ruff, Bandit, pre-commit and shard guard; obtain scoped independent review; commit and publish verified changes, reply to each new inline finding, request exactly one fresh exact-head review, and update the heartbeat. Stage 4 remains In Progress until merge gates are genuinely satisfied.

### Exact-Head Review on 7e15ca42d6

Review 5325936683 is complete with eight new inline findings; the prior 35 remain implemented. Clarification 5846439303 confirms all 27 omitted index entries are implemented and only the eight current inline findings remain active. Do not request a duplicate full review on this head.

1. **Credential and persistence safety (Complete):** Reproduce cross-origin forwarding of stored credentials in verify/discovery; pin reused credentials to the persisted account origin while retaining deliberate complete replacement credentials. Reproduce orphan secrets on account-insert failure and wrap account plus secret creation in one repository transaction. Controller owns provider_operations.py, Calendar API and integration tests. Initial regressions: 22 failed, then 124 API/service tests passed. Independent review additionally reproduced a fallback URL acting as its own origin for legacy unpinned credentials: two regressions failed before requiring a saved origin for all reuse, then 126 API/service tests passed. Narrow independent re-review approved with eight targeted tests and 16 helper checks.
2. **Temporal correctness (Complete):** Reproduce omitted timezone for local wall-clock item creation and inherit the selected calendar timezone. Controller owns calendar_service.py and its unit tests. A temporal implementer preserved embedded custom VTIMEZONE definitions and payload-scoped timezone reconstruction, with four later-window regressions failing before the fix. Preserve DST, folds, durations, exclusions, detached identity and refresh stability. Pre-resolution guards reject nonproductive rules, observance exclusions, oversized definitions and excessive observance counts. Independent review reproduced dense historical transitions and two rule-family gaps; the controller added a cumulative 20,000-transition budget covering RRULE history and explicit dates, rejected mixed ordinal/plain weekdays, and corrected implicit month-day density. Their regressions failed first; final narrow review approved with 38 focused tests, 180 finite-rule comparisons without undercounts, and DST probes through 2100. All temporal agents are closed.
3. **Frontend input and migration boundaries (Complete):** A frontend implementer owns CalendarSyncSettings.tsx/CalendarItemDrawer.tsx and their tests, normalizing sync days to integers and verifying selected-calendar timezone payloads. Six regressions failed first; 35 focused and 70 full Calendar frontend tests pass under French locale, TypeScript clean. A separate migration implementer moved only the new Calendar SQL to a DB_Management helper without changing unrelated migration code. Query-boundary/helper tests failed first; 18 focused unit tests and 52 related migration tests pass. Two real-PostgreSQL checks skip because the existing fixture cannot reach PostgreSQL. Both slices have clear independent scoped reviews; implementers/reviewers are closed.
4. **Compliance and release (In Progress):** Controller annotated service tests/helpers, renamed the vague API scenario, and mechanically fixed two property-test lint issues. Final verification: 312 Calendar backend tests, 70 French-locale frontend tests, 11 default temporal-view tests, 3 shared-UI route tests, TypeScript, Ruff, pre-commit and diff checks; Bandit zero findings/errors across 12 touched backend source files. Migration verification: 18 focused unit tests and 52 related tests pass, with two existing-fixture PostgreSQL skips. Shard guard covers 4806 test files with zero newly uncovered. All independent scoped reviews approve and every agent is closed. Commit with normal hooks, publish, reply with individual evidence and request one exact-new-head full review. Required CI, live-provider smoke and human rationale gates still apply.

### Exact-Head Review on 09d43aa0a0

Review 5326092846 completed with two new inline correctness findings; prior 43 findings remain implemented. Clarification 5846842617 confirms the index has two active findings, 42 implemented and one dismissed, with no additional active omitted entries.

1. **Floating recurrence dates (Complete):** Parser and four import/persistence/refresh/reopened-repository agenda/week regressions reproduced UTC coercion. Preserve naive wall time when no TZID is supplied; retain explicit UTC and named-zone offsets. IANA and embedded custom-zone cases retain intended additions, exclusions and one-hour durations across DST.
2. **Annual timezone productivity (Complete):** Eight finite-rule/import regressions reproduced valid month-day sets rejected because some selected months lack a requested day. Accept sets with at least one valid non-leap-year transition, retaining syntax validation, rejection of wholly nonproductive combinations, and conservative cumulative transition bounds. Positive/negative days, mixed valid/invalid dates, implicit/explicit months, pre-parser rejection and finite count estimates are covered.
3. **Verification and publication (In Progress):** Initial focused run: 14 failed and 15 passed; corrected run: 29 passed. Full Calendar backend: 330 passed. Ruff, normal pre-commit and diff checks pass; Bandit zero findings/errors on both changed source files; shard guard 4806 test files, zero newly uncovered. Independent scoped review approved: 29 focused and 16 existing regression tests, 1032 accepted finite-rule comparisons without undercounts, and 192 nonproductive combinations rejected. Reviewer closed with no active sessions. Frontend and migration production/test files are unchanged from their previously verified suites; do not repeat them. Publish verified fixes, reply individually, and request one fresh exact-new-head review. Existing CI, live-provider smoke and human-summary merge gates remain unchanged.

### Exact-Head Review on 9e0e55eea8

Review 5326171306 reports zero bugs and one documentation finding (4111618387) covering two helpers. Clarification 5847009576 confirms no other active omitted findings. Prior 45 findings retain verified fixes/dispositions.

1. **Helper contracts (Complete):** Two static contract cases failed before documentation changes. Expanded only _validate_timezone_rule and CalDavProvider._component_dates docstrings using established Args/Returns/Raises style. Every parameter, actual return behavior/bounds and CalendarValidationError conditions are documented; production ASTs are unchanged excluding docstrings.
2. **Verification and publication (In Progress):** All 10 contract tests and 29 focused temporal regressions pass; Ruff, Bandit zero findings/errors, diff and shard guard zero newly uncovered pass. Independent scoped reviewer approved documentation, tests and AST equivalence, then closed all sessions. Run final normal pre-commit, publish with individual evidence, and request one exact-new-head full review. Prior 330-test Calendar runtime verification remains applicable; do not repeat unchanged broad suites or weaken remaining merge gates.

### Current-Dev Rebase on f5fa1f3a41

1. **Integration (Complete):** Dev advanced to `f5fa1f3a41855aa02871d8b76d0ec0cebbaf9e07` through PR #3006 (Sync blob-upload expiry and documentation). All 15 Calendar commits rebased without conflicts and compare equal by range-diff. Only the 12 upstream-owned files differ from published `710885e988`; Calendar production and test files are unchanged.
2. **Fresh verification (Complete):** Full Calendar backend: 332 passed. French/Europe-Paris frontend: 70 passed; default temporal: 11 passed; shared-UI route: 3 passed. TypeScript, Ruff, full PR-scope normal pre-commit, diff checks and shard guard (4807 test files, zero newly uncovered) pass. Bandit reports zero findings/errors across 27 PR-owned backend source files. Related AuthNZ/service verification: 48 passed, two failed and two PostgreSQL fixture skips. Both failures are the previously documented migration-090 helper assertion at `test_rbac_seed_helper.py:432` (expected version 96, registry version 98); relevant helper/registry files are unchanged from the prior published head. No tests were disabled and no unrelated fix was introduced. PostgreSQL skips are not live-database success.
3. **Publication and gates (In Progress):** Publish the rebase and tracking evidence using normal hooks and an explicit lease against `710885e988`, then request one full review on the exact published head. The prior head's Qodo execution remains unverified; neither its resolved-only summary nor receipt reaction establishes completed full review. Required exact-head CI, live CalDAV smoke and the human-owned rationale gate remain prerequisites. Stage 4 and TASK-13356 remain In Progress.

### Exact-Head Deep Review on aa48471476

Review 5327644148 completed at 22:04:52Z and posted six new inline candidates. The summary shows one bug/two rules and omits 37 entries; clarification 5850352501 confirms exactly these six active findings and no additional active omitted entries. There are now 52 reported findings and 51 inline threads. Do not duplicate the completed full review.

1. **Polling cadence (Complete):** Reproduced omitted-interval bindings remaining continuously due. Schema, repository and UI now use the smoke runbook's hourly default; explicit/legacy null remains manual-only and is excluded from periodic due queries. Ten cadence regressions failed before production edits; all 113 DB/worker tests pass, retaining native/AnyIO cancellation and atomic import behavior.
2. **Backend request contracts (Complete):** Invalid ZoneInfo keys are rejected before calendar writes. Manual sync overrides require aware ISO timestamps, positive UTC spans no longer than the existing combined lookback/lookahead maximum (7400 days), and at most 64 characters per boundary; invalid input creates neither Jobs nor queued audits. The bounded sync-history handler uses synchronous FastAPI worker execution, including ownership checks. All 31 permission dependencies are annotated with the actual AuthPrincipal return type. Controller regressions failed first; the 165-test service/API/contract suite passes.
3. **Verified account setup (Complete):** Generic account creation remains compatible through verify_before_create=False. The UI sends True with complete supplied CalDAV credentials; bounded off-loop verification precedes the account/secret transaction. False verification or provider failure creates neither account nor encrypted secret. UI failure preserves the draft and skips post-persistence verification. Frontend 13 red regressions preceded fixes; 29 focused and 83 full French/Europe-Paris tests pass, with TypeScript and scoped ESLint clean. Strengthened backend regressions prove exact credentials, off-loop execution, ordering and no orphan secrets.
4. **Verification and publication (In Progress):** Full Calendar backend: 385 passed. Ruff, normal 17-file pre-commit, diff and shard guard (4807 files, zero newly uncovered) pass; Bandit reports zero findings/errors across all six changed backend source files. Independent backend review approved 115 targeted tests and nine boundary/authorization/rollback/cancellation probes; independent frontend review approved 29 focused tests and six Unicode/password/retry/serialization probes. All implementers/reviewers and their sessions are closed. Commit and publish with individual replies to all six findings and one exact-new-head full review. Existing hosted CI, live CalDAV smoke and human-owned rationale remain merge gates. Unchanged migration verification retains 48 related passes, two known migration-090 assertion failures and two unavailable-fixture PostgreSQL skips, not live-PG success.

### Exact-Head Deep Review on 573b90de8a

Review 5327879344 completed at 22:54:51Z and posted eight new inline candidates. Findings-index clarification 5850714013 confirms exactly eight active findings and no other active omitted entries; the remaining 52 are implemented or dismissed. The completed request must not be duplicated.

1. **Canonical rate limits (Complete):** Missing catalog policy produced three failing cases; catalog-only enforcement still failed three HTTP 429 assertions because rate dependencies ran before authentication. Authentication-first router dependencies and all 32 per-user budgets now enforce existing standard read (120/minute, burst 240) and privileged write/sync (60/minute, burst 120) classes. Nine integration cases cover all routes, auth-kind sharing, user isolation and stricter/higher database overrides. Independent review approved 178 related tests, 12 boundary probes and two seed checks; no live PostgreSQL success claimed.
2. **Off-loop polling (Complete):** The complete synchronous scan/account/queue phase uses the existing cancellation-drained DB phase; 12 regressions failed before the move. Independent review reproduced a manual/scheduler active-check/create race; shared same-binding admission and the manual HTTP off-loop phase now pass two race regressions and seven baseline-failing API cancellation/responsiveness cases. Independent approval includes 32 focused tests, 20 concurrent manual requests with scheduler contention and a symlinked repository producing one job/audit, unrelated binding progress, idle-lock reclamation and 21 repeated native cancels plus AnyIO draining of a real uncommitted audit. Preserve bounded scans and hourly/manual-only cadence. The weak-registry binding gate is process-local only; it is not cross-process atomicity or a transaction across separate Jobs/Calendar stores.
3. **Temporal consistency and coverage (Complete):** Six DST/fold mismatch regressions and one malformed P1DT regression failed before fixes; 63 focused and 66 temporal/property tests pass. Direct and set timed expansions share UTC elapsed-duration calculations. Thirty-seven direct duration cases passed baseline and explicitly protect civil days/weeks, elapsed subday time, validation and folds. Independent review reproduced second-fold DTSTART loss/duplication and same-zone fold-window rejection. Fold-aware candidates retain the exact initial instant/count slot; datetime query ordering/span use UTC. Twenty-eight baseline failures plus UNTIL/spring/exact-seed follow-up failures preceded fixes; initial 49 new cases and 237 related tests pass, including persisted agenda/week. Final review additionally reproduced same-zone UNTIL premature exclusion of first-fold candidates: five failures/two passes changed to seven passes after giving dateutil a UTC UNTIL bound, retaining civil/floating interpretation first. Final independent approval covers 58 focused tests and 13 civil/naive/COUNT/second-fold probes; no remaining actionable temporal findings.
4. **Frontend validation and coverage (Complete):** Projection URLs use the established safe-URL helper; local field-level schedule validation retains drafts without API calls. Thirteen initial regressions failed before fixes. Scoped review then reproduced due-only display fallback, hidden kind-switch draft and Honolulu test assumptions: three drawer cases, four persisted provenance cases and one Honolulu case failed before their respective fixes. Authorized view metadata now distinguishes canonical nullable item_start_at from a due-derived display timestamp; active-kind serialization preserves actual start-only todos and drops hidden event drafts. Full French/Europe-Paris frontend 125, Honolulu temporal 13 and TypeScript pass. Scoped ESLint has zero errors and three unchanged baseline warnings. Independent narrow review approved 12 probes, nine drawer cases, two Honolulu cases and scoped TypeScript; reviewer is closed. Direct filters/ownership coverage passes; same-day events already render in both views on unchanged production code, supporting a reasoned disposition rather than a false fix.
5. **Integration and publication (In Progress):** Final full Calendar backend: 530 passed with 13 warnings and no failures. Frontend 125 French/Europe-Paris, Honolulu temporal 13 and TypeScript pass. Ruff, scoped ESLint (zero errors, three unchanged warnings), diff checks and shard guard (4810 files, zero newly uncovered) pass; Bandit reports zero findings/errors in all six changed backend source files. Independent rate, frontend and temporal/admission reviewers approve; all agents and sessions are closed. Commit with normal hooks, publish and reply individually to all eight findings, including the baseline-verified same-day disposition. Then request one full review on the exact new head. Hosted required CI, live CalDAV smoke and human-owned rationale remain merge gates; Stage 4 and TASK-13356 remain In Progress until actual normal merge.

### Wave9: Durable Sync Admission (Approved 2026-09-27)

**Spec:** `Docs/superpowers/specs/2026-09-27-calendar-sync-admission-design.md`.
**Goal:** Fix the three exact-c87 findings without changing provider or UI scope.
**Status:** In Progress

1. **Atomic reservation and audit (Complete).** Modify
   `core/DB_Management/Calendar_DB.py` with a typed admission row, binding-unique
   reservation, conditional Job attachment/release and bounded pending lookup.
   Keep SQL in the repository. Add failing tests in
   `tests/Calendar/unit/test_calendar_sync_worker.py` for audit failure leaving
   zero runnable Jobs, dispatch failure retaining one recoverable audit, and
   simultaneous processes creating exactly one Job/audit. Run those exact tests
   before changing worker admission; implement and rerun them.
2. **Dispatch and lifecycle (Complete).** Replace worker process-local locking
   with committed reservation plus stable idempotent Jobs dispatch. Use exact
   scoped live/archive lookup and conditional release only for terminal states.
   Replace the capped legacy lookup with existing keyset pagination. Scheduler
   recovers pending admissions before ordinary due scans, including manual-only
   bindings. Red/green tests cover old Jobs beyond 100, post-dispatch crash,
   processing/delayed retries, archive/missing Job behavior, same-window terminal
   resubmission, tenant scope and unrelated bindings during blocked dispatch.
   Independent review reproduced stale-dispatch resurrection after terminal
   pruning, status-separated legacy retry omission, and completed manual-only
   recovery resubmission. Three controller regressions RED then GREEN using the
   existing Jobs durable receipt API, status-independent keyset traversal and
   exact recovery distinct from fresh triggers. Receipt commands explicitly use
   priority 5 to preserve the Jobs 1..10 contract; initial default-100 failures
   were corrected before final integration. Final independent review approved
   the exact source hashes and six race/encryption/authority/fairness probe groups.
3. **Typed core trigger (Complete).** Add
   `CalendarService.trigger_binding_sync(...)->CalendarSyncJobResponse` and type
   `queue_binding_sync`. Move configured-window resolution and ownership into
   core; endpoint delegates through `_run_db_phase` with validated fields.
   Add an endpoint delegation regression and preserve aware ISO <=64-character,
   positive UTC <=7400-day validation and native/AnyIO cancellation regressions.
4. **Verification and publication (In Progress).** Current-dev rebase completed
   without conflicts onto `718c191082f1d6372fb6fe000ac763dcc07ffcbd`; local rebased
   head `1509d98dd7504d946dcb001c5a00d7a4034452e1`, published c87 unchanged. Run
   full Calendar backend, current frontend/TypeScript and migration boundaries,
   Ruff, touched-source Bandit, shard guard, diff and normal pre-commit. Obtain
   fresh independent scoped review, fix findings with TDD, commit/publish only
   verified work with an explicit remote-head lease. Reply individually with
   evidence and verify thread resolution; request exactly one full new-head
   review. Required CI, latest dev, provider smoke and human rationale still gate
   normal merge. Never publish an intermediate tracking-only review head.
   Final current-base verification: 558 Calendar backend tests (13 warnings),
   68 existing SQLite Jobs receipt/prune tests (2 warnings), 18 Calendar migration
   unit tests (38 warnings), 13 Honolulu temporal tests, TypeScript, Ruff, diff,
   normal 11-file pre-commit and shard guard (4829 files, zero newly uncovered)
   pass. Bandit reports zero findings/errors in five changed backend sources.
   Prior current-frontend French/Europe-Paris run passed 125 tests on dev9668;
   subsequent dev718c changes have no frontend overlap. Bacon independent final
   scoped review approved and closed; no live PostgreSQL success is claimed.

## Decisions and Evidence

- 2026-09-26: Rebase onto dev `59bd584503` completed; original three commits unchanged by range-diff. Published head `6602a4c5956e99be6ed5d11fef879102d4732a36`.
- Baseline on rebased head: Calendar backend 112 passing, frontend 30 passing, shared-UI route tests 3 passing, TypeScript clean, Bandit zero findings, shard guard zero new uncovered files.
- Qodo review posted 21 findings. Finding 21 was omitted from the initial summary and was supplied in issue comment 5843301983: endpoint docstrings.
- External provider smoke remains unrun without credentials. Preserve the requester-authored Change summary verbatim; do not manufacture human rationale.
- Follow-up Qodo review on a4c76e9c00 resolved the original findings and raised four new findings: typed recurrence property helper, recurrence regression docstring, explicit truncation warning, and old high-frequency recurrence seeking. These are included in Stage 4 fixes.
- Ruling: Expand only productive provider rules (frequency/interval/count/until, weekly weekday selection, week start); preserve complex BYxxx rules verbatim with partial-result warnings rather than evaluating potentially non-yielding dateutil generators. This follows the spec's pragmatic recurrence/preserve-complex-provider-data boundary.
- Ruling: Seek second/minute/hour rules arithmetically before dateutil, preserving interval phase, COUNT, UNTIL, and duration-adjusted overlap. Dateutil between() itself still scans from DTSTART and does not provide a work bound.
- Remediation verification: 152 Calendar backend tests; 34 frontend tests; 3 route tests from the shared-UI cwd; TypeScript and full Calendar-source Ruff pass; touched-scope Bandit has zero findings/errors; pre-commit and shard coverage guard pass.
- Additional regression: provider all-day RRULE UNTIL dates are normalized for aware recurrence expansion while raw provider metadata remains unchanged.
- Refetched dev remains `59bd5845038342013a2d84d0130f6164f14b54fd`. Desktop/mobile Calendar drawer screenshots use isolated, nonsecret API fixtures, not a live provider connection.
- Final follow-up verification: 173 backend tests and 58 Calendar frontend tests pass; TypeScript passes. Scoped reviewers identified and prompted fixes for DST duration arithmetic, custom VTIMEZONE offsets, agenda window clamping, and DST-crossing edits. Added regressions failed before fixes.
- DURATION uses structured icalendar content-line parsing to retain P1D versus PT24H, then shared nominal-day/elapsed-hour arithmetic. Raw duration is preserved and reapplied to each recurrence instance. See RFC 5545 section 3.3.6.
- CalendarPage tests now pin Date to their June fixture window without faking async timers; previously they displayed mocked out-of-window records using the current date.
- Scoped frontend review approved after an additional overnight-window regression failed and was fixed; final frontend suite has 59 passing tests. Midnight-exclusive boundaries retain correct placement.
- Final backend fold regressions: compare recurrence durations and window overlap as UTC instants; normalize recurrence-set candidates/additions/exclusions before ordering so repeated local hours remain distinct. Zero nominal-day duration additions preserve fold. Both regressions failed first; full Calendar backend now has 175 passing tests.
- Independent scoped re-reviews are clear: frontend spec/quality approved; backend reports no remaining P1/P2 and seven targeted fold/exclusion/all-day checks pass. Required hosted CI and live-provider/human-summary gates remain external prerequisites.
- Fresh full Qodo review on 34e644438e reported no bugs and one locale-dependent test finding. French-locale reproduction failed 10 of 11 temporal-view cases; replacing hard-coded English labels with a locale-aware test formatter preserves civil-date assertions without changing production behavior. All 59 Calendar frontend tests and TypeScript pass under French locale after the fix.
- Integration checkpoint: dev advanced to `a2826f103f02a67f57adb40ed048dbfa2ecfc6e5` with unrelated VZ startup-drill changes. All ten Calendar commits rebased without conflicts and compare equal by range-diff; every PR-owned file is unchanged from published `eb1bb493a1`.
- Rebase verification: 175 backend tests, 59 French-locale frontend tests, 11 default-locale temporal-view tests, and 3 shared-UI route tests pass. TypeScript, Calendar-source Ruff, and diff checks pass; touched-backend Bandit has zero findings/errors; shard coverage has zero newly uncovered files among 4803 test files. Publish with an explicit lease, then obtain exact-new-head Qodo review and required CI. Live-provider smoke and human-summary policy gates remain outstanding; Stage 4 is not complete.
- New review wave checkpoint: 168 stable service/provider/API/DB/documentation-contract tests pass; 64 French-locale frontend tests, 11 default temporal tests, 3 shared-UI route tests and TypeScript pass. Scoped authorization/URL/query review is clear; large agendas require 8/9 SELECTs for 51 candidates, preserving tenant/private-provider/actor-overlay boundaries. DB/schema AST comparison confirms documentation-only changes except two reviewed batch helpers and return annotations. Worker cancellation regression remains in progress before full-module verification and publication.
- Final new-wave verification: 245 Calendar backend tests (43 worker), 64 French-locale frontend tests, 11 default temporal tests, 3 shared-UI route tests and TypeScript pass. Ruff, normal pre-commit, diff and shard checks pass; Bandit reports zero findings/errors in touched backend source. Both scoped re-reviews are clear and all agents are closed. All 35 reported Qodo findings have verified fixes, including the nine new findings plus collection-resolution/native-cancellation/AnyIO-drain gaps found by independent review. Publish, reply to the nine threads, then request one fresh exact-head full review; required hosted CI, live-provider smoke and human-summary rationale gates remain outstanding. Stage 4 stays In Progress.
