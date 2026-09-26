# Calendar PR #3019 Qodo Remediation

**Goal:** Address all 21 Qodo findings, verify the rebased Calendar module, and merge PR #3019 only when its required gates are satisfied.
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
