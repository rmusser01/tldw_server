---
id: TASK-13356
title: Integrate Calendar module onto dev for PR
status: In Progress
created_date: 2026-09-25 19:44
labels:
- calendar
- integration
documentation:
- Docs/superpowers/plans/2026-09-26-calendar-pr3019-qodo-remediation.md
updated_date: 2026-09-26 12:32
references:
- https://github.com/rmusser01/tldw_server/pull/3019
modified_files:
- .github/workflows/ci.yml
- apps/packages/ui/src/routes/__tests__/calendar-route.test.ts
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Port the completed Calendar module from the isolated feature branch onto current origin/dev without unrelated historical commits, verify integration, and open a PR against dev.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 PR diff contains Calendar module work and necessary integration only.
- [x] #2 Focused backend and frontend Calendar tests pass on current dev.
- [x] #3 Security scan and route/build verification are recorded.
- [x] #4 Pull request targets dev and documents manual provider-smoke limits.
- [x] #5 Calendar tests are assigned to the Python 3.12 and 3.13 full-suite shards and the shard coverage guard passes.
- [x] #6 PR #3019 is rebased onto latest dev, actionable Qodo feedback is addressed with verification, and merge status or remaining gates are recorded.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Approved CI follow-up: reproduce the shard coverage failure; add Calendar to gap-verified-2 in the Python 3.12 and 3.13 matrices; run the guard, focused Calendar tests and workflow validation; commit and push to PR #3019.
User-authorized follow-up: fetch latest dev and PR head; rebase the isolated Calendar branch without losing remote changes; obtain and verify Qodo review feedback; implement regression-tested fixes; run focused backend/frontend tests, Bandit and required CI; merge only after all required review and repository policy gates are met.
<!-- SECTION:PLAN:END -->
## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Ported Calendar onto origin/dev in an isolated worktree. Final focused verification: 142 backend/AuthNZ/lifecycle tests passed (2 stale migration-version assertions deselected), 30 Calendar frontend tests passed, frontend typecheck passed, targeted app ESLint passed, Bandit 0 findings, and local Next.js /calendar route returned HTTP 200. Broader router-contract tests hit unrelated optional-router import failures (media.ingest_jobs and prompt-studio websocket). The UI package has no standalone ESLint config. External Fastmail/CalDAV smoke remains unrun. Bounded polling deliberately does not infer remote deletion; explicit delta-sync deletion detection is deferred and documented. CalDAV is restricted to public HTTPS and same-origin discovered/bound collections.
PR #3019 CI failed Shard coverage guard because 10 Calendar test files are unassigned. Reproduced locally: shards=811, test_files=4802, new_uncovered=10, exit=1. User approved the scoped CI fix. Backlog MCP/CLI global searches stalled on active branch scanning; searched local task records and retained existing TASK-13356.
CI follow-up verification: added Calendar to gap-verified-2 in both Linux Python 3.12/3.13 matrices. Guard passed (shards=812, test_files=4802, new_uncovered=0); direct YAML assertions confirmed both assignments; 112 Calendar tests passed on local Python 3.11; workflow-contract suite had 56 passed and 1 pre-existing chunking-coverage assertion failure at line 1511, reproduced against HEAD workflow identical to origin/dev. Pre-commit checks passed. Bandit is not applicable to this YAML/task-only change; no Python source changed. Initial hook run hit disk exhaustion; removed only this Calendar worktree generated .next cache (1.1 GB), with no preview listener on port 13008.
User requested rebase onto latest dev, address all Qodo issues/comments once posted, then merge. Initial PR state: draft, BEHIND, head c8295ff2b155ae5db01b75c777e51b50ddb1c1b8. No PR reviews or inline review comments have been posted yet.
Rebased cleanly onto origin/dev 59bd584503. Range-diff confirms the three original Calendar implementation/docs/CI commits are unchanged. Frontend verification from apps/tldw-frontend reproduced 2 route-test failures caused by process.cwd assumptions; replaced them with the existing fileURLToPath(import.meta.url) pattern. Red/green: frontend runner changed from 28 passed/2 failed to 30 passed, and shared UI runner route tests also passed (3). Backend Calendar suite: 112 passed; shard guard: new_uncovered=0; TypeScript: passed; Bandit on all touched backend source: 0 findings, 0 errors. pnpm exec attempted auto-install and could not find @tldw/ui; used installed Vitest/TypeScript binaries directly without tracked dependency changes. Qodo review remains unposted before publishing the rebased head.
Qodo posted 21 findings (including omitted endpoint docstrings). Executing the linked 4-stage remediation plan with TDD before fresh review and merge-gate validation.
Independent review on a4c76e9c00 found eight additional issues: provider recurrence no-yield CPU bound, occurrence edits shifting masters, timestamp offset loss, missing native-item DELETE route, item timezone overlap, date-only all-day display, VEVENT DURATION omission, and silent recurrence truncation. Addressing with failing regressions under Stage 4 before merge. All 20 original Qodo inline threads are now resolved and have implementation/test replies; finding 21 acknowledged and fresh /agentic_review requested.
Final review follow-up fixes verified: 173 Calendar backend tests, 58 frontend tests, TypeScript and full touched-source Ruff pass. Bandit0 findings/errors. Fixes include bounded productive provider recurrence and arithmetic history seeking, explicit truncation warnings, native soft-delete API, item-zone/custom-offset view handling, lexical VEVENT duration and DST-safe per-occurrence arithmetic, timestamp-preserving text edits, occurrence time/kind locking, civil-date all-day spans, and visible-window agenda clamping. Added failing regressions before fixes; scoped independent re-review ongoing. Merge remains blocked by queued required GitHub checks, unrun live CalDAV/Fastmail smoke, and the repo human-summary rationale gate. Requester-authored text remains unchanged.
Final focused verification advanced to 175 backend and 59 frontend tests, TypeScript, Ruff, Bandit0 findings/errors, shardguard0 new uncovered, and normal pre-commit checks. Frontend scoped reviewer approved after overnight/exclusive-midnight coverage. Backend DST-fold regression fixes passed exact red/green tests; narrow final re-review pending before publishing.
Both independent scoped re-reviews are now clear: frontend spec/quality approved; backend no remaining P1/P2, seven targeted temporal probes pass. Publishing verified review fixes and requesting final exact-head Qodo review. No merge attempted while required CI/smoke/human-summary gates remain unsatisfied.
Fresh full Qodo review on34e644438e reports bugs0 and one new test-only rule finding: CalendarTemporalViews labels hard-code English while production intentionally follows host locale. Reproducing under French locale, then changing expected labels to actual locale formatting; no production behavior change. Current merge gates unchanged.
Locale regression reproduced with LANG/LC_ALL fr_FR:10 temporal-view tests failed. Test-only formatter now generates expected labels in the active locale while retaining numeric civil-date and negative-offset assertions. Full59 Calendar frontend tests plus TypeScript pass in French. Production backend/source unchanged from verified175-test head. Preparing normal test-only commit and exact-head follow-up review; heartbeat calendar-pr-3019-follow-up active for pending external gates.
2026-09-26 rebase checkpoint: dev advanced to a2826f103f02a67f57adb40ed048dbfa2ecfc6e5 via unrelated VZ startup-drill work. Rebased all ten Calendar commits without conflicts; range-diff reports every commit unchanged, and all PR-owned file contents match previous published eb1bb493a1025e2d6f1440ac3fe7b8d246647d78. Revalidated 175 Calendar backend tests, 59 frontend tests under French locale, 11 default-locale temporal-view tests, 3 shared-UI route tests, TypeScript, Calendar-source Ruff and diff checks. Bandit over all touched backend sources: zero findings/errors; shard guard: 4803 test files, zero newly uncovered. Publishing the rebased head after normal hooks and requesting review only once on the new head. All 25 inline threads remain resolved, no new actionable comments; previous review/CI do not establish exact-new-head approval. Task remains In Progress: required hosted checks, final exact-head review, authorized live-provider smoke and human-summary rationale gates are outstanding. Requester text preserved verbatim.
Exact-head Qodo review 5325803386 on 38c29d84bd (11:42:27Z) is complete and raised nine new actionable candidates: occurrence deletion targets the master; unvalidated organization creation; blocking worker SQLite phases; schema/DB/worker docstrings; DB test annotations; DB unit markers; malformed URL ports; inclusive scheduled-task end boundary; per-candidate agenda authorization queries. Review summary shows 19 omitted entries; these numerically match older resolved entries omitted from the rendered summary, but request explicit clarification rather than assuming. Extending Stage 4 with focused verified fixes and TDD; disjoint frontend/worker work may proceed while controller owns authorization, URL, view/DB compliance. No merge while new findings or prior gates remain.
Scoped review of the nine new Qodo fixes found two additional regression gaps before publish: malformed collection URL resolution can raise ValueError before validation, and native asyncio Task.cancel can abandon an in-flight DB phase. Add failing regressions, translate URL-resolution errors, and drain database work plus failure bookkeeping before propagating cancellation. Compliance agent completed contracts/docstrings/types/unit markers and batch overlay tests; no commits yet.
New review wave verification: 243 Calendar backend tests, 64 French-locale frontend tests, 11 default temporal-view tests, 3 shared-UI route tests and TypeScript pass. Ruff/pre-commit/diff checks pass; Bandit0 findings/errors; shard guard4804 files/new_uncovered0. Native cancellation regressions6failed before fix then6passed; full worker41passed. URL-resolution regressions4failed then6focusedpassed. Qodo clarification5846051515 confirms only9active findings. Scoped auth/query/URL reviewer approved and closed; DB/schema AST changes are doc-only except reviewed batch helpers/return annotations. Worker/frontend reviewer approved drawer and is narrowing re-review of cancellation and AnyIO scope behavior; no commit/publish yet. Dev refetched unchangeda2826f103f. Hosted required CI remains queued, live-provider smoke and human-summary rationale gates unchanged; no merge or waiver claimed.
Narrow cancellation re-review verifies native cancellation fix but reproduces an AnyIO CancelScope hot retry loop: one level-triggered cancellation during a blocked200ms DB phase produces14000-15000 shield retries (heartbeat progresses; CPU issue, not demonstrated total starvation). Record before edits: add deterministic bounded-retry scope cancellation regression and shield drain from AnyIO cancellation while preserving repeated native-cancel handling; reverify scoped and full Calendar tests/Bandit before publish.
All nine new Qodo findings are now verified fixed, plus three scoped-review gaps (collection URL resolution, native cancellation abandonment, AnyIO cancellation hot loop). Final Calendar backend245, worker43, frontend64French/defaulttemporal11/sharedroute3/TypeScript pass. AnyIO regression2red->2green; four independent200ms probes now use one drain retry with responsive heartbeat, complete success/rollback/failure audits/chaining, no orphan tasks or transaction ContextVar leakage. Both scoped reviewers PASS and all agents CLOSED. Ruff, normal pre-commit, diff, Bandit0findings0errors and shardguard0 pass. Preparing normal verified commit/push and individual review-thread evidence, then one new-head full Qodo review. Required hosted checks, live CalDAV/Fastmail smoke, and human-summary rationale remain merge gates; requester text remains verbatim. Task/Stage4 remain In Progress until actual merge.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Opened draft PR #3019 against dev from codex/calendar-dev-pr. The PR ports local Calendar CRUD, agenda/week UI, linked scheduled-task views, and read-only CalDAV VEVENT import with opt-in Jobs. Verification: 142 focused backend/AuthNZ/lifecycle tests and 30 frontend tests passed; frontend typecheck and targeted app ESLint passed; Bandit reported 0 findings; local /calendar route returned HTTP 200. Two stale AuthNZ migration-version assertions and unrelated optional-router import failures are documented in the PR. Real-provider smoke and human-authored Change summary are required before merge. Remote deletion inference is deferred until complete delta sync.
Approved CI follow-up assigns all Calendar tests to the existing gap-verified-2 Python 3.12 and 3.13 full-suite shards without changing runtime behavior or exclusions. Local shard coverage now passes with zero uncovered files; 112 Calendar tests and all applicable pre-commit checks passed. Broader workflow-contract verification: 56 passed, 1 unrelated baseline chunking-coverage assertion failure, reproduced with the unchanged dev workflow. Hosted CI will validate the pushed commit.
<!-- SECTION:FINAL_SUMMARY:END -->
## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->
