---
id: TASK-12970
title: Harden playlist ingest review boundaries
status: Done
assignee: []
created_date: ''
updated_date: '2026-07-15 20:03'
labels:
  - media-ingestion
  - security
  - reliability
  - code-review
dependencies: []
references:
  - TASK-12113
  - 'https://github.com/rmusser01/tldw_server/pull/2738'
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Resolve the four Important findings from the final playlist-ingestion code review before PR 2738 is ready: idempotent run creation, playlist-table PostgreSQL RLS, durable cancellation retries, and queue selection from authoritative merged processing options.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Same-owner playlist run creation is replay-safe and conflicts on a changed canonical request.
- [x] #2 Quick Ingest reuses one durable client request identity across ambiguous retries.
- [x] #3 All seven playlist authority tables are protected by PostgreSQL RLS and a least-privilege runtime role.
- [x] #4 Durable cancellation requests retry safely and cannot cancel a rebound, cross-owner, UUID-changed, or payload-type-changed job.
- [x] #5 Run-bound queue routing and payloads use the same validated authoritative processing options.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Completed the five-stage test-first remediation: replay-safe backend run creation, stable Quick Ingest identity, PostgreSQL RLS and least-privilege role isolation, durable atomically fenced cancellation, and authoritative merged processing options.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented test-first across commits 003a673f2a..ef64e164a4. Review follow-ups added token-fenced collection initialization and recovery, database-wide RLS role isolation, atomic Jobs binding locks for cancellation, type-sensitive payload and UUID fencing, and validation-before-reservation for persisted run options.

Fresh stable verification on ef64e164a4: backend remediation matrix 553 passed with RUN_JOBS=1, including PostgreSQL migrations, RLS, and parity; frontend focused suites 169 passed; repository-pinned scoped ESLint passed; Python compileall passed; scoped Ruff passed after excluding only blame-verified pre-existing Collections_DB.py and Jobs/manager.py findings; Bandit reported 0 findings across 27,666 production lines; git diff --check passed. Independent whole-range review reported zero Critical, Important, or Minor findings. Known baseline: package-wide TypeScript still reports the repository issues documented under TASK-12113, so the established scoped frontend gates remain authoritative. External merge gate: PR 2738 still requires the requester's own Change summary and post-push CI.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Resolved every playlist-ingestion review finding. Run creation now replays by owner-scoped client identity with canonical fingerprints and renewable token-fenced initialization recovery; WebUI and extension retries reuse the durable Quick Ingest session identity; PostgreSQL playlist authority tables run under enforced RLS with a least-privilege role; cancellation remains durable and is atomically fenced against owner, UUID, binding, and type-sensitive payload changes; and run-bound jobs validate one authoritative merged AddMediaForm for both queue routing and payload construction. All remediation commits passed focused, PostgreSQL, frontend, lint, compilation, Bandit, and final independent review gates.
<!-- SECTION:FINAL_SUMMARY:END -->

<!-- SECTION:FINAL_SUMMARY:END -->

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
