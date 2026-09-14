---
id: TASK-13238
title: >-
  Repair SQLite custom-schema migration preservation regression and publish
  auth/chat PR
status: Done
assignee: []
created_date: '2026-09-10 04:29'
updated_date: '2026-09-10 05:00'
labels: []
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/pull/2939'
documentation:
  - Docs/Design/ISSUES_2935_2938_REGRESSION_REPAIR.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Fix the outdated exact-column expectation in test_sqlite_upgrade_preserves_custom_users_schema_objects_and_foreign_keys. Preserve assertions for legacy columns, data, constraints, triggers, indexes, AUTOINCREMENT, and parent/child foreign keys; exercise migration91 alone and the latest startup upgrade. Update the audit record and publish the complete issues2935-2938 branch as a PR targeting dev.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Custom users schema preservation passes for migration91 alone and the latest migration chain
- [x] #2 Existing data, constraints, defaults, indexes, triggers, sequence, and foreign-key behavior remain asserted
- [x] #3 Focused and broader migration tests, lint, Bandit, and independent review are recorded
- [x] #4 Complete branch is published as a pull request targeting dev with accurate validation and human Change summary gate
- [x] #5 Array property regression remains reliable after the wider auth suite without reducing its input domain or suppressing health checks
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Stage 1 — Complete: reproduce the exact-column failure and identify intended migration 93 additions.
Stage 2 — Complete: preserve schema behavior across migration 91 and the complete upgrade; diagnose and repair the property-test timing issue without suppressing checks.
Stage 3 — Complete: validate, independently review, update the audit, commit, push, and publish PR #2939 against dev.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Migration repair: reproduced the stale exact-column expectation on the rebased branch. Parameterizing migration 91 and the latest upgrade first produced one pass and one failure. Migration 93 intentionally appends seven user fields. The repaired prefix assertion preserves original column order and all existing data, constraints, defaults, indexes, triggers, sequence, and foreign-key behavior. The full profile migration file passes 22 tests; independent review found no production migration defect or outstanding findings.

Array property reliability: two wider runs passed 99 tests but failed Hypothesis input-generation timing, while the recorded seed passed alone. Profiling a smaller reproduction showed _get_local_constants scanning 2,159 modules for 1.324 seconds. Applied the reviewed public @settings(deadline=1000), retaining a finite deadline, the full array domain and length bound, 100 examples, the assertion, and every health check. No private Hypothesis API or health-check suppression was added.

Final wider auth/profile/migration validation passed all 100 tests using test-order seed 182453889 and Hypothesis seed 219652731782752022987833063022846811844. Hypothesis completed 100 passing examples with typical runtimes below 1 ms. Fresh chat resolution/default/payload and full simplified endpoint validation passed 361 tests with one existing TestClient streaming skip; new streaming cases ran. Focused UI middleware/auth validation passed 11 tests.

Ruff, compilation, repository test guards, and whitespace checks pass. The array test passes Black. Bandit reports zero findings/errors across all 12 changed backend production files and both follow-up test files (B101 assertions excluded only for tests). Existing unrelated lint/format drift, the existing chat skip, and full-suite/large-table verification limits are documented in the retained audit.

Rebased the three unpublished issue-fix commits onto origin/dev at 177d58ac6fee87678d65ce3a9db0216021b6b68e without conflicts. Follow-up touched the two test files and Docs/Design/ISSUES_2935_2938_REGRESSION_REPAIR.md. PR publication is the remaining step.

PR #2939 is open against dev. Verified the published head and target through GitHub; final code/test commit is 46b1fd6934. All acceptance criteria are complete.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Published https://github.com/rmusser01/tldw_server/pull/2939 against dev with all issue 2935–2938 repairs, related-pattern fixes, and the unrelated migration-test correction. The custom-users regression covers migration 91 and the latest upgrade while retaining preservation checks. The array property uses a reviewed finite timing allowance for measured Hypothesis constant-discovery overhead, with full coverage and health checks intact. Fresh validation totals 494 passing targeted tests and one pre-existing chat skip; Bandit has zero findings/errors. GitHub checks have started. The PR clearly records the required human-written Change summary before merge.
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
