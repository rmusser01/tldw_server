---
id: TASK-13173
title: Address PR 2868 review findings and complete merge
status: In Progress
assignee:
  - '@codex'
created_date: '2026-09-05 02:25'
updated_date: '2026-09-05 02:39'
labels: []
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/pull/2868'
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Resolve Qodo review findings and CI issues for the rebased Personal Context relay PR, then integrate the verified change into dev.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Conflict-list clients can retrieve every page while selected Personal Context data remains proof-gated
- [x] #2 Post-commit relay failures produce content-free diagnostics without changing committed mutation success
- [x] #3 Reported exception placement, test markers, typing, and documentation issues are corrected
- [ ] #4 Relevant regression and PR checks pass, every review finding has a recorded resolution, and the PR is merged after required human summary is present
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce pagination and diagnostic findings. 2. Apply minimal fixes and review-rule cleanup. 3. Run focused tests and security/static checks. 4. Reply to Qodo findings and verify current-head CI/review. 5. Merge the latest verified head. ADR required: no new ADR; ADR-002 governs authority/privacy and existing offset paging is exposed without changing storage or proof policy.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Rebased 61 patch-identical commits on dev c5dfe0ff73d17e177380551c946c109008d0c2cd and pushed with an exact force-with-lease. Qodo remediation adds public bounded offset paging with protected-page regression coverage; content-free after-commit relay diagnostics isolated from sink errors; centralized exception definitions with old module imports preserved; test markers, docstrings and fixture typing. Regenerated canonical OpenAPI fingerprint and local frontend schema types; drift check passes (2068 paths, 3130 schemas). Ruff and Bandit pass; Bandit emits existing nosec/parser warnings only. Independent review of the remediation diff found no issues. Initial mixed test run used an invalid /tmp basetemp and hit existing trusted-storage-path guards; rerun under native pytest temp root is in progress without modifying validation. Human-written Change summary and current-head CI/review remain merge gates. ADR: existing ADR-002; no new storage or sync policy.

Final targeted nine-file regression gate: 449 passed, 74 warnings in 277.63s with TLDW_TEST_POSTGRES_REQUIRED=1 and four workers; no skips or deselections. This includes all 25 certification cases and the real two-connection PostgreSQL authority race. Final diagnostic test rerun after annotation: 2 passed. API paging documentation now covers continuation, proof-gated response shapes, and concurrent mutation caveats. All seven Qodo findings have implemented resolutions ready to publish. Task remains In Progress until remote current-head checks/review and the human summary permit merge.
<!-- SECTION:NOTES:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
