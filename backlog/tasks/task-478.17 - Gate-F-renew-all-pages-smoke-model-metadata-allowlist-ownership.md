---
id: TASK-478.17
title: 'Gate F: renew all-pages smoke model metadata allowlist ownership'
status: Done
assignee: []
created_date: ''
updated_date: '2026-05-26 06:40'
labels: []
milestone: Research Workspace UAT Remediation
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/pull/2055'
  - >-
    https://github.com/rmusser01/tldw_server/actions/runs/26436020317/job/77818951309
parent_task_id: TASK-478
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Fix the remaining UX Smoke Gate failure on PR #2055 where the all-pages hard-gate allowlist metadata check rejects two model-metadata noise entries because their `expiresOn` dates are stale (`2026-03-31`). Scope: keep the allowlist metadata current only if the entries are still intentionally scoped and owned; do not mask new page failures.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Expired all-pages smoke allowlist metadata entries are renewed only when they remain scoped and owned.
- [x] #2 The model-metadata rate-limit and abort allowlist entries keep their existing route scoping and ownership.
- [x] #3 Focused all-pages allowlist metadata test fails before the change and passes after it.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Root cause: the all-pages hard-gate metadata test aborted before route execution because m5-model-metadata-rate-limit-log-noise and m5-model-metadata-abort-noise had expiresOn set to 2026-03-31.
- Both entries remain intentionally narrow: the rate-limit rule is scoped to /content-review, /claims-review, and /research-workspace; the abort rule is scoped to /research-workspace only.
- Renewed only those two expiresOn values to 2026-09-30, matching the current review horizon used by the rest of SMOKE_HARD_GATE_ALLOWLIST.
- RED verification reproduced the CI failure: bunx playwright test e2e/smoke/all-pages.spec.ts --reporter=line --grep "hard-gate allowlist entries have current ownership metadata" failed with both expired entries.
- GREEN verification passed with 1 passed after renewal.
- git diff --check passed.
- Bandit not applicable: only TypeScript smoke-test metadata and Backlog task metadata were touched.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Renewed the two scoped model-metadata all-pages smoke allowlist entries whose ownership metadata had expired, preserving their existing owner/rationale/route scoping and aligning them with the current smoke allowlist review horizon.
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
