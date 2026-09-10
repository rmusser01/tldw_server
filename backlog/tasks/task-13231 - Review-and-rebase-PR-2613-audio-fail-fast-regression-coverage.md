---
id: TASK-13231
title: Review and rebase PR 2613 audio fail-fast regression coverage
status: Done
assignee: []
created_date: '2026-09-10 00:41'
updated_date: '2026-09-10 01:06'
labels:
  - audit
  - media
  - tests
  - pr-followup
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/pull/2613'
  - >-
    backlog/archive/tasks/task-13001 -
    Refresh-audit-oversized-audio-download-regression-PR.md
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Review PR #2613 against current dev, retain useful test-only fail-fast assertions, reconcile task-ID collisions, verify the result, and assess merge readiness.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Rebased PR retains incremental downloader contract coverage on current dev without production changes.
- [x] #2 The audit task has a unique active ID and existing UserProfiles records remain unchanged.
- [x] #3 Focused tests, Ruff, Bandit, diff checks and independent review are recorded.
- [x] #4 Rebased head is pushed and current CI and human-summary merge gates are reported.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Stage 1: inspect PR, dev, and review threads (complete). Stage 2: rebase and archive the colliding audit task through Backlog CLI from its unambiguous original revision (complete). Stage 3: verify focused behavior and independent review, push with an explicit lease, and report CI and human-summary merge gates (complete).
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Supersedes the archived PR-specific TASK-13001 audit record, which collides with current dev UserProfiles TASK-13001. MCP reads were unresponsive, so official Backlog CLI is used. Rebased onto origin/dev 40345571a2cfc8b3a8893545836097d27e4ee86c. The original no-op regression repair is already in dev; remaining value is correct URL, stream=True and no body iteration for oversized Content-Length in the injected downloader path. Three focused tests pass with four warnings; Ruff passes; Bandit has seven LOW B101 findings, all ordinary pytest assertions, zero errors. Independent review found only the now-archived active task collision. Both GitHub review threads are resolved. Old July backend-required failure was OpenAPI fingerprint drift, unrelated to the test-only diff. A human-written Change summary is still required by repository merge policy.

Published the rebase with explicit force-with-lease. GitHub confirms the PR is mergeable without conflicts against dev 40345571a2. Independent follow-up review confirms TASK-13231 resolves uniquely and UserProfiles TASK-13001 plus TASK-13001.1 are unchanged. Fresh workflows started successfully; frontend and license checks passed at the initial query, with remaining checks pending. The PR description contains the review and will carry the final live CI assessment; this task completes the review/rebase work, not authorization to merge.

Qodo review follow-up: PR marked ready for review and human Change summary supplied. Qodo raised two items: (1) alleged coupling to fake response counter, evaluated as a false positive because it observes the injected downloader interface and a mutation experiment proves early body consumption is detected; (2) duplicated archived FINAL_SUMMARY boundaries, reproduced in the archived TASK-13001 record after legacy CLI serialization. Fix the malformed task serialization through supported Backlog tooling, preserve historical notes and unrelated UserProfiles records, verify parsed content and focused tests, then reply to and resolve both Qodo threads.

Qodo follow-up verified: the archived audit task now has one matching marker pair per section; all three PR task records parse without nested marker content. Historical notes, the September paragraph and unrelated UserProfiles task/child are unchanged. Repaired through installed Backlog-py configured-editor scratch workflow on an isolated original revision, then cherry-picked. The counter objection was answered with the existing dependency-boundary contract and controlled mutation evidence, and its thread was resolved. Independent re-review accepts both dispositions with no actionable findings. Focused suite: 3 passed, 4 warnings; Ruff and diff checks pass; Bandit remains seven LOW B101 pytest assertion warnings, zero errors. Human summary is supplied and PR is non-draft. Final thread resolutions and any later Qodo review results are recorded on PR #2613.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
PR #2613 remains useful as a small test-only improvement: it checks URL, streamed request mode and no body iteration before oversized-header rejection. Rebased onto current dev without conflicts, retained the incremental assertions, fixed the active Backlog ID collision using official tooling, and published the branch. Three focused tests pass, Ruff and diff/compile checks pass, and Bandit reports only seven expected LOW pytest-assert findings. Independent review has no remaining actionable findings. Merge is conditional on fresh required CI and the human requester replacing the Change summary placeholder in their own words; no merge was performed.

Qodo follow-up: fixed malformed archived task section markers through supported Backlog tooling and documented why the body-iteration counter finding is a false positive. No test or production behavior changed. Independent review and focused validation pass.
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
