---
id: TASK-13422
title: Confirm claims integrity across server WebUI and extension
status: In Progress
assignee: []
created_date: '2026-10-02 23:09'
updated_date: '2026-10-02 23:34'
labels: []
dependencies: []
ordinal: 14969
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Investigate the existing Claims subsystem and confirm suspected evidence-integrity gaps before feature design or implementation. Preserve server, WebUI, and extension findings plus reproducible evidence in review PR #3090. Scope is investigation and documentation; no product fixes or architectural approval are implied.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Server claims, scope, repair, and verdict findings are classified with exact source references and focused reproduction evidence where feasible.
- [x] #2 WebUI and extension request construction, shared-code reuse, defaults, evidence display, and persistence are traced with precise qualifications.
- [ ] #3 Confirmed and unconfirmed findings, existing safeguards, test results and environment limitations are saved to PR #3090 without changing product code.
<!-- AC:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Trace server and client boundaries at dev snapshot 8140e493. 2. Reproduce verdict edge cases using isolated deterministic probes and run focused existing checks where available. 3. Reconcile evidence, document findings and limitations, verify references, commit and push the review addendum.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Completed server, WebUI/shared UI, and extension traces with independent follow-up review.
<!-- SECTION:NOTES:END -->
