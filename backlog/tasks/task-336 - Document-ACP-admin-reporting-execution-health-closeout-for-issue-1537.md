---
id: TASK-336
title: Document ACP admin reporting execution-health closeout for issue 1537
status: Done
assignee: []
created_date: '2026-05-14 05:09'
updated_date: '2026-05-14 05:31'
labels:
  - acp
  - docs
  - admin-reporting
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/issues/1537'
  - 'https://github.com/rmusser01/tldw_server/issues/1537#issuecomment-4447890500'
  - 'https://github.com/rmusser01/tldw_server/pull/1654'
  - 'https://github.com/rmusser01/tldw_server/pull/1686'
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Document the ACP admin execution-health reporting contract and current product surfaces so GitHub issue #1537 has a concrete closeout path. This follows the merged Agent Registry execution-health UI/API slice from PR #1654 and should keep remaining work split into follow-up implementation issues rather than hidden in roadmap prose.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Existing ACP docs identify execution-health summary API, metric groups, failure buckets, setup-health dimensions, retention/redaction posture, and compatibility evidence.
- [x] #2 Docs identify current and planned UI/API reporting surfaces for Agent Registry, Agent Tasks, ACP Playground diagnostics, admin/ops, and docs.
- [x] #3 Docs link dependencies #1512 retention cleanup, #1513 redacted views, #1529 admin/deployment baseline, and PR #1654 evidence.
- [x] #4 Issue #1537 can be updated with concrete completion evidence and any remaining follow-up split.
- [x] #5 Docs-only verification is recorded.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Audit current ACP execution-health docs and implementation surfaces. Status: Complete.
2. Update PRD/readiness/operator docs with the #1537 closeout contract, surface matrix, and dependency split. Status: Complete.
3. Run docs-only verification and update Backlog/GitHub with closeout evidence. Status: Complete.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Docs updated in ACP PRD, ACP production readiness matrix, and ACP operator guide. Verification: git diff --check passed; targeted rg confirmed #1537 dependencies and no stale 'Backend contract added under' status remains. Bandit skipped because this slice only changes Markdown documentation.

Draft PR #1686 is open for the closeout docs. Issue #1537 was updated with closeout evidence in comment https://github.com/rmusser01/tldw_server/issues/1537#issuecomment-4447890500. Keep the issue open until PR #1686 merges; close manually after merge if GitHub does not auto-close dev-targeted PRs. The PR remains draft until the requester adds the required human-authored Change summary.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Documented the ACP admin execution-health reporting closeout across the product PRD, operator guide, and production readiness matrix. The docs now define the summary endpoint, metric groups, failure buckets, setup-health dimensions, compatibility evidence, retention/redaction posture, product surface split, and dependency boundaries for #1512, #1513, #1529, #1563, and #1564. Verification: git diff --check plus targeted rg checks. Skips: Bandit and backend/frontend tests were skipped because the slice only changes Markdown documentation. Draft PR #1686 is open, and issue #1537 has closeout evidence in comment #4447890500; keep #1537 open until PR #1686 merges and the human Change summary gate is satisfied.
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
