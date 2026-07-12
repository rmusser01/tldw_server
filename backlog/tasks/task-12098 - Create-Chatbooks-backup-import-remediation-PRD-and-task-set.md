---
id: TASK-12098
title: Create Chatbooks backup import remediation PRD and task set
status: In Progress
assignee: []
created_date: ''
updated_date: '2026-07-12 13:27'
labels:
  - chatbooks
  - prd
  - ux
  - uat
  - backup
  - import
dependencies: []
references:
  - Docs/Reviews/CHATBOOKS_BACKUP_IMPORT_UAT_UX_REVIEW_2026_07_09.md
  - >-
    Docs/superpowers/plans/2026-07-09-chatbooks-full-account-backup-import-implementation-plan.md
  - Docs/Reviews/CHATBOOKS_POST_MERGE_UAT_UX_REVIEW_2026_07_09.md
  - >-
    Docs/superpowers/plans/2026-07-09-chatbooks-post-merge-uat-remediation-plan.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Create an umbrella PRD/spec and milestone Backlog tasks for addressing all findings from the 2026-07-09 Chatbooks backup/import UX UAT review. Scope covers P0 backup/restore correctness, P1 UX clarity, and P2 acceptance coverage, with precise all-export selection semantics and extension/WebUI parity requirements.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 PRD/spec exists at Docs/superpowers/specs/2026-07-09-chatbooks-backup-import-remediation-prd-design.md and covers all P0/P1/P2 findings from the 2026-07-09 review.
- [ ] #2 Milestone Backlog tasks exist for P0 correctness, P1 UX clarity, and P2 acceptance coverage.
- [ ] #3 Spec review records no blocking issues before the user review gate.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Docs/superpowers/specs/2026-07-09-chatbooks-backup-import-remediation-prd-design.md
Docs/superpowers/plans/2026-07-09-chatbooks-full-account-backup-import-implementation-plan.md
Docs/superpowers/plans/2026-07-09-chatbooks-post-merge-uat-remediation-plan.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Spec review pass recorded after PRD/task package verification. Implementation plan created at Docs/superpowers/plans/2026-07-09-chatbooks-full-account-backup-import-implementation-plan.md.

PR #2714 review remediation started 2026-07-12. Branch is already based on current origin/dev. Scope includes all verified inline and summary-level reviewer findings, focused regression coverage, CI triage, reviewer replies, and thread resolution. Review-fix plan: Docs/superpowers/plans/2026-07-12-pr-2714-review-remediation-plan.md.

2026-07-12 review remediation verification: 144 consolidated backend/UAT tests passed; 199 consolidated WebUI/shared/extension tests passed; 29 post-format backend tests passed; extension TypeScript compile and touched Python compilation passed; Bandit reported zero findings across all touched production Python paths; git diff --check passed. Full WebUI typecheck retains only the documented unchanged QuickIngestWizardModal.tsx:1813 baseline. Broad legacy-file Ruff debt remains unchanged; new touched test import-order findings were fixed. origin/dev remains an ancestor of the PR branch after the final fetch. Independent final review and GitHub thread closure are pending.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
