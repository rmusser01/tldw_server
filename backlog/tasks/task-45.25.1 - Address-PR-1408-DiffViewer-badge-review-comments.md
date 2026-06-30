---
id: TASK-45.25.1
title: Address PR 1408 DiffViewer badge review comments
status: Done
assignee: []
created_date: '2026-05-09 06:15'
updated_date: '2026-05-09 06:20'
labels:
  - design-system
  - webui
  - review
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/pull/1408'
  - apps/packages/ui/src/components/Agent/DiffViewer.tsx
  - apps/packages/ui/src/components/Common/StatusBadge.tsx
parent_task_id: TASK-45.25
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Follow-up review-fix task for PR #1408. Resolve the accessibility review comments on Agent DiffViewer file status badge screen-reader labels and address the maintainability feedback about duplicated severity-to-Badge variant mapping where it can be handled within this review slice.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 File status badges expose screen-reader text that describes the file operation rather than the canonical design-system state label
- [x] #2 DiffViewer does not introduce a new local severity-to-Badge variant mapping when a shared helper can provide the mapping
- [x] #3 Focused DiffViewer badge test covers visible variants and accessible file-status labels
- [x] #4 Focused tests, design-system guard/verifier, diff check, and touched-file TypeScript filter are recorded
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Review sweep for PR #1408 found three unresolved inline threads from Gemini, Qodo, and CodeRabbit on the same accessibility bug: DiffViewer FileStatusBadge exposed canonical state labels such as Ready/Error/Empty as Badge srLabel. Qodo also raised a PR-level maintainability issue that DiffViewer introduced another local design-system severity to Badge variant mapping.

Red evidence: bunx vitest run src/components/Agent/__tests__/DiffViewer.file-status-badge.test.tsx --reporter=dot failed because NEW still contained hidden text Ready instead of New file.

Implementation: Added file-operation srLabel values to FILE_STATUS_CONFIG, changed DiffViewer to pass config.srLabel to Badge, added getBadgeVariantForDesignSystemSeverity in components/ui/primitives/badgeUtils.ts, exported it from primitives, and updated both DiffViewer and Common/StatusBadge to use the shared helper instead of local severity mappings.

Verification: bunx vitest run src/components/Agent/__tests__/DiffViewer.file-status-badge.test.tsx src/components/Common/__tests__/StatusBadge.design-system.test.tsx --reporter=dot passed 5/5; bunx vitest run src/design-system/__tests__/product-state-guard.test.ts --reporter=dot passed 46/46; bun run verify:design-system-state passed with baseline exceptions 511 and local-status-badge 5; git diff --check passed. Full bunx tsc --noEmit --pretty false still fails on existing repo-wide frontend baseline errors, but the touched-file TypeScript filter returned no matches after fixing the DiffViewer test helper typing issue.

Bandit: skipped because this review-fix slice changes TypeScript/TSX and Backlog metadata only.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Resolved PR #1408 review feedback by replacing generic design-system srLabel text on DiffViewer file status badges with file-operation labels and by extracting the design-system severity to Badge variant mapping into a shared primitive helper reused by DiffViewer and Common StatusBadge. Focused tests, product-state guard tests, design-system verifier, diff check, and touched-file TypeScript filter passed; full frontend tsc remains blocked by unrelated baseline errors.
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
