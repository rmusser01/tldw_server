---
id: TASK-45.44.13.3
title: Repair LlamacppRuntimePanel product-state guard drift
status: Done
assignee:
- Codex
labels:
- design-system
- webui
- product-state
- guard
priority: medium
parent_task_id: TASK-45.44.13
references:
- apps/packages/ui/src/components/Option/Admin/LlamacppRuntimePanel.tsx
- apps/packages/ui/scripts/design-system-product-state-baseline.json
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Current origin/dev has an unbaselined AntD Alert product-state finding in LlamacppRuntimePanel. Replace that warning alert with the shared design-system Alert primitive so the product-state verifier is green for the next design-system migration slice.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 LlamacppRuntimePanel no longer imports or renders AntD Alert for the error notice.
- [x] #2 The warning notice uses the shared design-system Alert primitive with non-urgent status semantics.
- [x] #3 The design-system product-state verifier no longer reports LlamacppRuntimePanel as a blocked finding.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Import the shared Alert primitive under a non-conflicting alias and remove AntD Alert from the component imports.
2. Replace the LlamacppRuntimePanel error notice with the shared Alert primitive using variant=warning, role=status, and aria-live=polite.
3. Run the design-system verifier together with the WatchlistsHealthBar focused checks.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Migrated the new LlamacppRuntimePanel runtime inventory error notice from AntD Alert to the shared design-system Alert primitive with variant="warning", role="status", and aria-live="polite". Added focused component coverage proving the error notice renders through the design-system Alert primitive.

Verification: bun run verify:design-system-state first failed on the unbaselined LlamacppRuntimePanel AntD Alert finding, then passed after the migration. The focused LlamacppRuntimePanel/WatchlistsHealthBar/product-state guard suite passed 56/56; git diff --check passed; section marker sanity check passed. Full UI TypeScript still exits 2 on existing repo-wide type debt outside touched files, with no touched-file errors observed in the output. Bandit is not applicable because this repair touches frontend TypeScript and Backlog metadata only.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Repaired the current origin/dev product-state guard drift by replacing LlamacppRuntimePanel's runtime inventory warning notice with the shared design-system Alert primitive and adding focused coverage for the design-system Alert marker.
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
