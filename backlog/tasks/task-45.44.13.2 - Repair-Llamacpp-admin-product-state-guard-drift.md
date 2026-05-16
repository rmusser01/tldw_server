---
id: TASK-45.44.13.2
title: Repair Llamacpp admin product-state guard drift
status: Done
assignee: []
created_date: '2026-05-16 01:28'
updated_date: '2026-05-16 03:24'
labels:
  - design-system
  - ui
  - guard
dependencies: []
references:
  - apps/packages/ui/scripts/design-system-product-state-baseline.json
  - apps/packages/ui/src/components/Option/Admin/LlamacppAdminPage.tsx
  - apps/packages/ui/src/components/Option/Admin/LlamacppInventoryPanel.tsx
  - apps/packages/ui/src/components/Option/Admin/LlamacppLaunchPanel.tsx
  - apps/packages/ui/src/components/Option/Admin/LlamacppReadinessPanel.tsx
parent_task_id: TASK-45.44.13
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Current dev has unbaselined Llamacpp admin AntD Alert product-state findings. Migrate the small Llamacpp admin alert surface to the design-system Alert so the product-state guard is green for the Chatbooks slice.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Llamacpp admin panels no longer import or render AntD Alert for product-state messages.
- [x] #2 Design-system product-state verifier is green after Chatbooks and Llamacpp alert migrations.
- [x] #3 Focused guard or TypeScript verification is recorded.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Migrated the small Llamacpp admin Alert drift from AntD Alert to the design-system Alert primitive across AdminPage, InventoryPanel, LaunchPanel, and ReadinessPanel. Removed stale LlamacppAdminPage baseline entries. Verification: focused Llamacpp admin Vitest files passed; product-state guard test passed; verify:design-system-state passed; git diff --check passed. Full UI tsc remains blocked by pre-existing repo-wide unrelated TypeScript errors outside touched files. Bandit skipped because this task only touches frontend TypeScript/JSON/backlog files.

PR #1738 review fixes: marked non-urgent llama.cpp admin notices as role=status with aria-live=polite while leaving error alerts assertive. Verification: focused Llamacpp admin Vitest files passed; product-state guard test passed; verify:design-system-state passed; git diff --check passed. Full UI tsc remains blocked by pre-existing unrelated errors outside touched paths. Bandit not applicable for frontend TypeScript/backlog-only changes.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Llamacpp admin product-state alerts now use the shared Alert primitive, restoring the design-system guard to green on current dev.
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
