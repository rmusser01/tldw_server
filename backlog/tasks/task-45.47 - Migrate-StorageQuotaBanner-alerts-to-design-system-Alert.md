---
id: TASK-45.47
title: Migrate StorageQuotaBanner alerts to design-system Alert
status: Done
assignee: []
created_date: '2026-05-15 19:25'
updated_date: '2026-05-15 19:36'
labels:
  - design-system
  - webui
  - product-state
dependencies: []
references:
  - apps/packages/ui/src/components/Common/StorageQuotaBanner.tsx
  - apps/packages/ui/src/components/Common/__tests__/StorageQuotaBanner.test.tsx
  - apps/packages/ui/scripts/design-system-product-state-baseline.json
  - https://github.com/rmusser01/tldw_server/pull/1728
documentation:
  - Docs/Design/tldw_web_design_system_inventory.md
parent_task_id: TASK-45
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Continue the tldw_server WebUI design-system migration by replacing StorageQuotaBanner's remaining AntD Alert product-state usage with the shared design-system Alert primitive while preserving warning dismissal and exceeded-quota behavior.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 StorageQuotaBanner renders warning and exceeded quota banners through the shared design-system Alert primitive instead of AntD Alert.
- [x] #2 Existing warning dismissal and exceeded non-dismissable behavior remain covered by focused tests.
- [x] #3 The design-system product-state baseline no longer contains StorageQuotaBanner AntD Alert exceptions.
- [x] #4 Focused tests and design-system verifier results are recorded before completion.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implementation completed on branch codex/design-system-next-slice-7 in the dedicated design-system worktree.

StorageQuotaBanner now imports the shared design-system Alert primitive and maps exceeded quota to variant="error" and warning quota to variant="warning" with dismissible/onDismiss behavior.

Focused coverage asserts warning and exceeded banners render the design-system Alert marker while preserving session dismissal and exceeded non-dismissable behavior.

Removed the two StorageQuotaBanner AntD Alert exceptions from apps/packages/ui/scripts/design-system-product-state-baseline.json.

PR opened against dev: https://github.com/rmusser01/tldw_server/pull/1728.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Migrated StorageQuotaBanner warning and exceeded quota banners from AntD Alert to the shared design-system Alert primitive, added focused assertions that both states render the design-system primitive, and removed the StorageQuotaBanner AntD Alert baseline exceptions. Verification recorded: red test failed on missing design-system Alert marker, focused StorageQuotaBanner tests passed after implementation, combined StorageQuotaBanner/product-state guard suite passed, design-system verifier passed with 477 remaining AntD product-state baseline exceptions, baseline JSON parse passed, git diff --check passed, and package TypeScript still has existing unrelated diagnostics with no touched-file matches. Bandit is not applicable because this slice touched TypeScript, test, and JSON frontend files only.
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
