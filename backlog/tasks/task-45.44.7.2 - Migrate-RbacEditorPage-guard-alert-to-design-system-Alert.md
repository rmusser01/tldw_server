---
id: TASK-45.44.7.2
title: Migrate RbacEditorPage guard alert to design-system Alert
status: Done
labels:
- design-system
- webui
- admin
- product-state
priority: medium
parent_task_id: TASK-45.44.7
references:
- apps/packages/ui/src/components/Option/Admin/RbacEditorPage.tsx
- apps/packages/ui/scripts/design-system-product-state-baseline.json
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Continue the Admin and health expansion design-system migration by replacing RbacEditorPage's remaining AntD Alert guard state with the shared design-system Alert primitive while preserving the existing access-restricted copy.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 RbacEditorPage renders admin guard feedback through the shared design-system Alert primitive instead of AntD Alert.
- [x] #2 Focused coverage asserts the access-restricted guard feedback uses the design-system Alert marker and preserves existing copy.
- [x] #3 The RbacEditorPage AntD Alert baseline exception is removed without introducing new product-state guard findings for the component.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Red verification: `bunx vitest run src/components/Option/Admin/__tests__/RbacEditorPage.design-system.test.tsx` failed because the access-restricted guard had no `data-ds-component="Alert"` ancestor.
- Replaced the RbacEditorPage guard AntD Alert with the shared design-system Alert primitive and preserved the existing `Access Restricted` title plus guard detail text.
- Removed the `RbacEditorPage` AntD Alert baseline exception from `apps/packages/ui/scripts/design-system-product-state-baseline.json`.
- Verification: focused RbacEditorPage design-system Vitest passed; product-state guard unit test passed (54 tests); baseline JSON parsed; `git diff --check` passed; scoped verifier log at `/tmp/rbac-editor-design-state.log` has no `RbacEditorPage` findings and reports baseline exceptions at 189.
- Full `bun run verify:design-system-state` still exits 1 on current-dev drift outside this slice: IntegrationPolicyPanel, WritingActionBar, Notes, ResearchWorkspace, plus stale IntegrationPolicyPanel baseline entries.
- Bandit skipped because this slice only touches frontend TypeScript/TSX, JSON baseline, and Backlog metadata.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Migrated RbacEditorPage admin guard feedback from AntD Alert to the shared design-system Alert primitive, added focused regression coverage for the design-system Alert marker and existing copy, and removed the matching RbacEditorPage product-state baseline exception. Focused component coverage, product-state guard unit coverage, baseline JSON parsing, whitespace checks, and scoped verifier inspection were recorded. The full product-state verifier remains blocked by unrelated current-dev drift, with no RbacEditorPage findings.
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
