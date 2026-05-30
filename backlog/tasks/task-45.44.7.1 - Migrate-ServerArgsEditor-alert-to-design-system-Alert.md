---
id: TASK-45.44.7.1
title: Migrate ServerArgsEditor alert to design-system Alert
status: Done
assignee: []
created_date: '2026-05-30 15:49'
labels:
  - design-system
  - webui
  - admin
  - product-state
dependencies: []
references:
  - apps/packages/ui/src/components/Option/Admin/ServerArgsEditor.tsx
  - apps/packages/ui/scripts/design-system-product-state-baseline.json
parent_task_id: TASK-45.44.7
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Continue the Admin and health expansion design-system migration by replacing ServerArgsEditor's remaining AntD Alert product-state usage with the shared design-system Alert primitive while preserving the existing JSON validation copy.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 ServerArgsEditor renders JSON validation feedback through the shared design-system Alert primitive instead of AntD Alert.
- [x] #2 Focused coverage asserts the validation feedback uses the design-system Alert marker and preserves the existing copy.
- [x] #3 The ServerArgsEditor AntD Alert baseline exception is removed without introducing new product-state guard findings for the component.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Red verification: `bunx vitest run src/components/Option/Admin/__tests__/ServerArgsEditor.design-system.test.tsx` failed because the JSON validation feedback had no `data-ds-component="Alert"` ancestor.
- Replaced the JSON validation AntD Alert with the shared design-system Alert primitive and preserved the existing `Invalid JSON` / `Must be a JSON object` title copy.
- Removed the `ServerArgsEditor` AntD Alert baseline exception from `apps/packages/ui/scripts/design-system-product-state-baseline.json`.
- Verification: focused ServerArgsEditor design-system Vitest passed; product-state guard unit test passed (54 tests); baseline JSON parsed; scoped verifier log at `/tmp/server-args-editor-design-state.log` has no `ServerArgsEditor` findings and reports baseline exceptions at 190.
- Full `bun run verify:design-system-state` still exits 1 on current-dev drift outside this slice: IntegrationPolicyPanel, WritingActionBar, Notes, ResearchWorkspace, plus stale IntegrationPolicyPanel baseline entries.
- Bandit skipped because this slice only touches frontend TypeScript/TSX, JSON baseline, and Backlog metadata.
- Review fix: Qodo flagged the focused test's generic `switch` / `textbox` selectors as brittle. Added an accessible name to the JSON mode switch and tightened the test to query the named switch plus the JSON editor placeholder. Red verification failed on the unnamed switch, then focused Vitest passed after the component fix.
- Review-fix verification: focused ServerArgsEditor design-system Vitest passed; product-state guard unit test passed (54 tests); baseline JSON parsed; `git diff --check` passed; scoped verifier log still has no `ServerArgsEditor` findings.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Migrated ServerArgsEditor JSON validation feedback from AntD Alert to the shared design-system Alert primitive, added focused coverage for the design-system Alert marker and preserved copy, removed the matching ServerArgsEditor product-state baseline exception, and tightened the review-fix selectors by naming the JSON mode switch and querying the JSON editor directly. Focused component coverage, product-state guard unit coverage, baseline JSON parsing, whitespace checks, and scoped verifier inspection were recorded. The full product-state verifier remains blocked by unrelated current-dev drift, with no ServerArgsEditor findings.
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
