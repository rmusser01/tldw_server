---
id: TASK-2315
title: Fix Research Workspace capabilities test type fixture
status: Done
modified_files:
- apps/packages/ui/src/components/Option/ResearchWorkspace/__tests__/WorkspaceCapabilityRemediation.test.tsx
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Post-rebase UI package typecheck fails because WorkspaceCapabilityRemediation.test.tsx builds a WorkspaceCapabilitiesResponse fixture without the now-required workspace_profile field.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 UI package typecheck no longer fails on WorkspaceCapabilityRemediation.test.tsx workspace_profile typing.
- [x] #2 Research Workspace remediation component tests still pass.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Added the required `workspace_profile: "research"` field to the WorkspaceCapabilitiesResponse test fixture so the helper matches the current API contract. Bandit is not applicable for this TypeScript-only test fixture change.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Fixed the post-rebase UI package typecheck failure in `WorkspaceCapabilityRemediation.test.tsx` by completing the typed capabilities fixture. The focused Research Workspace test passes, and the UI package TypeScript check now exits successfully.
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
