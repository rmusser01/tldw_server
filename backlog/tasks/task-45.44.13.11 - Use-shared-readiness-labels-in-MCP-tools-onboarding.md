---
id: TASK-45.44.13.11
title: Use shared readiness labels in MCP tools onboarding
status: Done
labels:
- design-system
- webui
- product-state
references:
- TASK-45.44.13
- https://github.com/rmusser01/tldw_server/issues/1670
- apps/packages/ui/src/design-system/states.ts
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Continue GitHub #1670 with the two canonical-state-label findings in the MCP tools setup step. Read Ready and Unavailable from the shared design-system state registry and preserve all external-status wording and validation behavior.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Use the canonical design-system label for ready and unavailable external tool status.
- [x] #2 Cover both labels in rendered validation feedback and verify the other status cases through the setup tests.
- [x] #3 Clear exactly the two MCP tools findings without altering the verifier baseline.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
No ADR is required. This uses the existing shared labels in the setup UI. The Models row is already covered by the pending Kanban and Scheduled Tasks PR #3671.
Verified against remote dev at 5f9472f9b440fb99e68f67475d2e8219c686c3a0. The #1670 path map went from 26 to 24 blocked findings. The verifier comparison isolated the two MCP tools labels. The repository-wide verifier continues to exit 1 with 95 other blockers and 28 stale entries.
All 112 focused setup, registry, and guard tests passed. Replacing the registry labels in the test proved the two displayed labels come from the shared definitions. The setup tests also cover the Not configured and Incomplete values. Shared UI type checking and scoped ESLint passed. Bandit found zero Python files in the changed scope.
Pull request https://github.com/rmusser01/tldw_server/pull/3679
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
The MCP tools setup step uses the canonical Ready and Unavailable labels in external validation feedback. The two canonical-state-label findings are resolved.
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
