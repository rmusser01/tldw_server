---
id: TASK-2399
title: Implement MCP Unified internal RC artifact pipeline
status: In Progress
labels:
- mcp
- packaging
- uat
- release
- implementation
documentation:
- Docs/superpowers/specs/2026-06-22-mcp-unified-internal-rc-artifact-pipeline-design.md
- Docs/superpowers/plans/2026-06-22-mcp-unified-internal-rc-artifact-pipeline-implementation-plan.md
modified_files:
- apps/mcp-unified
- Helper_Scripts/mcp_unified_rc.py
- Helper_Scripts/Testing-related/mcp_standalone_user_guide_uat.py
- tldw_Server_API/app/core/MCP_unified/tests/test_runtime_package_boundary.py
- tldw_Server_API/app/core/MCP_unified/tests/test_mcp_unified_rc_harness.py
- .github/tests/test_mcp_unified_artifact_gate.py
- Makefile
- .github/workflows/pypi-package.yml
- .github/workflows/publish-pypi.yml
- .github/workflows/mcp-unified-rc.yml
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Execute the approved implementation plan for moving the standalone MCP package under apps/mcp-unified, updating artifact boundary tests, adding the private RC harness, Make targets, CI workflow, installed-wheel UAT, and validation/security checks.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Implementation started using subagent-driven development after the design/spec and implementation plan were approved and re-reviewed. Worktree: codex/mcp-unified-internal-rc-spec.

Task 1 worker completed package relocation in commit `92283dc17f`. Focused relocation tests passed: 3 passed, 4 warnings, using the root project virtualenv. The worker added root pytest `pythonpath` for `apps/mcp-unified/src`, which appears necessary because many MCP tests import `mcp_unified` directly after the root package directory is removed. Duplicate worker-created task `TASK-2400` was removed as redundant with `TASK-2399`.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

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
