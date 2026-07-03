---
id: TASK-12118
title: Prepare MCP Unified 0.2.0 package release
status: In Progress
labels:
- mcp
- packaging
- pypi
- release
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Prepare the standalone mcp-unified package for the next PyPI release after the docs corpus/server mounting work landed on main. Bump the package version, run release-candidate checks, publish through TestPyPI first, smoke install, then publish to PyPI after explicit release confirmation.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 mcp-unified package version is bumped from 0.1.1 to 0.2.0 before any publish attempt
- [x] #2 local MCP Unified RC and publish dry-run checks pass from a clean release worktree
- [ ] #3 TestPyPI publish is triggered only with explicit MCP_UNIFIED_PUBLISH confirmation and smoke-installed successfully
- [ ] #4 PyPI publish is triggered only after TestPyPI smoke succeeds and final confirmation is explicit
- [ ] #5 post-publish PyPI smoke install verifies CLI entry points
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Verified release prep on the 0.2.0 worktree. Version bumped in apps/mcp-unified/pyproject.toml and apps/mcp-unified/src/mcp_unified/__init__.py. Ran package-boundary tests (47 passed), Ruff on touched package init, full make mcp-unified-rc (RC status ok), make mcp-unified-publish-dry-run (RC status ok), and Bandit on touched code (zero findings).
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
