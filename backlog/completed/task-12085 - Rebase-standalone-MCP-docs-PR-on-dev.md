---
id: TASK-12085
title: Rebase standalone MCP docs PR on dev
status: Done
labels:
- mcp
- docs
- git
priority: medium
modified_files:
- apps/mcp-unified/pyproject.toml
- apps/mcp-unified/src/mcp_unified/docs
- pyproject.toml
- tldw_Server_API/app/core/MCP_unified/adapters/__init__.py
- tldw_Server_API/app/core/MCP_unified/adapters/docs
- tldw_Server_API/app/core/MCP_unified/modules/implementations/docs_module.py
- tldw_Server_API/app/core/MCP_unified/tests/test_runtime_package_boundary.py
- tldw_Server_API/tests/MCP_unified/docs
- backlog/tasks/task-12085 - Rebase-standalone-MCP-docs-PR-on-dev.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Rebase PR #2565 for the standalone MCP docs feature onto the latest origin/dev, dropping unrelated interleaved branch commits and resolving any conflicts introduced by the rebase.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Fetch latest origin/dev and inspect PR branch state.
2. Preserve a local backup ref for the pre-rebase branch.
3. Rebase the MCP docs commit range onto origin/dev while dropping unrelated interleaved commits.
4. Resolve conflicts minimally, rerun focused verification, force-push with lease, and retarget the draft PR to dev.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
PR #2565 branch was rebased onto latest origin/dev. The docs package now lives under the current standalone MCP package source tree, standalone package metadata includes the docs subpackages and schema resource, and the runtime/package boundary tests were updated accordingly. Verification passed: docs suite 153 passed; docs plus runtime package boundary 200 passed; focused MCP regression 185 passed, 548 deselected; black check clean; Bandit errors/results empty; git diff --check clean.
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
