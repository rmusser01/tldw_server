---
id: TASK-2401
title: Resolve PR 1982 dev-to-main conflicts with dev precedence
status: In Progress
labels:
- merge-conflict
- pr-1982
- dev-main
priority: high
modified_files:
- apps/mcp-unified/pyproject.toml
- Docs/Plans/2026-06-23-pr1982-dev-main-conflict-resolution.md
- tldw_Server_API/app/core/MCP_unified/tests/test_runtime_package_boundary.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Resolve the open PR #1982 (`dev` -> `main`) conflict state by merging current `origin/main` into current `origin/dev` while preserving `dev` for overlapping conflicts, then verify and push the resulting merge back to `dev` if clean.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] Confirm the current PR #1982 conflict surface between `origin/dev` and `origin/main`.
- [x] Merge `origin/main` into the PR head with `dev` winning overlapping conflicts.
- [x] Verify no unresolved merge paths or conflict markers remain before pushing.
- [x] Push the verified merge back to `dev` and confirm PR #1982 merge state/checks update.
- [x] Address current PR #1982 MCP Unified Internal RC package-boundary failure.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Docs/Plans/2026-06-23-pr1982-dev-main-conflict-resolution.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- `git merge-tree --write-tree origin/dev origin/main` identified a single content conflict in `README.md`.
- Merged `origin/main` into the work branch from `origin/dev` with `git merge origin/main -X ours --no-edit`; Git auto-merged `README.md` with the `dev` side winning the overlap.
- Verification before push: `git status --short --branch` showed no unresolved paths; `rg -n '<<<<<<<|=======|>>>>>>>' README.md` returned no matches; `git diff --check HEAD~1 HEAD` exited 0; `git diff --quiet HEAD:README.md origin/dev:README.md` exited 0.
- Pushed the conflict-resolution branch to `dev` at `3e9756f5f2bd7c1d577f439b5780eba386aed801`; GitHub PR #1982 moved from `DIRTY` to `UNSTABLE`, confirming conflicts were cleared and checks were running.
- The current PR #1982 `MCP Unified Internal RC` job failed because the built `mcp-unified` wheel omitted `mcp_unified.policy_grants`; gateway imports then failed with `ModuleNotFoundError: No module named 'mcp_unified.policy_grants'`.
- Added `mcp_unified.policy_grants` to the standalone setuptools package list and extended package-boundary coverage so the wheel/sdist must include that package.
- Verification for the package fix: targeted package-boundary pytest failed before the fix with the missing package assertions, passed after the fix, and `PYTHON=/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python make mcp-unified-rc` exited 0 with `RC status: ok`.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->
