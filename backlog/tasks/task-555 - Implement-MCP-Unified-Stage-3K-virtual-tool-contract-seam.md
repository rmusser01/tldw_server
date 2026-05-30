---
id: TASK-555
title: Implement MCP Unified Stage 3K virtual tool contract seam
status: Done
assignee:
  - codex
created_date: ''
updated_date: '2026-05-30 02:25'
labels:
  - mcp-unified
  - standalone-extraction
  - stage-3
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
<!-- SECTION:DESCRIPTION:BEGIN -->
<!-- SECTION:DESCRIPTION:END -->

<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Host external server manager reuses the package-owned VirtualExternalTool contract instead of defining a duplicate dataclass.
- [x] #2 Package VirtualExternalTool exposes caller-owned copy behavior for nested input_schema and metadata.
- [x] #3 Focused package-boundary and external federation tests cover contract identity and copy isolation.
- [x] #4 Existing host external server manager behavior remains compatible for discovered tool listing and execution.
- [x] #5 Focused pytest, Ruff, Bandit, and git diff --check verification are recorded before PR closeout.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Use TDD. First add failing package-boundary tests proving host VirtualExternalTool is not package-owned and package copy isolation is absent. Then move host manager to import the package contract, add copy support to the package model, preserve host exports, and run focused MCP federation/manager tests plus lint/security checks.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented Stage 3K with a narrow contract seam. Host external server manager now imports mcp_unified.federation.models.VirtualExternalTool instead of defining a duplicate dataclass. Package VirtualExternalTool now exposes copy() with caller-owned nested input_schema and metadata. Added package-boundary regression tests for host/package identity and virtual-tool copy isolation.

Verification: focused MCP suite 37 passed, 2 skipped, 3 warnings; Ruff all checks passed; Bandit 0 findings on touched implementation files; git diff --check clean.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Stage 3K complete. Host external federation now reuses the standalone package VirtualExternalTool contract, the package contract supports caller-owned copies, and focused MCP package-boundary/external federation tests plus Ruff, Bandit, and git diff --check verification pass.
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
