---
id: TASK-2255
title: Design MCP Git read-only inspection tools
status: In Progress
labels:
- mcp
- design
- git
- profiles
modified_files:
- Docs/superpowers/specs/2026-06-04-mcp-git-read-tools-design.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Create the design spec for active-workspace Git read-only MCP tools, including metrics/traces/evaluation requirements for standalone MCP tool-use analysis.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Docs/superpowers/specs/2026-06-04-mcp-git-read-tools-design.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Drafted the approved MCP Git read-only inspection tools design, including the active-repo-only default behavior and observability/evaluation metadata contract for standalone MCP tool-use comparisons.

Manual spec review pass integrated fixes for: external diff/textconv execution risk, ambiguous `git.diff` `head` scope, default author email exposure, strict workspace-relative path handling, and missing `git.conflicts.read` alignment with `_GIT_READ_TOOLS`.
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
