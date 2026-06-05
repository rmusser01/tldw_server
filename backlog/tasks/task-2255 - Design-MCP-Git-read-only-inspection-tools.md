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
references:
- backlog/tasks/task-2256 - Apply-MCP-tool-observability-and-evaluation-contract-across-all-tools.md
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

Second design review integrated fixes for: making observability/evaluation a shared all-tool MCP contract instead of a Git-only pattern, removing ignored-file listing from first-slice `git.status`, removing author email from first-slice Git outputs, requiring machine-readable/NUL-delimited Git output where available, requiring `--` before pathspecs, bounding `git.conflicts.read` by file bytes and hunk count, and creating follow-up TASK-2256 for MCP-wide adoption.

Verification: `git diff --check` passed for the branch diff; ASCII punctuation check passed. Bandit skipped because this task only adds design documentation and Backlog metadata.
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
