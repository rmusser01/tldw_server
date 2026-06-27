---
id: TASK-2282
title: Add NotebookEdit-style notebook file tools
status: In Progress
labels:
- mcp
- filesystem
- notebooks
- tools
- agentic-execution
references:
- https://code.claude.com/docs/en/tools-reference
documentation:
- Docs/Design/2026-06-27-mcp-notebook-edit-tools-design.md
modified_files:
- Docs/Design/2026-06-27-mcp-notebook-edit-tools-design.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Design and implement notebook-safe MCP tools modeled after Claude Code NotebookEdit. Support reading notebook structure and editing cells by cell id with replace, insert, and delete modes; require notebook path grants; preserve JSON validity; avoid raw whole-notebook overwrites for cell edits; include validation, diff summaries, redacted telemetry, and tests for Jupyter notebooks.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Design approved with refinements: keep notebook parsing/editing in a focused helper, expose NotebookEdit-style read/edit tools through the MCP filesystem module, require path grants and preimage checks, clear stale code-cell outputs by default, reject missing/duplicate target cell ids, preserve notebook JSON shape where practical, keep source reads bounded and intentional, treat cell deletion as file-policy edit, and keep telemetry/error payloads redaction-safe.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

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
