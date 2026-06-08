---
id: TASK-2302
title: Add file-policy audit event reporting
status: To Do
labels:
- mcp
- policy
- observability
- filesystem
- followup
references:
- Docs/superpowers/specs/2026-06-07-mcp-fs-patch-write-safe-edit-tools-design.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Extend MCP/tool-use observability with file-policy decision events. Record safe metadata such as requested action, normalized workspace-relative path, grant outcome/source, before/after hash fields when permitted, lock id when applicable, denial reason, and redaction state. Never persist raw file content, raw diffs, receipts, or absolute host paths.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

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
