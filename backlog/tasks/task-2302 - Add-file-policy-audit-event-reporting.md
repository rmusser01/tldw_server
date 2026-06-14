---
id: TASK-2302
title: Add file-policy audit event reporting
status: Done
assignee: []
created_date: ''
updated_date: '2026-06-14 21:26'
labels:
  - mcp
  - policy
  - observability
  - filesystem
  - followup
dependencies: []
references:
  - >-
    Docs/superpowers/specs/2026-06-07-mcp-fs-patch-write-safe-edit-tools-design.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Extend MCP/tool-use observability with file-policy decision events. Record safe metadata such as requested action, normalized workspace-relative path, grant outcome/source, before/after hash fields when permitted, lock id when applicable, denial reason, and redaction state. Never persist raw file content, raw diffs, receipts, or absolute host paths.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
['Docs/superpowers/plans/2026-06-14-mcp-file-policy-audit-event-reporting-plan.md']
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added metadata-only file-policy audit reporting for MCP tool-use events. Tool-use events now carry sanitized path decision metadata, grant outcomes, and boolean hash/lock-lease presence flags while rejecting absolute paths and sensitive payload values. Protocol capture now carries redacted scope payloads through prepared calls and prepare-time governance denials. Verification: reporting suite 42 passed; adjacent path/filesystem suite 112 passed; production Bandit on touched production files passed. Bandit over pytest files was intentionally not used as a completion gate because it reports normal B101 assert_used test noise.
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
