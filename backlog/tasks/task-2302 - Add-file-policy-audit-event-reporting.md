---
id: TASK-2302
title: Add file-policy audit event reporting
status: Done
assignee: []
created_date: ''
updated_date: 2026-06-14 21:26
labels:
- mcp
- policy
- observability
- filesystem
- followup
dependencies: []
references:
- Docs/superpowers/specs/2026-06-07-mcp-fs-patch-write-safe-edit-tools-design.md
documentation:
- 'PR #2360 review follow-up covers CodeRabbit 3418020607 and 3418020616'
- 'PR #2360 review follow-up covers Qodo 3418017882 and 3418017883 and 3418017886'
- Rebased branch from old base 7c775a3e187b15fe79a09a4d51f3cb3f297a9ec4 onto origin/dev
  a6cd8b0f76c90070dff5c0228b91fdde6a7fcad2
modified_files:
- Docs/superpowers/plans/2026-06-16-pr2360-review-rebase-fixes.md
- mcp_unified/tool_use_reporting/models.py
- tldw_Server_API/app/core/MCP_unified/protocol.py
- tldw_Server_API/app/core/MCP_unified/tests/test_tool_use_reporting_models.py
- tldw_Server_API/app/core/MCP_unified/tests/test_tool_use_reporting_protocol.py
- backlog/tasks/task-2302 - Add-file-policy-audit-event-reporting.md
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
['Docs/superpowers/plans/2026-06-14-mcp-file-policy-audit-event-reporting-plan.md', 'Docs/superpowers/plans/2026-06-16-pr2360-review-rebase-fixes.md']
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added PR #2360 review fixes after rebasing onto origin/dev a6cd8b0f76c90070dff5c0228b91fdde6a7fcad2. File-policy paths now reject URI-like values before slash normalization. Protocol extraction now copies at most MAX_FILE_POLICY_DECISIONS entries and derives event grant_outcome across all decisions with denied and not_granted precedence while preserving evaluator metadata precedence. Empty hash and lock containers no longer set presence booleans. Test fakes now include docstrings. Verification: red tests failed for the four behavior bugs before fixes. Focused reporting tests passed with 32 passed. Full tool-use reporting suite passed with 60 passed. Adjacent path-scope and enforcement suite passed with 47 passed. Bandit over touched production paths reported zero findings in /tmp/bandit_pr2360.json.
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
