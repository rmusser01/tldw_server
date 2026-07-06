---
id: TASK-12879
title: Design first-run MCP tool packs setup
status: Done
assignee: []
created_date: ''
updated_date: '2026-07-04 23:14'
labels:
  - design
  - mcp
  - setup
  - first-run
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Write the approved design spec for adding an optional first-run MCP Tool Packs step that seeds a visible First-run default MCP Hub profile, validates safe sample tools, and links users to MCP Hub for external server setup.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Design captures outcome packs, default low-risk read-only behavior, external server validation behavior, profile upsert/conflict rules, frontend flow, error handling, tests, and rollout constraints.
- [x] #2 Spec is saved under Docs/superpowers/specs/ with the current date and reviewed per brainstorming workflow.
- [x] #3 Backlog task records touched files, verification, and final summary.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Spec authored and reviewed for first-run MCP tool packs setup.

Spec path: Docs/superpowers/specs/2026-07-04-first-run-mcp-tool-packs-design.md

Review iterations:
- Pass 1 found missing completion acceptance for not_run, incomplete pack/add-on IDs, and undefined safe external tool metadata.
- Pass 2 found external validation/add-on ambiguity, missing high-risk confirmation persistence, and no state for external discovery timeout.
- Pass 3 approved with no blocking issues.

Follow-up review patch before implementation planning:
- Changed persisted profile_id and assignment_id to string/null to match MCP profile storage.
- Clarified default first-run profile assignment as the visible default assignment for single-user v1.
- Tightened default packs to current registered read-only tool names where known.
- Added a guard that mixed-risk modules must use explicit read-only tool allowlists or narrow tool_patterns, not module-wide grants.

Touched files:
- Docs/superpowers/specs/2026-07-04-first-run-mcp-tool-packs-design.md
- backlog/tasks/task-12131 - Design-first-run-MCP-tool-packs-setup.md
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Completed the approved first-run MCP tool packs design spec, review loop, and follow-up spec tightening. Verification: spec-review subagent approved on the third pass; follow-up patch passed scoped git diff --check. Bandit was not run because this task only changed documentation and Backlog metadata, not backend code.
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
