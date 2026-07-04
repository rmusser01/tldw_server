---
id: TASK-12131
title: Design first-run MCP tool packs setup
status: Done
assignee: []
created_date: ''
updated_date: '2026-07-04 23:08'
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
<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Spec authored and reviewed for first-run MCP tool packs setup.

Spec path: Docs/superpowers/specs/2026-07-04-first-run-mcp-tool-packs-design.md

Review iterations:
- Pass 1 found missing completion acceptance for not_run, incomplete pack/add-on IDs, and undefined safe external tool metadata.
- Pass 2 found external validation/add-on ambiguity, missing high-risk confirmation persistence, and no state for external discovery timeout.
- Pass 3 approved with no blocking issues.

Final design highlights:
- Optional first-run `mcp_tools` step with five default read-only packs.
- Default excludes local file reads and other high-risk capabilities.
- Built-in validation is deterministic and does not require user data.
- External validation is a setup-only readiness check, separate from ongoing profile permissions.
- Generated `First-run default` MCP Hub profile uses stable metadata and manual-edit hash protection.
- Strong add-ons require server-enforced confirmation fields.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Completed the approved first-run MCP tool packs design spec and review loop. Verification: spec-review subagent approved on the third pass; scoped git diff --cached --check passed for the spec and Backlog task files. Bandit was not run because this task only changed documentation and Backlog metadata, not backend code.
<!-- SECTION:FINAL_SUMMARY:END -->

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
