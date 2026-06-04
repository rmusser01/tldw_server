---
id: TASK-2257
title: Plan MCP Git read-only inspection tools implementation
status: Done
labels:
- mcp
- planning
- git
- profiles
references:
- Docs/superpowers/specs/2026-06-04-mcp-git-read-tools-design.md
modified_files:
- Docs/superpowers/plans/2026-06-04-mcp-git-read-tools-implementation-plan.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Create the implementation plan for the approved active-workspace Git read-only MCP tools, including TDD steps, profile metadata updates, docs, verification, and shared observability/evaluation contract hooks.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Drafted the MCP Git read-only inspection tools implementation plan from the approved spec. Local plan review integrated fixes for: real-Git test skips while keeping fake-runner tests active, unambiguous `git.log` field/record separators, exact `eval` response metadata key, isolated Git module registration test environment, ASCII-only docs text, Bandit subprocess guidance, and final verification coverage.

Verification: `git diff --check` passed for the worktree; ASCII punctuation check passed. Bandit skipped because this task only adds an implementation plan and Backlog metadata.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Created and locally reviewed the implementation plan for MCP Git read-only inspection tools. The plan covers Backlog setup, shared tool observability helpers, Git schemas and validation, async Git runner and repo resolution, status/branches/conflicts, diff/log/blame/conflict-read behavior, optional server registration, profile grants, docs, focused tests, adjacent regression tests, Bandit, and final verification. Bandit is not applicable to this planning-only task.
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
