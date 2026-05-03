---
id: TASK-1
title: Adopt Backlog.md task tracking
status: In Progress
assignee: []
created_date: '2026-05-03 15:28'
updated_date: '2026-05-03 15:28'
labels: []
dependencies: []
documentation:
  - Docs/superpowers/specs/2026-05-03-backlog-md-task-tracking-design.md
  - >-
    Docs/superpowers/plans/2026-05-03-backlog-md-task-tracking-implementation-plan.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Initialize Backlog.md in this repository and add root AGENTS.md instructions requiring Backlog.md tasks for future repo file changes.

This task begins after the approved spec and first setup commit, which are the agreed bootstrap exception.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Backlog.md is initialized with repo-local storage
- [ ] #2 Backlog.md config keeps auto_commit=false and bypass_git_hooks=false
- [ ] #3 Root AGENTS.md requires a Backlog.md task for repo file changes
- [ ] #4 Root AGENTS.md preserves superpowers/spec/plan/test/security requirements
- [ ] #5 CLI fallback and MCP-first guidance are documented
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Initialize Backlog.md with project-local storage and MCP-first setup.
2. Configure Definition of Done defaults without enabling auto-commit or hook bypass.
3. Add root AGENTS.md Backlog.md task-tracking instructions.
4. Verify Backlog.md config, AGENTS.md wording, and diff sanity.
5. Record final summary and mark the adoption task done.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Initialized Backlog.md and added the root AGENTS.md task-tracking policy. Backlog.md auto-commit and hook bypass remain disabled. MCP was registered with Codex after non-interactive init skipped client setup. remote_operations was disabled to keep local task operations from attempting sandbox-blocked fetches.
<!-- SECTION:NOTES:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Final summary added
- [ ] #5 Known skips or blockers documented
- [ ] #6 Bandit run for touched code when applicable or document non-code/environment skip
<!-- DOD:END -->
