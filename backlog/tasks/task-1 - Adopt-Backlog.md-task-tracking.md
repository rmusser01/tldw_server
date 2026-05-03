---
id: TASK-1
title: Adopt Backlog.md task tracking
status: Done
assignee: []
created_date: '2026-05-03 15:28'
updated_date: '2026-05-03 15:29'
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
- [x] #1 Backlog.md is initialized with repo-local storage
- [x] #2 Backlog.md config keeps auto_commit=false and bypass_git_hooks=false
- [x] #3 Root AGENTS.md requires a Backlog.md task for repo file changes
- [x] #4 Root AGENTS.md preserves superpowers/spec/plan/test/security requirements
- [x] #5 CLI fallback and MCP-first guidance are documented
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

Verification:
- backlog task list --plain: passed
- grep config checks: passed
- grep AGENTS.md checks: passed
- git diff --check HEAD~2..HEAD: passed
- codex mcp list: backlog server registered and enabled
- Bandit: skipped, docs/config/process-only change
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Initialized Backlog.md with repo-local storage, registered the Backlog.md MCP server in Codex, kept auto-commit and hook bypass disabled, added root AGENTS.md instructions requiring Backlog.md tasks for future repo file changes, and verified config plus documentation sanity. remote_operations is disabled to avoid sandbox-blocked fetches during local task operations. Bandit was skipped because the adoption changed only docs/config/process files.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Final summary added
- [x] #5 Known skips or blockers documented
- [x] #6 Bandit run for touched code when applicable or document non-code/environment skip
<!-- DOD:END -->
