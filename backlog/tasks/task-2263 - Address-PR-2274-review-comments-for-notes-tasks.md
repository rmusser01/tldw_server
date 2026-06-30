---
id: TASK-2263
title: Address PR 2274 review comments for notes tasks
status: Done
assignee: []
created_date: ''
updated_date: '2026-06-06 01:49'
labels:
  - notes
  - review
  - pr-2274
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Review and address actionable feedback on PR #2274 after rebasing the clean notes task to-do list branch on latest dev. Keep changes limited to notes to-do list work and Backlog task metadata.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 PR #2274 is rebased on latest dev and remains scoped to notes to-do list work plus Backlog metadata.
- [x] #2 Actionable PR review comments are addressed with regression coverage where behavior changed.
- [x] #3 Backend, frontend, OpenAPI, and Bandit verification results are recorded.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Rebased codex/notes-task-todos-clean-pr on origin/dev; git reported the branch was already up to date. Addressed PR review feedback for CRLF-safe projection deletion, flat i18n keys, fail-closed rate limiting, injected task service dependencies, read-only task list endpoints, accurate stale-note counts, note-scoped activity reads, active-note dock refresh guards, public projection helpers, DELETE query parameters, invalid status validation, raw SQL relocation into the task store, unreachable code cleanup, and Backlog metadata repairs.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
PR #2274 review comments addressed on the clean notes task worktree. Verification: backend focused suite 78 passed; backend broad notes/task/MCP suite 181 passed; UI focused suite 20 passed across 7 files; bun run verify:openapi passed with the repo's 10 reviewed OSS exceptions; Bandit touched backend scope produced zero results and zero errors in /tmp/bandit_notes_tasks_pr2274_review.json. Known skips or blockers: none for the touched scope.
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
