---
id: TASK-12014
title: Address PR 2523 context integrity review feedback
status: In Progress
assignee: []
created_date: ''
updated_date: '2026-06-26 05:03'
labels:
  - security
  - review
  - context-integrity
dependencies: []
references:
  - PR-2523
  - TASK-12017
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Address live CodeRabbit and Qodo review feedback on PR #2523 for context integrity foundation, including valid security/correctness/docstring/test-isolation items and documented technical disposition for non-actionable items.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 All live CodeRabbit context-integrity comments are fixed or dispositioned with technical rationale.
- [ ] #2 All live Qodo context-integrity findings are fixed or dispositioned with technical rationale.
- [ ] #3 Targeted tests and Bandit are run on the touched scope after fixes.
- [ ] #4 PR #2523 branch is updated with review-fix commit(s) and reviewer threads/comments are answered where appropriate.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Fixed valid CodeRabbit items: Unicode-normalized filesystem digest paths with collision detection; accepted mapping-backed signed manifest payloads; froze boot-state findings; detected duplicate live asset IDs as verification errors; rejected unknown startup modes; cleared global resolver before TestClient startup; and filtered integrity-allowed skills before pagination/counting.
- Fixed valid Qodo items: added admin module and manifest helper docstrings; narrowed startup settings exception handling with debug logging; added the missing fixture return type; hardened prompt and skill runtime file reads with no-follow lstat/fstat checks; skipped symlinked SKILL.md during registry sync; and added a short file-fingerprint cache for repeated discovery decisions.
- Backlog hygiene follow-up: renumbered branch-local context tasks from TASK-2363/TASK-2365/TASK-2366 to TASK-12015/TASK-12016/TASK-12017 after the dev rebase exposed ID collisions, updated references, and completed missing AC/DoD fields.
- Reviewed but did not centralize Context Integrity exceptions because this codebase already uses feature-local exceptions in core modules (for example Skills and MCP); local resolver exceptions match the surrounding package boundary. Reviewed but did not remove the global resolver because it is the current compatibility bridge for legacy runtime paths without app/request DI, while SkillsService still supports explicit resolver injection.
- Verification so far: focused 88-test regression run passed; broader focused suite passed with 318 passed and 6 warnings; Bandit on touched app scope exited 0 with zero findings; final quick regression run passed with 8 passed; git diff --check is clean.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
<!-- SECTION:NOTES:END -->

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
