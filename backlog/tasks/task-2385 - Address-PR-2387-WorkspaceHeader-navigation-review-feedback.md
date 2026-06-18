---
id: TASK-2385
title: Address PR 2387 WorkspaceHeader navigation review feedback
status: In Progress
labels:
- workspace
- frontend
- review
priority: High
references:
- https://github.com/rmusser01/tldw_server/pull/2387
- https://github.com/rmusser01/tldw_server/pull/2387#issuecomment-4738026921
- https://github.com/rmusser01/tldw_server/pull/2387#discussion_r3433030173
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Track follow-up work for review comments left on PR #2387 after it merged. Harden WorkspaceHeader ACP history navigation assertions and respond to the backlog filename convention comment.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 WorkspaceHeader ACP history navigation assertions enforce call order and no extra navigation calls.
- [x] #2 The focused WorkspaceHeader test passes.
- [ ] #3 Gemini backlog filename comment is answered with repository convention context.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- PR #2387 was already merged, so this is a follow-up branch from latest `origin/dev` rather than an in-place PR rebase.
- Qodo feedback is valid: `toHaveBeenCalledWith` does not enforce order or call exclusivity.
- Gemini backlog filename feedback conflicts with existing Backlog.md generated filename convention in this repository; respond rather than rename generated Backlog records.
- Verification: `./node_modules/.bin/vitest run ../packages/ui/src/components/Option/ResearchWorkspace/__tests__/WorkspaceHeader.test.tsx` passed with `43 passed`.
- Bandit is not applicable because this branch only changes a frontend test and Backlog task metadata, not production backend code.
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
