---
id: TASK-2385
title: Address PR 2387 WorkspaceHeader navigation review feedback
status: Done
labels:
- workspace
- frontend
- review
priority: High
references:
- https://github.com/rmusser01/tldw_server/pull/2387
- https://github.com/rmusser01/tldw_server/pull/2387#issuecomment-4738026921
- https://github.com/rmusser01/tldw_server/pull/2387#discussion_r3433030173
- https://github.com/rmusser01/tldw_server/pull/2389
- https://github.com/rmusser01/tldw_server/pull/2387#issuecomment-4742827932
- https://github.com/rmusser01/tldw_server/pull/2387#discussion_r3436459284
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Track follow-up work for review comments left on PR #2387 after it merged. Harden WorkspaceHeader ACP history navigation assertions and respond to the backlog filename convention comment.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 WorkspaceHeader ACP history navigation assertions enforce call order and no extra navigation calls.
- [x] #2 The focused WorkspaceHeader test passes.
- [x] #3 Gemini backlog filename comment is answered with repository convention context.
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
- Hardened the ACP history navigation assertions in `WorkspaceHeader.test.tsx` by checking call order and total call count.
- Opened follow-up PR #2389 because PR #2387 was already merged before review follow-up work began.
- Replied to Qodo on #2387 with the follow-up PR and verification results.
- Replied to the Gemini inline comment explaining that Backlog.md generated task filenames intentionally follow this repository's `task-<id> - <Title>.md` convention.
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
