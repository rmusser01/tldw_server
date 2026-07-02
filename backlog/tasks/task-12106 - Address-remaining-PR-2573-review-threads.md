---
id: TASK-12106
title: Address remaining PR 2573 review threads
status: Done
labels:
- webui
- chat
- code-review
priority: High
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Resolve still-open GitHub review feedback on PR #2573 after the placeholder-auth review fixes, including chat stale-prop attribution, chat-settings 404 matching, avatar data-url handling feedback if valid, extension env opt-out behavior, and Backlog marker cleanup.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Open review threads are verified against current code and valid issues are fixed.
- [x] #2 Regression tests cover behavioral fixes where practical.
- [x] #3 Focused UI/backend verification, diff check, and Bandit status are recorded.
- [x] #4 PR branch is committed and pushed after fixes.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Verified the unresolved PR #2573 review threads against current code. Fixed valid issues in greeting persistence feedback, collapsed composer advanced controls, mobile status live-region accessibility, stale character-chat id reuse, remote chat-settings 404 matching, mixed base64 alphabet validation, extension-safe quickstart env lookup, frontend env-auth opt-out on credential clear, Character.metadata typing, and duplicate Backlog marker cleanup. The placeholder-auth thread was already addressed by the prior commit.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Verification: shared UI focused suite passed (13 files, 204 tests); frontend focused suite passed (3 files, 44 tests); backend provider/capability suite passed (22 tests); deprecated AntD prop scan found no non-test UI matches; git diff --check passed; bad metadata-cast/stale-assignment source scan found no matches. Bandit was not rerun for TASK-12106 because this slice did not touch Python code; the prior placeholder-provider Python change was already checked with Bandit and had zero findings.
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
