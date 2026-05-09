---
id: TASK-145.10
title: Address Qodo PR review findings for embeddings apply
status: In Progress
assignee: []
created_date: '2026-05-09 17:34'
updated_date: '2026-05-09 17:38'
labels:
  - evals
  - backend
  - review-fix
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/pull/1421#discussion_r3213492997'
  - 'https://github.com/rmusser01/tldw_server/pull/1421#discussion_r3213493000'
  - 'https://github.com/rmusser01/tldw_server/pull/1421#discussion_r3213493002'
parent_task_id: TASK-145
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Fix Qodo review findings on PR #1421: visible policy-helper fallback handling, narrow config updater injection for live apply, and structured HTTP errors for file/permission/OS failures.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Policy helper import/runtime fallback is no longer silent
- [x] #2 Live apply helper supports injected config update behavior instead of hard-coding only the setup_manager singleton
- [x] #3 Apply endpoint converts FileNotFoundError/PermissionError/OSError apply failures into sanitized HTTP 500 responses
- [ ] #4 Qodo review threads are replied to or resolved after push
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Inspect existing helper/endpoint tests, add focused regressions for the three review findings, implement the narrow backend changes, run focused backend tests and hygiene checks, update task, commit, push, and reply/resolve review threads.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented Qodo review fixes for the embeddings live-apply backend path. Policy helper fallback now logs a warning when import/runtime helper loading fails instead of silently returning permissive defaults. apply_embedding_recipe_recommendation now receives a config_updater callable, and the FastAPI endpoint exposes get_recipe_config_updater as a dependency returning setup_manager.update_config, so tests and callers can override config mutation behavior without patching the helper module singleton. The apply endpoint now catches OSError subclasses from config writes, logs them, and returns sanitized HTTP 500 details through sanitize_error_message.

Verification: red test run first failed on missing get_recipe_config_updater import before production changes. After implementation, targeted regressions passed 3 tests with 5 warnings; focused backend eval recipe tests passed 41 tests with 5 warnings. Bandit on touched backend source wrote /tmp/bandit_embeddings_qodo_review.json with results 0/errors 0/skipped 0. git diff --check passed.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Addressed the Qodo backend review findings by making embedding policy fallback visible in logs, injecting the live apply config updater through the endpoint dependency boundary, and sanitizing file/permission/OS config-write failures into structured 500 responses. Added focused unit/integration regressions for the new behaviors and recorded backend verification plus Bandit results.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
