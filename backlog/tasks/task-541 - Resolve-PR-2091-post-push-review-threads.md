---
id: TASK-541
title: Resolve PR 2091 post-push review threads
status: Done
assignee: []
created_date: ''
updated_date: '2026-05-28 19:43'
labels:
  - research-workspace
  - review-fix
  - workspaces
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/pull/2091'
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Re-check PR #2091 after the rebased review-fix push, make any remaining minimal repo edits, resolve fixed review threads, and record verification.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 PR #2091 is confirmed rebased on latest origin/dev.
- [x] #2 All unresolved review threads are verified against current code and either minimally patched or resolved as already fixed.
- [x] #3 Focused verification is recorded after repo edits.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Fetched and rebased codex/workspaces-next; first rebase was already up to date, then origin/dev advanced while resolving threads so this follow-up will be rebased again before push.

Verified the live PR review inventory. The substantive comments were already addressed in 0c54ad8a7: i18n strings, malformed bundle guard, Redis precedence and close logging, sandbox diagnostics async/threading/source_label/cancellation/schema docs, migration tombstone sequencing, Agent Tasks stale request guard, ACP mcpServers fallback, portable TASK-478 command, and dependency override cleanup.

Made one minimal follow-up edit for the remaining still-current test-title comment: renamed the local tombstone preflight failure test to `fails safely when tombstone preflight write throws before local deletion`.

Resolved all 31 unresolved PR review threads after verifying current code against each category.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Rechecked PR #2091 after rebasing on latest origin/dev. Verified all 31 unresolved review threads against current code, made the remaining test-title clarity edit, and resolved the threads on GitHub. Follow-up verification: `bunx vitest run src/store/__tests__/workspace-migration.test.ts --reporter=dot` -> 1 file and 12 tests passed; `git diff --check` -> passed. Bandit skipped for this follow-up because only a frontend test title and Backlog task file changed. Known unrelated files left untouched: the two untracked watchlist template files under tldw_Server_API/Config_Files/templates/watchlists/.
<!-- SECTION:FINAL_SUMMARY:END -->

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
