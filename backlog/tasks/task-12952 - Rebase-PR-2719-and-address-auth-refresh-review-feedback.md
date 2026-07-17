---
id: TASK-12952
title: Rebase PR 2719 and address auth-refresh review feedback
status: Done
labels:
- auth
- webui
- browser-extension
- review
priority: high
references:
- https://github.com/rmusser01/tldw_server/pull/2719
- TASK-12953
documentation:
- Docs/superpowers/specs/2026-07-12-legacy-api-key-refresh-migration-design.md
modified_files:
- apps/packages/ui/src/services/__tests__/tldw-api-client.quickstart-auth.test.ts
- apps/tldw-frontend/e2e/extension-api-key-persistence.spec.ts
- apps/tldw-frontend/e2e/helpers/manual-api-key-fixture.ts
- apps/tldw-frontend/e2e/manual-api-key-persistence.spec.ts
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Rebase PR #2719 onto latest dev, evaluate and address every actionable review comment, verify the shared WebUI/browser-extension auth-refresh fix, resolve review threads, and update the PR branch.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 PR #2719 is rebased onto the latest origin/dev without unresolved conflicts.
- [x] #2 Every actionable inline and out-of-diff review comment is either fixed or answered with verified technical reasoning.
- [x] #3 Focused unit, browser, build, compile, diff, and applicable security checks pass.
- [x] #4 The rebased branch is pushed and all addressed review threads are resolved.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Post-rebase verification on 2026-07-12 at HEAD `4b741152bf9caab302856c52a286134762ce6658`:
- Baseline: worktree clean before verification; local `origin/dev` and `git merge-base HEAD origin/dev` both `5634ea4a04ebcb6322a218469d6ff972b2435450`.
- Shared auth Vitest matrix: 3 files, 50/50 passed.
- `NEXT_PUBLIC_X_API_KEY=ambient-test-key` isolation: 22/22 passed after the unchanged rerun was allowed to create Vitest temp files.
- `VITE_TLDW_API_KEY=ambient-test-key` isolation: 22/22 passed after the unchanged rerun was allowed to create Vitest temp files.
- Advanced WebUI persistence Playwright with fixture-backed readiness on API port 19041/web port 18084: 3/3 passed in 46.5s.
- Advanced Chrome production extension build: exit 0 in 33.8s; token sync OK.
- Packaged extension persistence Playwright: 3/3 passed in 42.2s.
- Extension `bun run compile`: exit 0, no diagnostics.
- Frontend `bun run typecheck`: sandboxed attempt exited 2 only because `tsconfig.tsbuildinfo` could not be written (TS5033 EPERM); the unchanged escalated rerun exited 0 with no TypeScript diagnostics. The previously recorded untouched QuickIngestWizardModal TS2322 baseline is not present post-rebase.
- Touched-scope ESLint from `apps/tldw-frontend` using its installed `eslint.config.mjs`: exit 0, 0 errors, 2 warnings. Both `apps/packages/ui` files were ignored because they are outside the config base path; the three frontend E2E files were accepted with no diagnostics.
- `git diff --check origin/dev...HEAD` and `git diff --check`: exit 0.
- Post-command `git status --short --untracked-files=all`: no entries. No generated `.output`, `test-results`, Playwright report, or `tsconfig.tsbuildinfo` path is tracked or newly untracked.
- No Python files differ from `origin/dev`; Bandit N/A.
- Sandbox-only first-attempt baselines: the two ambient Vitest commands were blocked before collection by temp-directory mkdir EPERM; the first WebUI Playwright attempt was blocked writing `test-results/.last-run.json`. All three unchanged escalated reruns passed as recorded above.
- Known non-failing build/tool warnings: Node localStorage/module.register/NO_COLOR warnings; stale Browserslist data; extension duplicate-import, circular-chunk, and oversized-chunk warnings.

Final pre-push fetch on 2026-07-12 confirmed `origin/dev` remained `5634ea4a04ebcb6322a218469d6ff972b2435450`; HEAD `4b5d19c1281e5c3778616eb7a4c33b787a70c070` had merge-base `5634ea4a04` and `git merge-base --is-ancestor origin/dev HEAD` exited 0. The branch was 21 commits ahead and 0 behind, so no second rebase or verification repeat was required.

Force-pushed the rebased branch with --force-with-lease. Posted direct replies to all eight inline comments, resolved all eight review threads, and posted a top-level technical response explaining why the two out-of-diff RAG wait suggestions conflict with the valid no-key initialization path. Updated the PR body with current task IDs and fresh verification, and restored draft status because the human-written Change summary remains pending. The newly triggered GitHub Actions jobs were inspected and remained queued with no branch-caused failure reported at finalization; CodeRabbit passed/skipped review for the dev target.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Rebased PR #2719 onto current dev, resolved the Backlog task-ID collision without altering the unrelated dev task, addressed every valid review finding, documented technical pushback for the invalid RAG waits, and verified the auth migration across unit tests, WebUI, and the packaged extension. The rebased branch is pushed, all review threads are resolved, and the PR remains draft pending the required human-written Change summary.
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
