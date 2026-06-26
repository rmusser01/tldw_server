---
id: TASK-2401
title: Resolve PR 1982 dev-to-main conflicts with dev precedence
status: In Progress
assignee: []
created_date: ''
updated_date: 2026-06-23 18:24
labels:
- merge-conflict
- pr-1982
- dev-main
dependencies: []
priority: high
modified_files:
- apps/mcp-unified/pyproject.toml
- Docs/Plans/2026-06-23-pr1982-dev-main-conflict-resolution.md
- tldw_Server_API/app/core/MCP_unified/tests/test_runtime_package_boundary.py
- .github/workflows/ui-watchlists-extension-e2e.yml
- apps/packages/ui/src/components/Notes/__tests__/task-markdown.test.ts
- apps/packages/ui/src/components/Notes/task-markdown.ts
- tldw_Server_API/tests/CI/test_required_workflow_contracts.py
- tldw_Server_API/tests/Helper_Scripts/test_mcp_standalone_user_guide_uat.py
- tldw_Server_API/tests/RAG_NEW/unit/test_analytics_db_dev_reconciliation.py
- tldw_Server_API/app/core/DB_Management/media_db/runtime/fts_ops.py
- tldw_Server_API/tests/DB_Management/test_media_postgres_support.py
- apps/extension/tests/e2e/utils/extension-build.test.ts
- apps/extension/tests/e2e/utils/extension-build.ts
- apps/extension/tests/e2e/utils/extension-id.test.ts
- apps/extension/tests/e2e/utils/extension-id.ts
- apps/extension/tests/e2e/utils/extension-paths.test.ts
- apps/extension/tests/e2e/utils/extension-paths.ts
- apps/extension/tests/e2e/utils/extension.launch.test.ts
- apps/extension/tests/e2e/utils/extension.ts
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Resolve the open PR #1982 (`dev` -> `main`) conflict state by merging current `origin/main` into current `origin/dev` while preserving `dev` for overlapping conflicts, then verify and push the resulting merge back to `dev` if clean.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Confirm the current PR #1982 conflict surface between `origin/dev` and `origin/main`.
- [x] #2 Merge `origin/main` into the PR head with `dev` winning overlapping conflicts.
- [x] #3 Verify no unresolved merge paths or conflict markers remain before pushing.
- [x] #4 Push the verified merge back to `dev` and confirm PR #1982 merge state/checks update.
- [x] #5 Address current PR #1982 MCP Unified Internal RC package-boundary failure.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Docs/Plans/2026-06-23-pr1982-dev-main-conflict-resolution.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- `git merge-tree --write-tree origin/dev origin/main` identified a single content conflict in `README.md`.
- Merged `origin/main` into the work branch from `origin/dev` with `git merge origin/main -X ours --no-edit`; Git auto-merged `README.md` with the `dev` side winning the overlap.
- Verification before push: `git status --short --branch` showed no unresolved paths; `rg -n '<<<<<<<|=======|>>>>>>>' README.md` returned no matches; `git diff --check HEAD~1 HEAD` exited 0; `git diff --quiet HEAD:README.md origin/dev:README.md` exited 0.
- Pushed the conflict-resolution branch to `dev` at `3e9756f5f2bd7c1d577f439b5780eba386aed801`; GitHub PR #1982 moved from `DIRTY` to `UNSTABLE`, confirming conflicts were cleared and checks were running.
- The current PR #1982 `MCP Unified Internal RC` job failed because the built `mcp-unified` wheel omitted `mcp_unified.policy_grants`; gateway imports then failed with `ModuleNotFoundError: No module named 'mcp_unified.policy_grants'`.
- Added `mcp_unified.policy_grants` to the standalone setuptools package list and extended package-boundary coverage so the wheel/sdist must include that package.
- Verification for the package fix: targeted package-boundary pytest failed before the fix with the missing package assertions, passed after the fix, and `PYTHON=/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python make mcp-unified-rc` exited 0 with `RC status: ok`.
- Current PR #1982 CI failures also include four `rag-new-unit-core-misc` shards failing because `FakePostgresBackend` lacked `table_exists`; added that fake backend method and verified `test_analytics_db_dev_reconciliation.py` passes locally.
- `core-utils-tooling` shards failed because the UAT test still referenced the removed root-level `mcp_unified/pyproject.toml` package layout and did not pass the standalone package install args; updated the test to use `apps/mcp-unified`, assert the `src` package layout, and cover the additional tool-event UAT steps.
- `Watchlists Extension E2E (No Skips)` failed because the extension launch skipped all tests when no system Chrome extension target appeared; updated the workflow to use Playwright Chromium and minimal locales, with a workflow contract test covering that behavior.
- GitHub Advanced Security CodeQL reported checklist label metadata sanitization in `task-markdown.ts`; replaced the HTML-comment regex with a scanner that removes multiline and dangling checklist metadata comments, with Vitest regressions.
- Local verification after these fixes: `bunx vitest run src/components/Notes/__tests__/task-markdown.test.ts` passed; focused Python pytest covering UAT, RAG reconciliation, workflow contract, and MCP package-boundary checks passed with 12 tests; `git diff --check` passed; Bandit on touched Python files passed with no issues.

Old-head CI later exposed Full Suite shard integrations failing test_reconcile_content_update_updates_media_fts_versions_and_binding because SQLite external-content FTS deleted using the already-updated media_fts row. Changed _update_fts_media to delete with the supplied old title/content payload, preserving synonym expansion for old and new content, and updated the unit expectation. Local verification: focused failing integration plus FTS unit passed, full test_media_postgres_support.py passed, and test_sync_coordinator.py passed.
- New-head Watchlists Extension E2E was canceled after `bunx playwright install --with-deps chromium` spent about 43 minutes downloading browsers and exhausted the 45-minute job window before the backend could start. Reverted that job to the runner's system Chrome channel while keeping `TLDW_E2E_EXTENSION_MINIMAL_LOCALES=1`; local verification: targeted workflow contract passed, full workflow-contract test file passed (36 tests), focused prior-failure Python regression set passed (16 tests), `git diff --check` passed, and Bandit on the touched CI contract test found no issues.
- Latest Watchlists Extension E2E reached the Playwright spec but skipped all 14 tests after repeatedly logging `[E2E_DEBUG] No service worker found after waiting`; the strict no-skip gate then failed with `passed=0 skipped=14`. Added deterministic staged extension IDs via a test manifest key, resolved extension IDs from that manifest key when no MV3 service worker/background target is active, and preserved the real default locale catalog while pruning non-default locales. Local verification: the extension utility Vitest set passed (12 tests) and `bun run compile` passed; local headed/headless browser probes could not complete in this macOS sandbox before browser startup timeout, so CI remains the end-to-end confirmation for the full Watchlists workflow.
- The same push exposed `Extension Research Workspace Parity` timing out after launch under `launchWithBuiltExtension`. Scoped deterministic manifest-key injection to the Watchlists/general `launchWithExtension` path only, leaving built-extension parity staging on its prior extension ID behavior while still preserving default locale catalogs. Local verification after scoping: extension utility Vitest set passed (13 tests), `bun run compile` passed, and `git diff --check` passed.
- Follow-up parity root cause: headed Chromium hung inside `chromium.launchPersistentContext` when `launchWithBuiltExtension` staged the full default `messages.json` under `TLDW_E2E_EXTENSION_MINIMAL_LOCALES=1`; restoring the built-extension launcher to the lightweight default-locale stub while leaving the general Watchlists launcher on the real default locale catalog made the reproduced headed parity spec pass. Local verification: extension utility Vitest set passed (14 tests), headed `bunx playwright test tests/e2e/research-workspace.parity.spec.ts --reporter=line` passed (1 test), strict headed `bun run test:e2e:workspace-parity:strict` passed with `[playwright-no-skips] passed=1 skipped=0 unexpected=0 flaky=0`, `bun run compile` passed, and `git diff --check` passed. Bandit was not applicable because this follow-up only touched TypeScript test utilities.
- 2026-06-26 CI fan-out triage: PR #2520 historical `CI` run `28178992004` had 726 jobs; PR #1982 current `CI` run `28222746881` was stuck at `pending` with zero jobs because stale queued workflow runs from older PR #1982 head SHAs and merged PR #2520 were still occupying/colliding with the Actions queue. Normal cancellation cleared most queued runs; force-cancel cleared the stuck old `ci-1982` run `28219062432`, after which current `CI` moved to 8 starter jobs and then expanded to 726 jobs. Current PR #1982 now shows 777 total checks. New failure roots identified from logs: shard coverage has 22 unassigned test files; `core-audit-unified` fails on `CancelledError` escaping fixture teardown through `UnifiedAuditService.stop()`; Windows `core-utils-tooling` prompt inventory tests return no prompt assets; and `UI Watchlists Extension E2E` remains failed separately.
- 2026-06-26 follow-up CI fixes before pushing another 700+ check run: added the five missing shard path groups to `.github/workflows/ci.yml`; restored the audit test's monkeypatched `flush` before fixture teardown; added portable prompt inventory fallbacks for Windows when fd-relative traversal is unavailable; aligned Watchlists extension launch staging with the lightweight default-locale path and covered it with Vitest; and made AuthNZ migrations opt into Redis-to-file migration-lock fallback only in test mode so CI jobs that export `REDIS_URL` without starting Redis do not fail unrelated shards. Verification: shard coverage passed with `shards=721 test_files=3882 ignored=4 baseline=130 new_uncovered=0`; focused Python regression set passed 57 tests; Redis-unavailable admin conflict subset passed 9 tests; Redis-unavailable Prompt Studio startup representative passed 1 test; extension utility Vitest passed 12 tests; `bun run compile` passed; `bun run build:chrome:prod` passed after temporarily pointing the local generated `antd` symlink at the installed Bun package target, then restoring it; `git diff --check` passed; Bandit over touched Python scope produced only existing pytest `B101` assert findings, and rerun with `-s B101` produced 0 findings.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->
