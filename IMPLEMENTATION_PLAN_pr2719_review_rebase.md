# PR 2719 Review Remediation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development`. Also follow `superpowers:test-driven-development`, `superpowers:receiving-code-review`, and `superpowers:verification-before-completion`.

**Goal:** Rebase PR #2719 onto current `dev`, address every verified review finding with minimal changes, and return the PR to a fully verified state.

**Architecture:** Keep the production auth migration unchanged. Limit remediation to test-harness reliability, mock-contract accuracy, test environment isolation, and Backlog/PR bookkeeping. Reuse existing Playwright and Vitest primitives and avoid new dependencies.

**Tech Stack:** TypeScript, Vitest, Playwright, Chrome MV3 extension harness, Backlog.md, GitHub Actions.

---

## Stage 1: Tracking and review inventory
**Goal**: Establish unique Backlog tracking after the rebase and record every actionable comment.
**Success Criteria**: TASK-12953 replaces the colliding auth TASK-12950 and remains In Progress pending the human Change summary; TASK-12952 tracks review remediation; the unrelated Quick Ingest TASK-12950 is byte-for-byte unchanged from `origin/dev`.
**Tests**: `test "$(rg -l '^id: TASK-12950$' backlog/tasks -g '*.md' | wc -l | tr -d ' ')" = 1`; `git diff --exit-code origin/dev -- 'backlog/tasks/task-12950 - Fix-Quick-Ingest-Standard-and-Deep-analysis-provider-presets.md'`; `git diff --check`.
**Status**: Complete

- [x] Rebase onto `origin/dev` and inspect all PR reviews, comments, threads, and checks.
- [x] Create TASK-12952 for the review-remediation work.
- [x] Create replacement feature task TASK-12953 through Backlog MCP, preserve the completed implementation criteria, and leave the human Change-summary gate unchecked with status In Progress.
- [x] Obtain the requester’s explicit AGENTS.md exception approval for the one manual Backlog deletion required by the duplicate ID.
- [x] Delete only `backlog/tasks/task-12950 - Preserve-legacy-single-user-API-key-across-media-page-refresh.md`, which Backlog MCP/CLI cannot safely target because both files have the same ID; its full implementation and verification record is preserved in TASK-12953.
- [x] Change the design header from TASK-12950 to TASK-12953 and update PR tracking to TASK-12953/TASK-12952.
- [x] Prove the unrelated Quick Ingest TASK-12950 is unchanged from `origin/dev` and only one active TASK-12950 remains.

## Stage 2: Test-harness review fixes
**Goal**: Make the extension and WebUI regression harnesses deterministic and contract-consistent.
**Files**:
- Modify: `apps/tldw-frontend/e2e/extension-api-key-persistence.spec.ts`
- Modify: `apps/tldw-frontend/e2e/helpers/manual-api-key-fixture.ts`
- Modify: `apps/tldw-frontend/e2e/manual-api-key-persistence.spec.ts`

**Success Criteria**: One shared helper waits for the extension service worker in both callers; startup cleanup reports/preserves close failures; the fixture keeps `/api/v1/media/` in OpenAPI for capability detection, does not advertise unimplemented POST search, and serves both Media list slash forms; both CodeRabbit out-of-diff negative RAG checks wait until `/api/v1/rag/health` is recorded before asserting no authenticated request.
**Tests**: `cd apps/tldw-frontend && bunx playwright test e2e/manual-api-key-persistence.spec.ts --reporter=line`; `cd apps/tldw-frontend && bunx playwright test e2e/extension-api-key-persistence.spec.ts --reporter=line`; `cd apps/extension && bun run compile`. Expected: 3/3 WebUI tests, 3/3 extension tests, and compile exit 0.
**Status**: Complete

- [x] In `extension-api-key-persistence.spec.ts`, add `getExtensionServiceWorker(context)` and reuse it in `resolveExtensionId` and `setExtensionStorage`; replace silent `context.close()` suppression with diagnostic handling that preserves the startup error.
- [x] In `manual-api-key-fixture.ts`, remove the unimplemented `/api/v1/media/search` advertisement and return the empty Media payload for both `/api/v1/media` and `/api/v1/media/`.
- [x] In both persistence specs, poll until a post-offset `/api/v1/rag/health` request exists before asserting that none of those requests is authenticated. These are CodeRabbit review-body comments that GitHub could not attach inline.
- [x] Run both focused Playwright suites and extension compile with the exact commands above.

## Stage 3: Vitest environment isolation
**Goal**: Use Vitest-managed environment stubs in all newly added auth-migration cases.
**Files**:
- Modify: `apps/packages/ui/src/services/__tests__/tldw-api-client.quickstart-auth.test.ts`

**Success Criteria**: The two reviewed newly added direct environment writes (parameterized deployment mode and active quickstart cookie-session case) use `vi.stubEnv`; existing unrelated legacy test style is not broadened.
**Tests**: From `apps/packages/ui`, run `bunx vitest run src/services/__tests__/tldw-api-client.quickstart-auth.test.ts src/services/__tests__/tldw-api-client.auth-source.test.ts src/services/__tests__/tldw-api-client.request-config.test.ts`; repeat the quickstart file with `NEXT_PUBLIC_X_API_KEY=ambient-test-key` and with `VITE_TLDW_API_KEY=ambient-test-key`. Expected: 50 tests for the matrix and 22 tests for each ambient-key run.
**Status**: Not Started

- [ ] Replace only lines currently assigning `process.env.NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE` in the new migration parameterized case and quickstart-cookie precedence case with `vi.stubEnv`.
- [ ] Run the focused and ambient auth test commands.

## Stage 4: Verification and PR update
**Goal**: Verify, commit, force-update the rebased branch safely, and close addressed review threads.
**Success Criteria**: Focused tests/builds pass; diff/security gates are recorded; a final fetch proves HEAD descends from current `origin/dev`; branch is pushed with `--force-with-lease`; all actionable threads are resolved or answered with technical rationale; current CI has no unaddressed branch-caused failures. TASK-12953 remains In Progress until the requester writes the Change summary; TASK-12952 may be completed once review remediation itself is verified.
**Tests**: Auth matrix, WebUI Playwright, `cd apps/extension && bun run build:chrome:prod`, packaged extension Playwright, extension compile, frontend typecheck/lint where available, `git diff --check`, generated-artifact scan, Bandit applicability check, GitHub checks.
**Status**: Not Started

- [ ] Run complete fresh verification.
- [ ] Update TASK-12952 notes/checklists and append the fresh post-rebase verification to TASK-12953 while keeping TASK-12953 In Progress pending the human Change summary.
- [ ] Run `git fetch origin dev`, verify `git merge-base --is-ancestor origin/dev HEAD`, and compare `origin/dev` to the previously verified base. If `dev` advanced, rebase again and repeat the complete verification matrix.
- [ ] Commit and push with `--force-with-lease`.
- [ ] Reply in and resolve review threads, then inspect the new CI run.
- [ ] After every stage is complete, remove this task-specific plan file and commit/push that finalization.
