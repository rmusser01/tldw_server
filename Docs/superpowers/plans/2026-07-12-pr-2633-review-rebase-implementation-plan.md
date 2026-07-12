# PR #2633 Review Rebase Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebase PR #2633 onto audited `origin/dev`, preserve its still-additive Research Workspace artifact-verification behavior, and close every review finding with code/test evidence or a documented technical disposition.

**Architecture:** Rebase the existing commit series in an isolated repair worktree, resolve conflicts in favor of current `dev` interfaces, then apply minimal TDD fixes grouped by frontend and backend behavior. Track all 21 findings in a ledger, verify the full touched scope, and update the existing PR ref with a guarded force push.

**Tech Stack:** Git, GitHub CLI/GraphQL, Backlog.md MCP, TypeScript/React/Vitest/Playwright, Python/FastAPI/Pydantic/pytest/Hypothesis, Bandit.

---

Task: TASK-12148
Spec: `Docs/superpowers/specs/2026-07-12-pr-2633-review-rebase-design.md`

## File Structure

Planning and tracking:

- Modify: `backlog/tasks/task-12148 - Rebase-PR-2633-and-address-review-feedback.md`
- Create: `Docs/superpowers/plans/2026-07-12-pr-2633-review-rebase-implementation-plan.md`

Expected frontend implementation/test files after rebase:

- Modify: `apps/packages/ui/src/services/acp/connection.ts`
- Modify: `apps/packages/ui/src/services/acp/__tests__/connection.test.ts`
- Modify: `apps/packages/ui/src/components/Option/ResearchWorkspace/StudioPane/hooks/useArtifactGeneration.tsx`
- Modify: `apps/packages/ui/src/components/Option/ResearchWorkspace/__tests__/StudioPane.stage1.test.tsx`
- Modify: `apps/packages/ui/src/components/Option/ResearchWorkspace/__tests__/StudioPane.stage2.test.tsx`
- Modify: `apps/packages/ui/src/components/Option/ResearchWorkspace/__tests__/StudioPane.stage3.test.tsx`
- Create if duplication survives rebase: `apps/packages/ui/src/components/Option/ResearchWorkspace/__tests__/studio-test-fixtures.ts`
- Modify: `apps/tldw-frontend/e2e/workflows/research-workspace.real-backend.spec.ts`

Expected backend implementation/test files after rebase:

- Modify: `tldw_Server_API/app/api/v1/endpoints/flashcards.py`
- Modify: `tldw_Server_API/app/api/v1/schemas/research_workspace_artifacts.py`
- Modify: `tldw_Server_API/app/core/Claims_Extraction/artifact_verification.py`
- Modify: `tldw_Server_API/app/core/config.py`
- Modify: `tldw_Server_API/tests/Claims/test_artifact_verification.py`
- Modify: `tldw_Server_API/tests/Flashcards/test_flashcards_endpoint_integration.py`
- Modify: `tldw_Server_API/tests/Quizzes/test_quiz_generator_test_mode.py`
- Modify: `tldw_Server_API/tests/Slides/test_slides_api.py`
- Create or modify: `tldw_Server_API/tests/Config/test_claims_verification_env.py`

Mechanical Backlog fixes:

- Modify: `backlog/tasks/task-12142 - Add-Research-Workspace-slides-presentation-artifact-coverage.md`
- Modify: `backlog/tasks/task-12143 - Validate-generated-Research-Workspace-slides-are-real-presentations.md`

The actual touched list must be updated after conflict resolution. Files already correct on `dev` must not be changed merely because they appear above.

## Review Ledger

Maintain the following ledger in TASK-12148 implementation notes. Each entry needs `fixed`, `satisfied_by_dev`, or `rejected_with_rationale`, plus evidence, test command, reply status, and resolution status.

| # | Thread ID | Root comment ID | Finding |
|---|---|---:|---|
| 1 | `PRRT_kwDOL1aGf86OXBy2` | `3523713694` | ACP runtime fallback return type |
| 2 | `PRRT_kwDOL1aGf86OXBy4` | `3523713696` | E2E auth assertion false-positive risk |
| 3 | `PRRT_kwDOL1aGf86OXCx0` | `3523718795` | `activateMenuItem` catch-all |
| 4 | `PRRT_kwDOL1aGf86OXCx2` | `3523718797` | Conflicting ACP runtime-precedence request |
| 5 | `PRRT_kwDOL1aGf86OZNRB` | `3524462054` | Repeated claim-verification fixture |
| 6 | `PRRT_kwDOL1aGf86OZNRC` | `3524462056` | Slides fallback incorrectly requires `presentationId` |
| 7 | `PRRT_kwDOL1aGf86OZNRD` | `3524462057` | ACP precedence and multi-user tests |
| 8 | `PRRT_kwDOL1aGf86OZNRF` | `3524462060` | CodeRabbit duplicate catch-all finding |
| 9 | `PRRT_kwDOL1aGf86OZNRI` | `3524462068` | Hard-coded legacy runtime storage key |
| 10 | `PRRT_kwDOL1aGf86OZNRK` | `3524462071` | Duplicate TASK-12142 final-summary marker |
| 11 | `PRRT_kwDOL1aGf86OZNRM` | `3524462073` | Duplicate TASK-12143 final-summary marker |
| 12 | `PRRT_kwDOL1aGf86OZNRN` | `3524462074` | Empty flashcard output reaches verifier |
| 13 | `PRRT_kwDOL1aGf86OZNRO` | `3524462075` | Unbounded Research Workspace `media_ids` |
| 14 | `PRRT_kwDOL1aGf86OZNRQ` | `3524462078` | `no_claims` drops unit truncation metadata |
| 15 | `PRRT_kwDOL1aGf86OZNRW` | `3524462084` | Truncated units can remain grounded |
| 16 | `PRRT_kwDOL1aGf86OZNRa` | `3524462088` | Claims verifier env overrides are unreachable |
| 17 | `PRRT_kwDOL1aGf86OZNRd` | `3524462092` | Missing `MISLEADING` verdict test |
| 18 | `PRRT_kwDOL1aGf86OZNRe` | `3524462094` | Flashcards monkeypatches use `raising=False` |
| 19 | `PRRT_kwDOL1aGf86OZNRf` | `3524462096` | Quiz monkeypatches use `raising=False` |
| 20 | `PRRT_kwDOL1aGf86OZNRg` | `3524462098` | Slides monkeypatches use `raising=False` |
| 21 | CodeRabbit review body, outside diff | n/a | Quiz source/media association shifts after filtering |

## Stage 1: Rebase and Semantic Conflict Resolution

**Goal:** Replay the PR series onto the approved `dev` commit without restoring superseded code.

**Success Criteria:** HEAD is based on `5a309be86b043f5a67b65324a81819f59aa860fc`, all rebase conflicts are resolved, and the rebased diff contains only intended PR behavior plus TASK-12148 records.

**Tests:** Rebase ancestry checks, `git range-diff`, focused pre-existing frontend/backend suites.

**Status:** In Progress

- [x] **Step 1: Verify the committed planning checkpoint**

Completed before plan re-review. Before the rebase, verify the planning history and the clean worktree:

```bash
git log -2 --oneline -- Docs/superpowers/plans/2026-07-12-pr-2633-review-rebase-implementation-plan.md 'backlog/tasks/task-12148 - Rebase-PR-2633-and-address-review-feedback.md'
git status --short --branch
```

Expected status: no tracked or untracked changes.

- [ ] **Step 2: Record immutable rebase inputs**

Run:

```bash
git rev-parse HEAD origin/dev origin/codex/issue-2605-research-workspace-uat
git status --short --branch
```

Expected original PR commit before the design commit: `07292d91aa046f60902d0a81cd0ab354ed991871`; expected base: `5a309be86b043f5a67b65324a81819f59aa860fc`.

- [ ] **Step 3: Rebase the local repair branch onto the audited base**

Run:

```bash
git rebase 5a309be86b043f5a67b65324a81819f59aa860fc
```

Resolve one conflicted commit at a time. Use `git checkout --ours` only for files wholly superseded by current `dev`; otherwise edit the conflict manually and preserve both current interfaces and additive PR behavior. After resolving every path reported by `git diff --name-only --diff-filter=U`, stage exactly that set with `git diff --name-only --diff-filter=U -z | xargs -0 git add`, run `git diff --check`, then continue with `GIT_EDITOR=true git rebase --continue`.

- [ ] **Step 4: Verify ancestry and commit-series intent**

Run:

```bash
git merge-base HEAD origin/dev
git log --oneline origin/dev..HEAD
git range-diff fd5c152b065c408e4e8ee5f08da41589f21cb7f5..07292d91aa046f60902d0a81cd0ab354ed991871 origin/dev..HEAD
git diff --check origin/dev...HEAD
```

Expected merge base: `5a309be86b043f5a67b65324a81819f59aa860fc`. Range-diff differences should be attributable to conflict resolution, already-landed `dev` changes, or TASK-12148 records.

- [ ] **Step 5: Re-run the clean baseline on the rebased tree**

Frontend, from `apps/tldw-frontend`:

```bash
bunx vitest run ../packages/ui/src/services/acp/__tests__/connection.test.ts ../packages/ui/src/components/Option/ResearchWorkspace/__tests__/StudioPane.stage1.test.tsx ../packages/ui/src/components/Option/ResearchWorkspace/__tests__/StudioPane.stage2.test.tsx ../packages/ui/src/components/Option/ResearchWorkspace/__tests__/StudioPane.stage3.test.tsx
```

Backend, from repository root after activating the shared project virtual environment:

```bash
source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate
python -m pytest -q tldw_Server_API/tests/Claims/test_artifact_verification.py tldw_Server_API/tests/Claims/test_artifact_verification_properties.py tldw_Server_API/tests/ResearchWorkspace/test_artifact_generation_service.py
```

If a failure is caused by conflict resolution, stop and repair it before review work. If it is a reproducible `dev` baseline failure, record it in TASK-12148 before continuing.

- [ ] **Step 6: Commit conflict-only follow-up if the rebase requires one**

Most resolutions are embedded in rebased commits. If semantic reconciliation requires a separate change, commit only that reconciliation:

Use `git diff --name-only -z | xargs -0 git add` after verifying the worktree contains only semantic reconciliation changes, then run `git commit -m "chore: reconcile PR 2633 with current dev"`.

## Stage 2: Frontend Review Findings

**Goal:** Fix still-valid ACP, Research Workspace generation, and E2E findings against the rebased code.

**Success Criteria:** Focused tests prove the approved ACP contract, fallback slides complete only through the fallback contract, normalized quiz questions keep their original media mapping, and E2E helpers fail loudly for unknown click errors.

**Tests:** ACP connection Vitest, StudioPane Stage 1/2/3 Vitest, Playwright spec listing/static execution.

**Status:** Not Started

- [ ] **Step 1: Audit frontend findings against the rebased tree**

For ledger items 1-9 and 21, inspect the rebased implementation and current `dev` history. Mark any item already satisfied by `dev` before editing. Preserve the approved configured-key precedence even though item 4 requests the opposite.

- [ ] **Step 2: Write failing ACP precedence/isolation tests**

In `apps/packages/ui/src/services/acp/__tests__/connection.test.ts`, add focused cases showing:

```ts
setRuntimeSingleUserApiKeyOverride("runtime-key")

expect(buildACPAuthHeaders({ authMode: "single-user", apiKey: " stored-key " }))
  .toMatchObject({ "X-API-KEY": "stored-key" })
expect(buildACPAuthParams({ authMode: "single-user", apiKey: " stored-key " }))
  .toMatchObject({ api_key: "stored-key" })

expect(buildACPAuthHeaders({ authMode: "multi-user", accessToken: "token" }))
  .toEqual({ Authorization: "Bearer token" })
expect(buildACPAuthParams({ authMode: "multi-user", accessToken: "token" }))
  .toEqual({ token: "token", api_key: undefined })
```

Run the single test file and confirm any missing branch fails for the expected reason before changing production code.

- [ ] **Step 3: Implement the minimal ACP contract**

In `apps/packages/ui/src/services/acp/connection.ts`, keep or restore only:

```ts
if (config?.authMode !== "single-user") return null
const configured = String(config.apiKey || "").trim()
if (configured && !isPlaceholderApiKey(configured)) return configured
return getRuntimeSingleUserApiKeyOverride() ?? null
```

Do not add another auth abstraction. Re-run the ACP test file and confirm green.

- [ ] **Step 4: Write a failing slide-fallback regression test**

In `StudioPane.stage1.test.tsx` or the smallest existing StudioPane test file, mock Slides API failure and a valid grounded markdown fallback. Assert the artifact completes with fallback content and without a `presentationId`. Confirm the test fails with `SLIDES_UNUSABLE_PRESENTATION_MESSAGE` before implementation.

- [ ] **Step 5: Distinguish API slide results from fallback results**

In `useArtifactGeneration.tsx`, add the smallest stable discriminator produced by `generateSlidesFromApi`/`generateSlidesFallback`. Enforce `presentationId` only for API-backed slide results; run normal usable-text validation for fallback results. Do not weaken `requireUsableSlidesPresentation` for API results. Re-run the focused slide test and Stage 1 suite.

- [ ] **Step 6: Write a failing quiz source-association regression test**

In the existing StudioPane quiz-generation tests, provide three generated questions where the middle candidate is invalid and filtered out, with distinct source citations/media IDs on the first and third questions. Assert retained question two uses the third original question's media ID. Confirm the current index-based implementation fails.

- [ ] **Step 7: Preserve each quiz candidate's original entry**

In `useArtifactGeneration.tsx`, normalize candidates as `{ original, normalized }`, filter pairs whose normalized value is absent, slice the retained pairs, and call `resolveQuizQuestionSourceMediaId(pair.original, uniqueMediaIds)`. Re-run the new test and the containing StudioPane suite.

- [ ] **Step 8: Add focused E2E click-fallback evidence**

The catch behavior lives in test infrastructure rather than production code, but it still needs executable regression evidence. Add `apps/tldw-frontend/e2e/utils/click-fallback.ts` with a pure exported predicate used by both `clickActionable` and `activateMenuItem`, and add `apps/tldw-frontend/__tests__/e2e/click-fallback.test.ts`.

First write the test so an unknown `Error("detached locator")` is incorrectly classified as retryable by the temporary current predicate and the assertion fails, while an error containing `nextjs-portal` is retryable. Then narrow the predicate to:

```ts
export const isNextJsPortalClickError = (error: unknown): boolean =>
  String(error).includes("nextjs-portal")
```

Run the focused Vitest file and confirm red, then green.

- [ ] **Step 9: Apply E2E helper/assertion fixes**

In `research-workspace.real-backend.spec.ts`:

- Use `isNextJsPortalClickError` in both click helpers; keep the `activateMenuItem` keyboard fallback only for that known error and rethrow all other errors.
- Replace the boolean OR auth assertion with `expect([...]).toContain(TEST_CONFIG.apiKey)`; no separate truthiness assertion is required because `resolveE2eApiKey` returns `string` or throws.
- After rebase, do not export a removed legacy runtime-session key merely to satisfy a stale comment. If the legacy key is still intentionally inspected, centralize it only if a current exported source-of-truth constant already exists; otherwise remove that legacy evidence branch and explain the stale comment.

The matcher rewrite is test-only hardening rather than a production behavior change. Its premise is already covered by `apps/tldw-frontend/__tests__/e2e/e2e-auth.test.ts`, which proves local resolution returns a non-empty string and remote resolution without a key throws. Run that suite alongside the click-fallback test:

```bash
bunx vitest run __tests__/e2e/e2e-auth.test.ts __tests__/e2e/click-fallback.test.ts
```

Verify parsing/listing without requiring live services:

```bash
bunx playwright test e2e/workflows/research-workspace.real-backend.spec.ts --list
```

- [ ] **Step 10: Deduplicate claim-verification fixtures only if still repeated**

If the same full grounded payload remains duplicated across Stage 1/2/3 after rebase, create `studio-test-fixtures.ts` with one `makeGroundedClaimVerification(metadataOverrides = {})` factory and replace identical objects. If payloads intentionally differ, leave them local and reply with the difference rather than forcing an abstraction. Re-run all three StudioPane suites.

- [ ] **Step 11: Commit the frontend review fixes**

```bash
git add apps/packages/ui/src/services/acp apps/packages/ui/src/components/Option/ResearchWorkspace apps/tldw-frontend/e2e/workflows/research-workspace.real-backend.spec.ts apps/tldw-frontend/e2e/utils/click-fallback.ts apps/tldw-frontend/__tests__/e2e/click-fallback.test.ts
git commit -m "fix: address PR 2633 frontend review findings"
```

## Stage 3: Backend Review Findings

**Goal:** Fix empty-output handling, validation bounds, Claims metadata/verdict behavior, environment overrides, and strict test patches.

**Success Criteria:** Each behavior has a red-green regression test; mechanical test/doc fixes are strict and minimal.

**Tests:** Focused Claims, Config, Flashcards, Quizzes, Slides, and Research Workspace pytest suites.

**Status:** Not Started

- [ ] **Step 1: Write a failing empty-flashcards endpoint test**

In `test_flashcards_endpoint_integration.py`, make the generation adapter return no usable cards and assert a direct 422 validation error whose detail is not `claim_verification_failed`. Also assert the verifier fake was not called. Run only the new test and confirm it fails because the verifier currently receives an empty unit list.

- [ ] **Step 2: Add the empty-output guard before verification**

In `flashcards.py`, after normalization and before `verify_generated_artifact_against_sources`, reject an empty `generated_cards` list with the endpoint's existing validation-error style. Re-run the new test and adjacent generation tests.

- [ ] **Step 3: Write failing schema-bound coverage**

Add a Research Workspace request-schema test proving 1-50 media IDs validate and 51 IDs fail before endpoint work. Use `max_length=50`, matching the review's practical cap and the endpoint's single-prompt processing budget.

- [ ] **Step 4: Bound `media_ids` in the Pydantic schema**

Change only:

```python
media_ids: list[int] = Field(..., min_length=1, max_length=50)
```

Retain the existing positive-ID deduplication validator. Re-run the focused schema/service tests.

- [ ] **Step 5: Write failing Claims metadata/truncation tests**

In `test_artifact_verification.py`, add separate tests proving:

- `no_claims` unit results retain `text_truncated`/length metadata and add `reason="no_claims"`.
- A verified unit with `text_truncated` or `claims_truncated` is `needs_revision`, while an uncapped verified unit remains `grounded`.
- `VerificationStatus.MISLEADING` maps to `needs_revision` in the existing parameter table.

Run each new test and confirm the first two fail for the expected current behavior.

- [ ] **Step 6: Preserve metadata and downgrade capped units**

In `artifact_verification.py`:

```python
metadata={**unit.metadata, "reason": "no_claims"}
```

When assembling normal unit results, compute the status verdict once and change `grounded` to `needs_revision` if that unit's metadata has `text_truncated` or `claims_truncated`. Re-run Claims unit/property tests.

- [ ] **Step 7: Write a failing Claims verifier env test**

In `tldw_Server_API/tests/Config/test_claims_verification_env.py`, use `monkeypatch.setenv` and module reload/cleanup patterns from existing Config tests to prove `CLAIMS_VERIFICATION_PROVIDER` and `CLAIMS_VERIFICATION_MODEL` override config-file defaults. Confirm the test fails because the env allowlist omits both keys.

- [ ] **Step 8: Add the two verifier keys to the Claims env allowlist**

In `config.py`, add only `CLAIMS_VERIFICATION_PROVIDER` and `CLAIMS_VERIFICATION_MODEL` to the `_env` input list used by the existing resolution branches. Do not refactor the large configuration expression. Re-run the new Config test and `test_claims_env_int_parsing.py`.

- [ ] **Step 9: Tighten monkeypatch targets and Backlog markers**

- Remove `raising=False` only where the target symbol is verified to exist: all five flashcards verifier patches, both quiz verifier patches, and every slides verifier patch found after rebase.
- Remove the second `<!-- SECTION:FINAL_SUMMARY:END -->` from TASK-12142 and TASK-12143.
- Run the affected test files to prove strict patches resolve.

- [ ] **Step 10: Commit the backend/mechanical review fixes**

```bash
git add tldw_Server_API/app tldw_Server_API/tests backlog/tasks/task-12142* backlog/tasks/task-12143*
git commit -m "fix: address PR 2633 backend review findings"
```

## Stage 4: Verification, Security, and Self-Review

**Goal:** Prove the rebased branch is coherent and introduces no touched-scope regressions or security findings.

**Success Criteria:** All focused suites pass, frontend types/lint are clean for touched files, Bandit reports zero new findings, and the review ledger has evidence for all 21 entries.

**Tests:** Commands below plus diff/range inspection.

**Status:** Not Started

- [ ] **Step 1: Run the complete focused frontend set**

From `apps/tldw-frontend`:

```bash
bunx vitest run ../packages/ui/src/services/acp/__tests__/connection.test.ts ../packages/ui/src/components/Option/ResearchWorkspace/__tests__/StudioPane.stage1.test.tsx ../packages/ui/src/components/Option/ResearchWorkspace/__tests__/StudioPane.stage2.test.tsx ../packages/ui/src/components/Option/ResearchWorkspace/__tests__/StudioPane.stage3.test.tsx
bun run typecheck
bunx vitest run __tests__/e2e/e2e-auth.test.ts __tests__/e2e/click-fallback.test.ts
bunx playwright test e2e/workflows/research-workspace.real-backend.spec.ts --list
```

Then, from repository root, lint every TypeScript/TSX file actually changed by the PR rather than a fixed hand-maintained subset:

```bash
git diff --name-only -z origin/dev...HEAD -- ':(glob)apps/**/*.ts' ':(glob)apps/**/*.tsx' | xargs -0 apps/tldw-frontend/node_modules/.bin/eslint --config apps/tldw-frontend/eslint.config.mjs
```

- [ ] **Step 2: Run the complete focused backend set**

Activate the shared project virtual environment explicitly, then run:

```bash
source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate
python -m pytest -q \
  tldw_Server_API/tests/Claims/test_artifact_verification.py \
  tldw_Server_API/tests/Claims/test_artifact_verification_properties.py \
  tldw_Server_API/tests/Config/test_claims_verification_env.py \
  tldw_Server_API/tests/Config/test_claims_env_int_parsing.py \
  tldw_Server_API/tests/Flashcards/test_flashcards_endpoint_integration.py \
  tldw_Server_API/tests/Quizzes/test_quiz_generator_test_mode.py \
  tldw_Server_API/tests/ResearchWorkspace/test_artifact_generation_service.py \
  tldw_Server_API/tests/Slides/test_slides_api.py
```

- [ ] **Step 3: Run Bandit on touched Python implementation paths**

From repository root, activate the shared virtual environment, derive the complete changed Python implementation scope from Git, and pass every path to Bandit:

```bash
source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate
git diff --name-only -z origin/dev...HEAD -- ':(glob)tldw_Server_API/app/**/*.py' | xargs -0 python -m bandit -f json -o /tmp/bandit_task12148.json
```

Read the JSON result and fix any new finding in touched code before continuing.

- [ ] **Step 4: Run Git and content hygiene checks**

```bash
git diff --check origin/dev...HEAD
git status --short --branch
git diff --name-only -z origin/dev...HEAD | xargs -0 rg -n '^(<<<<<<<|=======|>>>>>>>)'
git diff --stat origin/dev...HEAD
git diff origin/dev...HEAD
```

The conflict-marker scan should produce no matches; `rg` exit code 1 is the expected clean result.

Confirm no unrelated worktree files, databases, secrets, generated test output, or dependency directories are tracked.

- [ ] **Step 5: Complete the review ledger and Backlog evidence**

Update TASK-12148 with the 21 dispositions, test results, Bandit result path/summary, touched files, known skips, and the exact pre-push remote head. Mark plan stages complete only after evidence exists.

- [ ] **Step 6: Request code review and address findings**

Dispatch a focused reviewer against `origin/dev...HEAD`. Fix Critical/Important issues, rerun affected tests, and repeat up to three review iterations.

- [ ] **Step 7: Commit final tracking/verification updates**

```bash
git add Docs/superpowers/plans/2026-07-12-pr-2633-review-rebase-implementation-plan.md 'backlog/tasks/task-12148 - Rebase-PR-2633-and-address-review-feedback.md'
git commit -m "docs: record PR 2633 review remediation"
```

## Stage 5: Guarded PR Update and Thread Closure

**Goal:** Update the existing PR without overwriting concurrent work and close the review loop with evidence.

**Success Criteria:** The PR head points to the verified rebased commit, all review threads have replies/dispositions, addressed threads are resolved, and PR metadata/checks are re-inspected.

**Tests:** Remote-head lease check, GitHub PR/check/thread queries.

**Status:** Not Started

- [ ] **Step 1: Re-fetch and verify the remote lease**

```bash
git fetch origin dev codex/issue-2605-research-workspace-uat
git rev-parse origin/dev origin/codex/issue-2605-research-workspace-uat
```

If `origin/dev` moved, rebase onto the new exact `origin/dev` SHA, resolve conflicts, and then repeat Stage 1 Steps 4-5 plus every Stage 4 verification/self-review step before returning here. Do not reuse evidence from the pre-rewrite commit series. If the PR head is no longer `07292d91aa046f60902d0a81cd0ab354ed991871`, stop and inspect the concurrent commits instead of pushing.

- [ ] **Step 2: Push with an explicit lease**

```bash
git push origin HEAD:codex/issue-2605-research-workspace-uat --force-with-lease=refs/heads/codex/issue-2605-research-workspace-uat:07292d91aa046f60902d0a81cd0ab354ed991871
```

- [ ] **Step 3: Reply to all review threads**

Use each root comment ID in the ledger with the inline reply endpoint. The executable form is:

```bash
gh api --method POST repos/rmusser01/tldw_server/pulls/2633/comments/3523713694/replies -f body='Verified after rebase: getRuntimeSingleUserApiKeyOverride already returns string|null, and the focused ACP suite covers the fallback contract.'
```

Repeat that exact command form for root IDs `3523713696`, `3523718795`, `3523718797`, `3524462054`, `3524462056`, `3524462057`, `3524462060`, `3524462068`, `3524462071`, `3524462073`, `3524462074`, `3524462075`, `3524462078`, `3524462084`, `3524462088`, `3524462092`, `3524462094`, `3524462096`, and `3524462098`, substituting the final ledger evidence as the literal body. Keep replies factual: what changed and which test proves it, or why a stale/conflicting suggestion was not applied.

Item 21 has no inline root comment. Post its result explicitly:

```bash
gh pr comment 2633 --repo rmusser01/tldw_server --body 'Addressed the CodeRabbit outside-diff quiz finding: retained normalized quiz questions now preserve their matching original generated question for source-media resolution. Focused StudioPane regression coverage includes an invalid filtered middle candidate.'
```

- [ ] **Step 4: Resolve addressed inline threads**

Use the GraphQL `resolveReviewThread` mutation for the 20 thread IDs after their replies are posted. The following command is executable as written and resolves exactly the audited set:

```bash
for thread in \
  PRRT_kwDOL1aGf86OXBy2 PRRT_kwDOL1aGf86OXBy4 \
  PRRT_kwDOL1aGf86OXCx0 PRRT_kwDOL1aGf86OXCx2 \
  PRRT_kwDOL1aGf86OZNRB PRRT_kwDOL1aGf86OZNRC \
  PRRT_kwDOL1aGf86OZNRD PRRT_kwDOL1aGf86OZNRF \
  PRRT_kwDOL1aGf86OZNRI PRRT_kwDOL1aGf86OZNRK \
  PRRT_kwDOL1aGf86OZNRM PRRT_kwDOL1aGf86OZNRN \
  PRRT_kwDOL1aGf86OZNRO PRRT_kwDOL1aGf86OZNRQ \
  PRRT_kwDOL1aGf86OZNRW PRRT_kwDOL1aGf86OZNRa \
  PRRT_kwDOL1aGf86OZNRd PRRT_kwDOL1aGf86OZNRe \
  PRRT_kwDOL1aGf86OZNRf PRRT_kwDOL1aGf86OZNRg
do
  gh api graphql -F thread="$thread" -f query='mutation($thread:ID!){resolveReviewThread(input:{threadId:$thread}){thread{id isResolved}}}'
done
```

Do not run the loop until the ledger confirms all 20 replies succeeded. Do not resolve a thread without a disposition and evidence.

- [ ] **Step 5: Recheck PR state and checks**

```bash
gh pr view 2633 --repo rmusser01/tldw_server --json headRefOid,baseRefOid,mergeable,mergeStateStatus,reviewDecision,statusCheckRollup,url
gh pr checks 2633 --repo rmusser01/tldw_server
```

If GitHub Actions starts, inspect any failure before claiming completion. External reviewers are reported by URL/status only.

- [ ] **Step 6: Finalize TASK-12148**

Mark acceptance criteria and Definition of Done complete, record the pushed commit/PR URL and final check state, and explicitly note that the human-authored `Change summary` remains the requester's merge-gate responsibility.
