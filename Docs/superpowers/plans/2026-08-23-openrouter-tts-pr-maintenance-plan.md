# OpenRouter TTS PR Maintenance Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebase PR #2751 onto the latest `origin/dev`, resolve every actionable review and CI issue, and merge only after all repository gates pass.

**Architecture:** Preserve the approved explicit, config-first TTS gateway contract while replaying it over current `dev`. Validate review feedback against the rebased implementation, make only root-cause fixes with focused regression coverage, and use current post-rebase CI rather than the obsolete July run as the merge authority.

**Tech Stack:** Git, GitHub CLI, Python/FastAPI/pytest/Bandit, TypeScript/React/Vitest/ESLint, Backlog.md MCP.

**Spec:** `Docs/superpowers/specs/2026-07-15-openrouter-tts-gateway-design.md`

## Global Constraints

- Preserve legacy TTS inference whenever callers omit the explicit backend.
- Keep gateway endpoint URLs administrator-owned; BYOK supplies credentials only.
- Allow fallback only within the configured bounded pre-audio policy.
- Convert formats only after fully buffering provider output.
- Never log API keys or include secrets in discovery/config generations.
- Rebase with `--force-with-lease`; never overwrite an unexpectedly advanced remote branch.
- Do not mark AI-generated prose as the requester-written change summary required by repository policy.

---

### Task 1: Reconcile the branch with current dev

**Files:**
- Modify as conflicts require: files already changed by `codex/openrouter-tts-gateways-pr`
- Modify: `backlog/tasks/task-12116.1 - Implement-OpenRouter-and-generic-TTS-gateways.md`

**Interfaces:**
- Consumes: remote branch `origin/codex/openrouter-tts-gateways-pr` at the recorded pre-rebase SHA and `origin/dev`
- Produces: a clean local branch whose merge base is current `origin/dev`

- [x] **Step 1: Record recovery and ancestry evidence**

Run: `git status --short --branch && git rev-parse HEAD origin/codex/openrouter-tts-gateways-pr origin/dev && git rev-list --left-right --count origin/dev...HEAD`

Expected: clean feature worktree; local and remote feature SHA match before rebase.

- [x] **Step 2: Rebase onto the fetched dev tip**

Run: `git rebase origin/dev`

Expected: replay completes, or stops at explicit conflicts without discarding either side.

- [x] **Step 3: Resolve conflicts by contract**

For every conflict, compare the feature commit intent with the current `origin/dev` implementation. Preserve current-dev API/runtime improvements and retain only the approved gateway behavior listed under Global Constraints. After each resolution, run `git add <resolved-files>` and `git rebase --continue`.

- [x] **Step 4: Validate the replayed range**

Run: `git status --short && git diff --check origin/dev...HEAD && git diff --name-status origin/dev...HEAD && git range-diff 29acaca8c781213e27b12066372df13855e2e7a6..b5660be12734c9cafe8a322e5dbe7206101013b4 origin/dev...HEAD`

Expected: clean tree, no conflict markers or whitespace errors, and every intentional feature change is represented.

### Task 2: Resolve verified review findings

**Files:**
- Modify: `apps/packages/ui/src/components/Option/Settings/TTSModeSettings.tsx`
- Test: `apps/packages/ui/src/components/Option/Settings/__tests__/TTSModeSettings.test.tsx`
- Modify: `apps/packages/ui/src/components/Media/read-along/useMediaReadAlongSession.ts`
- Test: `apps/packages/ui/src/components/Media/read-along/__tests__/useMediaReadAlongSession.test.tsx`

**Interfaces:**
- Consumes: rebased backend capability/default data and read-along session state
- Produces: stable form synchronization and null-safe cacheability evaluation

- [x] **Step 1: Verify each reviewer claim against rebased code**

Inspect the complete effect around gateway-default synchronization and the complete read-along session construction path. Determine whether setters can receive equal values repeatedly and whether `providerContext` is structurally optional at runtime.

- [x] **Step 2: Add focused regressions only for reproducible risks**

For an equal-value setter loop, add a component test that renders a backend with already-resolved defaults and asserts the form-update callback is not re-issued. For missing provider context, add a read-along test with `providerContext` omitted and assert synthesis proceeds without throwing while cache defaults remain enabled.

- [x] **Step 3: Prove the regressions fail before production changes**

Run: `bunx vitest run src/components/Option/Settings/__tests__/TTSModeSettings.test.tsx src/components/Media/read-along/__tests__/useMediaReadAlongSession.test.tsx --maxWorkers=1 --no-file-parallelism`

Expected: any newly added regression for a verified issue fails for the reviewed reason. If current-dev already fixed an issue, record it as non-actionable instead of changing production code.

- [x] **Step 4: Implement the minimal root-cause fixes**

Guard default synchronization with field equality before `setFormValues`. The read-along claim is non-actionable because `providerContext` is required, constructed before session use, and dereferenced earlier on the same path; do not weaken that invariant with a later optional chain. Do not refactor unrelated settings or read-along behavior.

- [x] **Step 5: Verify focused frontend behavior**

Run: `bunx vitest run src/components/Option/Settings/__tests__/TTSModeSettings.test.tsx src/components/Media/read-along/__tests__/useMediaReadAlongSession.test.tsx --maxWorkers=1 --no-file-parallelism`

Expected: both files pass with zero failures.

### Task 3: Diagnose and clear CI issues

**Files:**
- Modify only files implicated by reproducible post-rebase failures
- Test alongside each verified root-cause fix

**Interfaces:**
- Consumes: old July CI logs for comparison and new post-rebase GitHub Actions results
- Produces: passing required checks or documented external/non-actionable failures with direct URLs

- [ ] **Step 1: Classify the obsolete July failures**

Run the GitHub Actions inspection helper against PR #2751 and preserve failing job names, run URLs, and root error snippets. Separate aggregate jobs that failed only because shards were cancelled from actionable leaf failures.

- [x] **Step 2: Run local feature-focused verification after the rebase**

Activate `.venv`, run the gateway/config/BYOK/audio/http/audiobook backend tests listed in the Backlog task, then run the touched frontend Vitest suites and pinned ESLint. Use an 8 GB heap for TypeScript and compare diagnostics in touched files to `origin/dev`.

- [x] **Step 3: Run security and syntax gates**

Run compileall for touched Python modules and `python -m bandit -r tldw_Server_API/app/api/v1/endpoints/audio tldw_Server_API/app/api/v1/endpoints/user_keys.py tldw_Server_API/app/api/v1/schemas/audio_schemas.py tldw_Server_API/app/api/v1/schemas/audiobook_schemas.py tldw_Server_API/app/api/v1/schemas/user_keys.py tldw_Server_API/app/core/Audio/tts_service.py tldw_Server_API/app/core/AuthNZ/byok_helpers.py tldw_Server_API/app/core/AuthNZ/byok_runtime.py tldw_Server_API/app/core/AuthNZ/byok_testing.py tldw_Server_API/app/core/Infrastructure/provider_registry.py tldw_Server_API/app/core/TTS tldw_Server_API/app/core/http_client.py tldw_Server_API/app/services/audiobook_jobs_worker.py tldw_Server_API/app/services/startup_heavy_init.py -f json -o /tmp/bandit_task_12116_1_pr_maintenance.json` from the project virtual environment.

Expected: compileall exits 0 and no new medium/high Bandit findings exist in changed code.

- [ ] **Step 4: Push with lease and inspect fresh CI**

Run: `git push --force-with-lease origin codex/openrouter-tts-gateways-pr`

Then run the GitHub Actions inspection helper and `gh pr checks 2751 --watch --fail-fast=false` until required checks finish.

- [ ] **Step 5: Fix only reproducible post-rebase failures**

For each actionable failure, trace the error to its source, add the smallest failing regression, implement one fix, and rerun the exact failing job scope locally before pushing with lease again. Stop after three failed fix attempts and reassess architecture.

### Task 4: Close review and merge gates

**Files:**
- Modify: `backlog/tasks/task-12116.1 - Implement-OpenRouter-and-generic-TTS-gateways.md`
- Update remotely: PR #2751 body, review threads, draft state, and merge state

**Interfaces:**
- Consumes: green local verification, completed required CI, and resolved review decisions
- Produces: merged PR and finalized Backlog record, or one explicit blocking gate

- [ ] **Step 1: Record maintenance verification in Backlog.md**

Append the rebased dev SHA, review dispositions, exact test counts, Bandit result, CI result, and remaining skips. Keep the task In Progress until merge is confirmed.

- [ ] **Step 2: Reply in and resolve every inline thread**

Reply to each comment using its GitHub thread/reply endpoint with the verified disposition and commit SHA, then resolve the corresponding review thread through GraphQL.

- [ ] **Step 3: Update the PR description accurately**

Refresh verification data and remove stale July-only status text. Preserve clear attribution: generated overview remains AI-generated, and a human-authored change summary must come from the requester.

- [ ] **Step 4: Confirm merge readiness**

Run: `gh pr view 2751 --json isDraft,mergeable,mergeStateStatus,reviewDecision,statusCheckRollup` and query unresolved review-thread count.

Expected: not draft, mergeable, no unresolved threads, no changes requested, and all required checks successful. If the human-authored change-summary gate is still missing, stop and request that exact input rather than bypassing policy.

- [ ] **Step 5: Merge and finalize**

Run `gh pr merge 2751 --repo rmusser01/tldw_server --squash --delete-branch=false`, verify PR state is `MERGED`, then mark `TASK-12116.1` Done with the merge URL/SHA and final summary. Preserve the PR worktree unless cleanup is separately requested.
