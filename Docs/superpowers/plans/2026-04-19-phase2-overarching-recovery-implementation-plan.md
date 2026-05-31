# Phase 2 Overarching Recovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Recover the Phase 2 refactor program by preserving the clean `2.3` branch, replacing the stale `2.1`/`2.2`/`2.4`/`2.5` branches with fresh `origin/dev`-based worktrees, and finishing the overall decomposition plan without merging conflict-heavy stale PRs.

**Architecture:** Treat the current Phase 2 PR set as two groups. Group A is `Phase 2.3`, which is current against `origin/dev` and should continue in place. Group B is `Phase 2.1`, `2.2`, `2.4`, and `2.5`, which are all cut from an old `#1010`-era base and must be superseded by new branches created from current `origin/dev`, with only the still-useful deltas re-applied in smaller slices.

**Tech Stack:** Git worktrees, GitHub PR workflow against `dev`, FastAPI application startup/lifespan code, router registration, ChaChaNotes DB decomposition, pytest, Bandit

---

## Current State Summary

- `PR #1113` (`worktree-phase2.1-lifespan-extraction`) was the stale `Phase 2.1` PR; it is now superseded by replacement `PR #1123` from `worktree-phase2.1-lifespan-extraction-redux`.
- `PR #1110` (`worktree-phase2.2-router-groups`) targets `dev`, is `UNSTABLE`, and is `263 behind / 2 ahead` of `origin/dev`.
- `PR #1112` (`worktree-phase2.4-config-sections`) targets `dev`, is `UNSTABLE`, and is `263 behind / 1 ahead` of `origin/dev`.
- `PR #1111` (`worktree-phase2.5-unified-errors`) targets `dev`, is `UNKNOWN`, and is `263 behind / 1 ahead` of `origin/dev`.
- `PR #1115` (`worktree-phase2.3-chacha-decomp`) targets `dev`, is `UNSTABLE`, and is `0 behind / 4 ahead` of `origin/dev`.
- `Phase 2.3` is the only phase branch that is still on a current base and should remain the active continuation branch.

## Branch Strategy

- Do **not** rebase the stale `Phase 2.1` / `2.2` / `2.4` / `2.5` branches in place.
- Do **not** merge the stale PRs as-is.
- Preserve the stale PRs as historical references until replacement PRs exist, then close them with a supersession note.
- Continue `Phase 2.3` on the existing branch/worktree.
- Recreate `Phase 2.1`, `2.2`, `2.4`, and `2.5` from fresh `origin/dev`-based worktrees with replacement branch names.

## Recommended Merge Order

1. `Phase 2.5 redux` — smallest surface, quickest confidence win.
2. `Phase 2.4 redux` — similarly small and isolated.
3. `Phase 2.3 current` — already active and locally verified.
4. `Phase 2.2 redux` — moderate scope, stale but salvageable.
5. `Phase 2.1 redux` — largest and riskiest; keep last so it can be re-landed against the then-current `dev`.

This order is recommended for risk reduction, not because of numeric phase ordering. If repo maintainers insist on numeric merge order, keep the same replacement-branch strategy but defer `2.3` merge until `2.1` and `2.2` redux branches are ready.

### Task 1: Freeze The Old Phase Branches And Record Supersession Strategy

**Files:**
- Modify: `Docs/superpowers/plans/2026-04-19-phase2-overarching-recovery-implementation-plan.md`
- Reference only: GitHub PRs `#1113`, `#1110`, `#1112`, `#1111`, `#1115`

- [x] **Step 1: Verify branch and PR state before replacement work begins**

Run:
```bash
gh pr view 1113 --repo rmusser01/tldw_server --json baseRefName,headRefName,mergeStateStatus,state,url
gh pr view 1110 --repo rmusser01/tldw_server --json baseRefName,headRefName,mergeStateStatus,state,url
gh pr view 1112 --repo rmusser01/tldw_server --json baseRefName,headRefName,mergeStateStatus,state,url
gh pr view 1111 --repo rmusser01/tldw_server --json baseRefName,headRefName,mergeStateStatus,state,url
gh pr view 1115 --repo rmusser01/tldw_server --json baseRefName,headRefName,mergeStateStatus,state,url
git rev-list --left-right --count origin/dev...worktree-phase2.1-lifespan-extraction
git rev-list --left-right --count origin/dev...worktree-phase2.2-router-groups
git rev-list --left-right --count origin/dev...worktree-phase2.3-chacha-decomp
git rev-list --left-right --count origin/dev...worktree-phase2.4-config-sections
git rev-list --left-right --count origin/dev...worktree-phase2.5-unified-errors
```

Expected:
- `2.1`, `2.2`, `2.4`, and `2.5` are materially behind `origin/dev`
- `2.3` is not behind `origin/dev`

Verified on 2026-04-19:
- `PR #1113` -> `dev` from `worktree-phase2.1-lifespan-extraction`, `DIRTY`, `OPEN`
- `PR #1110` -> `dev` from `worktree-phase2.2-router-groups`, `UNSTABLE`, `OPEN`
- `PR #1112` -> `dev` from `worktree-phase2.4-config-sections`, `UNSTABLE`, `OPEN`
- `PR #1111` -> `dev` from `worktree-phase2.5-unified-errors`, `UNSTABLE`, `OPEN`
- `PR #1115` -> `dev` from `worktree-phase2.3-chacha-decomp`, `UNSTABLE`, `OPEN`
- `origin/dev...worktree-phase2.1-lifespan-extraction` = `263 5`
- `origin/dev...worktree-phase2.2-router-groups` = `263 2`
- `origin/dev...worktree-phase2.3-chacha-decomp` = `0 4`
- `origin/dev...worktree-phase2.4-config-sections` = `263 1`
- `origin/dev...worktree-phase2.5-unified-errors` = `263 1`

- [x] **Step 2: Add a supersession note template for the stale PRs**

Use this exact PR comment template later when replacement PRs exist:

```text
Superseded by PR #<new-pr-number>.

This branch was cut from an old pre-#1072 base and is too stale to merge safely.
The replacement PR was recreated from current `dev` and carries forward only the
still-relevant slice from this branch.
```

- [ ] **Step 3: Commit the recovery plan document before replacement execution begins**

Execution note:
- Deferred for now because the root `phase-3-v1-terminal-stack` worktree has unrelated modified and untracked files.
- Do not create a doc-only commit on that branch without isolating it first. Continue the recovery execution in dedicated redux worktrees instead.

Run:
```bash
git add Docs/superpowers/plans/2026-04-19-phase2-overarching-recovery-implementation-plan.md
git commit -m "docs: add phase 2 recovery and supersession plan"
```

### Task 2: Recreate Phase 2.5 On A Fresh `origin/dev` Base

**Files:**
- Create worktree: `.claude/worktrees/phase2.5-unified-errors-redux`
- Reference old branch: `worktree-phase2.5-unified-errors`
- Likely modify: touched files from old branch diff only

- [x] **Step 1: Create a replacement branch and worktree from `origin/dev`**

Run:
```bash
git worktree add .claude/worktrees/phase2.5-unified-errors-redux -b worktree-phase2.5-unified-errors-redux origin/dev
```

Expected:
- new clean worktree rooted at current `origin/dev`

Completed on 2026-04-19:
- created `.claude/worktrees/phase2.5-unified-errors-redux`
- branch `worktree-phase2.5-unified-errors-redux` now tracks `origin/dev`

- [x] **Step 2: Inspect the old one-commit delta before porting it**

Run:
```bash
git log --oneline origin/dev..worktree-phase2.5-unified-errors
git diff --stat origin/dev...worktree-phase2.5-unified-errors
git diff --unified=5 origin/dev...worktree-phase2.5-unified-errors
```

Expected:
- one small cohesive change set

Observed:
- exactly one old branch commit: `2a9d58812 feat: unified DB error hierarchy and generalized map_db_error_to_http`
- touched files:
  - `tldw_Server_API/app/api/v1/utils/http_errors.py`
  - `tldw_Server_API/app/core/DB_Management/db_errors.py`
- no existing `db_errors.py` file on current `origin/dev`
- no current callers still pass `not_found_status=...` to `map_db_error_to_http()`

- [x] **Step 3: Re-apply the `2.5` change manually or by cherry-pick if clean**

Preferred:
```bash
git cherry-pick 2a9d58812
```

Fallback:
- manually port the change into the redux worktree if the cherry-pick conflicts or drags stale context

Completed on 2026-04-19:
- cherry-pick applied cleanly onto the replacement branch
- replacement commit in redux branch: `39e787c13`
- follow-up coverage commit in redux branch: `523aa35b0`

- [x] **Step 4: Run the exact tests relevant to the touched error-mapping surface**

Run only the smallest targeted slice that covers the modified files. Record the command and result in the replacement branch plan/PR.

Verification completed on 2026-04-19:
- `source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m pytest tldw_Server_API/tests/Utils/test_api_v1_utils.py -k http_error_mapping -v`
  - result: `18 passed, 6 deselected`
- `source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m pytest tldw_Server_API/tests/Utils/test_api_v1_utils.py -v`
  - result: `24 passed`
- `source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m bandit -f json -o /tmp/bandit_phase2_5_redux.json tldw_Server_API/app/api/v1/utils/http_errors.py tldw_Server_API/app/core/DB_Management/db_errors.py`
  - result: `0 findings`

- [x] **Step 5: Open a replacement PR, then close or supersede `#1111`**

Completed on 2026-04-19:
- pushed replacement branch: `worktree-phase2.5-unified-errors-redux`
- opened replacement PR: `#1120` -> `https://github.com/rmusser01/tldw_server/pull/1120`
- closed stale PR `#1111` with the supersession note pointing to `#1120`

### Task 3: Recreate Phase 2.4 On A Fresh `origin/dev` Base

**Files:**
- Create worktree: `.claude/worktrees/phase2.4-config-sections-redux`
- Reference old branch: `worktree-phase2.4-config-sections`
- Likely modify: config section files from old branch diff only

- [x] **Step 1: Create a replacement worktree from `origin/dev`**

Run:
```bash
git worktree add .claude/worktrees/phase2.4-config-sections-redux -b worktree-phase2.4-config-sections-redux origin/dev
```

Completed on 2026-04-19:
- created `.claude/worktrees/phase2.4-config-sections-redux`
- branch `worktree-phase2.4-config-sections-redux` now tracks `origin/dev`

- [x] **Step 2: Inspect the old one-commit delta**

Run:
```bash
git log --oneline origin/dev..worktree-phase2.4-config-sections
git diff --stat origin/dev...worktree-phase2.4-config-sections
git diff --unified=5 origin/dev...worktree-phase2.4-config-sections
```

Observed:
- exactly one old branch commit: `c431ae917 feat: add 4 typed config sections (database, server, logging, embeddings)`
- touched files:
  - `tldw_Server_API/app/core/config_sections/__init__.py`
  - `tldw_Server_API/app/core/config_sections/database.py`
  - `tldw_Server_API/app/core/config_sections/embeddings.py`
  - `tldw_Server_API/app/core/config_sections/logging.py`
  - `tldw_Server_API/app/core/config_sections/server.py`
- config keys still match current `config.txt` section names and option names on `origin/dev`

- [x] **Step 3: Re-apply only the typed config-section change**

Preferred:
```bash
git cherry-pick c431ae917
```

Fallback:
- manual re-port if cherry-pick is noisy

Completed on 2026-04-19:
- cherry-pick applied cleanly onto the replacement branch
- replacement commit in redux branch: `c6d6e79a6`
- follow-up coverage commit in redux branch: `e4ae901fd`

- [x] **Step 4: Run targeted config-loading tests and startup validation slices**

Use the smallest current test slice covering the modified config helpers.

Verification completed on 2026-04-19:
- `source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m pytest tldw_Server_API/tests/Config/test_config_sections_typed_loaders.py -v`
  - result: `4 passed`
- `source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m pytest tldw_Server_API/tests/Audio/test_stt_vnext_config_flags.py -v`
  - result: `7 passed`
- `source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m pytest tldw_Server_API/tests/Config/test_config_precedence_contract.py -v`
  - result: `5 passed`
- `source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m bandit -f json -o /tmp/bandit_phase2_4_redux.json tldw_Server_API/app/core/config_sections/__init__.py tldw_Server_API/app/core/config_sections/database.py tldw_Server_API/app/core/config_sections/embeddings.py tldw_Server_API/app/core/config_sections/logging.py tldw_Server_API/app/core/config_sections/server.py`
  - result: `0 findings`

- [x] **Step 5: Open a replacement PR, then close or supersede `#1112`**

Completed on 2026-04-19:
- pushed replacement branch: `worktree-phase2.4-config-sections-redux`
- opened replacement PR: `#1121` -> `https://github.com/rmusser01/tldw_server/pull/1121`
- closed stale PR `#1112` with the supersession note pointing to `#1121`

### Task 4: Continue Phase 2.3 On The Existing Branch

**Files:**
- Existing worktree: `.claude/worktrees/phase2.3-chacha-decomp`
- Existing branch: `worktree-phase2.3-chacha-decomp`
- Existing PR: `#1115`

- [x] **Step 1: Keep all further `2.3` work on the existing clean branch**

Do **not** recreate `2.3` unless it later falls materially behind `origin/dev`.

Status on 2026-04-19:
- existing worktree retained: `.claude/worktrees/phase2.3-chacha-decomp`
- existing branch retained: `worktree-phase2.3-chacha-decomp`
- branch remains the active current `2.3` continuation branch

- [x] **Step 2: Update the PR title and description to match current scope**

Current title only mentions `CharacterStore`, but the branch now includes:
- `CharacterStore`
- `MessageStore`
- `NoteStore`
- `KeywordStore`
- `PersonaStateStore`
- conservative monolith shrink follow-up work

Suggested replacement title:

```text
Phase 2.3: continue ChaChaNotes store extraction and monolith shrink
```

Completed on 2026-04-19:
- updated PR `#1115` title to `Phase 2.3: continue ChaChaNotes store extraction and monolith shrink`
- refreshed PR body so it now reflects the landed remote scope (`CharacterStore`, `MessageStore`, `NoteStore`, `KeywordStore`) and the bounded continuation rules for the remaining conservative monolith shrink

- [ ] **Step 3: Finish only the remaining verified decomposition work in this branch**

Bound the rest of `2.3` to:
- removing only tested dead duplication
- adding tests before deletion
- avoiding unrelated startup/router/config/error work

- [x] **Step 4: Re-run the current verified local slices before any PR update**

Use the already-proven slices from the branch-local plan:
- store tests
- graph DB query tests
- facade DB tests
- character DB tests
- Bandit on touched scope

Completed earlier on 2026-04-19 before the PR metadata refresh:
- store tests: `40 passed`
- character tag search tests: `23 passed`
- graph DB query tests: `24 passed`
- facade DB focused slice: `56 passed`
- character DB focused slice: `95 passed`
- Bandit on touched scope: `0 findings`

### Task 5: Recreate Phase 2.2 On A Fresh `origin/dev` Base

**Files:**
- Create worktree: `.claude/worktrees/phase2.2-router-groups-redux`
- Reference old branch: `worktree-phase2.2-router-groups`
- Likely modify: router group registration files touched by the old diff

- [x] **Step 1: Create a new redux worktree from `origin/dev`**

Run:
```bash
git worktree add .claude/worktrees/phase2.2-router-groups-redux -b worktree-phase2.2-router-groups-redux origin/dev
```

Completed on 2026-04-19:
- created `.claude/worktrees/phase2.2-router-groups-redux`
- branch `worktree-phase2.2-router-groups-redux` now tracks `origin/dev`

- [x] **Step 2: Inspect the old diff carefully before porting**

Run:
```bash
git log --oneline origin/dev..worktree-phase2.2-router-groups
git diff --stat origin/dev...worktree-phase2.2-router-groups
git diff --unified=5 origin/dev...worktree-phase2.2-router-groups
```

Observed:
- old branch contains two commits, already split on sensible batch boundaries:
  - `bf5cb8b83 feat: populate core router group with 9 infrastructure endpoints`
  - `56a045461 feat: populate content and admin router groups (batch 2)`
- touched files:
  - `tldw_Server_API/app/api/v1/router_groups/core.py`
  - `tldw_Server_API/app/api/v1/router_groups/content.py`
  - `tldw_Server_API/app/api/v1/router_groups/admin.py`
  - `tldw_Server_API/app/api/v1/router_groups/spec.py`
  - `tldw_Server_API/app/api/v1/router_registry.py`
  - `tldw_Server_API/app/main.py`
- current `main.py` already uses `include_router_idempotent()` in the full-app helper path, so grouped registration remains safe to land before the legacy explicit registrations

- [x] **Step 3: Re-land router-group changes in smaller batches than the stale PR**

Prefer:
- one commit for core router grouping
- one commit for content/admin batch

Do **not** carry forward stale unrelated context from the old branch.

Completed on 2026-04-19:
- re-landed batch 1 via cherry-pick: `58e977e47` (core group + spec/registry/main wiring)
- re-landed batch 2 via cherry-pick: `0c5c418f8` (content + admin groups)
- added targeted coverage commit: `e955aab43`
- added Bandit-driven follow-up fix: `ff0539976`

- [x] **Step 4: Run route-registration and endpoint smoke tests**

Use the smallest test set that proves:
- app startup still succeeds
- grouped routers still register
- moved endpoints remain reachable

Verification completed on 2026-04-19:
- red/green TDD slice:
  - `source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m pytest tldw_Server_API/tests/Services/test_router_groups_contract.py -v`
  - result after port: `4 passed`
- full router contract slice:
  - `source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m pytest tldw_Server_API/tests/Services/test_main_router_contract.py -v`
  - result: `5 passed`
- Bandit:
  - `source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m bandit -f json -o /tmp/bandit_phase2_2_redux.json tldw_Server_API/app/api/v1/router_groups/admin.py tldw_Server_API/app/api/v1/router_groups/content.py tldw_Server_API/app/api/v1/router_groups/core.py tldw_Server_API/app/api/v1/router_groups/spec.py tldw_Server_API/app/api/v1/router_registry.py tldw_Server_API/app/main.py`
  - result: `0 findings`

- [x] **Step 5: Open a replacement PR, then close or supersede `#1110`**

Completed on 2026-04-19:
- pushed replacement branch: `worktree-phase2.2-router-groups-redux`
- opened replacement PR: `#1122` -> `https://github.com/rmusser01/tldw_server/pull/1122`
- closed stale PR `#1110` with the supersession note pointing to `#1122`

Post-PR CI repair completed on 2026-04-19:
- GitHub Actions `e2e-smoke (ubuntu-latest, py3.11)` exposed duplicate route registrations during non-pytest module import.
- Root cause was redundant direct includes in `main.py` for routers now owned by grouped registration:
  - `claims` was already populated by `tldw_Server_API/app/api/v1/router_groups/content.py`
  - `vlm` was already populated by `tldw_Server_API/app/api/v1/router_groups/core.py`
- landed follow-up commit on `worktree-phase2.2-router-groups-redux`: `7bb548323` (`fix: remove duplicate grouped router includes`)
- added subprocess import regression coverage in `tldw_Server_API/tests/Services/test_main_router_contract.py` to exercise the non-pytest startup path directly
- verification after the fix:
  - `source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m pytest tldw_Server_API/tests/Services/test_main_router_contract.py tldw_Server_API/tests/Services/test_router_groups_contract.py tldw_Server_API/tests/Config/test_route_and_cors_guards.py -v`
  - result: `36 passed`
  - `source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m bandit tldw_Server_API/app/main.py -f json -o /tmp/bandit_phase2_2_router_fix.json`
  - result: `0 findings`
- PR `#1122` checks were re-triggered by the push and were pending immediately after the fix was pushed

### Task 6: Recreate Phase 2.1 On A Fresh `origin/dev` Base

**Files:**
- Create worktree: `.claude/worktrees/phase2.1-lifespan-extraction-redux`
- Reference old branch: `worktree-phase2.1-lifespan-extraction`
- Likely modify:
  - `tldw_Server_API/app/main.py`
  - `tldw_Server_API/app/services/startup_auth.py`
  - `tldw_Server_API/app/services/startup_validation.py`
- Add tests for the extracted startup/lifespan seam before broad worker migration

- [x] **Step 1: Create the redux worktree from `origin/dev`**

Run:
```bash
git worktree add .claude/worktrees/phase2.1-lifespan-extraction-redux -b worktree-phase2.1-lifespan-extraction-redux origin/dev
```

Completed on 2026-04-19:
- created `.claude/worktrees/phase2.1-lifespan-extraction-redux`
- branch `worktree-phase2.1-lifespan-extraction-redux` now tracks `origin/dev`

- [x] **Step 2: Split `2.1` into smaller salvage slices before touching code**

Required decomposition:
- Slice A: inspect `worker_registry.py` first, but discard it if current `main.py` already owns worker/job-poller lifecycle differently
- Slice B: add `startup_auth.py` extraction only
- Slice C: add `startup_validation.py` only if still needed after inspecting current `main.py`

Completed on 2026-04-19:
- rejected the stale `WorkerRegistry` slice after inspecting current `main.py`; current lifecycle ownership already uses managed job-poller helpers and no longer matches the old abstraction
- landed `startup_auth.py` extraction in redux commit `63a198cbf`
- landed `startup_validation.py` extraction in redux commit `ac0a79e41`

- [x] **Step 3: Inspect current `main.py` first and discard any obsolete assumptions from the old branch**

Run:
```bash
rg -n "lifespan|startup|worker|create_task|stop_event" tldw_Server_API/app/main.py
git diff --unified=5 origin/dev...worktree-phase2.1-lifespan-extraction -- tldw_Server_API/app/main.py
```

Expected:
- identify only the still-useful seams

Observed on 2026-04-19:
- current `main.py` already has `_ManagedJobPoller`, `_register_owned_job_poller`, and `_replace_owned_job_poller_inventory`
- the old `WorkerRegistry` design is stale against current startup/shutdown ownership and would be a redesign rather than a safe salvage
- the still-valid seams were the first-time setup plus AuthNZ integrity preflight block and the AuthNZ service-init block

- [x] **Step 4: Re-land only the still-valid 2.1 infrastructure slices**

Goal:
- preserve current behavior on current `origin/dev`
- shrink the monolithic lifespan startup where the seams are still clean
- avoid re-introducing stale abstractions that no longer match current lifecycle ownership

Completed on 2026-04-19:
- added `tldw_Server_API/app/services/startup_auth.py`
- added `tldw_Server_API/app/services/startup_validation.py`
- updated `tldw_Server_API/app/main.py` to delegate both seams while leaving the newer worker/job-poller lifecycle code intact

- [x] **Step 5: Add or extend startup/lifespan tests before worker migration batches**

The redux branch must have explicit coverage for:
- startup success path
- shutdown success path
- the extracted startup validation path
- the extracted auth service-init path

Verification completed on 2026-04-19:
- `source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m pytest tldw_Server_API/tests/Services/test_startup_auth.py -v`
  - result: `3 passed`
- `source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m pytest tldw_Server_API/tests/Services/test_startup_validation.py -v`
  - result: `4 passed`
- `source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m pytest tldw_Server_API/tests/Services/test_startup_auth.py tldw_Server_API/tests/Services/test_startup_validation.py tldw_Server_API/tests/Services/test_main_lifecycle_contract.py -v`
  - result: `16 passed`
- `source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m bandit -r tldw_Server_API/app/services/startup_auth.py tldw_Server_API/app/main.py -f json -o /tmp/bandit_phase2_1_auth_redux.json`
  - result: `0 findings`
- `source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m bandit -r tldw_Server_API/app/services/startup_validation.py tldw_Server_API/app/main.py -f json -o /tmp/bandit_phase2_1_validation_redux.json`
  - result: `0 findings`

- [x] **Step 6: Port `startup_auth.py` only if it still reduces duplication cleanly**

Do not port it blindly. If current `main.py` has drifted enough that the extraction seam moved, adapt the design to the current code rather than restoring the stale exact block.

Completed on 2026-04-19:
- `startup_auth.py` still matched a clean seam after adapting it to current logging/import behavior

- [x] **Step 7: Open a replacement PR, then close or supersede `#1113`**

Completed on 2026-04-19:
- pushed replacement branch: `worktree-phase2.1-lifespan-extraction-redux`
- opened replacement PR: `#1123` -> `https://github.com/rmusser01/tldw_server/pull/1123`
- closed stale PR `#1113` with the supersession note pointing to `#1123`

### Task 7: Program Closeout And Tracking

**Files:**
- Modify: `Docs/superpowers/plans/2026-04-19-phase2-overarching-recovery-implementation-plan.md`
- Optionally create: a lightweight tracking issue comment or checklist in the repo workflow of record

- [x] **Step 1: Track replacement PR numbers in this plan**

Add:
- replacement branch names
- replacement PR numbers
- stale PR numbers they supersede

Tracked replacements so far:
- `worktree-phase2.1-lifespan-extraction-redux` -> `PR #1123` supersedes stale `#1113`
- `worktree-phase2.5-unified-errors-redux` -> `PR #1120` supersedes stale `#1111`
- `worktree-phase2.4-config-sections-redux` -> `PR #1121` supersedes stale `#1112`
- `worktree-phase2.2-router-groups-redux` -> `PR #1122` supersedes stale `#1110`
- current `worktree-phase2.3-chacha-decomp` remains active as `PR #1115` (latest pushed continuation commit: `77402c311`)

- [x] **Step 2: Mark the old stale PRs closed only after replacement PRs are open**

Never leave the program without a live replacement branch.

Completed on 2026-04-19:
- stale PRs `#1111`, `#1112`, `#1110`, and `#1113` are now all closed only after replacement PRs `#1120`, `#1121`, `#1122`, and `#1123` were opened

- [ ] **Step 3: Re-evaluate merge order after each replacement PR lands**

If `dev` moves significantly during the recovery effort, re-run:
```bash
git rev-list --left-right --count origin/dev...<replacement-branch>
```

If a replacement branch becomes materially stale before merge, refresh it immediately rather than repeating the old accumulation pattern.
