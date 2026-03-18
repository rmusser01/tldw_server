# PR 908 Merge Conflict Resolution Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Rebase PR 908 onto the current `feat/production-readiness-gaps` head, resolve the overlapping file conflicts in favor of PR 908, and restore GitHub mergeability.

**Architecture:** Keep the current base branch as the foundation, replay the PR 908 boundary-redesign commits on top, and resolve only the true overlapping files by preferring the PR 908 side. Verification stays scoped to the Jobs and metering files that participate in the conflict set.

**Tech Stack:** Git worktrees, git rebase, Python 3.11, pytest, Bandit, GitHub CLI

---

### Task 1: Start The Rebase And Capture The Conflict Set

**Files:**
- Modify: `tldw_Server_API/app/core/Jobs/manager.py`
- Modify: `tldw_Server_API/app/services/stripe_metering_service.py`
- Modify: `tldw_Server_API/tests/Jobs/test_fair_share_integration.py`
- Modify: `tldw_Server_API/tests/test_stripe_metering.py`
- Reference: `Docs/Plans/2026-03-17-pr908-merge-conflict-resolution-design.md`

**Step 1: Start the rebase**

Run:

```bash
git -C /Users/macbook-dev/Documents/GitHub/tldw_server2/.worktrees/codex-pr898-boundary-redesign rebase feat/production-readiness-gaps
```

Expected: the rebase stops on the overlapping Jobs/metering files.

**Step 2: Record the conflict set**

Run:

```bash
git -C /Users/macbook-dev/Documents/GitHub/tldw_server2/.worktrees/codex-pr898-boundary-redesign diff --name-only --diff-filter=U
```

Expected: only the overlapping files appear.

### Task 2: Resolve Overlaps With PR 908 As Authoritative

**Files:**
- Modify: `tldw_Server_API/app/core/Jobs/manager.py`
- Modify: `tldw_Server_API/app/services/stripe_metering_service.py`
- Modify: `tldw_Server_API/tests/Jobs/test_fair_share_integration.py`
- Modify: `tldw_Server_API/tests/test_stripe_metering.py`

**Step 1: Resolve each overlapping file**

For the overlapping files, keep the rebased PR 908 side and remove conflict markers. During rebase this means selecting the replayed commit side for the files where behavior differs, then reading the result to confirm it matches the intended repository-boundary version.

**Step 2: Stage the resolved files**

Run:

```bash
git -C /Users/macbook-dev/Documents/GitHub/tldw_server2/.worktrees/codex-pr898-boundary-redesign add \
  tldw_Server_API/app/core/Jobs/manager.py \
  tldw_Server_API/app/services/stripe_metering_service.py \
  tldw_Server_API/tests/Jobs/test_fair_share_integration.py \
  tldw_Server_API/tests/test_stripe_metering.py
```

**Step 3: Continue the rebase**

Run:

```bash
git -C /Users/macbook-dev/Documents/GitHub/tldw_server2/.worktrees/codex-pr898-boundary-redesign rebase --continue
```

Repeat until the rebase completes cleanly.

### Task 3: Verify The Resolved Boundary Scope

**Files:**
- Test: `tldw_Server_API/tests/Billing/test_authnz_metering_repository.py`
- Test: `tldw_Server_API/tests/Jobs/test_fair_share_integration.py`
- Test: `tldw_Server_API/tests/Jobs/test_jobs_repository.py`
- Test: `tldw_Server_API/tests/test_stripe_metering.py`

**Step 1: Run the focused regression suite**

Run:

```bash
source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate
python -m pytest \
  tldw_Server_API/tests/Billing/test_authnz_metering_repository.py \
  tldw_Server_API/tests/Jobs/test_fair_share_integration.py \
  tldw_Server_API/tests/Jobs/test_jobs_repository.py \
  tldw_Server_API/tests/test_stripe_metering.py -v
```

Expected: pass.

**Step 2: Run Bandit on the touched backend files**

Run:

```bash
source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate
python -m bandit -r \
  tldw_Server_API/app/core/Jobs/manager.py \
  tldw_Server_API/app/core/DB_Management/Jobs_Repository.py \
  tldw_Server_API/app/core/DB_Management/AuthNZ_Metering_Repository.py \
  tldw_Server_API/app/services/stripe_metering_service.py \
  -f json -o /tmp/bandit_pr908_merge_conflict_resolution.json
```

Expected: `0` findings and `0` errors in the JSON output.

### Task 4: Publish The Rebasing Result

**Files:**
- Modify: `Docs/Plans/2026-03-17-pr908-merge-conflict-resolution-design.md`
- Modify: `Docs/Plans/2026-03-17-pr908-merge-conflict-resolution.md`

**Step 1: Update the plan docs with final results**

Record the exact rebase outcome, final verification output, and any notable conflict-resolution notes.

**Step 2: Force-push the rebased branch**

Run:

```bash
git -C /Users/macbook-dev/Documents/GitHub/tldw_server2/.worktrees/codex-pr898-boundary-redesign push --force-with-lease origin codex/pr898-boundary-redesign
```

**Step 3: Confirm mergeability**

Run:

```bash
gh pr view 908 --repo rmusser01/tldw_server --json mergeStateStatus,mergeable,url
```

Expected: PR 908 is no longer `DIRTY` / `CONFLICTING`.
