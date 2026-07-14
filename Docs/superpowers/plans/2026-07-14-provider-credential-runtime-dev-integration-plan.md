# Provider Credential Runtime Dev Integration Plan

**Goal:** Rebase the completed shared provider credential runtime onto current `origin/dev` without carrying unrelated source-branch commits, repair the Backlog task ID collision, and revalidate the integrated result.

**Backlog:** `TASK-12963`

## Stage 1: Preserve and prepare

**Goal:** Preserve the reviewed source branch and establish a clean integration branch and audit trail.

**Success Criteria:** The source branch remains at `05a0817b03`; the integration branch contains this plan and a current In Progress Backlog record.

**Tests:** Git branch/status/provenance checks.

**Status:** Complete

## Stage 2: Replay onto current dev

**Goal:** Replay only commits after implementation base `4f88741711` onto current `origin/dev`.

**Success Criteria:** The three unrelated source-branch planning commits are absent; conflicts preserve current-dev changes and the reviewed provider credential behavior.

**Tests:** Diff/provenance audit, `git diff --check`, focused tests for every conflicted production file.

**Status:** Complete

## Stage 3: Repair task identity

**Goal:** Replace colliding `TASK-12112` with target-branch-safe `TASK-12963` and update durable references.

**Success Criteria:** Exactly one credential-runtime task exists under its new ID, target branch's existing `TASK-12112` remains untouched, and the design/task references agree.

**Tests:** Backlog CLI view/search plus repository reference scan.

**Status:** Complete

## Stage 4: Integrated verification

**Goal:** Re-run the complete backend, frontend, browser, static, and security verification gates on current dev.

**Success Criteria:** Focused and full feature suites pass, Bandit introduces no findings, and independent review has no unresolved Critical or Important findings.

**Tests:** Recorded in the Backlog task, including known skips and baseline-only warnings.

**Status:** In Progress

## Stage 5: Finalize

**Goal:** Finalize tracking and preserve the clean branch for the human-authored PR Change summary.

**Success Criteria:** Backlog is Done, this completed plan is removed, the branch is clean apart from pre-existing unrelated untracked files, and no PR is opened without the required human summary.

**Tests:** Final status, log, diff, and Backlog validation.

**Status:** Not Started
