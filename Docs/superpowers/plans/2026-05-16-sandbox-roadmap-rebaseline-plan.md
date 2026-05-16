# Sandbox Roadmap Rebaseline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebaseline the sandbox roadmap after the initial VZ Linux stability queue landed so contributors can see completed foundations, remaining host-gated boundaries, and the next pragmatic work.

**Architecture:** Keep this slice documentation-only. Update the existing roadmap spec as the source of truth, point to current evidence docs/tasks instead of restating implementation detail, and record the next queue as prepared-host evidence first. Do not change runtime code, workflow triggers, helper behavior, or public API contracts.

**Tech Stack:** Markdown docs, Backlog.md task `TASK-405`, existing sandbox roadmap/operator/policy docs.

---

### Task 1: Rebaseline Roadmap Status

**Files:**
- Modify: `Docs/superpowers/specs/2026-05-02-sandbox-module-roadmap-design.md`
- Modify: `backlog/tasks/task-405 - Rebaseline-sandbox-roadmap-after-VZ-stability-PRs.md`

- [x] **Step 1: Inspect current roadmap and evidence docs**

  Review the immediate queue in the sandbox roadmap plus current sandbox docs and Backlog tasks for image-store metadata, runtime inventory, helper lifecycle, boot/resource diagnostics, cleanup contracts, security policy, host-gated policy, and public-doc reconciliation.

- [x] **Step 2: Patch the roadmap status**

  Replace the stale pending immediate queue with a status table that records current completion state, source-of-truth evidence, and residual boundaries.

- [x] **Step 3: Add the next pragmatic queue**

  Document prepared-host acceptance evidence as the next work item, followed by remaining lifecycle drill gaps, operator/admin status consolidation, and expansion-only-after-evidence guidance.

### Task 2: Verify Documentation-Only Change

**Files:**
- Verify: `Docs/superpowers/specs/2026-05-02-sandbox-module-roadmap-design.md`
- Verify: `Docs/superpowers/plans/2026-05-16-sandbox-roadmap-rebaseline-plan.md`
- Verify: `backlog/tasks/task-405 - Rebaseline-sandbox-roadmap-after-VZ-stability-PRs.md`

- [x] **Step 1: Run markdown/reference smoke checks**

  Run targeted `rg` checks for the updated queue headings and source-of-truth paths.

  Verification: `rg -n "Immediate Queue Rebaseline|Next Pragmatic Queue|Prepared-host acceptance evidence|Current Handoff|sandbox-runtime-capability-inventory|vz-linux-host-gated-ci-acceptance-policy" ...` found the expected roadmap and README references. `ls` verified referenced source-of-truth docs and the host-gated workflow exist.

- [x] **Step 2: Run diff hygiene**

  Run `git diff --check` for whitespace/path hygiene.

  Verification: `git diff --check` passed.

- [x] **Step 3: Record Bandit skip**

  Because this slice changes only Markdown and Backlog task metadata, record that Bandit is not applicable.

  Bandit is skipped for this slice because no Python/runtime source changed.

- [x] **Step 4: Close the Backlog task**

  Check acceptance criteria and Definition of Done, add verification notes/final summary, and mark `TASK-405` Done.

  `TASK-405` is marked Done with acceptance criteria and Definition of Done checked.
