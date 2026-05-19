# AI PR Change Summary Policy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a repo-local hard merge gate requiring human-authored change summaries for AI-generated pull requests.

**Architecture:** Keep the rollout documentation-only. Add one canonical policy file under `Docs/superpowers/`, then add a narrow enforcement hook in `AGENTS.md` that points back to the policy instead of duplicating the full rationale everywhere.

**Tech Stack:** Markdown documentation

---

### Task 1: Add The Canonical Policy Document

**Files:**
- Create: `Docs/superpowers/AI_GENERATED_PR_CHANGE_SUMMARY_POLICY_2026_04_17.md`
- Read for style: `Docs/Monitoring/WATCHLISTS_ERROR_PREVENTION_POLICY_2026_02_18.md`

- [x] **Step 1: Draft the policy**

Write the standing policy with sections for purpose, scope, merge gate, required summary content, unacceptable summaries, and reviewer enforcement.

- [x] **Step 2: Verify the policy language**

Check that the document makes the rule unambiguous:

- merge is blocked when the summary is missing or inadequate
- the summary must cover both what changed and why those choices were made
- inability of the human requester to explain the rationale means the PR is not merge-ready

### Task 2: Add The Enforcement Hook In AGENTS

**Files:**
- Modify: `AGENTS.md`

- [x] **Step 1: Update quality gates**

Add a short `AI-Generated PR Merge Gate` subsection under `Quality Gates` and add a `Definition of Done` checklist item that references the new requirement.

- [x] **Step 2: Keep the hook concise**

Point to the canonical policy file instead of copying the full policy text into `AGENTS.md`.

### Task 3: Verify The Documentation Change Set

**Files:**
- Modify: `Docs/superpowers/plans/2026-04-17-ai-pr-change-summary-policy-implementation-plan.md`

- [x] **Step 1: Check touched-file diff**

Run: `git diff -- AGENTS.md Docs/superpowers/`
Expected: Only the new policy/spec/plan docs and the intended `AGENTS.md` section are changed.

- [x] **Step 2: Run Bandit on the touched scope**

Run: `source .venv/bin/activate && python -m bandit -r AGENTS.md Docs/superpowers -f json -o /tmp/bandit_ai_pr_change_summary_policy.json`
Expected: Command completes without reporting executable-code findings in the touched scope.
Observed: `0` findings. Bandit reported AST parse errors for the touched Markdown files, which is expected because this change set is documentation-only.

- [x] **Step 3: Mark plan progress**

Update this plan so completed steps are checked off before handoff.
