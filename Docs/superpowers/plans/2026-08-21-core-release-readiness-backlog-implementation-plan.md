# Core Release Readiness Backlog Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Materialize the approved reusable core release-readiness program as a complete public Backlog graph without implementing product changes or disclosing downstream private details.

**Architecture:** TASK-13013 owns the public program. Ten focused child tasks cover CI, release lineage, security, deployment defaults, supply chain, data isolation, capacity, and tracker integrity. Existing TASK-12116 remains the frontend-safety dependency, and TASK-12983 becomes the final downstream handoff.

**Tech Stack:** Backlog.md CLI, Markdown specifications, Git worktrees, Git validation.

**Spec:** `Docs/superpowers/specs/2026-08-21-core-release-readiness-program-design.md`

## Global Constraints

- Use remote `origin/dev`, never the dirty local checkout, as the authoritative baseline.
- Do not add private repository URLs, infrastructure, customer, commercial, or proprietary patch details.
- This plan changes task and documentation records only; each implementation child must create its own focused plan before code or configuration edits.

---

### Task 1: Establish the public program boundary

**Files:**
- Create: `backlog/tasks/task-13013 - Prepare-a-verifiable-core-release-candidate-for-downstream-deployments.md`
- Create: `Docs/superpowers/specs/2026-08-21-core-release-readiness-program-design.md`

**Interfaces:**
- Consumes: the approved remote-dev audit and repository ownership decision.
- Produces: TASK-13013 and a public, reusable readiness specification.

- [x] Create TASK-13013 from the exact audited `origin/dev` head.
- [x] State the reusable public scope and explicit downstream exclusions.
- [x] Record the release-candidate exit criteria.

### Task 2: Create focused public child tasks

**Files:**
- Create: `backlog/tasks/task-13013.1*` through `backlog/tasks/task-13013.10*`

**Interfaces:**
- Consumes: TASK-13013 and the approved program decomposition.
- Produces: independently reviewable tasks with measurable acceptance criteria.

- [x] Create CI and release tasks TASK-13013.1 through TASK-13013.3.
- [x] Create security, deployment, and supply-chain tasks TASK-13013.4 through TASK-13013.7.
- [x] Create isolation, capacity, and tracker-integrity tasks TASK-13013.8 through TASK-13013.10.

### Task 3: Correct the downstream handoff record

**Files:**
- Rename and modify: `backlog/tasks/task-12983*`

**Interfaces:**
- Consumes: TASK-13013.3 through TASK-13013.8 and existing TASK-12116.
- Produces: a public release handoff with no commercial deployment scope or orphan dependency.

- [x] Replace the private customer deployment description with the reusable release-handoff contract.
- [x] Remove the missing TASK-12982 dependency and link the real readiness prerequisites.
- [x] Preserve the scope-change rationale in implementation notes.

### Task 4: Validate and commit the public task graph

**Files:**
- Modify: `backlog/tasks/task-13013*`
- Modify: `backlog/tasks/task-12983*`
- Modify: this plan only if verification finds a task-graph defect.

**Interfaces:**
- Consumes: all records created or reconciled by Tasks 1-3.
- Produces: a clean committed branch containing a parseable, non-cyclic, privacy-safe task graph.

- [x] Run Backlog task listing and inspect TASK-13013 and TASK-12983.
- [x] Verify every local dependency exists and the release graph has no cycle.
- [x] Run `git diff --check` and scan changed files for prohibited private details or sensitive values.
- [ ] Commit on `codex/core-release-readiness-backlog`.
