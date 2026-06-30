# Knowledge QA Stage 0 Baseline Reconciliation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reconcile the follow-on Knowledge QA remediation work against `TASK-528`, latest `origin/dev`, current tests, docs, and live QA findings before code changes begin.

**Architecture:** This is a documentation and planning slice only. It creates a gap matrix that maps each live finding to existing coverage, a follow-on owner task, and a release gate. It must not duplicate completed readiness, empty-state, and baseline UAT work from `TASK-528`.

**Tech Stack:** Markdown, Backlog.md MCP, ripgrep, git diff hygiene checks.

**Backlog Task:** TASK-2279.1

---

## Boundaries

- Do not change runtime code in this stage.
- Do not add flashcard, deck, spaced repetition, or study-set behavior to `/knowledge`.
- Treat `TASK-528` and its linked child plans as baseline.
- Treat extension runtime E2E blocked by the WXT build stall as an explicit verification risk.

## Files

- Create: `Docs/Plans/2026-06-07-knowledge-qa-follow-on-gap-matrix.md`
- Modify: `Docs/Plans/2026-06-07-knowledge-qa-uat-checklist.md`
- Modify: `backlog/tasks/task-2279.1 - Reconcile-Knowledge-QA-follow-on-baseline-and-release-gates.md`
- Reference: `Docs/superpowers/specs/2026-06-07-knowledge-qa-follow-on-trust-remediation-design.md`
- Reference: `backlog/tasks/task-528 - Plan-knowledge-QA-WebUI-and-extension-remediation-after-UX-audit.md`

## Task 1: Inventory Existing Baseline

- [ ] **Step 1: Read the parent and child TASK-528 records**

Run:

```bash
rg -n "TASK-528|Known blocker|WXT|verification|Knowledge QA" backlog/tasks Docs/superpowers/plans Docs/Plans
```

Expected: output includes `TASK-528`, `TASK-528.1` through `TASK-528.8`, the UAT checklist, and the WXT extension E2E blocker.

- [ ] **Step 2: Read current Knowledge QA test inventory**

Run:

```bash
find apps/packages/ui/src/components/Option/KnowledgeQA -path '*__tests__*' -type f | sort
find apps/tldw-frontend/e2e -type f | rg 'knowledge'
find apps/extension/tests/e2e -type f | rg 'knowledge'
```

Expected: output lists shared UI Vitest files, WebUI Knowledge QA E2E specs, and extension Knowledge QA E2E specs.

## Task 2: Create Gap Matrix

- [ ] **Step 1: Draft the matrix file**

Create `Docs/Plans/2026-06-07-knowledge-qa-follow-on-gap-matrix.md` with this structure:

```markdown
# Knowledge QA Follow-On Gap Matrix

## Scope Boundary

/knowledge is a Knowledge QA workflow for searching a personal library and reviewing grounded answers with citations. Flashcards, decks, spaced repetition, and study-set behavior are out of scope.

## Matrix

| Live finding | Existing TASK-528 coverage | Remaining gap | Owner task | Release gate |
| --- | --- | --- | --- | --- |
| WebUI search returned sources but zero citations | Results/evidence UI guardrails exist | Need trust taxonomy and citation enforcement | TASK-2279.2, TASK-2279.4 | Normal answer requires valid citations |
```

- [ ] **Step 2: Add every reviewed finding**

Include rows for:

- five sources with zero citations
- uncited general answer with web fallback disabled
- full source content unavailable
- zero percent match supporting answer
- extension setup/offline flapping
- extension search success with sync timeout
- export/history preserving degraded answers
- WXT runtime E2E blocked before browser launch

- [ ] **Step 3: Map follow-on owner tasks**

Expected owner mapping:

- TASK-2279.2 for trust taxonomy
- TASK-2279.3 for evidence materialization
- TASK-2279.4 for citation enforcement and abstention
- TASK-2279.5 for extension runtime and sync reliability
- TASK-2279.6 for scoped-search round-trip
- TASK-2279.7 for export/history propagation
- TASK-2279.8 for live UAT gates
- TASK-2279.9 for non-blocking evidence workflow improvements

## Task 3: Update UAT Checklist

- [ ] **Step 1: Add a follow-on release-gate section**

Modify `Docs/Plans/2026-06-07-knowledge-qa-uat-checklist.md` with a short section that links the gap matrix and states that follow-on release signoff requires trust, origin, evidence, and extension runtime gates.

- [ ] **Step 2: Add WXT blocker language**

Expected text concept:

```markdown
Extension runtime E2E is not a soft skip. If the WXT build or runtime harness cannot launch `options.html#/knowledge`, release signoff must record the blocker and owner.
```

## Task 4: Verify

- [ ] **Step 1: Run markdown hygiene**

Run:

```bash
git diff --check -- Docs/Plans/2026-06-07-knowledge-qa-follow-on-gap-matrix.md Docs/Plans/2026-06-07-knowledge-qa-uat-checklist.md
```

Expected: exit 0.

- [ ] **Step 2: Run scope guard**

Run:

```bash
rg -n "deck|spaced repetition|study set" Docs/Plans/2026-06-07-knowledge-qa-follow-on-gap-matrix.md Docs/Plans/2026-06-07-knowledge-qa-uat-checklist.md
```

Expected: matches appear only in explicit out-of-scope guardrail text.

- [ ] **Step 3: Update Backlog**

Use Backlog MCP to add implementation notes to `TASK-2279.1` with matrix path, verification commands, and known skips.

- [ ] **Step 4: Commit**

```bash
git add Docs/Plans/2026-06-07-knowledge-qa-follow-on-gap-matrix.md Docs/Plans/2026-06-07-knowledge-qa-uat-checklist.md "backlog/tasks/task-2279.1 - Reconcile-Knowledge-QA-follow-on-baseline-and-release-gates.md"
git commit -m "docs: reconcile knowledge qa follow-on baseline"
```
