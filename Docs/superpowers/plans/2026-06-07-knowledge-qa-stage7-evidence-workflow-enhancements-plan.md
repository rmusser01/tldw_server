# Knowledge QA Stage 7 Evidence Workflow Enhancements Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Track non-blocking future Knowledge QA evidence workflow improvements without blocking the core trust remediation release.

**Architecture:** This is a post-release planning slice. It separates semantic claim-to-source mapping, confidence summaries, evidence audit views, and Research Workspace handoff from the core stages 1 through 6. It may produce design docs and Backlog tasks, but should not alter `/knowledge` runtime behavior unless separately approved.

**Tech Stack:** Markdown, Backlog.md MCP, optional future React/Python plans.

**Backlog Task:** TASK-2279.9

---

## Boundaries

- Do not block stages 1 through 6 on this work.
- Do not add semantic LLM judges into the release gate unless a later approved task changes scope.
- Do not add flashcard behavior to `/knowledge`.
- Do not turn `/knowledge` into Research Workspace, Chat, Notes, or Media.

## Files

- Create: `Docs/Plans/2026-06-07-knowledge-qa-stage7-evidence-workflow-backlog.md`
- Modify: `Docs/superpowers/specs/2026-06-07-knowledge-qa-follow-on-trust-remediation-design.md` only if accepted scope changes are needed
- Modify: `backlog/tasks/task-2279.9 - Plan-non-blocking-Knowledge-QA-evidence-workflow-improvements.md`

## Task 1: Write Non-Blocking Enhancement Backlog

- [ ] **Step 1: Draft enhancement document**

Create `Docs/Plans/2026-06-07-knowledge-qa-stage7-evidence-workflow-backlog.md` with:

```markdown
# Knowledge QA Stage 7 Evidence Workflow Backlog

## Non-Blocking Status

These enhancements are intentionally outside the release scope for stages 1 through 6.

## Candidates

| Candidate | User value | Dependencies | Risks | Proposed owner |
| --- | --- | --- | --- | --- |
| Claim-to-source mapping | Helps power users audit each sentence | Stage 1B, Stage 2 | False confidence if semantic judging is weak | Future task |
```

- [ ] **Step 2: Include candidates**

Add rows for:

- claim-to-source mapping
- evidence confidence and coverage summaries
- power-user evidence audit view
- Research Workspace handoff
- Chat handoff with preserved citations
- Notes save with trust metadata
- Media open-original deep links

## Task 2: Define Promotion Criteria

- [ ] **Step 1: Add promotion criteria**

Each candidate must state what evidence would justify pulling it into a release:

- real user demand
- deterministic evaluation path
- no degradation of `/knowledge` beginner workflow
- no flashcard overlap
- clear owner surface

- [ ] **Step 2: Add explicit non-goals**

Non-goals:

- deck management
- spaced repetition
- study-set behavior
- replacing Research Workspace
- replacing Notes or Media CRUD

## Task 3: Verify

- [ ] **Step 1: Run docs hygiene**

```bash
git diff --check -- Docs/Plans/2026-06-07-knowledge-qa-stage7-evidence-workflow-backlog.md
```

- [ ] **Step 2: Run scope guard**

```bash
rg -n "flashcard|deck|spaced repetition|study set" Docs/Plans/2026-06-07-knowledge-qa-stage7-evidence-workflow-backlog.md
```

Expected: matches only in explicit non-goal/out-of-scope text.

- [ ] **Step 3: Update Backlog and commit**

```bash
git add Docs/Plans/2026-06-07-knowledge-qa-stage7-evidence-workflow-backlog.md "backlog/tasks/task-2279.9 - Plan-non-blocking-Knowledge-QA-evidence-workflow-improvements.md"
git commit -m "docs: plan knowledge qa evidence workflow backlog"
```
