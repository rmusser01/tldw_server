# Scheduled Tasks Phase 4D Backlog Identity Normalization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give every Scheduled Tasks Phase 4D task a collision-free Backlog ID without changing product scope, status, history, or dependency order.

**Architecture:** Treat this as a documentation-only mechanical migration. Rename the eight Phase 4D task files, update their frontmatter and only the references that semantically point to this workstream, then validate repository-wide ID uniqueness and stale-reference absence.

**Tech Stack:** Backlog.md, Markdown, Git, ripgrep.

**Spec:** `Docs/superpowers/specs/2026-08-24-scheduled-tasks-phase4d-agent-task-execution-design.md`

## Global Constraints

- Preserve every task title, status, acceptance criterion, implementation note, final summary, and Definition of Done state.
- Preserve the approved Phase 4D implementation order and all product/security boundaries.
- Do not modify unrelated task records that use the collided IDs.
- Do not change Watchlists, standalone Agent Tasks, backend behavior, APIs, or client behavior.
- Use this fixed mapping:

| Existing ID | Replacement ID | Task |
| --- | --- | --- |
| `TASK-13112` | `TASK-13126` | Design Scheduled Tasks Phase 4D Agent Task execution |
| `TASK-13113` | `TASK-13127` | Fix Agent Task Jobs consumer missing-definition crash |
| `TASK-13116` | `TASK-13128` | Plan Scheduled Tasks Phase 4D prerequisite and feasibility implementation |
| `TASK-13117` | `TASK-13129` | Implement Scheduled Tasks Phase 4D.0F execution feasibility gate |
| `TASK-13118` | `TASK-13130` | Add scheduled execution isolation attestation and hostile runtime proof |
| `TASK-13119` | `TASK-13131` | Add ACP scheduled-mode secure transcripts and leakage gates |
| `TASK-13120` | `TASK-13132` | Add ACP dispatch recovery and monotonic execution evidence |
| `TASK-13121` | `TASK-13133` | Add scheduled execution identity credentials and pre-action mediation |

---

## Stage 1: Reserve And Record The Migration
**Goal:** Establish an auditable task and prove the replacement range is free.
**Success Criteria:** `TASK-13125` tracks this work and `TASK-13126` through `TASK-13133` do not exist on the starting `dev` revision.
**Tests:** Exact frontmatter ID scan over active, completed, and archived Backlog records.
**Status:** Complete

- [x] Create `TASK-13125` through the Backlog.md MCP workflow.
- [x] Confirm the replacement IDs are absent from current `origin/dev`.
- [x] Record the fixed old-to-new mapping above.

## Stage 2: Rename Phase 4D Task Records
**Goal:** Make each Phase 4D task filename and frontmatter ID agree.
**Success Criteria:** Eight renamed files exist under `backlog/tasks/`; the old Phase 4D filenames no longer exist; task content is otherwise unchanged.
**Tests:** `git diff --summary`, frontmatter scans, and content-diff review with ID lines normalized.
**Status:** Complete

- [x] Rename the eight files with `git mv`.
- [x] Replace each task's frontmatter ID with its fixed replacement.
- [x] Update Phase 4D dependencies and references inside those records.

## Stage 3: Update Approved Workstream References
**Goal:** Keep the approved spec, plans, and adjacent Phase 4D task metadata navigable.
**Success Criteria:** The Phase 4D spec, both approved implementation plans, `TASK-13122`, and `TASK-13125` use replacement IDs; unrelated collided-ID records are unchanged.
**Tests:** Path-scoped stale-reference scan and review of every changed Markdown file.
**Status:** Complete

- [x] Update the approved Phase 4D spec.
- [x] Update the prerequisite and feasibility implementation plans.
- [x] Update the adjacent `TASK-13122` dependency/reference.
- [x] Record the replacement mapping and verification evidence in `TASK-13125`.

## Stage 4: Validate And Publish
**Goal:** Prove the migration is complete and reviewable.
**Success Criteria:** Replacement IDs are unique, no Phase 4D old-ID references remain, Markdown and whitespace checks pass, and a focused PR targets `dev`.
**Tests:** Duplicate-ID scan, stale-reference scan, path existence checks, `git diff --check`, and Backlog MCP task lookup.
**Status:** Complete

- [x] Verify all eight replacement IDs occur exactly once as frontmatter IDs.
- [x] Verify all renamed task paths exist and all old Phase 4D task paths are absent.
- [x] Verify no stale old-ID reference remains in the scoped Phase 4D files.
- [x] Review `git diff --check` and the complete diff.
- [x] Mark `TASK-13125` Done with final evidence, commit, push, and open PR #2826 against `dev`.
