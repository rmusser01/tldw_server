---
id: TASK-304.9
title: Document Research Studio capability health contract
status: Done
assignee: []
created_date: '2026-05-12 23:29'
updated_date: '2026-05-12 23:32'
labels:
  - implementation
  - research-studio
  - webui
  - health
  - docs
dependencies:
  - TASK-304.8
documentation:
  - >-
    Docs/superpowers/plans/2026-05-12-research-studio-ux-remediation-implementation-plan.md
parent_task_id: TASK-304
priority: medium
---

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Current backend health endpoints used by Research Studio are inventoried
- [x] #2 Capability matrix distinguishes proven-safe states from unresolved states
- [x] #3 Frontend implementation decision is documented without inventing unsupported health semantics
- [x] #4 Follow-up backend contract requirements are implementation-ready
- [x] #5 Bandit/test skips are recorded when only docs/task files change
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Inspect aggregate, RAG, LLM, audio, slides, and frontend connection health contracts.
2. Document the capability matrix and unsupported semantics in the Research Studio operations runbook.
3. Update the Backlog task with evidence, verification, and final summary.
4. Run diff hygiene and commit the docs-first health follow-up.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Inspected aggregate /api/v1/health, /api/v1/health/live, RAG /api/v1/rag/health, LLM /api/v1/llm/health, audio /api/v1/audio/health and /api/v1/audio/transcriptions/health, slides /api/v1/slides/health, frontend connection store, ServerReadinessGate, and WorkspaceStatusBar.

Documented that current payloads support broad app entry and degraded status only; they do not prove per-action safety for source browsing, chat, artifact generation, slides, audio, export, or sync.

Implementation decision: no action-level disabling in this slice because doing so would invent unsupported frontend semantics. Follow-up requires backend-owned capability payload fixtures.

Verification: git diff --check passed. Tests and Bandit skipped because this slice changes docs/task records only.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Documented the Research Studio capability-aware health contract gap and a backend-owned capability payload needed before frontend action gating can be implemented safely.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->
