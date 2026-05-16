---
id: TASK-297.7
title: Design /knowledge source health and evidence controls
status: Done
assignee: []
created_date: '2026-05-16 00:13'
updated_date: '2026-05-16 00:19'
labels:
  - webui
  - knowledge
  - ux
  - design
dependencies: []
documentation:
  - >-
    Docs/superpowers/specs/2026-05-16-knowledge-source-health-evidence-controls-design.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Create the approved design/spec for the next /knowledge QA-only improvement slice. Scope is read-only source health plus stronger evidence actions using existing handoff surfaces where available. Do not add durable evidence persistence or turn /knowledge into the canonical CRUD/import hub.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Spec states the QA-only boundary and explicitly excludes server-backed saved evidence for this slice.
- [x] #2 Spec defines source health metadata, UI placement, evidence actions, answer trust summary, recovery copy, and verification strategy.
- [x] #3 Spec is self-contained for a future implementation agent and references current KnowledgeQA surfaces.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Spec written at Docs/superpowers/specs/2026-05-16-knowledge-source-health-evidence-controls-design.md. First review found scope-hardening issues around ambiguous Add Sources copy, save-to-note scope, nearest-match scope, source-health V1 requirements, and source-id compatibility. Spec was updated to address those issues; second review approved it. Additional self-review fixed implementation risks: pre-query source health is now explicitly separate from existing post-query metadata.source_status, endpoint placement prefers a focused read-only RAG source-health endpoint, embedding_status supports not_applicable for non-vector sources, SourceCard action duplication is discouraged, and the erroneous parent_task_id was removed because TASK-297 is unrelated in this clean dev worktree. Verification: git diff --check passed. Bandit not applicable because this is a docs/backlog-only design slice.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Created and review-hardened the /knowledge source health and evidence controls design spec. The spec preserves the QA-only boundary, limits the slice to read-only source health plus existing evidence handoffs, defers durable evidence persistence, and defines staged implementation, recovery copy, privacy constraints, and verification expectations.
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
