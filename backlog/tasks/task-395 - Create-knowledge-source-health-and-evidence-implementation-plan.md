---
id: TASK-395
title: Create /knowledge source health and evidence implementation plan
status: Done
assignee: []
created_date: '2026-05-16 00:27'
labels:
  - webui
  - knowledge
  - ux
  - plan
dependencies: []
documentation:
  - >-
    Docs/superpowers/specs/2026-05-16-knowledge-source-health-evidence-controls-design.md
  - >-
    Docs/superpowers/plans/2026-05-16-knowledge-source-health-evidence-controls-plan.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Create a staged implementation plan for the approved /knowledge source health and evidence controls slice. Scope stays QA-only: read-only pre-query source health, clearer evidence actions, answer trust summary, and recovery copy using existing handoffs. Do not implement code in this task.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Plan references the approved design spec and preserves the QA-only, no-new-persistence boundary.
- [x] #2 Plan names exact backend, frontend, test, docs, and verification files/commands for each stage.
- [x] #3 Plan separates pre-query source health from existing post-query metadata.source_status diagnostics.
- [x] #4 Plan uses TDD-style bite-sized steps and defines focused verification for backend, frontend, extension parity, diff check, and Bandit when backend code is touched.
<!-- AC:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Created `Docs/superpowers/plans/2026-05-16-knowledge-source-health-evidence-controls-plan.md` from the approved `/knowledge` source health and evidence controls design spec.

The plan keeps `/knowledge` QA-only, excludes source CRUD/import and durable evidence persistence, separates pre-query source health from existing post-query `metadata.source_status`, and decomposes the work into backend contract, frontend normalization, provider/source-picker UI, evidence/trust UI, recovery/parity, and final verification tasks.

Local review found and corrected a backend implementation risk in the first draft: the source-health endpoint snippet originally used search pipeline `*_path` keys when constructing `MultiDatabaseRetriever`. The final plan now uses constructor keys such as `media_db`, `notes_db`, `character_cards_db`, `world_books_db`, `chat_dictionaries_db`, `prompts_db`, and `kanban_db`, and explicitly warns implementers not to pass the search endpoint's pipeline argument names directly.

Verification recorded for this planning-only task: `git diff --check` passed; targeted `rg` checks confirmed the obsolete `media:read` permission string was removed and the old `*_path` key names only remain in the explicit warning text. Bandit is not applicable because this task changed only docs and Backlog task metadata.

The plan-document-reviewer subagent was not dispatched because current session tool policy only permits delegated subagents when the user explicitly asks for them. A code-grounded local critique pass was performed instead, including checks against current RAG `DataSource`, `MultiDatabaseRetriever`, auth permission, and frontend `RagSource` contracts.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Created the `/knowledge` source health and evidence controls implementation plan at `Docs/superpowers/plans/2026-05-16-knowledge-source-health-evidence-controls-plan.md`. The plan is scoped to QA-only improvements, preserves the no-new-persistence boundary, maps exact backend/frontend/test files, uses bite-sized TDD steps, and includes focused backend pytest, frontend Vitest, extension parity, browser smoke, `git diff --check`, and Bandit verification gates. During review, the backend source-health endpoint steps were corrected to use the actual `MultiDatabaseRetriever` constructor contract. No product code was changed.
<!-- SECTION:FINAL_SUMMARY:END -->
