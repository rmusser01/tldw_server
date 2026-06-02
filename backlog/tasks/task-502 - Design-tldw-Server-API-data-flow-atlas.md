---
id: TASK-502
title: Design tldw_Server_API data flow atlas
status: In Progress
labels:
- docs
- architecture
priority: Medium
documentation:
- Docs/superpowers/specs/2026-06-02-data-flow-atlas-design.md
- Docs/superpowers/plans/2026-06-02-data-flow-atlas-implementation-plan.md
modified_files:
- Docs/superpowers/specs/2026-06-02-data-flow-atlas-design.md
- Docs/superpowers/plans/2026-06-02-data-flow-atlas-implementation-plan.md
- Docs/Code_Documentation/Data_Flow_Atlas.md
- Docs/Architecture.md
- Docs/Code_Documentation/Code_Map.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Create the approved design spec for a newcomer- and maintainer-oriented Mermaid data flow atlas for tldw_Server_API. The implementation target is a dedicated Docs/Code_Documentation/Data_Flow_Atlas.md linked from existing architecture docs.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Design spec captures the approved layered atlas structure and scope.
- [ ] #2 Spec documents Mermaid-only diagrams, target audience, verification, and maintenance approach.
- [ ] #3 Backlog task is kept current with spec path and verification results.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Implementation is proceeding via Docs/superpowers/plans/2026-06-02-data-flow-atlas-implementation-plan.md. TASK-502 is the authoritative implementation and verification task; TASK-503 is planning-only.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- 2026-06-02 Stage 1 foundation skeleton completed in `Docs/Code_Documentation/Data_Flow_Atlas.md`: added stable table of contents, reading guide, legend, and placeholders for all planned atlas sections. Text check passed with `rg -n "Data Flow Atlas|System Context|Router Coverage Matrix|How To Update This Atlas" Docs/Code_Documentation/Data_Flow_Atlas.md` returning hits on lines 1, 8, 14, 15, 33, 121, and 125.
- 2026-06-02 Stage 2 foundation maps completed in `Docs/Code_Documentation/Data_Flow_Atlas.md`: replaced the System Context, Request Lifecycle, Router Group Map, and Data Store Map placeholders with code-grounded Mermaid diagrams. Router anchors were verified with `rg -n "def (include_router_idempotent|register_router_specs|register_all_routers)|def resolve_router|def append_imported_router_spec|def iter_.*router_specs|register_router_specs\\(" tldw_Server_API/app/api/v1/router_registry.py tldw_Server_API/app/api/v1/router_groups/*.py tldw_Server_API/app/main.py`, which returned the expected registry helpers, `RouterSpec.resolve_router`, grouped iterators, and minimal registration calls. Foundation check passed with `rg -n "flowchart|sequenceDiagram|include_router_idempotent|register_router_specs|RouterSpec.resolve_router|AuthNZ DB|Media DB|ChaChaNotes|ChromaDB|Redis" Docs/Code_Documentation/Data_Flow_Atlas.md`.
- 2026-06-02 Stage 2 quality-review fix completed: corrected the Data Store Map so evaluations storage is shown under the per-user root as `evaluations/evaluations.db`, matching `DatabasePaths.get_evaluations_db_path(user_id)`. Focused check passed with `rg -n "Per-user evaluations storage|evaluations/evaluations.db" Docs/Code_Documentation/Data_Flow_Atlas.md`.
- 2026-06-02 Stage 2 quality re-review fix completed: clarified the System Context storage node label from `Evaluations DB` to `Per-user Evaluations DB` for consistency with the Data Store Map ownership. Focused check passed with `rg -n "Per-user Evaluations DB|Per-user evaluations storage" Docs/Code_Documentation/Data_Flow_Atlas.md`.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Design spec drafted and tightened at Docs/superpowers/specs/2026-06-02-data-flow-atlas-design.md. Verification: local text sanity checks confirmed required scope terms, TASK-502 backlink, Mermaid-only requirement, phased delivery, router registration helper anchors, coverage table requirement, verification section, and docs-only Bandit skip. Independent spec review loop returned Approved after the router anchor improvement. User asked to continue into implementation planning.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
