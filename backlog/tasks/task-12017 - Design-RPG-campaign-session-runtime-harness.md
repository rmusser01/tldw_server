---
id: TASK-12017
title: Design RPG campaign/session runtime harness
status: In Progress
created_date: 2026-06-25 02:10
labels:
- design
- rpg
- ttrpg
- backend
documentation:
- Docs/superpowers/specs/2026-06-25-rpg-campaign-session-runtime-design.md
modified_files:
- Docs/superpowers/specs/2026-06-25-rpg-campaign-session-runtime-design.md
- backlog/tasks/task-12017 - Design-RPG-campaign-session-runtime-harness.md
updated_date: 2026-06-25 02:14
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Brainstorm and document a backend-first RPG/TTRPG harness design for tldw_server. Scope: core campaign/session runtime, append-only event ledger, cached snapshots, rules adapters for D&D 5e SRD/PF2e/Fate, hybrid rules-pack retrieval, REST and MCP surfaces, authority policy, testing, and rollout. No implementation code in this task.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Approved design is written to Docs/superpowers/specs/2026-06-25-rpg-campaign-session-runtime-design.md
- [x] #2 Spec includes architecture, data model, REST/MCP API, service behavior, error handling, testing, rollout, non-goals, and licensing constraints
- [x] #3 Spec self-review finds no placeholders, contradictions, or ambiguous core requirements
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Capture approved brainstorming decisions in a design spec.
2. Self-review for placeholders, contradictions, scope creep, and ambiguity.
3. Run lightweight documentation verification.
4. Commit the Backlog task update and design document.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Spec written to Docs/superpowers/specs/2026-06-25-rpg-campaign-session-runtime-design.md. Self-review performed for unresolved markers/placeholders, contradictions, ambiguous scope, and whitespace issues. Verification: `rg -n "TBD|TODO|FIXME|placeholder|unclear|\?\?\?|\[ \]" Docs/superpowers/specs/2026-06-25-rpg-campaign-session-runtime-design.md` returned no matches; `git diff --check -- Docs/superpowers/specs/2026-06-25-rpg-campaign-session-runtime-design.md "backlog/tasks/task-12017 - Design-RPG-campaign-session-runtime-harness.md"` passed. Bandit skipped because this task only adds documentation and Backlog metadata, with no Python code changes.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Documented the approved RPG/TTRPG campaign/session runtime design. The spec defines the backend-first architecture, append-only event ledger, cached snapshots, D&D 5e SRD/PF2e/Fate rules adapter boundary, conservative rules-content licensing constraints, REST and MCP surfaces, authority policy, service behavior, error handling, testing strategy, and phased rollout. No implementation code was changed.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
