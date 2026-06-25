---
id: TASK-12017
title: Design RPG campaign/session runtime harness
status: Done
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
updated_date: 2026-06-25 15:07
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
Follow-up design review amendments added: pinned `dnd5e_srd` to SRD 5.1, deferred SRD 5.2.1/5.5e to a separate key, tightened PF2e bundled prose requirements, clarified the `RPG_DB.py` repository boundary, required atomic multi-event proposal apply, tightened committed roll randomness provenance, and added AuthNZ privilege registry rollout/testing coverage. Verification repeated: unresolved-marker scan returned no matches and `git diff --check` passed for the spec/task scope.
Status corrected to Done after confirming acceptance criteria, Definition of Done, design spec, final summary, and verification notes were already complete. Follow-on implementation tasks TASK-12018 through TASK-12028 completed the approved runtime plan.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Documented and amended the approved RPG/TTRPG campaign/session runtime design. The spec defines the backend-first architecture, append-only event ledger, cached snapshots, pinned D&D SRD 5.1/PF2e/Fate rules adapter boundary, conservative rules-content licensing constraints, REST and MCP surfaces, authority policy, service behavior, error handling, testing strategy, permission rollout, and phased implementation path. No implementation code was changed.
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
