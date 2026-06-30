---
id: TASK-193
title: Write persona visual pack duplicate implementation plan
status: Done
assignee: []
created_date: '2026-05-09 21:30'
updated_date: '2026-05-09 21:38'
labels:
  - persona
  - buddy
  - webui
  - plan
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/issues/1449'
  - 'https://github.com/rmusser01/tldw_server/issues/1450'
documentation:
  - >-
    Docs/superpowers/specs/2026-05-09-persona-visual-duplicate-to-persona-design.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Write an implementation plan for GitHub issue #1450: duplicate a Persona Visual pack from one same-user persona to a different same-user persona as a draft. The approved spec requires physical copying of manifest-referenced asset bytes into the target persona/pack storage path, remapping manifest asset IDs, preserving active packs, keeping the public response as PersonaVisualPackResponse, rejecting same-persona targets in V1, and keeping this Buddy/persona visual-pack work separate from VN/CYOA and shared-library work.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Plan decomposes backend DB/service/API, frontend UI/service/types, tests, and documentation into TDD-friendly steps.
- [x] #2 Plan references exact files and commands needed for implementation and verification.
- [x] #3 Plan preserves the tightened spec decisions: no public asset_id_map, no idempotency key, copy only manifest-referenced assets, reject same-persona target, and no auto-activation.
- [x] #4 Plan is reviewed before implementation begins.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Created implementation plan at Docs/superpowers/plans/2026-05-09-persona-visual-duplicate-to-persona-implementation-plan.md. Self-review checked code paths for PersonaVisualService, persona_state_store, persona endpoint/schema, VisualPackEditor, sidepanel persona switching, and existing test commands. Patched the plan to make the proposed DB status helper reject active status so activation validation cannot be bypassed.

Reviewed the plan against existing PersonaVisualService, persona_state_store, persona endpoint/schema, visual portability importer/exporter, VisualPackEditor, and sidepanel wiring. Patched the plan to normalize blank duplicate titles before DB creation, map missing/checksum-mismatched source assets to 409 conflict responses, and require failure-cleanup coverage so copied files are removed and no failed duplicate is exposed as a draft or active pack. Formal subagent review was not run in this turn.

Verification: git diff --check passed for the plan and Backlog task updates. Bandit was not run because this task changed only Markdown planning/tracking files and no Python implementation code.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implementation plan written and reviewed for GitHub issue #1450. Plan covers backend helper extraction, DB lineage/status support, duplicate service orchestration, API contract, frontend service/types/UI, docs, focused tests, Bandit scope, and whitespace verification. Review patches tightened title normalization, source-state conflict mapping, and partial-failure cleanup expectations.
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
