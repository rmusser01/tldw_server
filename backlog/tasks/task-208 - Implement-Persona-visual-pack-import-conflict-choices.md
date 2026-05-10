---
id: TASK-208
title: Implement Persona visual-pack import conflict choices
status: Done
assignee: []
created_date: '2026-05-10 02:04'
updated_date: '2026-05-10 02:34'
labels:
  - persona
  - webui
  - visual-packs
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/issues/1490'
documentation:
  - Docs/Product/WebUI/Persona_Live_Visual_Packs_PRD.md
  - Docs/Code_Documentation/Persona_Visual_Packs.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add the next Phase 3 Persona/Buddy visual-pack reuse slice tracked by GitHub issue #1490. Extend the existing manifest-backed import preview/commit workflow so target-persona conflicts produce explicit user choices while preserving draft review and no automatic activation.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Import preview reports actionable conflicts and allowed commit choices for the target persona.
- [x] #2 Import commit requires an explicit choice when conflicts are present and rejects ambiguous conflict commits.
- [x] #3 Supported choices preserve review-before-commit behavior and leave active visual packs unchanged until a later explicit activation.
- [x] #4 Persona Garden or Visual Pack editor UI presents the choices clearly without marketplace/shared-library framing.
- [x] #5 Backend, frontend, documentation, and focused tests cover the selected V1 conflict choices.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Plan: Docs/superpowers/plans/2026-05-10-persona-visual-import-conflicts.md. V1 adds preview conflict metadata, explicit commit choices, draft-only replacement, Persona Garden controls, docs, and focused tests.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented preview-backed Persona visual import conflict choices for target title matches. Backend preview reports allowed choices; commit requires explicit target mode for conflicted previews; replace_draft is limited to reviewed draft/review/failed target packs and leaves active packs unchanged. Persona Garden now shows conflict choices and disables commit until a user selects a policy. Verification: pytest persona visual portability/worker/jobs/API suite passed with 57 tests; VisualPackEditor Vitest passed with 21 tests; git diff --check passed; Bandit on touched backend production files wrote /tmp/bandit_persona_visual_import_conflicts.json with zero findings.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added V1 Persona/Buddy visual-pack import conflict choices. Preview now reports target title-match conflicts with create-new and draft-only replace options; import commit rejects ambiguous conflict commits and supports reviewed draft replacement with optional title override while preserving separate activation. Persona Garden surfaces the choices and requires an explicit user selection before conflicted commits. Docs and focused backend/frontend tests were updated.
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
