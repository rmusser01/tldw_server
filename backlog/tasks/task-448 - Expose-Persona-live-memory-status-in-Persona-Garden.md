---
id: TASK-448
title: Expose Persona live memory status in Persona Garden
status: Done
labels:
- persona
- frontend
priority: Medium
references:
- Docs/Product/Persona_Agent_Design.md
- TASK-447
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add a bounded Live Session memory status surface that makes the selected Persona runtime mode, retrieval toggle/top-k, and Persona state-context mode availability visible without adding new backend behavior or broad memory curation controls.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Live Session shows the selected Persona runtime mode from the profile/catalog response when available.
- [x] #2 Live Session shows memory retrieval on/off and selected top-k alongside the existing controls.
- [x] #3 Live Session distinguishes whether Persona state context can apply in the current runtime mode, matching existing backend persistent_scoped gating.
- [x] #4 Focused frontend coverage proves the memory status is rendered from catalog/profile state and updates when controls change.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Added `mode` to the Persona catalog/profile UI contract so Persona Garden can render the selected runtime mode without adding a new endpoint or memory behavior.
- Projected catalog profile modes through the existing backend runtime-mode allowlist, defaulting invalid or missing values to `session_scoped`.
- Added a compact Live Session memory status row for Persona mode, memory on/off, selected top-k, and state-context availability. The status is hidden in companion mode and does not add mutation, archive, or broad memory-curation controls.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
- RED: `bunx vitest run src/routes/__tests__/sidepanel-persona.test.tsx --testNamePattern "shows live memory mode status"` initially failed because the memory status row did not exist.
- GREEN: `bunx vitest run src/routes/__tests__/sidepanel-persona.test.tsx --testNamePattern "shows live memory mode status"` passed.
- GREEN: `bunx vitest run src/routes/__tests__/sidepanel-persona.test.tsx` passed, 75 tests.
- GREEN: `source ../../.venv/bin/activate && python -m pytest tldw_Server_API/tests/Persona/test_persona_catalog.py -q` passed, 5 tests.
- GREEN: `source ../../.venv/bin/activate && python -m bandit -r tldw_Server_API/app/api/v1/endpoints/persona.py tldw_Server_API/app/api/v1/schemas/persona.py -f json -o /tmp/bandit_persona_memory_status.json` passed with no findings.
- GREEN: `git diff --check` passed.
- KNOWN BASELINE: `bunx tsc --noEmit --pretty false` still fails on existing broad WebUI TypeScript debt outside this slice, including unrelated audio, chat composer, flashcards, workspace, route registry, and service test typing issues.
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
