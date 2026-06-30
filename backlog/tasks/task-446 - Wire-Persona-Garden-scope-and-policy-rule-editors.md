---
id: TASK-446
title: Wire Persona Garden scope and policy rule editors
status: Done
labels:
- persona
- frontend
priority: Medium
references:
- Docs/Product/Persona_Agent_Design.md
- TASK-445
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Replace the Persona Garden Scopes and Policies placeholder panels with focused editors backed by existing Persona scope-rules and policy-rules endpoints.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Scopes panel loads GET /api/v1/persona/profiles/{persona_id}/scope-rules for the selected persona and shows empty/error states.
- [x] #2 Scopes panel allows add/edit/remove of allowed rule types and saves with PUT /api/v1/persona/profiles/{persona_id}/scope-rules.
- [x] #3 Policies panel loads GET /api/v1/persona/profiles/{persona_id}/policy-rules for the selected persona and shows empty/error/pending-plan context.
- [x] #4 Policies panel allows add/edit/remove of mcp_tool/skill rules, allowed/confirmation/max-calls settings, and saves with PUT /api/v1/persona/profiles/{persona_id}/policy-rules.
- [x] #5 Focused frontend tests cover load, save payloads, and validation/error recovery for both panels.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Replaced the Scopes placeholder with a compact rule editor that loads/saves through the existing `scope-rules` endpoint and validates non-empty rule values locally.
- Replaced the Policies placeholder with a compact rule editor that loads/saves through the existing `policy-rules` endpoint and preserves current rules on save failures.
- Passed `selectedPersonaId` from the Persona Garden route into both panels; no backend behavior or permission scope was added.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented Persona Garden scope and policy rule editors for the current Persona PRD completion scope.

Verification:
- RED: `bunx vitest run src/components/PersonaGarden/__tests__/ScopePolicyEditors.test.tsx` failed before implementation because the placeholder panels had no editor controls.
- GREEN: `bunx vitest run src/components/PersonaGarden/__tests__/ScopePolicyEditors.test.tsx` passed 4 tests.
- Regression: `bunx vitest run src/components/PersonaGarden/__tests__/ScopePolicyEditors.test.tsx src/components/PersonaGarden/__tests__/PersonaGardenPanels.i18n.test.tsx` passed 6 tests.
- Regression: `bunx vitest run src/routes/__tests__/sidepanel-persona.test.tsx` passed 74 tests.
- `git diff --check` passed.
- `bunx tsc --noEmit --pretty false` still exits 2 on unrelated repo-wide baseline errors; visible output did not include the changed Persona Garden files.
- Bandit is not applicable to this frontend-only slice.
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
