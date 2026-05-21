---
id: TASK-465
title: Address PR 1905 Persona follow-up review findings
status: Done
labels:
- persona
- review-fix
priority: Medium
references:
- https://github.com/rmusser01/tldw_server/pull/1905
- tldw_Server_API/app/api/v1/endpoints/persona.py
- tldw_Server_API/app/api/v1/schemas/persona.py
- tldw_Server_API/tests/Persona/test_persona_sessions.py
- tldw_Server_API/tests/Persona/test_persona_profiles_api.py
- apps/packages/ui/src/components/PersonaGarden/ScopesPanel.tsx
- apps/packages/ui/src/components/PersonaGarden/PoliciesPanel.tsx
- apps/packages/ui/src/routes/hooks/usePersonaStateDocs.ts
- apps/packages/ui/src/routes/sidepanel-persona.tsx
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Resolve the remaining PR #1905 Cubic and Qodo review findings across Persona export, state archive, scope/policy editor loading states, transcript export confirmation lifecycle, metrics, and docstrings. Fix only still-valid issues and keep scope limited to Persona.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Scope and policy editors avoid stale loading/save races called out by review.
- [x] #2 State-history archive preserves unsaved-draft confirmation behavior before applying returned payloads.
- [x] #3 Transcript export confirmation cannot carry over across selected session context changes.
- [x] #4 Archive metrics classify archive actions correctly and archive request validation uses schema validation for blank IDs.
- [x] #5 Persona export/archive helpers and endpoint behavior are documented and empty transcript exports fail loudly.
- [x] #6 Focused backend/frontend tests plus Bandit and whitespace checks pass.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Disabled scope/policy saves while rules are loading or saving and reset loading when no persona is selected.
- Added an archive-state unsaved-draft guard before archive confirmation and payload application.
- Reset transcript export confirmation when live Persona session/persona context changes.
- Classified persona state archive metrics as archive, moved blank archive entry validation into the Pydantic request schema, documented touched helper/endpoint behavior, and made transcript export return 409 when no live snapshot exists.
- Added/updated focused backend and frontend regression tests for the review findings.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Resolved the remaining PR #1905 Persona review findings with focused UI/backend changes and regression coverage. Verification passed: `bunx vitest run src/components/PersonaGarden/__tests__/ScopePolicyEditors.test.tsx src/routes/__tests__/sidepanel-persona.test.tsx --testNamePattern "scope|export|restore|archives"`; `bunx vitest run src/components/PersonaGarden/__tests__/ScopePolicyEditors.test.tsx src/routes/__tests__/sidepanel-persona.test.tsx`; `source ../../.venv/bin/activate && python -m pytest tldw_Server_API/tests/Persona/test_persona_sessions.py tldw_Server_API/tests/Persona/test_persona_profiles_api.py -q`; `source ../../.venv/bin/activate && python -m bandit -r tldw_Server_API/app/api/v1/endpoints/persona.py tldw_Server_API/app/api/v1/schemas/persona.py -f json -o /tmp/bandit_pr1905_followup_fix.json`; `git diff --check`. No known skips or blockers.
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
