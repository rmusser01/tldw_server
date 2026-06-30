---
id: TASK-466
title: Address PR 1905 CodeRabbit follow-up review findings
status: Done
labels:
- persona
- review-fix
priority: Medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Resolve the remaining PR #1905 CodeRabbit follow-up findings after the first review-fix commit. Verify each item against current code, fix only still-valid issues, and keep scope limited to Persona.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Scope and policy editors do not clear existing rule state before async reload completes and still block save while loading.
- [x] #2 Sidepanel Persona tests do not issue unhandled connect-session requests.
- [x] #3 Persona mode values in API responses are consistently normalized through the bounded runtime-mode helper.
- [x] #4 Transcript export metadata drops non-dict top-level metadata instead of leaking scalar/list payloads.
- [x] #5 Persona transcript export honors the existing export RBAC toggle/permission behavior.
- [x] #6 Relevant task verification paths are corrected and focused tests/security checks pass.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Preserved existing scope/policy rule editor state while replacement persona rules are loading and kept save controls disabled during reload.
- Added regression coverage for preserving scope/policy rules during async persona switches.
- Added the missing sidepanel memory-mode connect-session mock to avoid unhandled request paths.
- Centralized Persona runtime-mode normalization for profile, catalog/session projection, and session summary/detail builders.
- Made transcript export reject direct calls when Persona export policy is disabled and added regression coverage.
- Redacted non-object top-level transcript metadata to an empty object with a redaction marker instead of exporting scalar/list payloads.
- Corrected the Persona state-archive Backlog verification path noted by review.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Resolved the still-valid CodeRabbit follow-up findings for PR #1905. Verification passed: `bunx vitest run src/components/PersonaGarden/__tests__/ScopePolicyEditors.test.tsx src/routes/__tests__/sidepanel-persona.test.tsx --testNamePattern "scope|policy|memory mode|export|restore|archives"`; `source ../../.venv/bin/activate && python -m pytest tldw_Server_API/tests/Persona/test_persona_sessions.py tldw_Server_API/tests/Persona/test_persona_profiles_api.py -k "export or mode or state_history" -q`; `bunx vitest run src/components/PersonaGarden/__tests__/ScopePolicyEditors.test.tsx src/routes/__tests__/sidepanel-persona.test.tsx`; `source ../../.venv/bin/activate && python -m pytest tldw_Server_API/tests/Persona/test_persona_sessions.py tldw_Server_API/tests/Persona/test_persona_profiles_api.py -q`; `source ../../.venv/bin/activate && python -m bandit -r tldw_Server_API/app/api/v1/endpoints/persona.py tldw_Server_API/app/api/v1/schemas/persona.py -f json -o /tmp/bandit_pr1905_coderabbit_fix.json`; `git diff --check`. No known skips or blockers.
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
