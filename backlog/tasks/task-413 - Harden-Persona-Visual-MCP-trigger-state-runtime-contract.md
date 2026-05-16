---
id: TASK-413
title: Harden Persona Visual MCP trigger-state runtime contract
status: Done
labels:
- persona
- buddy
- visual-packs
- mcp
- runtime
- issue-1787
references:
- https://github.com/rmusser01/tldw_server/issues/1787
- https://github.com/rmusser01/tldw_server/pull/1717
- https://github.com/rmusser01/tldw_server/pull/1794
- https://github.com/rmusser01/tldw_server/pull/1798
modified_files:
- Docs/Code_Documentation/Persona_Visual_Packs.md
- apps/packages/ui/src/components/Common/PersonaBuddy/BuddyShellHost.tsx
- apps/packages/ui/src/components/Common/PersonaBuddy/__tests__/BuddyShellHost.test.tsx
- apps/packages/ui/src/components/Common/PersonaBuddy/__tests__/personaVisualState.test.ts
- apps/packages/ui/src/components/Common/PersonaBuddy/personaVisualState.ts
- apps/packages/ui/src/routes/hooks/__tests__/usePersonaIncomingPayload.visuals.test.tsx
- apps/packages/ui/src/routes/hooks/usePersonaIncomingPayload.ts
- apps/packages/ui/src/store/persona-visual-runtime.ts
- apps/packages/ui/src/types/persona-visuals.ts
- tldw_Server_API/app/api/v1/endpoints/persona.py
- tldw_Server_API/app/core/MCP_unified/modules/implementations/persona_visuals_module.py
- tldw_Server_API/app/core/MCP_unified/tests/test_persona_visuals_module.py
- tldw_Server_API/app/core/Persona/visuals.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the next issue #1787 Buddy animation pipeline slice after the import-preview fixture work: harden the direct MCP/runtime persona_visuals.trigger_state path for exact tool/custom-state variants. Scope is limited to Persona Visual/Buddy runtime trigger behavior and excludes generation jobs, final art production, Persona Garden UX, VN/CYOA, and unrelated Persona backend work.

Acceptance criteria:
- Backend direct persona_visuals.trigger_state handling validates/bounds emitted runtime state identifiers and reasons rather than passing arbitrary unsafe values through.
- Direct trigger payloads can target declared custom state IDs while rejecting or ignoring unsafe/unknown states at the right boundary.
- Frontend runtime override handling remains aligned with the current pack/custom-state contract and does not bypass existing built-in/manual override restrictions.
- Focused backend/frontend tests cover accepted custom state IDs plus ignored unsafe/unknown values.
- Verification includes focused tests, syntax/type checks as applicable, git diff --check, and Bandit for touched Python paths.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Hardened the Persona Visual MCP trigger-state runtime contract for the Buddy animation pipeline. Direct MCP triggers can now target safe custom state IDs declared by the active visual pack, capabilities expose active custom runtime states, and the WebUI resolves custom runtime overrides only when the currently loaded active pack exposes the state. Added backend/frontend regression coverage and documented the contract.

Verification:
- Red tests first: backend custom trigger tests failed on built-in-only behavior; frontend custom override tests failed on rejected custom state/resolver fallback.
- `python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_persona_visuals_module.py -q` -> 13 passed, 3 warnings.
- `bunx vitest run -c vitest.config.ts src/components/Common/PersonaBuddy/__tests__/personaVisualState.test.ts src/routes/hooks/__tests__/usePersonaIncomingPayload.visuals.test.tsx src/components/Common/PersonaBuddy/__tests__/BuddyShellHost.test.tsx` -> 3 files passed, 38 tests passed; existing i18next warning only.
- `python -m py_compile` on touched Python files -> passed.
- `git diff --check` -> passed.
- Bandit on touched Python paths -> 0 findings in `/tmp/bandit_persona_visual_mcp_trigger.json`.
- `bunx tsc --noEmit -p tsconfig.json` in `apps/packages/ui` remains blocked by unrelated repo baseline errors outside touched Persona Visual files.
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
