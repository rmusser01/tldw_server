---
id: TASK-477
title: Harden Persona assistant selection contract tests
status: Done
labels:
- persona
- chat
- webui
- tests
priority: Medium
references:
- https://github.com/rmusser01/tldw_server/issues/1908
- Docs/superpowers/plans/2026-05-22-persona-backed-chat-startup-hardening.md
- https://github.com/rmusser01/tldw_server/pull/1935
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the assistant-selection contract slice from the Persona-backed Chat Startup plan: add focused coverage for Persona normalization and selector switching behavior, patching implementation only if tests reveal a current gap.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 AssistantSelection tests cover Persona numeric IDs, blank IDs, invalid object shapes, and legacy Character-like values.
- [x] #2 Assistant selector behavior confirms switching among Character, Persona, and none does not leak the previous assistant kind.
- [x] #3 Any implementation changes are minimal and only address failing contract cases.
- [x] #4 Verification commands and Bandit applicability are recorded.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Added `apps/packages/ui/src/types/__tests__/assistant-selection.test.ts` covering Persona numeric IDs, blank/missing IDs, invalid kinds, arrays/nulls, optional invalid text fields, and legacy Character-like normalization.
- Extended `apps/packages/ui/src/components/Common/__tests__/AssistantSelect.behavior.test.tsx` so the existing switching matrix also covers none-to-Character selection.
- No production implementation changes were needed; the new contract coverage passed against the current code.
- Verified with `bunx vitest run src/types/__tests__/assistant-selection.test.ts src/components/Common/__tests__/AssistantSelect.behavior.test.tsx --reporter=verbose` from `apps/packages/ui`.
- Bandit not applicable: no Python files changed.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added focused WebUI contract coverage for Persona assistant selection normalization and selector switching. This closes the assistant-selection test slice from the Persona-backed Chat Startup plan without changing runtime code.
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
