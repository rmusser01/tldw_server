---
id: TASK-602
title: Phase 2.3 ChaChaNotes persona delegation B
status: Done
labels:
- phase-2.3
- chachanotes
- persona
- refactor
priority: medium
references:
- https://github.com/rmusser01/tldw_server/issues/1116
- https://github.com/rmusser01/tldw_server/pull/2231
documentation:
- Docs/superpowers/specs/2026-05-03-phase2-followup-stack-design.md
- Docs/superpowers/plans/2026-05-03-phase2-followup-stack-implementation-plan.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Continue #1116 Phase 2.3 with a conservative PersonaStateStore/facade delegation slice. Inventory remaining persona-related methods, select one small covered method family distinct from completed exemplar normalization, add or strengthen public CharactersRAGDB facade coverage first, then move only covered behavior behind PersonaStateStore or a pure helper while preserving public signatures, sync logging, and schema behavior.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Selected persona method family has public CharactersRAGDB facade coverage before movement.
- [x] #2 Implementation moves only covered behavior into PersonaStateStore or a pure helper behind it.
- [x] #3 Public method names, signatures, sync logging, and schema behavior remain compatible.
- [x] #4 Focused ChaChaNotes/persona tests, Bandit touched-source scope, and git diff --check pass.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Selected persona memory filtering/query policy as the Phase 2.3 B slice. Added public CharactersRAGDB facade coverage for list/count/archive/delete/get memory lifecycle behavior, then consolidated repeated persona_memory_entries WHERE predicate construction behind PersonaStateStore._build_persona_memory_where_clause. Also made the existing live voice analytics unit fixture date-stable with a fixed started_at value and an explicit broad query window.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Consolidated persona memory filter construction in PersonaStateStore while preserving public CharactersRAGDB method signatures and behavior. Added store-level helper coverage plus public facade lifecycle coverage for memory filters, archive exclusion/include_archived behavior, and soft-delete retrieval. Review follow-up moved this task record from TASK-600 to TASK-602 to avoid a post-rebase Backlog id collision with dev. Verification: python -m pytest tldw_Server_API/tests/ChaChaNotesDB/test_chacha_persona_state_store.py -q; python -m pytest tldw_Server_API/tests/ChaChaNotesDB/test_persona_persistence_db.py -q; python -m pytest tldw_Server_API/tests/Persona/test_persona_profiles_api.py -k "persona_profile_state" -q; python -m bandit -r tldw_Server_API/app/core/DB_Management/chacha/persona_state_store.py -f json -o /tmp/bandit_phase2_3_persona_delegation_b.json; git diff --check. Known skips/blockers: none.
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
