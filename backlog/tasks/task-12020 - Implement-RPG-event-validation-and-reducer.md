---
id: TASK-12020
title: Implement RPG event validation and reducer
status: Done
created_date: 2026-06-25 03:21
labels:
- rpg
- ttrpg
- backend
- implementation
priority: high
references:
- TASK-12018
- TASK-12019
documentation:
- Docs/superpowers/plans/2026-06-25-rpg-campaign-session-runtime-implementation-plan.md
modified_files:
- tldw_Server_API/app/core/RPG/events.py
- tldw_Server_API/app/core/RPG/reducer.py
- tldw_Server_API/tests/RPG/test_rpg_events_reducer.py
updated_date: 2026-06-25 03:36
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the RPG event envelope validation, canonical request hashing, and pure snapshot reducer slice from the reviewed plan. Scope excludes persistence/service/API/MCP and dice/check resolution.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 canonical_request_hash is stable for key order
- [x] #2 validate_event_envelope rejects missing stable IDs, invalid sources, oversized payloads, and unknown event types
- [x] #3 reduce_events deterministically handles all V1 core event domains in RPGSnapshotState
- [x] #4 Focused tests are written test-first and pass
- [x] #5 Bandit/diff checks are recorded
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Write failing tests for event hash stability, envelope validation, unknown event rejection, and reducer coverage across scene, actors, npcs, quests, inventory, locations, factions, clocks, rolls, notes, rules references, and rulings.
2. Implement `events.py` and `reducer.py` with a shared supported-event registry.
3. Run focused tests, compileall, Bandit, and diff checks.
4. Record modified files and final notes.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
TDD RED confirmed before implementation: `python -m pytest tldw_Server_API/tests/RPG/test_rpg_events_reducer.py -q --tb=short` failed during collection with `ModuleNotFoundError: No module named 'tldw_Server_API.app.core.RPG.events'`.

Implemented deterministic event hashing, shared supported-event registry, envelope validation, and pure snapshot reduction for the V1 core event domains in the assigned files only.

Verification:
- `source /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/activate && python -m pytest tldw_Server_API/tests/RPG/test_rpg_events_reducer.py -q` -> 8 passed, 28 warnings.
- `source /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/activate && python -m compileall tldw_Server_API/app/core/RPG/events.py tldw_Server_API/app/core/RPG/reducer.py tldw_Server_API/tests/RPG/test_rpg_events_reducer.py` -> exit 0.
- `source /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/activate && python -m bandit -r tldw_Server_API/app/core/RPG/events.py tldw_Server_API/app/core/RPG/reducer.py tldw_Server_API/tests/RPG/test_rpg_events_reducer.py -f json -o /tmp/bandit_rpg_task_12020.json` -> exit 0, 0 findings.
- `git diff --check` -> exit 0.
Implemented event envelope validation, canonical request hashing, and deterministic snapshot reducer for the v1 core RPG event set. Verification: focused RED was confirmed by the worker; combined RPG focused tests passed (30 passed); compileall passed; Bandit on core RPG/DB touched scope reported 0 results; git diff --check passed.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added the TASK-12020 RPG event/reducer slice: `events.py` now exposes canonical request hashing, supported event types, and deterministic envelope validation; `reducer.py` now performs pure snapshot reductions and rejects unsupported event types. The focused tests were written first, verified RED, then passed after the minimal implementation.
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
