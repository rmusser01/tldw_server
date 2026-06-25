---
id: TASK-12019
title: Implement RPG core models and adapter registry
status: Done
created_date: 2026-06-25 03:15
labels:
- rpg
- ttrpg
- backend
- implementation
priority: high
references:
- TASK-12018
documentation:
- Docs/superpowers/plans/2026-06-25-rpg-campaign-session-runtime-implementation-plan.md
- Docs/superpowers/specs/2026-06-25-rpg-campaign-session-runtime-design.md
modified_files:
- tldw_Server_API/app/core/RPG/__init__.py
- tldw_Server_API/app/core/RPG/constants.py
- tldw_Server_API/app/core/RPG/errors.py
- tldw_Server_API/app/core/RPG/models.py
- tldw_Server_API/app/core/RPG/rules/__init__.py
- tldw_Server_API/app/core/RPG/rules/adapters.py
- tldw_Server_API/tests/RPG/test_rpg_core_models_adapters.py
updated_date: 2026-06-25 03:19
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the first RPG runtime slice from the reviewed plan: core constants, domain errors, dataclasses, bundled adapter protocol/registry for D&D 5e SRD, Pathfinder 2e, and Fate, and focused tests. This task intentionally excludes persistence, service orchestration, REST, MCP, and rules retrieval.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Core RPG constants, errors, models, and adapter registry are implemented under tldw_Server_API/app/core/RPG
- [x] #2 Bundled adapter registry lists dnd5e_srd, fate, and pf2e with license summaries, mechanics tags, and schemas
- [x] #3 Focused adapter/model tests are written test-first and pass
- [x] #4 Verification results are recorded, including Bandit status for touched code
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Write failing tests for adapter registry and model defaults.
2. Implement minimal RPG package constants, errors, models, and adapter registry to satisfy tests.
3. Run focused tests and syntax/security checks for touched scope.
4. Record modified files and final notes before commit.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
TDD evidence: wrote `tldw_Server_API/tests/RPG/test_rpg_core_models_adapters.py` first and confirmed RED via `ModuleNotFoundError: No module named 'tldw_Server_API.app.core.RPG'`. Implemented minimal core RPG package, constants, errors, dataclasses, and bundled adapter registry. Verification: `python -m pytest tldw_Server_API/tests/RPG/test_rpg_core_models_adapters.py -q` passed 4 tests; `python -m compileall -q tldw_Server_API/app/core/RPG tldw_Server_API/tests/RPG/test_rpg_core_models_adapters.py` passed; `python -m bandit -r tldw_Server_API/app/core/RPG tldw_Server_API/tests/RPG/test_rpg_core_models_adapters.py -f json -o /tmp/bandit_rpg_core_models.json` completed with no findings; `git diff --check` passed.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented the first RPG runtime slice: core constants, domain exceptions, slots dataclasses, and bundled rules adapter registry for D&D 5e SRD, Fate, and Pathfinder 2e. Added focused tests for adapter ordering/license summaries, mechanics metadata, defensive schema copies, unknown adapter errors, and snapshot default isolation.
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
