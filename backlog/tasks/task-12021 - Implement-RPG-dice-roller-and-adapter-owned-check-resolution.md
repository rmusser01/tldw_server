---
id: TASK-12021
title: Implement RPG dice roller and adapter-owned check resolution
status: Done
created_date: 2026-06-25 03:22
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
- tldw_Server_API/app/core/RPG/models.py
- tldw_Server_API/app/core/RPG/dice.py
- tldw_Server_API/app/core/RPG/checks.py
- tldw_Server_API/app/core/RPG/rules/adapters.py
- tldw_Server_API/tests/RPG/test_rpg_dice_checks.py
updated_date: 2026-06-25 03:36
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement deterministic/testable dice rolling and adapter-owned check resolution for D20 and Fate adapters from the reviewed plan. Scope excludes persistence/service/API/MCP and event reducer work.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 DiceRoller parses bounded dice expressions and supports deterministic injected roll values
- [x] #2 Fate dice rolling supports deterministic injected fate values
- [x] #3 CheckResult and DiceRollResult models are added
- [x] #4 resolve_check delegates to adapter-owned resolution logic instead of branching in core checks.py
- [x] #5 Focused tests are written test-first and pass
- [x] #6 Bandit/diff checks are recorded
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Write failing tests for deterministic d20 and Fate rolls and adapter-owned check resolution.
2. Implement `dice.py`, `checks.py`, CheckResult/DiceRollResult models, and adapter resolver callbacks/classes.
3. Run focused tests, compileall, Bandit, and diff checks.
4. Record modified files and final notes.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
TDD evidence: wrote `tldw_Server_API/tests/RPG/test_rpg_dice_checks.py` first. Initial focused pytest RED failed during collection with `ModuleNotFoundError: No module named 'tldw_Server_API.app.core.RPG.checks'`, confirming the missing dice/check slice before implementation.

Implementation notes: added bounded deterministic `DiceRoller`, `DiceRollResult`, `CheckResult`, a thin `checks.resolve_check()` delegation point, and bundled adapter-owned D20/Fate check resolvers. `checks.resolve_check()` delegates to `adapter.resolve_check(roller, payload)` and does not branch on mechanics tags.

Verification: `source /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/activate && python -m pytest tldw_Server_API/tests/RPG/test_rpg_dice_checks.py tldw_Server_API/tests/RPG/test_rpg_core_models_adapters.py -q` passed: 16 passed, 44 existing warnings. `python -m compileall` on touched Python files passed. `python -m bandit` on touched scope wrote `/tmp/bandit_task_12021.json` with 0 results and 0 errors; expected `# nosec` skips cover pytest asserts and non-cryptographic tabletop dice randomness. `git diff --check` passed.
Implemented deterministic-testable dice rolling and adapter-owned check resolution for d20/Fate mechanics. Verification: focused RED was confirmed by the worker; combined RPG focused tests passed (30 passed); compileall passed; Bandit on core RPG/DB touched scope reported 0 results; git diff --check passed.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented the RPG dice/check slice with deterministic, bounded dice rolling and adapter-owned D20/Fate check resolution. The core check helper remains a small delegation boundary so future systems can own mechanics in their adapters rather than expanding core branching.
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
