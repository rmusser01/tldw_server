---
id: TASK-2366
title: Implement context integrity foundation for skills and prompts
status: In Progress
labels:
- security
- skills
- prompts
- implementation
priority: high
references:
- TASK-2363
- TASK-2365
modified_files:
- tldw_Server_API/app/services/startup_context_integrity.py
- tldw_Server_API/app/services/lifespan_startup_sequence.py
- tldw_Server_API/app/services/lifespan_shutdown_sequence.py
- tldw_Server_API/tests/Services/test_startup_context_integrity.py
- tldw_Server_API/tests/Services/test_lifespan_startup_sequence.py
- tldw_Server_API/tests/Services/test_lifespan_shutdown_sequence.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the reviewed context integrity foundation for skill and prompt files, including manifests, startup verification, resolver enforcement, integration chokepoints, admin reporting, tests, docs, and verification.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Docs/superpowers/plans/2026-06-25-context-integrity-foundation-implementation-plan.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Task 5 slice implemented the startup verification producer and lifecycle wiring. Startup now inventories prompt files, env prompt overrides, and discovered/test-injected user skill roots through rich InventoryResult APIs; loads optional signed HMAC manifests from environment; distinguishes no manifest from valid empty manifests; attaches ContextIntegrityBootState and ContextIntegrityResolver to app state; sets the global resolver; registers context_integrity.* startup warnings; and clears app/global resolver state during lifespan shutdown.

Verification recorded for Task 5:
- RED run: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Services/test_startup_context_integrity.py tldw_Server_API/tests/Services/test_lifespan_startup_sequence.py tldw_Server_API/tests/Services/test_lifespan_shutdown_sequence.py -v` failed as expected before implementation with missing startup_context_integrity imports and shutdown resolver cleanup assertion failure.
- Focused Services suite: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Services/test_startup_context_integrity.py tldw_Server_API/tests/Services/test_lifespan_startup_sequence.py tldw_Server_API/tests/Services/test_lifespan_shutdown_sequence.py -v` passed with `12 passed, 6 warnings`.
- Context_Integrity unit suite: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Context_Integrity/unit -v` passed with `116 passed, 6 warnings`.
- Formatter: `source .venv/bin/activate && python -m black ...` completed; 2 files reformatted.
- Bandit: `source .venv/bin/activate && python -m bandit -r tldw_Server_API/app/services/startup_context_integrity.py tldw_Server_API/app/services/lifespan_startup_sequence.py tldw_Server_API/app/services/lifespan_shutdown_sequence.py -f json -o /tmp/bandit_context_integrity_task5.json` exited 0 with zero findings.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
