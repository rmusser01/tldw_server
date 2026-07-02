---
id: TASK-12093
title: Implement visual identity ZIP import draft jobs
status: Done
labels:
- visual-identities
- expression-packs
- jobs
priority: High
references:
- Docs/superpowers/specs/2026-07-01-visual-identity-expression-packs-design.md
- Docs/superpowers/plans/2026-07-01-visual-identity-expression-packs-implementation-plan.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->

<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Stage 5: add secure ZIP import parsing and Jobs integration for visual identity draft assets. Implement archive validation, deterministic expression-slot mapping, duplicate reporting, draft status updates, idempotent job creation, tests, Bandit, and code review loops before continuing.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Stage 5 implementation complete. RED evidence: targeted pytest run initially failed at collection because Visual_Identities.archive_import and Visual_Identities.jobs were missing; repository helper slice failed with AttributeError for update_draft_validation_summary. GREEN evidence: source .venv/bin/activate && python -m pytest -q tldw_Server_API/tests/Visual_Identities/test_visual_identity_archive_import.py tldw_Server_API/tests/Visual_Identities/test_visual_identity_jobs.py --tb=short -> 12 passed, 3 warnings; source .venv/bin/activate && python -m pytest -q tldw_Server_API/tests/Visual_Identities/test_visual_identity_db.py -k 'update_draft_validation_summary' --tb=short -> 2 passed, 18 deselected, 3 warnings; source .venv/bin/activate && python -m pytest -q tldw_Server_API/tests/Visual_Identities --tb=short -> 71 passed, 3 warnings. Bandit: source .venv/bin/activate && python -m bandit -r tldw_Server_API/app/core/Visual_Identities tldw_Server_API/app/core/DB_Management/VisualIdentity_DB.py -f json -o /tmp/bandit_visual_identity_stage5.json -> 0 findings. git diff --check 7eee48dc66..HEAD -> clean.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added secure visual identity expression ZIP draft import parsing and Jobs integration. The importer validates archive size before opening, rejects unsafe archive metadata and paths, enforces entry/size/decompression limits, skips unsupported non-image entries with validation-summary errors, maps aliases/custom slots deterministically, reports duplicate expression-key mappings, stores valid assets through the existing storage validator, updates draft slot maps, persists validation summaries, and sets draft status to ready_for_review or failed. Added visual identity import job helpers with the required domain/job type/queue, deterministic payload hash, batch group, and idempotency behavior. Added a narrow owner-scoped draft validation-summary repository helper with tests.
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
