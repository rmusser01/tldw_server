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
Added secure visual identity expression ZIP draft import parsing and Jobs integration, then addressed review findings before continuing. The importer now rejects raw backslashes, drive-letter path segments, encrypted/symlink directory entries, and replaces the visible draft asset set at the start of each owned import attempt so unsafe, malformed, missing, or oversized reimports cannot leave stale non-deleted assets. Plain directory entries remain allowed and recorded without false unsupported-entry errors. Validation summaries, slot maps, draft statuses, deterministic duplicate handling, and import job idempotency are covered by tests. Review-fix RED evidence: targeted regression subset failed with 6 failures before implementation; a follow-up plain-directory regression failed before the directory metadata fix; pre-open failure regressions for missing/oversized archives failed before cleanup ordering was corrected. GREEN evidence: regression subset 6 passed; plain-directory regression 1 passed; pre-open regressions 2 passed; Stage 5 archive/jobs/db tests 42 passed; full Visual_Identities suite 81 passed. Bandit: /tmp/bandit_visual_identity_stage5_review_fixes.json has 0 findings. git diff --check 7eee48dc66..HEAD is clean.
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
