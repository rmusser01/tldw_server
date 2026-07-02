---
id: TASK-12092
title: Implement visual identity asset storage validation
status: Done
references:
- Docs/superpowers/plans/2026-07-01-visual-identity-expression-packs-implementation-plan.md
- Docs/superpowers/specs/2026-07-01-visual-identity-expression-packs-design.md
modified_files:
- tldw_Server_API/app/core/Visual_Identities/storage.py
- tldw_Server_API/tests/Visual_Identities/test_visual_identity_storage.py
- tldw_Server_API/app/core/DB_Management/db_path_utils.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement Stage 3 from the Visual Identity Expression Packs plan: visual identity asset storage directory helper, image validation/storage helpers, animated metadata extraction, AVIF capability gating, generated-file copy helper, and focused storage tests.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Stage 3 storage slice implemented with TDD.

Changed:
- Added `DatabasePaths.get_user_visual_identities_dir(user_id)` for `Databases/user_databases/{user_id}/visual_identities`.
- Added `Visual_Identities.storage` with content-hash original storage, MIME/header/Pillow validation, dimension/frame limits, animated metadata, first-frame previews, safe relpath resolution, and lower-level generated-file record copy helper.
- Added focused Visual Identity storage tests.

Red evidence:
- `source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m pytest -q tldw_Server_API/tests/Visual_Identities/test_visual_identity_storage.py` failed with 7 failures for the expected missing `DatabasePaths.get_user_visual_identities_dir` helper and missing `Visual_Identities.storage` module.

Green evidence:
- `source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m pytest -q tldw_Server_API/tests/Visual_Identities/test_visual_identity_storage.py` passed: 7 passed, 3 warnings.
- `source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m pytest -q tldw_Server_API/tests/Visual_Identities/test_expression_slots.py tldw_Server_API/tests/Visual_Identities/test_visual_identity_capabilities.py tldw_Server_API/tests/Visual_Identities/test_visual_identity_db.py tldw_Server_API/tests/Visual_Identities/test_visual_identity_storage.py` passed: 34 passed, 3 warnings.

Bandit:
- `source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m bandit -r tldw_Server_API/app/core/Visual_Identities tldw_Server_API/app/core/DB_Management/db_path_utils.py -f json -o /tmp/bandit_visual_identity_stage3.json` exited 0; report totals show 0 findings.

Concern:
- Implemented lower-level `copy_generated_file_record_to_expression_asset(...)` for Stage 11 reuse instead of wiring an AuthnzGeneratedFilesRepo-backed service/API helper in Stage 3, because generated-file lookup is async and belongs at the later service/API layer.
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
